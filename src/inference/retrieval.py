"""
Ported retrieval logic (cluster-aware belief retrieval with suppression)
from V2/src/retrieval_belief_cluster_suppression.ipynb.

Hard constraints preserved:
- Embedding model: text-embedding-3-large
- Question-aware retrieval text when question lookup is provided
- Cosine similarity with optional L2 normalization
- Delta-from-top thresholding + global score floor
- Cluster suppression allowing up to 2 per cluster with redundancy check
- Final cap on beliefs

This module exposes minimal reusable functions without redesigning behavior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# OpenAI embeddings
from openai import OpenAI


# === Config copied from V2 notebook ===
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
NORMALIZE_EMBEDDINGS = True
EMBED_BATCH_SIZE = 128

DELTA_FROM_TOP = 0.1
MIN_SCORE_FLOOR = 0.40

MAX_FINAL_BELIEFS = 8
MAX_BELIEFS_PER_CLUSTER = 2

ALLOW_SECOND_BELIEF_PER_CLUSTER = True
SECOND_BELIEF_SCORE_MARGIN = 0.02
INTRA_CLUSTER_REDUNDANCY_THRESHOLD = 0.92

# Note: V2 notebook uses a global USE_QUESTION_AWARE_TEXT flag. We surface
# this as a parameter in flatten/load helpers; default True to preserve behavior.


@dataclass
class FlatBelief:
    cluster_id: int
    source_coarse_cluster_id: int
    cluster_size: int
    belief_id: str
    question_id: int
    belief_text: str
    retrieval_text: str


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_question_lookup(path: Optional[Path]) -> Dict[int, str]:
    """
    Robust question lookup loader supporting multiple formats, matching V2 logic.
    Accepted:
    - dict: {question_id: question_text}
    - list of dicts with keys like ('question_id'|'id'|'qid') and ('question'|'question_text'|'text')
    Returns: {int(question_id): str(question_text)}
    """
    if path is None:
        return {}
    p = path.expanduser().resolve()
    if not p.exists():
        print(f"[warn] Question lookup not found: {p}")
        return {}
    obj = _load_json(p)
    qmap: Dict[int, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                qmap[int(k)] = str(v)
            except Exception:
                pass
    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            qid = row.get("question_id", row.get("id", row.get("qid")))
            qtxt = row.get("question", row.get("question_text", row.get("text")))
            if qid is not None and qtxt is not None:
                try:
                    qmap[int(qid)] = str(qtxt)
                except Exception:
                    pass
    print(f"Loaded question lookup entries: {len(qmap)} from {p}")
    return qmap


def _make_retrieval_text(
    belief_text: str,
    question_id: int,
    question_lookup: Dict[int, str],
    use_question_aware: bool,
) -> str:
    if use_question_aware and question_id in question_lookup:
        q = question_lookup[question_id].strip()
        return f"Question: {q} Belief: {belief_text.strip()}"
    return belief_text.strip()


def flatten_belief_map(
    belief_map: List[Dict[str, Any]],
    question_lookup: Dict[int, str],
    use_question_aware: bool = True,
) -> pd.DataFrame:
    """
    Flatten clustered belief map into a DataFrame with retrieval_text field.
    Mirrors V2's flatten_belief_map implementation.
    """
    rows: List[FlatBelief] = []
    for cl in belief_map:
        cid = int(cl["cluster_id"])
        scid = int(cl.get("source_coarse_cluster_id", -1))
        csize = int(cl.get("size", len(cl.get("beliefs", []))))
        for b in cl.get("beliefs", []):
            belief_id = str(b["belief_id"])            
            qid = int(b["question_id"])              
            belief_text = str(b["belief"]).strip()
            rtext = _make_retrieval_text(belief_text, qid, question_lookup, use_question_aware)
            rows.append(FlatBelief(cid, scid, csize, belief_id, qid, belief_text, rtext))
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.reset_index(inplace=True)
    df.rename(columns={"index": "emb_idx"}, inplace=True)
    return df


def load_flattened_beliefs(
    belief_map_path: Path,
    question_lookup_path: Optional[Path] = None,
    use_question_aware_text: bool = True,
) -> pd.DataFrame:
    """
    Load belief_map.json and optional question lookup, then flatten to a DF.
    This preserves the V2 logic for question-aware retrieval text.
    """
    bp = belief_map_path.expanduser().resolve()
    if not bp.exists():
        raise FileNotFoundError(f"Belief map not found: {bp}")
    belief_map = _load_json(bp)
    qmap = load_question_lookup(question_lookup_path) if question_lookup_path else {}
    df = flatten_belief_map(belief_map, qmap, use_question_aware=use_question_aware_text)
    return df


def embed_flat_beliefs(
    flat_df: pd.DataFrame,
    client: Optional[OpenAI] = None,
    model_name: str = OPENAI_EMBEDDING_MODEL,
    normalize: bool = NORMALIZE_EMBEDDINGS,
    batch_size: int = EMBED_BATCH_SIZE,
) -> np.ndarray:
    """
    Encode flat_df['retrieval_text'] using OpenAI embeddings, with optional L2 norm.
    Exactly mirrors the V2 embedding approach.
    """
    if client is None:
        client = OpenAI()
    texts = flat_df["retrieval_text"].tolist()
    embs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model_name, input=batch)
        embs.extend([d.embedding for d in resp.data])
    belief_embeddings = np.array(embs, dtype=np.float32)
    if normalize and len(belief_embeddings) > 0:
        norms = np.linalg.norm(belief_embeddings, axis=1, keepdims=True) + 1e-12
        belief_embeddings = belief_embeddings / norms
    return belief_embeddings


def embed_query(
    question: str, client: OpenAI, model_name: str = OPENAI_EMBEDDING_MODEL, normalize: bool = True
) -> np.ndarray:
    resp = client.embeddings.create(model=model_name, input=[question])
    q_emb = np.array(resp.data[0].embedding, dtype=np.float32)
    if normalize:
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-12)
    return q_emb


def score_all_beliefs(query_emb: np.ndarray, belief_embeddings: np.ndarray, flat_df: pd.DataFrame) -> pd.DataFrame:
    if NORMALIZE_EMBEDDINGS:
        scores = belief_embeddings @ query_emb
    else:
        scores = cosine_similarity(belief_embeddings, query_emb.reshape(1, -1))[:, 0]
    out = flat_df.copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def threshold_beliefs(scored_df: pd.DataFrame, delta_from_top: float, min_score_floor: float) -> pd.DataFrame:
    if scored_df.empty:
        return scored_df
    top = float(scored_df["score"].iloc[0])
    kept = scored_df[
        (scored_df["score"] >= top - delta_from_top) & (scored_df["score"] >= min_score_floor)
    ].copy()
    if kept.empty:
        kept = scored_df.head(1).copy()
    kept = kept.sort_values("score", ascending=False).reset_index(drop=True)
    return kept


def suppress_within_clusters(
    retained_df: pd.DataFrame,
    full_embedding_matrix: np.ndarray,
    flat_df: pd.DataFrame,
    allow_second_belief: bool = True,
    second_belief_margin: float = 0.02,
    intra_cluster_redundancy_threshold: float = 0.92,
    max_beliefs_per_cluster: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster-level suppression, exactly as in V2.
    Keeps top belief per cluster; optionally keeps a second if margin + low redundancy.
    """
    if retained_df.empty:
        return retained_df.copy(), retained_df.copy(), pd.DataFrame()

    kept_rows: List[pd.Series] = []
    suppressed_rows: List[pd.Series] = []
    debug_rows: List[Dict[str, Any]] = []

    for cid, group in retained_df.groupby("cluster_id", sort=False):
        grp = group.sort_values("score", ascending=False).reset_index(drop=True)
        top = grp.iloc[0]
        kept = [top]
        second = grp.iloc[1] if len(grp) > 1 else None
        sim12: Optional[float] = None

        if second is not None and allow_second_belief and max_beliefs_per_cluster >= 2:
            if second["score"] >= top["score"] - second_belief_margin:
                idx1 = int(top["emb_idx"])   # indexes into full embedding matrix
                idx2 = int(second["emb_idx"]) 
                v1 = full_embedding_matrix[idx1]
                v2 = full_embedding_matrix[idx2]
                if NORMALIZE_EMBEDDINGS:
                    sim12 = float(v1 @ v2)
                else:
                    sim12 = float(cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])
                if sim12 < intra_cluster_redundancy_threshold:
                    kept.append(second)
                else:
                    suppressed_rows.append(second)
            else:
                suppressed_rows.append(second)

        for i in range(2, len(grp)):
            suppressed_rows.append(grp.iloc[i])

        debug_rows.append(
            {
                "cluster_id": cid,
                "candidate_count": len(grp),
                "kept_belief_ids": [str(r["belief_id"]) for r in kept],
                # Iterate rows explicitly to avoid iterating over column names
                "suppressed_belief_ids": [
                    str(r["belief_id"]) for _, r in grp.iloc[len(kept) :].iterrows()
                ],
                "top_score": float(top["score"]),
                "second_score": float(second["score"]) if second is not None else None,
                "sim_top_second": sim12,
            }
        )

        kept_rows.extend(kept)

    kept_df = (
        pd.DataFrame(kept_rows).reset_index(drop=True) if kept_rows else pd.DataFrame(columns=retained_df.columns)
    )
    suppressed_df = (
        pd.DataFrame(suppressed_rows).reset_index(drop=True)
        if suppressed_rows
        else pd.DataFrame(columns=retained_df.columns)
    )
    debug_df = pd.DataFrame(debug_rows)
    return kept_df, suppressed_df, debug_df


def finalize_results(kept_df: pd.DataFrame, max_final_beliefs: int) -> pd.DataFrame:
    if kept_df.empty:
        return kept_df
    out = kept_df.sort_values("score", ascending=False).reset_index(drop=True)
    if len(out) > max_final_beliefs:
        out = out.iloc[:max_final_beliefs].copy()
        out.reset_index(drop=True, inplace=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def pretty_print_results(df: pd.DataFrame, cols: Optional[List[str]] = None, max_rows: int = 20) -> None:
    if cols is None:
        cols = ["rank", "score", "cluster_id", "belief_id", "question_id", "belief_text"]
    if not df.empty:
        print(df[cols].head(max_rows).to_string(index=False))
    else:
        print("<empty>")


def retrieve_for_question(
    query_question: str,
    client: OpenAI,
    flat_df: pd.DataFrame,
    belief_embeddings: np.ndarray,
    *,
    delta_from_top: float = DELTA_FROM_TOP,
    min_score_floor: float = MIN_SCORE_FLOOR,
    allow_second_belief: bool = ALLOW_SECOND_BELIEF_PER_CLUSTER,
    second_belief_margin: float = SECOND_BELIEF_SCORE_MARGIN,
    intra_cluster_redundancy_threshold: float = INTRA_CLUSTER_REDUNDANCY_THRESHOLD,
    max_beliefs_per_cluster: int = MAX_BELIEFS_PER_CLUSTER,
    max_final_beliefs: int = MAX_FINAL_BELIEFS,
    print_debug: bool = False,
) -> Dict[str, Any]:
    """
    Single-query retrieval exactly matching V2 behavior.
    Returns dict with intermediate dataframes and final selection.
    """
    # Warn if question-aware text isn’t active
    if len(flat_df) and "Question:" not in str(flat_df["retrieval_text"].iloc[0]):
        # In V2 this warning triggers when USE_QUESTION_AWARE_TEXT=True but lookup missing
        print("[warn] Question lookup not provided; proceeding with belief-only retrieval text.")

    q_emb = embed_query(query_question, client, OPENAI_EMBEDDING_MODEL, normalize=NORMALIZE_EMBEDDINGS)
    scored_df = score_all_beliefs(q_emb, belief_embeddings, flat_df)
    retained_df = threshold_beliefs(scored_df, delta_from_top, min_score_floor)

    kept_df, suppressed_df, cluster_debug_df = suppress_within_clusters(
        retained_df,
        belief_embeddings,
        flat_df,
        allow_second_belief=allow_second_belief,
        second_belief_margin=second_belief_margin,
        intra_cluster_redundancy_threshold=intra_cluster_redundancy_threshold,
        max_beliefs_per_cluster=max_beliefs_per_cluster,
    )

    final_df = finalize_results(kept_df, max_final_beliefs)

    if print_debug:
        print("Retained before suppression:")
        pretty_print_results(retained_df)
        print("Kept after suppression:")
        pretty_print_results(final_df)
        print("Suppressed:")
        if not suppressed_df.empty:
            print(suppressed_df[["belief_id", "cluster_id", "score"]].to_string(index=False))
        else:
            print("<none>")

    return {
        "query": query_question,
        "scored_df": scored_df,
        "retained_df_before_suppression": retained_df,
        "kept_df_after_suppression": kept_df,
        "suppressed_df": suppressed_df,
        "final_df": final_df,
        "cluster_debug_df": cluster_debug_df,
    }

