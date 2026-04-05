#!/usr/bin/env python3
"""
Cluster beliefs using the exact logic from notebooks/belief_clusters.ipynb.

Pipeline (preserved):
- Flatten beliefs from beliefs_extracted records
- Embed belief texts with OpenAI embeddings (text-embedding-3-large)
- Coarse agglomerative clustering by cosine distance with a tightness threshold
- NLI-based sub-clustering within coarse clusters using DeBERTa MNLI
- Emit final cluster list preserving fields and orderings used in the notebook

Environment:
- OPENAI_API_KEY must be set (loaded from .env/V3/.env if present)
- Optional: OPENAI_EMBEDDING_MODEL (defaults to text-embedding-3-large)
- Optional: NLI_MODEL (defaults to MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, List, Dict
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# --- Config copied from notebook ---
EMBEDDING_MODEL = os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-large')
BATCH_SIZE = 128
CLUSTER_TIGHTNESS = 'very_loose'  # one of: tight, medium, loose, very_loose
_THRESHOLDS = {
    'tight': 0.25,
    'medium': 0.35,
    'loose': 0.45,
    'very_loose': 0.55,
}
ENTAIL_THRESHOLD = 0.65  # retained from notebook (matrix printing path)
NLI_DISTANCE_THRESHOLD = 0.70

# --- Env loader (minimal) ---
def _load_env_if_present() -> None:
    if os.environ.get('OPENAI_API_KEY'):
        return
    for candidate in (Path('.env'), Path('V3/.env')):
        try:
            if candidate.exists():
                for line in candidate.read_text(encoding='utf-8').splitlines():
                    if not line or line.strip().startswith('#'):
                        continue
                    if '=' in line:
                        k, v = line.split('=', 1)
                        if k.strip() == 'OPENAI_API_KEY' and v.strip():
                            os.environ['OPENAI_API_KEY'] = v.strip()
                            return
        except Exception:
            continue

# --- OpenAI embeddings ---
OpenAIClient = None
try:
    from openai import OpenAI  # type: ignore
    OpenAIClient = OpenAI
except Exception:
    OpenAIClient = None


def embed_texts(texts: List[str], model: str, batch_size: int = 128) -> np.ndarray:
    if OpenAIClient is None:
        raise RuntimeError('openai package not available')
    _load_env_if_present()
    client = OpenAIClient()
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
    return np.array(out, dtype=np.float32)

# --- Coarse clustering ---

def run_agglomerative(emb: np.ndarray, tightness: str = 'medium') -> np.ndarray:
    thresh = _THRESHOLDS.get(tightness, 0.35)
    sim = cosine_similarity(emb)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    model = AgglomerativeClustering(
        n_clusters=None, metric='precomputed', linkage='average', distance_threshold=thresh
    )
    return model.fit_predict(dist)

# --- Flatten beliefs (same fields as notebook) ---

def flatten_beliefs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for row in items:
        qid = int(row.get('id'))
        beliefs = row.get('beliefs', []) or []
        for i, b in enumerate(beliefs):
            flat.append({
                'belief_id': f'{qid}_{i}',
                'question_id': qid,
                'belief': str(b).strip(),
            })
    return flat

# --- NLI-based subclustering ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
import torch  # type: ignore


def _load_nli_model():
    model_name = os.environ.get('NLI_MODEL', 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Determine entailment index
    entail_idx = None
    if getattr(model.config, 'label2id', None):
        for k, v in model.config.label2id.items():
            if 'entail' in str(k).lower():
                entail_idx = int(v)
                break
    if entail_idx is None and getattr(model.config, 'id2label', None):
        for i, name in model.config.id2label.items():
            if 'entail' in str(name).lower():
                entail_idx = int(i)
                break
    if entail_idx is None:
        entail_idx = 2
    return tokenizer, model, device, entail_idx


@torch.no_grad()
def entail_prob(tokenizer, model, device, entail_idx: int, premise: str, hypothesis: str) -> float:
    inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(device)
    logits = model(**inputs).logits[0].float()
    probs = torch.softmax(logits, dim=-1)
    return float(probs[entail_idx].item())


def mean_entail_matrix(tokenizer, model, device, entail_idx: int, texts: List[str]) -> np.ndarray:
    n = len(texts)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        M[i, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            p1 = entail_prob(tokenizer, model, device, entail_idx, texts[i], texts[j])
            p2 = entail_prob(tokenizer, model, device, entail_idx, texts[j], texts[i])
            m = 0.5 * (p1 + p2)
            M[i, j] = M[j, i] = m
    return M

# --- Public API ---

def cluster_beliefs(extracted_beliefs: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
    """Return final clustered belief structure as in the notebook.

    Input must be the list loaded from beliefs_extracted.jsonl (each row has
    id, question, raw_answer, verdict, beliefs).
    """
    # Flatten beliefs
    flat = flatten_beliefs(extracted_beliefs)
    texts = [b['belief'] for b in flat]
    if not texts:
        return []

    # Embeddings and coarse clustering
    emb = embed_texts(texts, EMBEDDING_MODEL, BATCH_SIZE)
    labels = run_agglomerative(emb, CLUSTER_TIGHTNESS)

    # Build coarse map idxs
    coarse_map: Dict[int, List[int]] = defaultdict(list)
    for idx_belief, lab in enumerate(labels):
        coarse_map[int(lab)].append(idx_belief)

    # NLI model
    tokenizer, model, device, entail_idx = _load_nli_model()

    final_clusters: List[Dict[str, Any]] = []
    next_id = 0

    # Iterate coarse clusters by size desc
    for coarse_id, idxs in sorted(coarse_map.items(), key=lambda kv: len(kv[1]), reverse=True):
        print(f"[cluster] coarse_id={coarse_id}, size={len(idxs)}, progress={next_id}")
        if len(idxs) < 4:
            # Keep as single final cluster
            beliefs_payload = [
                {
                    'belief_id': flat[i]['belief_id'],
                    'question_id': int(flat[i]['question_id']),
                    'belief': flat[i]['belief'],
                }
                for i in idxs
            ]
            final_clusters.append({
                'cluster_id': next_id,
                'source_coarse_cluster_id': int(coarse_id),
                'size': len(beliefs_payload),
                'beliefs': beliefs_payload,
            })
            next_id += 1
            continue

        # Size >= 4: NLI within this cluster
        cluster_texts = [texts[i] for i in idxs]
        M = mean_entail_matrix(tokenizer, model, device, entail_idx, cluster_texts)
        D = 1.0 - M
        np.fill_diagonal(D, 0.0)
        sub_model = AgglomerativeClustering(
            n_clusters=None, metric='precomputed', linkage='average', distance_threshold=NLI_DISTANCE_THRESHOLD
        )
        sub_labels = sub_model.fit_predict(D)

        # Group by sub_labels in size-desc order
        sub_map: Dict[int, List[int]] = defaultdict(list)
        for loc_idx, lab in enumerate(sub_labels):
            sub_map[int(lab)].append(loc_idx)
        ordered_subs = sorted(sub_map.items(), key=lambda kv: len(kv[1]), reverse=True)
        for _, loc_indices in ordered_subs:
            beliefs_payload = [
                {
                    'belief_id': flat[idxs[li]]['belief_id'],
                    'question_id': int(flat[idxs[li]]['question_id']),
                    'belief': flat[idxs[li]]['belief'],
                }
                for li in loc_indices
            ]
            final_clusters.append({
                'cluster_id': next_id,
                'source_coarse_cluster_id': int(coarse_id),
                'size': len(beliefs_payload),
                'beliefs': beliefs_payload,
            })
            next_id += 1

    return final_clusters


def load_beliefs_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_clustered(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (list, dict)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    else:
        raise TypeError('Provide a JSON-serializable clustered structure when saving.')
