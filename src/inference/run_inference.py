from __future__ import annotations

"""
Thin orchestrator for Phase 4.
Reproduces the V2 notebook flow using saved artifacts under V3/data/processed.

Flow:
questions.json -> retrieval over belief_map -> verdict generation -> mimic generation -> output json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Import guard to support both package mode (-m) and direct script execution.
try:
    # Package-relative (when executed via -m V3.src.inference.run_inference)
    from .retrieval import (
        load_flattened_beliefs,
        embed_flat_beliefs,
        retrieve_for_question,
    )
    from .generate_verdicts import generate_verdict_and_reasons
    from .generate_mimic import (
        load_json as load_json_file,
        generate_mimic_answer,
        embed_texts as embed_texts_for_mimic,
    )
except Exception:
    # Direct script execution: load sibling modules by file path
    import importlib.util
    import types
    from pathlib import Path as _P

    _DIR = _P(__file__).resolve().parent

    def _load_local(name: str, filename: str) -> types.ModuleType:
        path = _DIR / filename
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {filename}")
        mod = importlib.util.module_from_spec(spec)
        import sys as _sys
        _sys.modules[name] = mod  # ensure availability during execution (dataclasses, etc.)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod

    _retrieval = _load_local("retrieval_local", "retrieval.py")
    _verdicts = _load_local("generate_verdicts_local", "generate_verdicts.py")
    _mimic = _load_local("generate_mimic_local", "generate_mimic.py")

    load_flattened_beliefs = _retrieval.load_flattened_beliefs
    embed_flat_beliefs = _retrieval.embed_flat_beliefs
    retrieve_for_question = _retrieval.retrieve_for_question

    generate_verdict_and_reasons = _verdicts.generate_verdict_and_reasons

    load_json_file = _mimic.load_json
    generate_mimic_answer = _mimic.generate_mimic_answer
    embed_texts_for_mimic = _mimic.embed_texts


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("Input must be a JSON list of objects with 'question'.")
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Path to questions JSON list")
    ap.add_argument("--output", required=True, type=str, help="Path to output JSON")
    ap.add_argument("--belief-map", default="V3/data/processed/belief_map.json", type=str)
    ap.add_argument("--question-lookup", default="V3/data/processed/beliefs_extracted.json", type=str)
    ap.add_argument("--persona-spec", default="V3/data/processed/persona_spec.json", type=str)
    ap.add_argument("--retrieval-dataset", default="V3/data/processed/dataset.json", type=str)
    ap.add_argument("--overwrite", action="store_true", help="Allow overwriting output")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    client = OpenAI()

    # Load artifacts for retrieval
    flat_df = load_flattened_beliefs(Path(args.belief_map), Path(args.question_lookup), use_question_aware_text=True)
    belief_embeddings = embed_flat_beliefs(flat_df, client=client)

    # Load persona + retrieval dataset for mimic
    persona_raw = load_json_file(Path(args.persona_spec))
    persona_spec = persona_raw.get("persona_spec", persona_raw)
    retrieval_raw = load_json_file(Path(args.retrieval_dataset))
    # normalize retrieval dataset into list of {question, raw_answer}
    retrieval_dataset: List[Dict[str, Any]] = []
    if isinstance(retrieval_raw, list):
        records = retrieval_raw
    elif isinstance(retrieval_raw, dict):
        if isinstance(retrieval_raw.get("data"), list):
            records = retrieval_raw["data"]
        else:
            # fallback: first list field
            records = None
            for k, v in retrieval_raw.items():
                if isinstance(v, list):
                    records = v
                    break
            if records is None:
                raise ValueError("Retrieval dataset must be a list or dict with a top-level list.")
    else:
        raise ValueError("Invalid retrieval dataset JSON")
    for r in records:
        if not isinstance(r, dict):
            continue
        q = r.get("question")
        a = r.get("raw_answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            item = {"question": q.strip(), "raw_answer": a.strip()}
            if r.get("id") is not None:
                item["id"] = r.get("id")
            retrieval_dataset.append(item)
    if not retrieval_dataset:
        raise ValueError("No usable (question, raw_answer) pairs found in retrieval dataset.")
    # Local import to keep module import side-effects minimal
    import numpy as np  # noqa: WPS433
    question_embeddings = np.array(
        embed_texts_for_mimic(client, [d["question"] for d in retrieval_dataset])
    )

    # Load input questions
    questions = read_json_list(in_path)

    outputs: List[Dict[str, Any]] = []
    for row in questions:
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        # Retrieval
        ret = retrieve_for_question(q, client, flat_df, belief_embeddings, print_debug=args.debug)
        final_df = ret["final_df"]
        retrieved_beliefs: List[str] = (
            final_df["belief_text"].tolist() if final_df is not None and not final_df.empty else []
        )

        # Verdict
        verdict_obj = generate_verdict_and_reasons(q, retrieved_beliefs, client=client)
        verdict = verdict_obj.get("verdict", "")
        cleaned_beliefs = verdict_obj.get("beliefs", [])

        # Mimic
        mimic_text = generate_mimic_answer(
            question=q,
            verdict=verdict,
            beliefs=cleaned_beliefs,
            persona_spec=persona_spec,
            retrieval_dataset=retrieval_dataset,
            question_embeddings=question_embeddings,
            client=client,
        )

        outputs.append(
            {
                "question": q,
                "verdict": verdict,
                "reasons": cleaned_beliefs,
                "mimic_answer": mimic_text,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved outputs to {out_path}")


if __name__ == "__main__":
    main()
