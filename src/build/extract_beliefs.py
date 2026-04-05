#!/usr/bin/env python3
"""
Belief extraction (exact port of V2/src/extract_beliefs.ipynb logic), with V3 defaults.

- Preserves the same helper functions, retry behavior, message structure, and output schema.
- Uses the V3 prompt at src/prompts/belief_extraction_prompt.txt by default.
- When imported, `extract_beliefs(dataset, prompt_template, model)` runs the same loop over a prepared dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# OpenAI client (new SDK)
OpenAIClient = None
try:
    from openai import OpenAI  # type: ignore
    OpenAIClient = OpenAI
except Exception:
    OpenAIClient = None

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class QAPair:
    id: int
    question: str
    raw_answer: str


@dataclass
class ExtractionResult:
    verdict: str
    canonical_beliefs: List[str]


def _load_env_if_present() -> None:
    if os.environ.get("OPENAI_API_KEY"):
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


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_data_txt(path: Path) -> List[QAPair]:
    text = path.read_text(encoding="utf-8")
    header_re = re.compile(r"^\s*(\d+)\.\s*(.*)$", re.MULTILINE)
    matches = list(header_re.finditer(text))
    qa_pairs: List[QAPair] = []
    for i, m in enumerate(matches):
        qid = int(m.group(1))
        question_line = m.group(2).rstrip("\n\r ")
        answer_start = m.end()
        answer_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer_block = text[answer_start:answer_end]
        answer_str = answer_block.strip("\n\r ")
        qa_pairs.append(QAPair(id=qid, question=question_line, raw_answer=answer_str))
    if qa_pairs and all(qa_pairs[i].id <= qa_pairs[i + 1].id for i in range(len(qa_pairs) - 1)):
        return qa_pairs
    return sorted(qa_pairs, key=lambda x: x.id)

def _coerce_json_from_text(txt: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(txt)
    except Exception:
        pass
    if "{" in txt and "}" in txt:
        try:
            start = txt.index("{")
            end = txt.rindex("}") + 1
            return json.loads(txt[start:end])
        except Exception:
            return None
    return None


def _dedupe_and_clean_beliefs(beliefs: List[str]) -> List[str]:
    seen, cleaned = set(), []
    for b in beliefs:
        b2 = str(b).strip()
        if not b2:
            continue
        key = b2.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(b2)
    return cleaned


def validate_extraction(data: Dict[str, Any]) -> ExtractionResult:
    if not isinstance(data, dict):
        raise ValueError("Extraction must be a JSON object.")
    if 'verdict' not in data or 'canonical_beliefs' not in data:
        raise ValueError("Missing required keys: verdict, canonical_beliefs.")
    verdict = data.get('verdict')
    beliefs = data.get('canonical_beliefs')
    if not isinstance(verdict, str) or not verdict.strip():
        raise ValueError("verdict must be a non-empty string")
    if not isinstance(beliefs, list):
        raise ValueError("canonical_beliefs must be a list")
    beliefs_clean = _dedupe_and_clean_beliefs([str(b) for b in beliefs])
    if len(beliefs_clean) > 4:
        beliefs_clean = beliefs_clean[:4]
    if len(beliefs_clean) < 2:
        raise ValueError("canonical_beliefs must contain between 2 and 4 items after cleaning")
    return ExtractionResult(verdict=verdict.strip(), canonical_beliefs=beliefs_clean)


def build_prompt_with_input(template: str, question: str, raw_answer: str) -> str:
    return template.replace("{{question}}", question).replace("{{raw_answer}}", raw_answer)


def call_extractor(client: Any, model: str, prompt_with_input: str, attempt: int) -> Optional[Dict[str, Any]]:
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt_with_input},
                {"role": "user", "content": "Return only the strict JSON object now (no commentary, no fences)."},
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content if resp.choices else ""
    except Exception as e:
        print(f"[warn] API call failed on attempt {attempt}: {e}")
        return None
    return _coerce_json_from_text(content or "")


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def save_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
def extract_beliefs(dataset: List[Dict[str, Any]], prompt_template: str, model: str = DEFAULT_MODEL) -> List[Dict[str, Any]]:
    _load_env_if_present()
    client = None
    if OpenAIClient is not None and os.environ.get("OPENAI_API_KEY"):
        try:
            client = OpenAIClient()
        except Exception as e:
            print(f"[warn] Failed to init OpenAI client: {e}")
            client = None

    results: List[Dict[str, Any]] = []
    total = len(dataset)
    for idx, row in enumerate(dataset, start=1):
        qid = row.get('id')
        question = row.get('question', '')
        raw_answer = row.get('raw_answer', '')
        print(f"[info] Processing id={qid} ({idx}/{total})...")

        filled_prompt = build_prompt_with_input(prompt_template, question, raw_answer)
        extraction = None
        last_error = None

        for attempt in range(1, 3):
            obj = call_extractor(client, model, filled_prompt, attempt)
            if obj is None:
                last_error = "model call failed or unavailable"
                continue
            try:
                extraction = validate_extraction(obj)
                break
            except Exception as ve:
                last_error = str(ve)
                filled_prompt += "\n\nReturn ONLY strict JSON with keys verdict and canonical_beliefs (2-4 items)."


        if extraction is None:
            print(f"[warn] Falling back for id={qid}: {last_error}")
            results.append({
                "id": qid,
                "question": question,
                "raw_answer": raw_answer,
                "verdict": "",
                "beliefs": [],
                "_error": last_error or "unknown failure",
            })
            continue

        results.append({
            "id": qid,
            "question": question,
            "raw_answer": raw_answer,
            "verdict": extraction.verdict,
            "beliefs": extraction.canonical_beliefs,
        })

    return results


# CLI wrapper that mirrors the notebook's main cell (with V3 defaults)

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Extract verdict + canonical beliefs from raw QA text.")
    parser.add_argument("--input", type=str, default="data/raw/Data.txt", help="Path to the raw Data.txt file")
    parser.add_argument("--prompt", type=str, default="src/prompts/belief_extraction_prompt.txt", help="Path to the belief extraction prompt file")
    parser.add_argument("--outdir", type=str, default="data/processed", help="Output directory for JSON and JSONL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name (overrides OPENAI_MODEL)")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    prompt_path = Path(args.prompt)
    outdir = Path(args.outdir)

    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}")
        sys.exit(1)
    if not prompt_path.exists():
        print(f"[error] Prompt file not found: {prompt_path}")
        sys.exit(1)

    _load_env_if_present()
    prompt_template = load_prompt(prompt_path)
    qa_list = parse_data_txt(input_path)
    ensure_outdir(outdir)

    # Convert to dataset dicts and reuse the same loop
    dataset = [
        {"id": qa.id, "question": qa.question, "raw_answer": qa.raw_answer}
        for qa in qa_list
    ]

    results = extract_beliefs(dataset, prompt_template, model=args.model)
    out_json = outdir / "beliefs_extracted.json"
    out_jsonl = outdir / "beliefs_extracted.jsonl"
    save_json(out_json, results)
    save_jsonl(out_jsonl, results)
    print("[done] Extraction finished.")


if __name__ == "__main__":
    main()
