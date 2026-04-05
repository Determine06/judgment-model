#!/usr/bin/env python3

"""
parse_dataset.py

Purpose:
- Parse data/raw/Data.txt into a structured Python list of dicts.
- Save to data/processed/dataset.json (I/O handled by caller or the save helper here).

Logic preservation:
- Reuses the exact parsing routine `parse_data_txt` defined in src/build/extract_beliefs.py.
- Output rows keep keys: id, question, raw_answer (no inference fields added here).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json

# Import the existing parsing function from build.extract_beliefs
try:
    from .extract_beliefs import parse_data_txt  # type: ignore
except Exception:
    # Fallback for script execution when run as a top-level file
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))  # project root
    from src.build.extract_beliefs import parse_data_txt  # type: ignore

def parse_dataset(input_path: Path) -> List[Dict]:
    pairs = parse_data_txt(Path(input_path))
    # Keep only the raw fields produced by parsing (no model outputs here)
    out: List[Dict] = []
    for p in pairs:
        out.append({
            'id': p.id,
            'question': p.question,
            'raw_answer': p.raw_answer,
        })
    return out

def save_dataset_json(output_path: Path, dataset: List[Dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Parse Data.txt into dataset.json using existing logic')
    ap.add_argument('--input', type=str, required=True, help='Path to data/raw/Data.txt')
    ap.add_argument('--output', type=str, required=True, help='Path to data/processed/dataset.json')
    args = ap.parse_args()
    ds = parse_dataset(Path(args.input))
    save_dataset_json(Path(args.output), ds)
    print(f"[done] Wrote {len(ds)} rows to {args.output}")
