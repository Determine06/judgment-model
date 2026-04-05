#!/usr/bin/env python3
"""
Run the V3 build pipeline end-to-end (or as much as the environment allows).

Example:
    python src/build/run_build.py --input data/raw/Data.txt

Behavior:
- Parses dataset from --input and writes data/processed/dataset.json.
- Runs belief extraction using the exact logic ported from the notebook,
  writing beliefs_extracted.json and beliefs_extracted.jsonl.
- Then attempts clustering and belief map construction (may be NotImplemented),
  followed by persona extraction (may be NotImplemented).
- No silent overwrites unless --overwrite is provided.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict
import json

# Allow imports when executed as a script (python src/build/run_build.py)
if __package__ is None:  # running as a script
    sys.path.append(str(Path(__file__).resolve().parents[2]))  # project root

from src.build.parse_dataset import parse_dataset, save_dataset_json  # type: ignore
from src.build.extract_beliefs import extract_beliefs, save_json as save_beliefs_json, save_jsonl as save_beliefs_jsonl, load_prompt  # type: ignore
from src.build.cluster_beliefs import load_beliefs_jsonl, cluster_beliefs, save_clustered  # type: ignore
from src.build.build_belief_map import build_belief_map, save_belief_map  # type: ignore
from src.build.extract_persona import extract_persona, save_persona  # type: ignore


DEFAULT_INPUT = 'data/raw/Data.txt'
PROCESSED_DIR = Path('data/processed')
DATASET_JSON = PROCESSED_DIR / 'dataset.json'
BELIEFS_JSONL = PROCESSED_DIR / 'beliefs_extracted.jsonl'
BELIEFS_JSON = PROCESSED_DIR / 'beliefs_extracted.json'
BELIEF_MAP_JSON = PROCESSED_DIR / 'belief_map.json'
PERSONA_JSON = PROCESSED_DIR / 'persona_spec.json'
PROMPT_PATH = Path('src/prompts/belief_extraction_prompt.txt')


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description='V3 build orchestrator (Phase 3).')
    ap.add_argument('--input', type=str, default=DEFAULT_INPUT, help='Path to raw Data.txt')
    ap.add_argument('--overwrite', action='store_true', help='Allow overwriting existing processed artifacts')
    args = ap.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input not found: {input_path}")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: parse dataset
    print('[build] Parsing dataset...')
    dataset: List[Dict] = parse_dataset(input_path)
    if DATASET_JSON.exists() and not args.overwrite:
        print(f"[info] {DATASET_JSON} exists; not overwriting. Use --overwrite to replace.")
    else:
        save_dataset_json(DATASET_JSON, dataset)
        print(f"[ok] Wrote {DATASET_JSON} ({len(dataset)} rows)")

    # Step 2: extract beliefs (exact notebook logic)
    if BELIEFS_JSONL.exists() and not args.overwrite:
        print(f"[info] {BELIEFS_JSONL} exists; not overwriting. Use --overwrite to replace.")
    else:
        if not PROMPT_PATH.exists():
            print(f"[error] Prompt not found: {PROMPT_PATH}")
            sys.exit(1)
        print('[build] Extracting beliefs...')
        prompt_template = load_prompt(PROMPT_PATH)
        results = extract_beliefs(dataset, prompt_template)
        save_beliefs_json(BELIEFS_JSON, results)
        save_beliefs_jsonl(BELIEFS_JSONL, results)
        print(f"[ok] Wrote {BELIEFS_JSON} and {BELIEFS_JSONL}")

    # Step 3: cluster beliefs
    try:
        print('[build] Clustering beliefs...')
        extracted = load_beliefs_jsonl(BELIEFS_JSONL)
        print('[info] Loaded beliefs. Running clustering algorithm')
        clustered = cluster_beliefs(extracted)
        clustered_out = PROCESSED_DIR / 'clustered_beliefs.json'
        if clustered_out.exists() and not args.overwrite:
            print(f"[info] {clustered_out} exists; not overwriting. Use --overwrite to replace.")
        else:
            save_clustered(clustered_out, clustered)
            print(f"[ok] Wrote {clustered_out}")

        # Step 4: build belief map
        print('[build] Building belief_map.json...')
        belief_map = build_belief_map(clustered)
        if BELIEF_MAP_JSON.exists() and not args.overwrite:
            print(f"[info] {BELIEF_MAP_JSON} exists; not overwriting. Use --overwrite to replace.")
        else:
            save_belief_map(BELIEF_MAP_JSON, belief_map)
            print(f"[ok] Wrote {BELIEF_MAP_JSON}")
    except NotImplementedError as nie:
        print(f"[skip] clustering/map: {nie}")

    # Step 5: persona extraction
    try:
        print('[build] Extracting persona...')
        # Use the parsed dataset as input
        if not DATASET_JSON.exists():
            dataset = parse_dataset(input_path)
        else:
            dataset = json.loads(DATASET_JSON.read_text(encoding='utf-8'))
        persona = extract_persona(dataset)
        if PERSONA_JSON.exists() and not args.overwrite:
            print(f"[info] {PERSONA_JSON} exists; not overwriting. Use --overwrite to replace.")
        else:
            save_persona(PERSONA_JSON, persona)
            print(f"[ok] Wrote {PERSONA_JSON}")
    except NotImplementedError as nie:
        print(f"[skip] persona: {nie}")

    print('[done] Build orchestrator finished.')


if __name__ == '__main__':
    main()
