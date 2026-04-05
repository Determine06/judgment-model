#!/usr/bin/env python3
"""
Build belief_map.json from clustered beliefs (ported from notebook).

Notebook behavior:
- The notebook saves the `final_clusters` structure directly as belief_map.json.
- We preserve that contract: the input clustered_beliefs should be the same
  list[dict] produced by cluster_beliefs(), and we return it unchanged.
"""
from __future__ import annotations
from typing import Any, Dict, List
import json
from pathlib import Path


def build_belief_map(clustered_beliefs: List[Dict[str, Any]], **kwargs: Any) -> List[Dict[str, Any]]:
    # Pass-through to match notebook output contract
    return clustered_beliefs


def save_belief_map(path: Path, belief_map: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(belief_map, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
