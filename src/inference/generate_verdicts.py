"""
Port of V2/src/generate_verdicts_from_retrieved_beliefs.ipynb core logic
for verdict + cleaned beliefs generation. Preserves prompt, model name,
API usage, and JSON parsing behavior.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


# Model config copied from V2 notebook
OPENAI_MODEL = "gpt-5-mini"
TEMPERATURE = 0.2

# Prompt path (source-of-truth prompt file in V2)
PROMPT_PATH = Path("V2/src/prompts/verdict_from_beliefs_prompt.txt")


def load_prompt_text(path: Optional[Path] = None) -> str:
    p = (path or PROMPT_PATH).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Prompt file not found: {p}")
    return p.read_text(encoding="utf-8")


def build_user_message(question: str, beliefs: List[str]) -> str:
    # Exact V2 behavior: concatenated lines without explicit newlines.
    lines = ["Question:", question.strip(), "", "Beliefs:"]
    for idx, b in enumerate(beliefs, start=1):
        lines.append(f"{idx}. {b.strip()}")
    return "".join(lines)


def extract_json(text: str) -> str:
    # Strip markdown fences (exact logic)
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    # Try direct
    if text.startswith("{") and text.endswith("}"):
        return text
    # Fallback: slice between first { and last }
    if "{" in text and "}" in text:
        s = text.index("{")
        e = text.rindex("}") + 1
        return text[s:e]
    return text


def call_llm(
    client: OpenAI,
    question: str,
    beliefs: List[str],
    prompt_text: str,
    retries: int = 1,
) -> Dict[str, Any]:
    system_msg = prompt_text
    user_msg = build_user_message(question, beliefs)
    for attempt in range(retries + 1):
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        content = resp.choices[0].message.content or ""
        try:
            payload = json.loads(extract_json(content))
            if not isinstance(payload, dict):
                raise ValueError("not a dict")
            if "verdict" not in payload or "beliefs" not in payload:
                raise ValueError("missing keys")
            payload["verdict"] = str(payload["verdict"]).strip()
            payload["beliefs"] = [str(b).strip() for b in payload.get("beliefs", []) if str(b).strip()]
            return payload
        except Exception:
            if attempt >= retries:
                raise
    raise RuntimeError("LLM parsing failed")


def generate_verdict_and_reasons(
    question: str,
    retrieved_beliefs: List[str],
    client: Optional[OpenAI] = None,
    prompt_path: Optional[Path] = None,
    retries: int = 1,
) -> Dict[str, Any]:
    """
    Wrapper to produce the verdict and cleaned beliefs using the exact V2 flow.
    Returns a dict with keys: verdict, beliefs.
    """
    if client is None:
        client = OpenAI()
    prompt_text = load_prompt_text(prompt_path)
    out = call_llm(client, question, retrieved_beliefs, prompt_text, retries=retries)
    return out


