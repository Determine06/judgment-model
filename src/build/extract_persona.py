#!/usr/bin/env python3
"""
Persona extraction (ported from notebooks/persona_extraction.ipynb).

Behavior preserved:
- Formats examples into a prompt block
- Uses the exact SYSTEM_PROMPT and USER_PROMPT_TEMPLATE
- Calls OpenAI Responses API with the same parameters
- Parses strict JSON (with fallback to first {...} block)
- Saves persona_spec.json as pretty JSON
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import json
import re
from textwrap import dedent

# OpenAI client (modern)
from openai import OpenAI  # type: ignore

MODEL_NAME: str = 'gpt-5'  # from notebook
TEMPERATURE: float = 0.0


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


def normalize_for_persona(dataset: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # Map our dataset rows (id, question, raw_answer, ...) to {question, answer}
    ex: List[Dict[str, str]] = []
    for r in dataset:
        q = str(r.get('question', '')).strip()
        a = str(r.get('raw_answer', '')).strip()
        if q and a:
            ex.append({'question': q, 'answer': a})
    if not ex:
        raise ValueError('No Q&A pairs found for persona extraction')
    return ex


def format_examples_for_prompt(examples: List[Dict[str, str]], max_examples: Optional[int] = None) -> str:
    if isinstance(max_examples, int) and max_examples > 0:
        examples = examples[:max_examples]
    blocks = []
    for idx, ex in enumerate(examples, 1):
        blocks.append(f"Example {idx}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}")
    return "\n\n".join(blocks)


def extract_json_from_response(text: str) -> Dict[str, Any]:
    if not text or not isinstance(text, str):
        raise ValueError('Empty response text.')
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find('{')
        end = s.rfind('}')
        if start != -1 and end != -1 and end > start:
            snippet = s[start:end+1]
            return json.loads(snippet)
        raise


# Persona extraction prompt (exact system and user prompts)
SYSTEM_PROMPT = dedent(
    """
You are extracting a communication persona from a set of question-answer pairs.

Your goal is NOT to summarize what the person is saying.
Your goal is to identify HOW the person consistently communicates and reasons.

You must infer stable patterns across multiple answers.

Focus only on observable communication patterns.

Return a JSON object that matches this exact schema:

{
  "persona_spec": {
    "identity_core": {
      "tone": {
        "formality": "informal | neutral | formal",
        "confidence": "low | medium | high",
        "directness": "low | medium | high",
        "warmth": "low | medium | high"
      },
      "discourse_structure": {
        "opens_with_verdict": true,
        "typical_reason_count": "1 | 2 | 3 | variable",
        "preferred_format": "paragraph | structured | mixed"
      },
      "reasoning_presentation": {
        "primary_mode": "causal | comparative | prioritization | descriptive",
        "secondary_mode": "causal | comparative | prioritization | descriptive",
        "focus": "key_factors | full_explanation",
        "tradeoff_style": "explicit | implicit | minimal"
      },
      "decision_style": {
        "optimizes_for": ["clarity", "practicality", "correctness", "completeness", "efficiency"],
        "uncertainty_behavior": "brief_acknowledgment_then_continue | brief_acknowledgment_then_commit | rarely_acknowledges_uncertainty | heavily_hedged",
        "hedging_level": "low | medium | high"
      }
    },
    "rhetorical_habits": {
      "common_transitions": [],
      "sentence_style": {
        "length": "short | medium | long",
        "cadence": "choppy | controlled | flowing | run-on and continuous"
      },
      "lexical_style": {
        "contractions": "rare | occasional | frequent",
        "jargon": "low | moderate | high",
        "intensity": "low | moderate | high"
      },
      "punctuation_style": {
        "dashes": "rare | occasional | frequent",
        "lists": "rare | occasional | frequent"
      }
    },
    "guardrails": {
      "preserve_verdict": true,
      "preserve_beliefs": true,
      "no_new_reasoning": true,
      "no_biography_roleplay": true,
      "avoid_caricature": true
    }
  }
}

Rules:
1. Only include patterns that appear consistently across multiple answers.
2. Do not include one-off behaviors.
3. Do not summarize content or opinions.
4. Do not infer vague traits like "smart", "analytical", or "thoughtful".
5. Focus only on stable, observable communication patterns.
6. If a trait is unclear, choose the most conservative reasonable option.
7. Only include repeated phrases in `common_transitions` if they genuinely recur multiple times.
8. Keep the output compact, specific, and schema-valid.
9. Output valid JSON only. No markdown fences. No commentary.
    """
)

USER_PROMPT_TEMPLATE = dedent(
    """
Below is a set of question-answer pairs.

Analyze the answers and derive the persona_spec JSON.

Q&A DATA:
{formatted_examples}
    """
)


def extract_persona(dataset: List[Dict[str, Any]], max_examples: Optional[int] = None) -> Dict[str, Any]:
    _load_env_if_present()
    client = OpenAI()
    examples = normalize_for_persona(dataset)
    formatted = format_examples_for_prompt(examples, max_examples)
    user_prompt = USER_PROMPT_TEMPLATE.format(formatted_examples=formatted)

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
        text={
            'format': {
                'type': 'json_object'
            }
        },
    )
    # Attempt to read unified output_text first
    raw_model_output = getattr(response, 'output_text', None)
    if raw_model_output is None:
        # Fallback reconstruction (not changing semantics)
        parts: List[str] = []
        for item in getattr(response, 'output', []) or []:
            for content in getattr(item, 'content', []) or []:
                if getattr(content, 'type', '') == 'output_text':
                    parts.append(getattr(content, 'text', ''))
        raw_model_output = '\n'.join(parts) if parts else None

    if raw_model_output is None:
        raise RuntimeError('Empty persona response from model')

    persona_obj = extract_json_from_response(raw_model_output)
    if 'persona_spec' not in persona_obj:
        raise KeyError("Missing top-level 'persona_spec' in model output.")
    return persona_obj


def save_persona(path: Path, persona: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(persona, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
