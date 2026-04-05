"""
Port of V2/src/mimic_generation.ipynb core generation logic for mimic answers.
Preserves:
- MODEL_NAME = "gpt-5"
- Responses API usage with instructions + input
- Persona injection and example retrieval format
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI


# Configuration copied from V2
MODEL_NAME: str = "gpt-5"
EMBEDDING_MODEL: str = "text-embedding-3-small"
K: int = 3
TEMPERATURE: Optional[float] = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_system_prompt() -> str:
    # Exact block from V2 (verbatim content preserved)
    return dedent(
        """
        You are a mimic renderer.

        Your job is to take a verdict and supporting beliefs and express them the way this person would actually say them out loud.

        Your top priority is faithful spoken imitation, not elegance, clarity, polish, coherence, or stylefulness.

        Hard constraints:
        - Do not change the meaning of the verdict.
        - Do not change the meaning of the beliefs.
        - Do not add new reasoning, beliefs, or facts.
        - Do not invent biography details.
        - Do not turn the answer into polished assistant prose.

        Priority order:
        1. Preserve the meaning of the verdict and beliefs.
        2. Match the raw reference answers as closely as possible in cadence, sentence flow, roughness, repetition, looseness, point of view, and level of polish.
        3. Match how the person opens, chains ideas, repeats themselves, wanders, and ends.
        4. Use persona_spec only as a secondary guide when the reference answers do not clearly decide something.

        Verdict rendering:
        - Do not restate the verdict in formal wording.
        - Do not begin with a neat summary sentence.
        - Render the verdict in the person's own natural phrasing, as if they are answering in real time.
        - The opening should sound immediate and spoken, not like a clean setup sentence.
        - It is okay for the opening to start slightly rough or mid-thought if that matches the references.

        Style rules:
        - The output should feel spoken, not written.
        - Prefer long, chained, slightly messy sentences over cleanly segmented ones when that matches the references.
        - It is okay to sound repetitive, a little awkward, slightly run-on, or mid-thought if that matches the references.
        - It is okay to branch mid-sentence, partially restart, or stack clauses with "and", "so", or "because" if that matches the references.
        - Preserve natural redundancy if that is part of how the person explains things.
        - If the person usually ends with practical advice or a corrective takeaway, do that instead of ending with a clean summary.
        - Match not only sentence flow but also the coherence level of the reference answers.
        - If the reference answers wander, circle back, or feel slightly loose, preserve that looseness.

        Critical anti-failure rules:
        - Do not improve the person's voice.
        - Do not make it sound smarter, sharper, smoother, more confident, more coherent, or more cohesive than the reference answers.
        - Do not make the response more polished than the reference answers.
        - Do not make the response more structured than the reference answers.
        - Do not make the response more grammatical than the reference answers if that changes the cadence.
        - Do not make the response more concise than the reference answers if that removes the person's natural repetition.
        - Do not use sharper, more idiomatic, more stylish, or more rhetorically impressive wording than the reference answers.
        - Do not replace plain or repetitive phrasing with tighter phrasing.
        - Do not introduce “good writing” behaviors like elegant transitions, cleaner summaries, or tighter sentence structure.
        - Do not use rhetorical flourishes, article-like phrasing, or analyst-style language.
        - Do not use dashes, semicolons, or overly neat punctuation unless the reference answers clearly do.
        - Do not use more basketball jargon than the reference answers use.
        - Do not add more filler words, slang, or colloquialisms than the reference answers use.
        - Do not compress several ideas into one cleaner sentence if the reference answers would normally spread them out more naturally.
        - Do not sound like a coach article, analyst writeup, or generic AI assistant.

        Reference-answer grounding:
        - If reference answers are provided, they are the strongest signal for style.
        - Match their level of polish, punctuation, sentence flow, repetition, awkwardness, looseness, and overall messiness.
        - Match how they begin and how they end.
        - Match their point of view: first person, second person, or third person.
        - Prefer their concrete, practical phrasing over generic paraphrases.
        - If the reference answers are rough, repetitive, or mildly ungrammatical, preserve that feel.
        - If there is any conflict between persona_spec and the reference answers, follow the reference answers.

        Constraints with references:
        - Do not copy their topic or specific content.
        - Do not quote them directly.
        - Use them to copy how the person talks, not what they said.

        Output format:
        - Return only the final natural-language response.
        - Default to one paragraph unless the references strongly suggest otherwise.

        When in doubt, choose the version that sounds more like the raw reference answers, even if it is less polished, less concise, less coherent, less cohesive, or slightly awkward.
        """
    ).strip()


def embed_texts(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in response.data]


def format_examples(examples: List[Dict[str, str]]) -> str:
    blocks: List[str] = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Reference Example {i}"
            f"Question: {ex['question']}"
            f"Raw Answer: {ex['raw_answer']}"
        )
    return "".join(blocks)


def retrieve_top_k(
    client: OpenAI,
    query: str,
    retrieval_dataset: List[Dict[str, Any]],
    question_embeddings: np.ndarray,
    *,
    query_id: Optional[Any] = None,
    query_question: Optional[str] = None,
    k: int = K,
) -> List[Dict[str, Any]]:
    query_embedding = embed_texts(client, [query])[0]
    sims = cosine_similarity([query_embedding], question_embeddings)[0]
    candidates: List[Tuple[int, float]] = []
    for i, sim in enumerate(sims):
        rec = retrieval_dataset[i]
        same_id = query_id is not None and rec.get("id") == query_id
        same_question = query_question is not None and rec.get("question") == query_question
        if same_id or same_question:
            continue
        candidates.append((i, float(sim)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in candidates[:k]]
    return [retrieval_dataset[i] for i in top_indices]


def get_output_text(response: Any) -> str:
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt
    try:
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    parts.append(getattr(content, "text", ""))
        if parts:
            return "\n".join([p for p in parts if p])
    except Exception:
        pass
    raise ValueError("Could not extract output text from response.")


def generate_mimic_answer(
    question: str,
    verdict: str,
    beliefs: List[str],
    persona_spec: Dict[str, Any],
    retrieval_dataset: List[Dict[str, Any]],
    question_embeddings: np.ndarray,
    *,
    client: Optional[OpenAI] = None,
) -> str:
    """
    Generate the mimic answer by reproducing the V2 responses.create flow.
    """
    if client is None:
        client = OpenAI()
    system_prompt = build_system_prompt()

    # Build user input (exact structure and wording ported)
    persona_block = json.dumps(persona_spec, indent=2, ensure_ascii=False)
    task_input = {
        "question": question,
        "verdict": verdict,
        "beliefs": beliefs or [],
        "counterweights": [],
    }
    task_block = json.dumps(task_input, indent=2, ensure_ascii=False)
    retrieved_examples = retrieve_top_k(
        client,
        query=question,
        query_id=None,
        query_question=question,
        k=K,
        retrieval_dataset=retrieval_dataset,
        question_embeddings=question_embeddings,
    )
    examples_block = format_examples(retrieved_examples)
    user_input = dedent(
        f"""
        You will be given:
        1. a persona specification JSON
        2. a task input JSON
        3. raw reference answers from this same person

        Your task is to answer the new question in a way that sounds like this person would actually say it.

        Important:
        - Preserve the meaning of the verdict and beliefs.
        - Match the raw reference answers especially in how they OPEN, how they CHAIN ideas, how REPETITIVE they are, how LOOSE they are, what POINT OF VIEW they use, and how they END.
        - Match their level of polish exactly.
        - Match their roughness, awkwardness, and coherence level.
        - If the references are rambling, rough, repetitive, or a little awkward, keep that feel.
        - Do not improve the writing.
        - Do not make it cleaner, tighter, smoother, smarter, more cohesive, or more idiomatic than the reference answers.
        - Do not add extra filler words, slang, or colloquialisms unless the references clearly use them.
        - Do not use tighter jargon than the references.
        - Do not copy the content of the reference answers. Use them only for style.

        PERSONA SPEC JSON:
        {persona_block}

        TASK INPUT JSON:
        {task_block}

        RAW REFERENCE ANSWERS:
        {examples_block}

        Generate the final response now.
        """
    ).strip()

    kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "instructions": system_prompt,
        "input": user_input,
        "text": {"format": {"type": "text"}},
    }
    if TEMPERATURE is not None:
        kwargs["temperature"] = float(TEMPERATURE)
    resp = client.responses.create(**kwargs)
    return get_output_text(resp)


