"""LLM-as-judge — rubric scoring for benchmark conversations.

Evaluates conversations using 5 binary rubrics via any OpenAI-compatible LLM.
Each rubric returns 0 (fail) or 1 (pass); total score per conversation is 0-5.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think>.*", re.DOTALL)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluation judge for e-commerce shopping conversations.
You evaluate conversations between a customer and an AI shopping assistant.

Score each conversation on these 5 binary rubrics (0 = fail, 1 = pass):

1. **Intent Understanding**: Did the assistant correctly understand what the customer wanted?
   - PASS (1): The assistant demonstrated understanding of the customer's shopping intent, product preferences, and constraints.
   - FAIL (0): The assistant misunderstood the request, ignored stated preferences, or responded to something the customer didn't ask for.

2. **Clarification Behaviour**: Did the assistant ask appropriate clarifying questions when needed?
   - PASS (1): The assistant asked relevant clarifying questions to narrow down preferences OR the customer's request was already clear enough that no clarification was needed.
   - FAIL (0): The assistant asked no questions when the request was ambiguous, OR asked irrelevant/excessive questions.

3. **Recommendation Quality**: Were the product recommendations relevant and helpful?
   - PASS (1): Recommended products matched the customer's stated preferences (style, size, budget, color, occasion, etc.).
   - FAIL (0): Recommendations were irrelevant, generic, or ignored customer constraints.

4. **Add-to-Cart Correctness**: Was the cart operation handled correctly?
   - PASS (1): Items were added to cart only when the customer explicitly requested it, with correct product details. OR no cart action was needed and none was attempted.
   - FAIL (0): Cart was modified without customer consent, wrong item was added, or the assistant falsely claimed items were added without actually doing so.

5. **Grounding / Coherence / Non-Hallucination**: Did the assistant stay grounded in actual catalog data?
   - PASS (1): All product information (names, prices, sizes, availability) came from actual tool/catalog results. The assistant did not invent products or details.
   - FAIL (0): The assistant hallucinated product information, made up items not in the catalog, or provided incorrect details that contradict tool results.

Return your evaluation as strict JSON:
{
  "intent_understanding": {"score": 0 or 1, "reason": "brief explanation"},
  "clarification": {"score": 0 or 1, "reason": "brief explanation"},
  "recommendation": {"score": 0 or 1, "reason": "brief explanation"},
  "add_to_cart": {"score": 0 or 1, "reason": "brief explanation"},
  "grounding": {"score": 0 or 1, "reason": "brief explanation"},
  "total": <sum of all scores 0-5>
}
"""


def _prepare_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize conversation into strict user/assistant alternation for the judge."""
    filtered: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        tool_calls = m.get("tool_calls", [])

        if role == "assistant":
            clean_content = _THINK_RE.sub("", content)
            clean_content = _THINK_UNCLOSED_RE.sub("", clean_content).strip()
            parts: list[str] = []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args = fn.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    parts.append(f"[Tool call: {name}({args})]")
            if clean_content:
                parts.append(clean_content)
            combined = "\n".join(parts)
            if combined:
                filtered.append({"role": "assistant", "content": combined})

        elif role == "tool":
            tool_content = content if content else "(no result)"
            if filtered and filtered[-1]["role"] == "assistant":
                filtered[-1]["content"] += f"\n[Tool result: {tool_content}]"

        elif role == "user":
            if content.strip():
                filtered.append({"role": "user", "content": content.strip()})

    # Merge consecutive same-role messages
    merged: list[dict[str, Any]] = []
    for m in filtered:
        if merged and m["role"] == merged[-1]["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(dict(m))

    # Ensure starts with user, ends with assistant
    while merged and merged[0]["role"] != "user":
        merged.pop(0)
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()

    return merged


def judge_conversation(
    messages: list[dict[str, Any]],
    *,
    model: str = "",
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: int = 180,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Judge a conversation using LLM-as-judge with 5 binary rubrics.

    Args:
        messages: The full conversation to evaluate.
        model: LLM model to use for judging (default: from JUDGE_MODEL env).
        base_url: OpenAI-compatible API base URL (default: from JUDGE_BASE_URL env).
        api_key: API key (default: from JUDGE_API_KEY env).

    Returns:
        Dict with 'judgement' (rubric scores) and 'error' (None or error string).
    """
    model = model or os.getenv("JUDGE_MODEL", "gpt-4.1")
    base_url = base_url or os.getenv("JUDGE_BASE_URL") or None
    api_key = api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""

    prepared = _prepare_messages(messages)
    if not prepared:
        return {"judgement": None, "error": "No valid messages after filtering"}

    conversation_text = "\n\n".join(
        f"{'Customer' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in prepared
    )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=float(timeout_s))

    last_error: str = "Unknown error"
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evaluate this conversation:\n\n{conversation_text}"},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            # Strip any thinking blocks from judge model
            content = _THINK_RE.sub("", content).strip()
            content = _THINK_UNCLOSED_RE.sub("", content).strip()
            result = json.loads(content)
            return {"judgement": result, "error": None}
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}"
            break
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(10 * (2 ** attempt))
            else:
                break

    return {"judgement": None, "error": last_error}
