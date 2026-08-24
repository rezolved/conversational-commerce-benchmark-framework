from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import LLMConfig


RERANK_PROMPT = """You are an e-commerce recommendation ranker.
You receive a user query and top-k catalog candidates.

Return strict JSON with:
{
  "summary": "one paragraph",
  "ranked_skus": ["sku1", "sku2", "..."],
  "reasons_by_sku": {"sku1":"...", "sku2":"..."}
}

Rules:
- Prioritize intent fit: product type, gender, color/style words, and any budget clues.
- Keep only SKUs that appear in the input candidates.
- Preserve at most 10 SKUs.
- If uncertain, keep original order.
"""


TRANSIENT_ERROR_MARKERS = (
    "error code: 429",
    "error code: 500",
    "error code: 502",
    "error code: 503",
    "error code: 504",
    "rate limit",
    "model busy",
    "retry later",
    "timeout",
    "temporarily unavailable",
    "inference error",
)


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


@dataclass
class RAGRefiner:
    llm_config: LLMConfig
    mock_mode: bool = False

    def _heuristic_refine(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        query_tokens = _tokenize(query)
        rescored: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            searchable = " ".join(
                [
                    candidate.get("title", ""),
                    " ".join(candidate.get("colors", [])),
                    " ".join(candidate.get("materials", [])),
                    candidate.get("features", ""),
                    candidate.get("description", ""),
                ]
            )
            overlap = len(query_tokens.intersection(_tokenize(searchable)))
            score = float(candidate.get("score", 0.0)) + (0.01 * overlap)
            rescored.append((score, candidate))

        rescored.sort(key=lambda item: item[0], reverse=True)
        ranked = [candidate["sku"] for _, candidate in rescored]
        reasons = {
            candidate["sku"]: "Heuristic overlap with query terms and original vector score"
            for _, candidate in rescored
        }
        return {
            "summary": "Ranked with heuristic overlap because LLM mode is disabled.",
            "ranked_skus": ranked,
            "reasons_by_sku": reasons,
        }

    def _llm_refine(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        client = OpenAI(api_key=self.llm_config.api_key, base_url=self.llm_config.base_url)
        payload = {"query": query, "candidates": candidates}
        max_attempts = 8
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                completion = client.chat.completions.create(
                    model=self.llm_config.model,
                    temperature=self.llm_config.temperature,
                    messages=[
                        {"role": "system", "content": RERANK_PROMPT},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_format={"type": "json_object"},
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts - 1 or not _is_transient_error(exc):
                    raise
                time.sleep(min(30, 5 * (attempt + 1)))
        else:
            raise RuntimeError("RAG refinement failed") from last_error

        content = completion.choices[0].message.content or "{}"
        sanitized = re.sub(r"[\x00-\x1f\x7f]", " ", content)
        try:
            data = json.loads(sanitized)
        except json.JSONDecodeError:
            return self._heuristic_refine(query, candidates)
        return {
            "summary": data.get("summary", ""),
            "ranked_skus": data.get("ranked_skus", []),
            "reasons_by_sku": data.get("reasons_by_sku", {}),
        }

    def refine(self, query: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {"summary": "No candidates to refine.", "results": []}

        use_llm = not self.mock_mode and bool(self.llm_config.api_key)
        if use_llm:
            raw_refined = self._llm_refine(query=query, candidates=candidates)
        else:
            raw_refined = self._heuristic_refine(query=query, candidates=candidates)

        by_sku = {candidate["sku"]: candidate for candidate in candidates}
        ordered_results: list[dict[str, Any]] = []
        for rank, sku in enumerate(raw_refined["ranked_skus"], start=1):
            if sku not in by_sku:
                continue
            candidate = dict(by_sku[sku])
            candidate["rank"] = rank
            candidate["reason"] = raw_refined["reasons_by_sku"].get(sku, "")
            ordered_results.append(candidate)

        if not ordered_results:
            ordered_results = list(candidates)

        return {
            "summary": raw_refined.get("summary", ""),
            "results": ordered_results[:10],
        }
