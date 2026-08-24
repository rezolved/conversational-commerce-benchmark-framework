"""Tool-using LLM assistant agent for conversational commerce."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import LLMConfig
from .rag_refinement import RAGRefiner
from .search_tool import CatalogSearchTool


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


TRANSIENT_LLM_ERROR_MARKERS = (
    "Error code: 429",
    "Error code: 500",
    "Error code: 502",
    "Error code: 503",
    "Error code: 504",
    "connection error",
    "rate limit",
    "model busy",
    "retry later",
    "timeout",
    "temporarily unavailable",
    "inference error",
)

KNOWN_TOOL_NAMES = {"catalog_search", "add_to_cart"}
_TOOL_ALIASES = {"search_catalog": "catalog_search"}


def _is_transient_llm_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker.lower() in message for marker in TRANSIENT_LLM_ERROR_MARKERS)


def _normalize_tool_name(name: str) -> str:
    candidate = (name or "").strip()
    if candidate in KNOWN_TOOL_NAMES:
        return candidate
    if candidate in _TOOL_ALIASES:
        return _TOOL_ALIASES[candidate]
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*", candidate)
    if not match:
        return candidate
    normalized = match.group(0)
    if normalized in _TOOL_ALIASES:
        return _TOOL_ALIASES[normalized]
    if normalized in KNOWN_TOOL_NAMES:
        return normalized
    return candidate


SYSTEM_PROMPT = """You are a friendly, knowledgeable shopping assistant for an Amazon fashion catalog. Talk like a real store associate would — warm, direct, and natural.

Operating rules:

SEARCH & CART
1) Always use catalog_search before recommending products. Base every product detail (name, price, size, SKU, availability) strictly on returned tool data — never invent or embellish any product information, URLs, capabilities, or services. If a tool call fails or returns an error, tell the customer honestly that something went wrong instead of claiming success.
2) When the customer asks to add an item, you MUST call the add_to_cart tool first and wait for its response. Only after add_to_cart returns a success status may you confirm the addition. Never write "in your cart", "added to cart", "been added", or any equivalent phrase unless that exact turn contains a successful add_to_cart tool response.
3) Before calling add_to_cart, confirm SKU, size, and quantity with the customer. If a product has multiple sizes available, present the options and wait for the customer to choose. After a successful add_to_cart, state what was added in a brief, conversational way.
4) When the customer refines their request with additional details — such as color, fit, wash, rise, size, or budget — perform a fresh catalog_search with the updated criteria.

RESPONSE STYLE
5) Write the way a helpful store associate speaks — plain, warm, conversational. No formatting at all: no bold, italic, headings, tables, links, inline code, or code fences. No lists of any kind. Present multiple products in flowing sentences or short separate paragraphs.
6) Never narrate, announce, or reference your own actions. Never use first-person verb phrases that describe what you are doing behind the scenes. Instead, state the product information or outcome directly.

NO-MATCH HANDLING
7) When results do not match the customer's intent, silently reformulate the query and re-search once before replying. If there is still no match, state the limitation once using impersonal language then pivot to closest alternatives.

CUSTOMER INTERACTION
8) Track preferences the customer has already stated (size, color, fit, budget) and apply them to later turns without re-asking.
9) Ask at most one clarifying question per turn.
"""


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "catalog_search",
            "description": "Returns top-k catalog matches from the fashion vector index",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language shopping query"},
                    "top_k": {
                        "type": "integer",
                        "description": "How many catalog candidates to return",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Adds a selected SKU to cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string"},
                    "size": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1, "default": 1},
                },
                "required": ["sku"],
            },
        },
    },
]


@dataclass
class ToolTraceLogger:
    path: Path

    def log(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False))
            stream.write("\n")


@dataclass
class PureLLMAssistantAgent:
    llm_config: LLMConfig
    search_tool: CatalogSearchTool
    refiner: RAGRefiner
    trace_logger: ToolTraceLogger
    system_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.system_prompt:
            self.system_prompt = SYSTEM_PROMPT
        self._client = OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            timeout=120.0,
        )

    def _create_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        allow_tools: bool = True,
        max_tokens_override: int | None = None,
    ) -> Any:
        last_error: Exception | None = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.llm_config.model,
                    "temperature": self.llm_config.temperature,
                    "messages": messages,
                }
                if max_tokens_override is not None:
                    request_kwargs["max_tokens"] = max_tokens_override
                elif self.llm_config.max_tokens is not None:
                    request_kwargs["max_tokens"] = self.llm_config.max_tokens
                if allow_tools and self.llm_config.native_tools:
                    request_kwargs["tools"] = TOOL_SCHEMAS
                return self._client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                last_error = exc
                if attempt == max_attempts - 1 or not _is_transient_llm_error(exc):
                    raise
                time.sleep(min(30, 5 * (attempt + 1)))

        raise RuntimeError("LLM completion failed") from last_error

    def _execute_tool(
        self, name: str, arguments: dict[str, Any], default_top_k: int,
    ) -> tuple[str, list[dict[str, Any]], str]:
        name = _normalize_tool_name(name)
        if name == "catalog_search":
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", default_top_k)
            raw_results = self.search_tool.search(query, top_k=top_k)
            self.trace_logger.log(
                {
                    "tool_name": "catalog_search",
                    "tool_input": {"query": query, "top_k": top_k},
                    "tool_output_count": len(raw_results),
                }
            )
            refined = self.refiner.refine(query=query, candidates=raw_results)
            tool_payload = json.dumps(refined["results"], ensure_ascii=False)
            return tool_payload, refined["results"], refined.get("summary", "")

        if name == "add_to_cart":
            self.trace_logger.log(
                {"tool_name": "add_to_cart", "tool_input": arguments}
            )
            result = {
                "status": "success",
                "sku": arguments.get("sku", ""),
                "size": arguments.get("size", ""),
                "quantity": arguments.get("quantity", 1),
            }
            return json.dumps(result, ensure_ascii=False), [], ""

        return json.dumps({"error": f"Unknown tool: {name}"}), [], ""

    def run(
        self,
        user_query: str,
        top_k: int = 10,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_query})

        tool_messages_out: list[dict[str, Any]] = []
        top_results: list[dict[str, Any]] = []
        refinement_summary = ""
        assistant_message = ""

        for _ in range(self.llm_config.max_tool_rounds):
            completion = self._create_completion(messages)
            response_msg = completion.choices[0].message

            if not response_msg.tool_calls:
                assistant_message = strip_thinking(response_msg.content or "")
                if not assistant_message and self.llm_config.max_tokens:
                    try:
                        retry_msgs = messages + [
                            {"role": "system", "content":
                             "Respond to the customer directly and concisely. "
                             "Use the information already in this conversation."},
                        ]
                        retry = self._create_completion(
                            retry_msgs, allow_tools=False,
                            max_tokens_override=self.llm_config.max_tokens * 2,
                        )
                        assistant_message = strip_thinking(
                            retry.choices[0].message.content or ""
                        )
                    except Exception:
                        pass
                break

            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": _normalize_tool_name(tc.function.name),
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response_msg.tool_calls
                ],
            }
            messages.append(assistant_dict)
            tool_messages_out.append(assistant_dict)

            for tc in response_msg.tool_calls:
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                result_json, results, summary = self._execute_tool(
                    _normalize_tool_name(tc.function.name), fn_args, default_top_k=top_k,
                )
                if results:
                    top_results = results
                if summary:
                    refinement_summary = summary

                tool_response: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                }
                messages.append(tool_response)
                tool_messages_out.append({
                    **tool_response,
                    "name": _normalize_tool_name(tc.function.name),
                })
        else:
            forced_answer_messages = [
                *messages,
                {"role": "system", "content":
                 "Answer the customer directly using the tool outputs already "
                 "present in this conversation. Do not call any more tools."},
            ]
            try:
                forced_completion = self._create_completion(
                    forced_answer_messages, allow_tools=False,
                    max_tokens_override=(
                        self.llm_config.max_tokens * 2 if self.llm_config.max_tokens else None
                    ),
                )
                assistant_message = strip_thinking(
                    forced_completion.choices[0].message.content or ""
                )
            except Exception:
                pass
            if not assistant_message:
                assistant_message = (
                    strip_thinking(response_msg.content or "")
                    or "Something went wrong on our end. Could you try that again?"
                )

        return {
            "assistant_message": assistant_message,
            "refinement_summary": refinement_summary,
            "top_results": top_results,
            "tool_messages": tool_messages_out,
        }
