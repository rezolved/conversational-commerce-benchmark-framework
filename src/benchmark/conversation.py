"""Conversation loop — multi-turn exchange between customer emulation and assistant."""

from __future__ import annotations

import os
import time
from typing import Any

from retrieval.agent_runtime import PureLLMAssistantAgent
from benchmark.customer import CustomerEmulationAgent, END_MARKER
from benchmark.judge import judge_conversation

MAX_TURNS = 10


def _tprint(*args: Any, **kwargs: Any) -> None:
    """Thread-safe print."""
    import threading
    _tprint._lock = getattr(_tprint, "_lock", threading.Lock())  # type: ignore[attr-defined]
    with _tprint._lock:  # type: ignore[attr-defined]
        print(*args, **kwargs)


SCENARIO_TIMEOUT = int(os.getenv("SCENARIO_TIMEOUT", "300"))


class ScenarioTimeoutError(Exception):
    """Raised when a scenario exceeds the wall-clock time limit."""


def run_conversation(
    customer: CustomerEmulationAgent,
    assistant: PureLLMAssistantAgent,
    golden_messages: list[dict[str, str]],
    max_turns: int = MAX_TURNS,
    include_tool_messages: bool = False,
    scenario_timeout: float = SCENARIO_TIMEOUT,
) -> dict[str, Any]:
    """Run a multi-turn conversation between customer emulation and assistant agent."""
    system_prompt = customer.build_system_prompt(golden_messages)
    conversation: list[dict[str, Any]] = []
    turn_details: list[dict[str, Any]] = []
    total_latency = 0.0
    wall_start = time.perf_counter()
    opening_msg: str | None = None

    for turn in range(max_turns):
        if time.perf_counter() - wall_start > scenario_timeout:
            raise ScenarioTimeoutError(
                f"Scenario exceeded {scenario_timeout}s wall-clock limit"
            )

        if turn == 0 and golden_messages and golden_messages[0]["role"] == "user":
            customer_msg = golden_messages[0]["content"]
            opening_msg = customer_msg
        else:
            customer_view = [
                m for m in conversation
                if m.get("role") in ("user", "assistant") and not m.get("tool_calls")
            ]
            customer_msg = customer.generate_turn(system_prompt, customer_view)
            if END_MARKER in customer_msg:
                break
            if opening_msg and customer_msg.strip() == opening_msg.strip():
                nudge = (
                    "\n\nCRITICAL: You just repeated your very first question "
                    "word-for-word. That is not allowed. Read the assistant's "
                    "last reply and respond to it naturally — ask a follow-up, "
                    "narrow your preferences, or end the conversation with "
                    "#END_OF_DIALOGUE if the goal cannot be met."
                )
                customer_msg = customer.generate_turn(
                    system_prompt + nudge, customer_view,
                )
                if END_MARKER in customer_msg:
                    break
                if customer_msg.strip() == opening_msg.strip():
                    break

        conversation.append({"role": "user", "content": customer_msg})

        t0 = time.perf_counter()
        agent_output = assistant.run(
            user_query=customer_msg,
            conversation_history=conversation[:-1],
        )
        latency = time.perf_counter() - t0
        total_latency += latency

        assistant_msg = agent_output["assistant_message"]
        if include_tool_messages:
            conversation.extend(agent_output.get("tool_messages", []))
        conversation.append({"role": "assistant", "content": assistant_msg})

        turn_details.append({
            "turn": turn + 1,
            "user": customer_msg,
            "assistant": assistant_msg,
            "refinement_summary": agent_output.get("refinement_summary", ""),
            "search_result_count": len(agent_output.get("top_results", [])),
            "latency_seconds": round(latency, 3),
        })

    return {
        "result_dialog": conversation,
        "turn_details": turn_details,
        "num_turns": len(turn_details),
        "total_latency_seconds": round(total_latency, 3),
    }


def run_benchmark(
    assistant: PureLLMAssistantAgent,
    customer: CustomerEmulationAgent,
    dialogs: list[dict[str, Any]],
    max_turns: int = MAX_TURNS,
    judge_model: str = "",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    skip_judge: bool = False,
    include_tool_messages: bool = False,
) -> dict[str, Any]:
    """Run full benchmark: conversation generation + judgement scoring."""
    results: list[dict[str, Any]] = []
    num_success = 0
    num_failed = 0

    for i, dialog in enumerate(dialogs):
        scenario_id = dialog.get("scenario_id", f"unknown_{dialog.get('index', 0)}")
        _tprint(f"[{i + 1}/{len(dialogs)}] Scenario: {scenario_id}")

        golden_messages = dialog["result_dialog"]

        try:
            conv = run_conversation(
                customer=customer,
                assistant=assistant,
                golden_messages=golden_messages,
                max_turns=max_turns,
                include_tool_messages=include_tool_messages,
            )
            _tprint(f"  Generated {conv['num_turns']} turns in {conv['total_latency_seconds']}s")
        except Exception as exc:
            _tprint(f"  FAILED (conversation): {exc}")
            results.append({
                "index": i,
                "scenario_id": scenario_id,
                "status": "error_conversation",
                "error": str(exc),
            })
            num_failed += 1
            continue

        judgement_data: dict[str, Any] = {"judgement": None, "error": None}
        has_valid_dialog = conv["num_turns"] >= 2
        if not skip_judge and has_valid_dialog:
            try:
                judgement_data = judge_conversation(
                    conv["result_dialog"],
                    model=judge_model,
                    base_url=judge_base_url,
                    api_key=judge_api_key,
                )
                if judgement_data["error"]:
                    _tprint(f"  Judge warning: {judgement_data['error']}")
                else:
                    _tprint(f"  Judged OK")
            except Exception as exc:
                judgement_data = {"judgement": None, "error": str(exc)}
                _tprint(f"  Judge error: {exc}")

        num_success += 1
        results.append({
            "index": i,
            "scenario_id": scenario_id,
            "status": "success",
            "result_dialog": conv["result_dialog"],
            "turn_details": conv["turn_details"],
            "num_turns": conv["num_turns"],
            "total_latency_seconds": conv["total_latency_seconds"],
            "judgement": judgement_data["judgement"],
            "judgement_error": judgement_data["error"],
        })

    return {
        "total": len(dialogs),
        "success": num_success,
        "failed": num_failed,
        "results": results,
    }
