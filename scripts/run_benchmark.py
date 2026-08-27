#!/usr/bin/env python3
"""Run the conversational commerce benchmark on a model.

Usage:
    PYTHONPATH=src python scripts/run_benchmark.py \
        --dialogs data/amazon_jeans_sample_20.json \
        --model Qwen3-32B \
        --runs 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from retrieval.agent_runtime import PureLLMAssistantAgent, ToolTraceLogger
from retrieval.config import AppConfig
from retrieval.rag_refinement import RAGRefiner
from retrieval.search_tool import CatalogSearchTool
from retrieval.txtai_index import load_txtai_index
from benchmark.conversation import run_benchmark
from benchmark.customer import CustomerEmulationAgent
from benchmark.models import resolve_model
from benchmark.output import write_model_outputs


def load_dialogs(path: Path, limit: int | None = None) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conversational commerce benchmark")
    parser.add_argument("--dialogs", required=True, help="Path to scenarios JSON")
    parser.add_argument("--model", required=True, help="Model key or identifier")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs")
    parser.add_argument("--limit", type=int, default=None, help="Limit scenarios")
    parser.add_argument("--output-dir", default="artifacts/results", help="Output base dir")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judging")
    parser.add_argument(
        "--omit-tool-messages",
        action="store_true",
        help="Omit tool calls/results from saved transcripts and judge input (not recommended)",
    )
    args = parser.parse_args()

    model_tag, model_id = resolve_model(args.model)
    print(f"Model: {model_tag} ({model_id})")

    config = AppConfig.from_env(PROJECT_ROOT)

    # Load txtai index
    print("Loading txtai index ...")
    embeddings = load_txtai_index(config)
    search_tool = CatalogSearchTool.from_files(embeddings, config.paths.metadata_path)
    refiner = RAGRefiner(llm_config=config.llm)
    print("  Index loaded")

    # Load scenarios
    dialogs = load_dialogs(Path(args.dialogs), limit=args.limit)
    print(f"Loaded {len(dialogs)} scenarios from {args.dialogs}")

    # Determine LLM config for the model under test
    from dataclasses import replace
    from retrieval.config import LLMConfig

    # Use AGENT_LLM_* env vars as base, override model name
    agent_llm = replace(config.llm, model=model_id.split(":", 1)[-1] if ":" in model_id else model_id)

    # Customer simulation uses its own LLM config
    customer_model = os.getenv("CUSTOMER_LLM_MODEL", "gpt-4.1")
    customer_base_url = os.getenv("CUSTOMER_LLM_BASE_URL", os.getenv("AGENT_LLM_BASE_URL"))
    customer_api_key = os.getenv("CUSTOMER_LLM_API_KEY", os.getenv("AGENT_LLM_API_KEY"))

    for run_idx in range(1, args.runs + 1):
        print(f"\n{'='*60}")
        print(f"  Run {run_idx}/{args.runs} — {model_tag}")
        print(f"{'='*60}\n")

        output_dir = Path(args.output_dir) / model_tag / f"run-{run_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)

        trace_logger = ToolTraceLogger(path=output_dir / "tool-traces.jsonl")

        assistant = PureLLMAssistantAgent(
            llm_config=agent_llm,
            search_tool=search_tool,
            refiner=refiner,
            trace_logger=trace_logger,
        )

        customer = CustomerEmulationAgent(
            llm_config_model=customer_model,
            llm_config_base_url=customer_base_url,
            llm_config_api_key=customer_api_key,
        )

        report = run_benchmark(
            assistant=assistant,
            customer=customer,
            dialogs=dialogs,
            judge_model=os.getenv("JUDGE_MODEL", "gpt-4.1"),
            judge_base_url=os.getenv("JUDGE_BASE_URL"),
            judge_api_key=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            skip_judge=args.skip_judge,
            include_tool_messages=not args.omit_tool_messages,
        )

        full_output = {
            "model_tag": model_tag,
            "model_id": model_id,
            "dialog_file": args.dialogs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_scenarios": report["total"],
            "successful": report["success"],
            "failed": report["failed"],
            "results": report["results"],
        }

        write_model_outputs(full_output, output_dir)
        print(f"\n  Results saved to {output_dir}")


if __name__ == "__main__":
    main()
