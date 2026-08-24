#!/usr/bin/env python3
"""Generate benchmark scenarios for a product catalog using LLM.

This is a simplified scenario generation template. For each product in the
catalog, it generates a multi-turn golden dialogue showing a realistic shopping
interaction. These dialogues then serve as the reference for customer simulation.

Usage:
    PYTHONPATH=src python scripts/generate_scenarios.py \
        --catalog data/amazon_jeans_shop.json \
        --output data/generated_scenarios.json \
        --num-scenarios 20

The generated scenarios can be used directly with run_benchmark.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")


SCENARIO_GEN_PROMPT = """You are a scenario designer for an e-commerce conversational AI benchmark.

Given a product from a fashion catalog, create a realistic multi-turn shopping dialogue between a Customer and an Assistant. The dialogue should:

1. Start with the customer describing what they want (without naming the exact product)
2. Include at least one clarification exchange
3. Show the assistant recommending the product based on search results
4. End with the customer asking to add the product to their cart

The dialogue should be natural, realistic, and test the assistant's ability to:
- Understand shopping intent
- Ask appropriate clarifying questions
- Make relevant recommendations
- Handle the add-to-cart flow correctly

Product to base the scenario on:
{product_info}

Generate a dialogue with 4-6 turns (8-12 messages total). Each message should be a JSON object with "role" (user/assistant) and "content" fields.

Return strict JSON:
{{
  "scenario_id": "<descriptive short id>",
  "product_hint": "<product SKU>",
  "difficulty": "hard",
  "result_dialog": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    ...
  ]
}}
"""


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_products(catalog: dict[str, Any], num: int, seed: int = 42) -> list[tuple[str, dict]]:
    """Select a diverse subset of products from the catalog."""
    rng = random.Random(seed)
    items = list(catalog.items())
    if len(items) <= num:
        return items
    return rng.sample(items, num)


def format_product_info(sku: str, product: dict[str, Any]) -> str:
    """Format product info for the prompt."""
    title = product.get("title", "Unknown")
    price = product.get("price", "N/A")
    md = product.get("md", "")
    return f"SKU: {sku}\nTitle: {title}\nPrice: ${price}\n\nDetails:\n{md[:1000]}"


def generate_scenario(client: OpenAI, model: str, product_info: str) -> dict[str, Any] | None:
    """Generate a single scenario using the LLM."""
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=model,
                temperature=0.7,
                messages=[
                    {"role": "system", "content": SCENARIO_GEN_PROMPT.format(product_info=product_info)},
                    {"role": "user", "content": "Generate the scenario dialogue."},
                ],
                response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            print(f"  Attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark scenarios")
    parser.add_argument("--catalog", required=True, help="Path to product catalog JSON")
    parser.add_argument("--output", required=True, help="Output scenarios JSON path")
    parser.add_argument("--num-scenarios", type=int, default=20, help="Number of scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for product selection")
    args = parser.parse_args()

    model = os.getenv("AGENT_LLM_MODEL", "gpt-4.1")
    base_url = os.getenv("AGENT_LLM_BASE_URL")
    api_key = os.getenv("AGENT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)

    catalog = load_catalog(Path(args.catalog))
    print(f"Loaded catalog with {len(catalog)} products")

    selected = select_products(catalog, args.num_scenarios, seed=args.seed)
    print(f"Selected {len(selected)} products for scenario generation")

    scenarios: list[dict[str, Any]] = []
    for i, (sku, product) in enumerate(selected):
        print(f"[{i + 1}/{len(selected)}] Generating scenario for {sku} ...")
        product_info = format_product_info(sku, product)
        scenario = generate_scenario(client, model, product_info)
        if scenario:
            scenario["index"] = i
            scenarios.append(scenario)
            print(f"  Generated: {scenario.get('scenario_id', 'unknown')}")
        else:
            print(f"  FAILED — skipping")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(scenarios)} scenarios → {output_path}")


if __name__ == "__main__":
    main()
