"""Model registry — maps friendly names to provider:model_id strings."""

from __future__ import annotations


MODEL_REGISTRY: dict[str, str] = {
    "gpt-oss-20B": "deepinfra:openai/gpt-oss-20b",
    "gpt-oss-120B": "nebius:openai/gpt-oss-120b",
    "Qwen3-30B-A3B": "nebius:Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen3-32B": "nebius:Qwen/Qwen3-32B",
    "Qwen3.5-35B-A3B": "dashscope:qwen3.5-35b-a3b",
    "Qwen3.5-27B": "dashscope:qwen3.5-27b",
    "Qwen3.5-122B-A10B": "dashscope:qwen3.5-122b-a10b",
    "Kimi-K2.5": "nebius:moonshotai/Kimi-K2.5",
    "Kimi-K2.6": "nebius:moonshotai/Kimi-K2.6",
    "Gemma-4-31B": "nebius:google/gemma-4-31b-it",
}


def resolve_model(model_key: str) -> tuple[str, str]:
    """Return (model_tag, model_id) from a registry key or raw model id."""
    if model_key in MODEL_REGISTRY:
        return model_key, MODEL_REGISTRY[model_key]
    tag = model_key.replace("/", "_").replace(":", "_").replace(" ", "_")
    return tag, model_key
