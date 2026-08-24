from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    batch_size: int


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str | None
    api_key: str | None
    temperature: float
    max_tool_rounds: int = 10
    max_tokens: int | None = None
    native_tools: bool = True


@dataclass(frozen=True)
class PathsConfig:
    feed_json_path: Path
    index_path: Path
    metadata_path: Path
    trace_path: Path
    settings_path: Path


@dataclass(frozen=True)
class AppConfig:
    embedding: EmbeddingConfig
    llm: LLMConfig
    paths: PathsConfig

    @staticmethod
    def from_env(project_root: Path) -> "AppConfig":
        artifacts_dir = project_root / "artifacts"
        index_path = Path(os.getenv("TXTAI_INDEX_PATH", artifacts_dir / "txtai-index"))
        metadata_path = Path(os.getenv("CATALOG_METADATA_PATH", artifacts_dir / "catalog-metadata.json"))
        trace_path = Path(os.getenv("AGENT_TRACE_PATH", artifacts_dir / "agent-tool-traces.jsonl"))
        settings_path = Path(os.getenv("INDEX_SETTINGS_PATH", artifacts_dir / "txtai-index-settings.json"))
        feed_json_path = Path(os.getenv("CATALOG_FEED_PATH", project_root / "data" / "amazon_jeans_shop.json"))

        embedding = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai_compatible"),
            model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
            api_key=os.getenv("EMBEDDING_API_KEY"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        )
        llm = LLMConfig(
            model=os.getenv("AGENT_LLM_MODEL", "gpt-4.1-mini"),
            base_url=os.getenv("AGENT_LLM_BASE_URL"),
            api_key=os.getenv("AGENT_LLM_API_KEY"),
            temperature=float(os.getenv("AGENT_LLM_TEMPERATURE", "0.0")),
            max_tool_rounds=int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "10")),
        )
        paths = PathsConfig(
            feed_json_path=feed_json_path,
            index_path=index_path,
            metadata_path=metadata_path,
            trace_path=trace_path,
            settings_path=settings_path,
        )
        return AppConfig(embedding=embedding, llm=llm, paths=paths)
