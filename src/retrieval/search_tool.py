from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from txtai import Embeddings


@dataclass
class CatalogSearchTool:
    embeddings: Embeddings
    metadata_by_sku: dict[str, dict[str, Any]]

    @classmethod
    def from_files(cls, embeddings: Embeddings, metadata_path: Path) -> "CatalogSearchTool":
        with metadata_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
        return cls(embeddings=embeddings, metadata_by_sku=metadata)

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        raw_results = self.embeddings.search(query, top_k)
        results: list[dict[str, Any]] = []

        for rank, item in enumerate(raw_results, start=1):
            if isinstance(item, dict):
                sku = str(item.get("id", ""))
                score = float(item.get("score", 0.0))
            else:
                sku = str(item[0])
                score = float(item[1])

            payload = dict(self.metadata_by_sku.get(sku, {}))
            payload.update(
                {
                    "score": round(score, 6),
                    "rank": rank,
                }
            )
            results.append(payload)

        return results
