#!/usr/bin/env python3
"""Build the txtai vector index from the product catalog."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from retrieval.catalog_schema import load_products
from retrieval.config import AppConfig
from retrieval.txtai_index import build_txtai_index


def main() -> None:
    config = AppConfig.from_env(PROJECT_ROOT)
    print(f"Loading products from {config.paths.feed_json_path} ...")
    products = load_products(config.paths.feed_json_path)
    print(f"  Found {len(products)} products")

    print("Building txtai index ...")
    settings = build_txtai_index(products, config)
    print(f"  Index built: {settings['feed_size']} products indexed")
    print(f"  Saved to: {config.paths.index_path}")


if __name__ == "__main__":
    main()
