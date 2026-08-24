from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FIELD_PATTERN = r"\*\*{field}:\*\*\s*(.*?)(?=\n\*\*[^*]+:\*\*|\Z)"


def extract_md_field(md_text: str, field_name: str) -> str:
    if not md_text:
        return ""
    pattern = FIELD_PATTERN.format(field=re.escape(field_name))
    match = re.search(pattern, md_text, flags=re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group(1).strip().split())


def split_values(text: str, separators: tuple[str, ...] = (",", "|")) -> list[str]:
    normalized = text
    for separator in separators:
        normalized = normalized.replace(separator, ",")
    values = [value.strip() for value in normalized.split(",")]
    return [value for value in values if value]


def extract_gender(audience: str) -> str:
    match = re.search(r"genders:\s*([a-zA-Z]+)", audience)
    return match.group(1).lower() if match else ""


def extract_age_group(audience: str) -> str:
    match = re.search(r"ageGroups:\s*([a-zA-Z]+)", audience)
    return match.group(1).lower() if match else ""


def extract_color(color_info: str) -> str:
    match = re.search(r"colors:\s*(.*)", color_info)
    return match.group(1).strip() if match else color_info.strip()


def extract_currency(price_info: str) -> str:
    match = re.search(r"currency:\s*([A-Z]{3})", price_info)
    return match.group(1) if match else "USD"


@dataclass(frozen=True)
class ProductDocument:
    sku: str
    title: str
    url: str
    price: float
    currency: str
    availability: str
    brand: str
    categories: list[str]
    gender: str
    age_group: str
    colors: list[str]
    materials: list[str]
    available_sizes: list[str]
    features: str
    description: str

    def retrieval_text(self) -> str:
        categories = ", ".join(self.categories)
        colors = ", ".join(self.colors)
        sizes = ", ".join(self.available_sizes)
        materials = ", ".join(self.materials)
        parts = [
            f"title: {self.title}",
            f"brand: {self.brand}",
            f"categories: {categories}",
            f"gender: {self.gender}",
            f"age_group: {self.age_group}",
            f"colors: {colors}",
            f"sizes: {sizes}",
            f"materials: {materials}",
            f"price: {self.price:.2f} {self.currency}",
            f"features: {self.features}",
            f"description: {self.description}",
        ]
        return "\n".join(part for part in parts if part.strip())

    def search_payload(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "title": self.title,
            "url": self.url,
            "price": self.price,
            "currency": self.currency,
            "availability": self.availability,
            "brand": self.brand,
            "categories": self.categories,
            "gender": self.gender,
            "age_group": self.age_group,
            "colors": self.colors,
            "materials": self.materials,
            "available_sizes": self.available_sizes,
            "features": self.features,
            "description": self.description,
        }


def _to_product(sku: str, payload: dict[str, Any]) -> ProductDocument:
    md = payload.get("md", "")
    audience = extract_md_field(md, "audience")
    color_info = extract_md_field(md, "colorInfo")
    price_info = extract_md_field(md, "priceInfo")

    return ProductDocument(
        sku=sku,
        title=payload.get("title", "").strip(),
        url=payload.get("url", "").strip(),
        price=float(payload.get("price", 0.0) or 0.0),
        currency=extract_currency(price_info),
        availability=extract_md_field(md, "Availability") or "UNKNOWN",
        brand=extract_md_field(md, "brands") or "Unknown",
        categories=[c for c in split_values(extract_md_field(md, "categories"), (">",))],
        gender=extract_gender(audience),
        age_group=extract_age_group(audience),
        colors=split_values(extract_color(color_info)),
        materials=split_values(extract_md_field(md, "materials")),
        available_sizes=split_values(extract_md_field(md, "availableSizes")),
        features=extract_md_field(md, "features"),
        description=extract_md_field(md, "description"),
    )


def load_products(feed_path: Path, limit: int | None = None) -> list[ProductDocument]:
    if not feed_path.exists():
        raise FileNotFoundError(
            f"Product feed not found at {feed_path}.\n"
            "Set CATALOG_FEED_PATH to a valid product feed JSON file."
        )
    with feed_path.open("r", encoding="utf-8") as stream:
        raw_data: dict[str, dict[str, Any]] = json.load(stream)

    products: list[ProductDocument] = []
    for index, (sku, payload) in enumerate(raw_data.items()):
        if limit is not None and index >= limit:
            break
        products.append(_to_product(sku, payload))
    return products
