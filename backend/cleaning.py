"""Data-cleaning pipeline: near-dedup, stage-1 relevance, keyword extraction."""

import re
from pathlib import Path

import pandas as pd
import yaml
from rapidfuzz import fuzz

from models import Item

# ── Product config ───────────────────────────────────────────────────────

_PRODUCTS_PATH = Path(__file__).parent / "products.yaml"


def load_product_config() -> dict[str, dict]:
    """Return {product: {aliases: [...], blocklist: [...]}}."""
    if not _PRODUCTS_PATH.exists():
        return {}
    with open(_PRODUCTS_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # Normalise keys and list entries to lowercase
    return {
        k.lower(): {
            "aliases": [a.lower() for a in v.get("aliases", [])],
            "blocklist": [b.lower() for b in v.get("blocklist", [])],
        }
        for k, v in raw.items()
    }


# ── Near-duplicate removal ───────────────────────────────────────────────


def deduplicate(items: list[Item], threshold: float = 82) -> list[Item]:
    """Remove near-duplicate syndicated copies by title similarity.

    Groups titles with rapidfuzz ``token_set_ratio`` above *threshold*
    and keeps the earliest item (by ``published_at``) in each group.
    """
    if not items:
        return []

    kept: list[Item] = []
    used = [False] * len(items)

    for i, item_i in enumerate(items):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            score = fuzz.token_set_ratio(item_i.title, items[j].title)
            if score >= threshold:
                group.append(j)
                used[j] = True

        # Keep the earliest published item in the group
        earliest = min(group, key=lambda idx: items[idx].published_at)
        kept.append(items[earliest])
        used[i] = True  # mark the whole group consumed

    return kept


# ── Stage-1 relevance ───────────────────────────────────────────────────


def is_relevant(item: Item, product: str, config: dict[str, dict] | None = None) -> bool:
    """Return True if the item's title or first 300 chars mention the product
    or one of its aliases, and none of the blocklist terms appear."""
    if config is None:
        config = load_product_config()

    product_lower = product.lower()
    entry = config.get(product_lower, {"aliases": [product_lower], "blocklist": []})
    aliases = entry["aliases"] or [product_lower]
    blocklist = entry["blocklist"]

    haystack = f"{item.title} {item.text[:300]}".lower()

    # Blocklist check first
    for blocked in blocklist:
        if blocked in haystack:
            return False

    # Alias match
    return any(alias in haystack for alias in aliases)


def filter_relevant(items: list[Item], product: str) -> list[Item]:
    """Keep only items that pass stage-1 relevance."""
    config = load_product_config()
    return [item for item in items if is_relevant(item, product, config)]


# ── Keyword extraction (copied from pipeline.py, unchanged) ─────────────

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","is","was","are","were","be","been","has","have",
    "had","will","would","could","should","may","might","it","its",
    "this","that","these","those","he","she","they","we","i","you",
    "s","t","said","says","new","also","one","two","after","before",
    "about","from","into","by","as","up","out","no","not","more"
}


def _clean_text(text: str) -> str:
    """Lowercase, strip URLs and punctuation — for keyword counting only."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(text.split()).lower()


def extract_keywords(texts: list[str], top_n: int = 10) -> list[dict]:
    all_words = []
    for text in texts:
        all_words.extend([w for w in text.split() if w not in STOPWORDS and len(w) > 2])
    freq = pd.Series(all_words).value_counts().head(top_n)
    return [{"word": word, "count": int(cnt)} for word, cnt in freq.items()]


# ── Convenience pipeline ────────────────────────────────────────────────


def clean(items: list[Item], product: str) -> list[Item]:
    """Full cleaning pass: near-dedup → stage-1 relevance."""
    deduped = deduplicate(items)
    return filter_relevant(deduped, product)
