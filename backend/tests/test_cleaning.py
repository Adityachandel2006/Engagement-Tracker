"""Tests for cleaning.py — 20 sample items including syndicated copies
and a same-name disambiguation case (Apple fruit vs Apple company)."""

import pytest

from models import Item
from cleaning import clean, deduplicate, extract_keywords, filter_relevant, is_relevant


# ── 20 sample items ─────────────────────────────────────────────────────


def _item(id_: int, title: str, text: str, product: str, published: str = "2026-01-01") -> Item:
    return Item(
        id=id_,
        source="news",
        source_name="Test",
        title=title,
        text=text,
        url=f"https://example.com/{id_}",
        published_at=published,
        country="us",
        product=product,
        content_hash=Item.make_hash(title, text),
    )


SAMPLES = [
    # 1-3: Original + 2 syndicated copies (near-duplicate titles)
    _item(1,  "Nike Running Shoes break sales records in 2026",
              "Nike's new running shoe line exceeded expectations.", "nike running shoes", "2026-01-01"),
    _item(2,  "Nike Running Shoes break sales records in 2026!",
              "Sales numbers are through the roof for Nike.", "nike running shoes", "2026-01-02"),
    _item(3,  "Nike running shoes break sales records in 2026 - syndicated",
              "A syndicated copy of the original article.", "nike running shoes", "2026-01-03"),

    # 4-6: Relevant Nike articles (unique titles)
    _item(4,  "Nike launches eco-friendly running shoe line",
              "The new Nike shoes use recycled materials.", "nike running shoes", "2026-01-04"),
    _item(5,  "Runner reviews Nike ZoomX Vaporfly",
              "A detailed look at the Nike running shoe.", "nike running shoes", "2026-01-05"),
    _item(6,  "Nike shoes dominate Boston Marathon 2026",
              "Nike running shoes were worn by top finishers.", "nike running shoes", "2026-01-06"),

    # 7-8: Irrelevant to Nike running shoes
    _item(7,  "Adidas announces new basketball shoe",
              "Adidas reveals its latest basketball sneaker.", "nike running shoes", "2026-01-07"),
    _item(8,  "Stock market rallies on tech earnings",
              "Tech stocks led gains on Wall Street today.", "nike running shoes", "2026-01-08"),

    # 9-11: Apple iPhone — relevant
    _item(9,  "Apple iPhone 17 Pro review: best camera yet",
              "The Apple iPhone 17 Pro has a stunning camera.", "apple iphone", "2026-02-01"),
    _item(10, "iPhone 17 battery life tested",
              "Apple's latest iPhone lasts a full day easily.", "apple iphone", "2026-02-02"),
    _item(11, "Apple iPhone sales surge in India",
              "iPhone shipments doubled compared to last year.", "apple iphone", "2026-02-03"),

    # 12-14: Apple the FRUIT — should be blocked for "apple iphone"
    _item(12, "Apple fruit prices rise due to drought",
              "Fresh apple harvest hit by poor weather.", "apple iphone", "2026-02-04"),
    _item(13, "Best apple cider recipes for autumn",
              "Try these delicious apple cider drinks.", "apple iphone", "2026-02-05"),
    _item(14, "Apple orchard tours are popular this fall",
              "Families flock to apple orchards for picking.", "apple iphone", "2026-02-06"),

    # 15-17: Apple iPhone syndicated copies
    _item(15, "Apple iPhone 17 Pro review: best camera yet!",
              "A republished version of the camera review.", "apple iphone", "2026-02-07"),
    _item(16, "Apple iPhone 17 Pro review - best camera yet (copy)",
              "Another syndicated copy of the camera review.", "apple iphone", "2026-02-08"),

    # 17-18: Tesla articles (no config entry — product name itself used)
    _item(17, "Tesla Model Y gets a price cut",
              "Tesla drops Model Y prices across all trims.", "tesla model y", "2026-03-01"),
    _item(18, "Tesla announces new Supercharger network expansion",
              "Tesla is building more Supercharger stations.", "tesla model y", "2026-03-02"),

    # 19-20: Totally unrelated to any product
    _item(19, "World Cup 2026 draws record viewership",
              "The FIFA World Cup attracted billions of viewers.", "nike running shoes", "2026-04-01"),
    _item(20, "New cooking show premieres on streaming platform",
              "Celebrity chefs compete in a new TV series.", "apple iphone", "2026-04-02"),
]


# ── Deduplication tests ─────────────────────────────────────────────────


def test_dedup_removes_syndicated_nike():
    """Items 1, 2, 3 are near-duplicates — only the earliest (item 1) survives."""
    nike_items = SAMPLES[0:3]
    result = deduplicate(nike_items)
    assert len(result) == 1
    assert result[0].id == 1  # earliest published


def test_dedup_removes_syndicated_apple():
    """Items 9, 15, 16 are near-duplicates — only item 9 survives."""
    apple_dupes = [SAMPLES[8], SAMPLES[14], SAMPLES[15]]
    result = deduplicate(apple_dupes)
    assert len(result) == 1
    assert result[0].id == 9


def test_dedup_keeps_unique_titles():
    """Distinct titles should all survive dedup."""
    unique_items = [SAMPLES[3], SAMPLES[4], SAMPLES[5]]  # items 4, 5, 6
    result = deduplicate(unique_items)
    assert len(result) == 3


def test_dedup_empty_list():
    assert deduplicate([]) == []


# ── Stage-1 relevance tests ─────────────────────────────────────────────


def test_relevant_nike_matches():
    """Items 1, 4, 5, 6 mention Nike / Nike shoes — should be relevant."""
    for item in [SAMPLES[0], SAMPLES[3], SAMPLES[4], SAMPLES[5]]:
        assert is_relevant(item, "nike running shoes"), f"Item {item.id} should be relevant"


def test_irrelevant_nike():
    """Items 7 (Adidas), 8 (stock market), 19 (World Cup) — not relevant."""
    for item in [SAMPLES[6], SAMPLES[7], SAMPLES[18]]:
        assert not is_relevant(item, "nike running shoes"), f"Item {item.id} should NOT be relevant"


def test_relevant_apple_iphone():
    """Items 9, 10, 11 mention iPhone / Apple iPhone — relevant."""
    for item in [SAMPLES[8], SAMPLES[9], SAMPLES[10]]:
        assert is_relevant(item, "apple iphone"), f"Item {item.id} should be relevant"


def test_apple_fruit_blocked():
    """Items 12, 13, 14 mention apple fruit/cider/orchard — blocklisted."""
    for item in [SAMPLES[11], SAMPLES[12], SAMPLES[13]]:
        assert not is_relevant(item, "apple iphone"), f"Item {item.id} should be BLOCKED"


def test_unrelated_apple():
    """Item 20 (cooking show) mentions neither iPhone nor apple aliases."""
    assert not is_relevant(SAMPLES[19], "apple iphone")


def test_fallback_product_match():
    """Tesla model y has no products.yaml entry — uses product name itself."""
    assert is_relevant(SAMPLES[16], "tesla model y")  # item 17 mentions Tesla


def test_irrelevant_tesla():
    """Item 18 mentions Tesla but NOT 'model y', and there's no alias config."""
    # This should be relevant because the fallback alias is "tesla model y"
    # which won't match "Tesla announces new Supercharger..."
    # Actually "tesla" substring IS in the title, but the alias is the full product name.
    # Let's check: the fallback alias is [product_lower] = "tesla model y"
    # "tesla model y" is NOT in "tesla announces new supercharger..."
    assert not is_relevant(SAMPLES[17], "tesla model y")


# ── filter_relevant ─────────────────────────────────────────────────────


def test_filter_relevant_nike():
    nike_items = [SAMPLES[i] for i in [0, 3, 4, 5, 6, 7, 18]]  # items 1,4,5,6,7,8,19
    result = filter_relevant(nike_items, "nike running shoes")
    ids = {item.id for item in result}
    assert ids == {1, 4, 5, 6}  # only Nike-relevant items


def test_filter_relevant_apple():
    apple_items = [SAMPLES[i] for i in [8, 9, 10, 11, 12, 13, 19]]
    result = filter_relevant(apple_items, "apple iphone")
    ids = {item.id for item in result}
    assert ids == {9, 10, 11}  # fruit items blocked, cooking show excluded


# ── Full clean() pipeline ───────────────────────────────────────────────


def test_clean_nike_end_to_end():
    """Full clean: dedup + relevance on Nike items 1-8 + 19."""
    nike_items = SAMPLES[0:8] + [SAMPLES[18]]  # 9 items total
    result = clean(nike_items, "nike running shoes")
    ids = {item.id for item in result}
    # Items 2,3 deduped (only 1 kept); items 7,8,19 irrelevant → 1,4,5,6
    assert ids == {1, 4, 5, 6}


def test_clean_apple_end_to_end():
    """Full clean: dedup + relevance on Apple items 9-16 + 20."""
    apple_items = SAMPLES[8:16] + [SAMPLES[19]]
    result = clean(apple_items, "apple iphone")
    ids = {item.id for item in result}
    # Items 15,16 deduped (only 9 kept); items 12,13,14 blocked; 20 irrelevant
    # Keeps: 9, 10, 11
    assert ids == {9, 10, 11}


# ── extract_keywords ────────────────────────────────────────────────────


def test_extract_keywords_basic():
    texts = ["nike shoes are great shoes", "these nike shoes rock"]
    result = extract_keywords(texts, top_n=3)
    words = [kw["word"] for kw in result]
    assert "nike" in words
    assert "shoes" in words


def test_extract_keywords_empty():
    assert extract_keywords([], top_n=5) == []
