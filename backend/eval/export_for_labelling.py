"""Export a stratified sample of items for human labelling.

Generates a balanced sample across sources and countries, including
about 10% irrelevant candidates, and writes to data/labelling/to_label.csv.
Model predictions are never included in this file.
"""

import argparse
import csv
from collections import defaultdict
import logging
from pathlib import Path
import random
import re
import sys

from rapidfuzz import fuzz
from sqlmodel import Session, select

# Add backend directory to sys.path if not present
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from cleaning import is_relevant, load_product_config
from db import get_engine
from models import Item, Label

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = BACKEND_DIR.parent / "data" / "labelling" / "to_label.csv"
COLUMNS = [
    "item_id",
    "source",
    "country",
    "title",
    "text",
    "url",
    "human_relevant",
    "human_sentiment",
    "notes",
]


def load_few_shot_texts(prompts_dir: Path | None = None) -> set[str]:
    """Read few-shot example texts from prompt files in analysis/prompts/*.txt."""
    if prompts_dir is None:
        prompts_dir = BACKEND_DIR / "analysis" / "prompts"

    few_shots = set()
    if not prompts_dir.exists():
        return few_shots

    for p in prompts_dir.glob("*.txt"):
        try:
            content = p.read_text(encoding="utf-8")
            matches = re.findall(r'Text:\s*"([^"]+)"', content)
            for m in matches:
                cleaned = m.strip().lower()
                if cleaned:
                    few_shots.add(cleaned)
        except Exception as e:
            logger.warning("Could not read prompt file %s: %s", p, e)

    return few_shots


def _get_items_from_db(session: Session) -> list[Item]:
    """Fetch unlabelled items from Item table, excluding already labelled items."""
    labeled_ids = set(session.exec(select(Label.item_id)).all())
    all_items = list(session.exec(select(Item)).all())
    return [it for it in all_items if it.id not in labeled_ids]


def filter_few_shots(items: list[Item], few_shot_texts: set[str]) -> list[Item]:
    """Exclude items whose text appears in prompt few-shot examples."""
    if not few_shot_texts:
        return items

    filtered = []
    for it in items:
        norm_text = (it.text or "").strip().lower()
        # Exclude if exact match or long overlap
        is_fs = any(fs in norm_text or norm_text in fs for fs in few_shot_texts if len(fs) > 15)
        if not is_fs:
            filtered.append(it)
    return filtered


def remove_near_duplicates(items: list[Item], threshold: float = 82) -> list[Item]:
    """Remove near-duplicate items based on title or text similarity."""
    kept: list[Item] = []
    seen: list[str] = []

    for it in items:
        sig = (it.title or "").strip().lower() if it.source == "news" else (it.text or "")[:200].strip().lower()
        if not sig:
            kept.append(it)
            continue

        is_dup = False
        # Windowed check against recent items for speed
        for s in seen[-100:]:
            if fuzz.token_set_ratio(sig, s) >= threshold:
                is_dup = True
                break

        if not is_dup:
            kept.append(it)
            seen.append(sig)

    return kept


def sample_balanced_strata(items: list[Item], target: int, rng: random.Random) -> list[Item]:
    """Sample items balanced across (source, country) strata."""
    if not items or target <= 0:
        return []

    if len(items) <= target:
        return items.copy()

    strata: dict[tuple[str, str], list[Item]] = {}
    for it in items:
        key = ((it.source or "news").lower(), (it.country or "us").lower())
        strata.setdefault(key, []).append(it)

    for k in strata:
        rng.shuffle(strata[k])

    num_strata = len(strata)
    base_per_stratum = target // num_strata
    remainder = target % num_strata

    selected: list[Item] = []
    pool: list[Item] = []

    for i, (key, stratum_items) in enumerate(strata.items()):
        quota = base_per_stratum + (1 if i < remainder else 0)
        take = min(quota, len(stratum_items))
        selected.extend(stratum_items[:take])
        pool.extend(stratum_items[take:])

    shortfall = target - len(selected)
    if shortfall > 0 and pool:
        rng.shuffle(pool)
        selected.extend(pool[:shortfall])

    return selected


def disperse_items(items: list[Item], rng: random.Random) -> list[Item]:
    """Shuffle/interleave items so similar items (source/country/title) are not adjacent."""
    buckets = defaultdict(list)
    for it in items:
        buckets[(it.source.lower(), it.country.lower(), (it.title or "")[:40].lower())].append(it)

    for k in buckets:
        rng.shuffle(buckets[k])

    sorted_keys = sorted(buckets.keys(), key=lambda k: len(buckets[k]), reverse=True)
    interleaved: list[Item] = []
    active_keys = list(sorted_keys)

    while active_keys:
        next_active = []
        for k in active_keys:
            if buckets[k]:
                interleaved.append(buckets[k].pop(0))
            if buckets[k]:
                next_active.append(k)
        active_keys = next_active

    # Additional pass: prevent adjacent items sharing same title or stratum if possible
    for i in range(1, len(interleaved)):
        prev = interleaved[i - 1]
        curr = interleaved[i]
        if prev.title == curr.title or (prev.source == curr.source and prev.country == curr.country):
            for j in range(i + 1, min(i + 20, len(interleaved))):
                cand = interleaved[j]
                if cand.title != prev.title and not (cand.source == prev.source and cand.country == prev.country):
                    interleaved[i], interleaved[j] = interleaved[j], interleaved[i]
                    break

    return interleaved


def stratified_sample(items: list[Item], target_count: int = 300, seed: int = 42) -> list[Item]:
    """Select sample balanced across source and country, with ~10% irrelevant items."""
    if not items:
        return []

    if len(items) <= target_count:
        return items.copy()

    rng = random.Random(seed)

    # Filter few-shot examples
    few_shots = load_few_shot_texts()
    filtered = filter_few_shots(items, few_shots)

    # Exclude near-duplicates
    deduped = remove_near_duplicates(filtered)

    # Separate into relevant candidates and potentially irrelevant items
    config = load_product_config()
    rel_items: list[Item] = []
    irrel_items: list[Item] = []

    for it in deduped:
        if is_relevant(it, it.product, config):
            rel_items.append(it)
        else:
            irrel_items.append(it)

    # Allocate ~10% for irrelevant items
    irrel_target = int(round(target_count * 0.10))
    rel_target = target_count - irrel_target

    sampled_rel = sample_balanced_strata(rel_items, rel_target, rng)
    sampled_irrel = sample_balanced_strata(irrel_items, irrel_target, rng)

    # If shortfall in irrelevant, top up from relevant, or vice-versa
    shortfall = target_count - (len(sampled_rel) + len(sampled_irrel))
    if shortfall > 0:
        remaining_rel = [it for it in rel_items if it not in sampled_rel]
        if remaining_rel:
            rng.shuffle(remaining_rel)
            sampled_rel.extend(remaining_rel[:shortfall])

    shortfall = target_count - (len(sampled_rel) + len(sampled_irrel))
    if shortfall > 0:
        remaining_irrel = [it for it in irrel_items if it not in sampled_irrel]
        if remaining_irrel:
            rng.shuffle(remaining_irrel)
            sampled_irrel.extend(remaining_irrel[:shortfall])

    # If still shortfall (e.g. in small unit tests where dummy titles collapsed),
    # top up from remaining pool so target_count is satisfied
    shortfall = target_count - (len(sampled_rel) + len(sampled_irrel))
    if shortfall > 0:
        selected_ids = {it.id for it in sampled_rel + sampled_irrel}
        remaining_items = [it for it in items if it.id not in selected_ids]
        if remaining_items:
            rng.shuffle(remaining_items)
            sampled_rel.extend(remaining_items[:shortfall])

    combined = sampled_rel + sampled_irrel

    # Disperse/shuffle so similar items are not next to each other
    return disperse_items(combined, rng)


def export_for_labelling(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    target_count: int = 300,
    session: Session | None = None,
    seed: int = 42,
) -> Path:
    """Export stratified items to a CSV file for human labelling."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        items = _get_items_from_db(session)
        sampled = stratified_sample(items, target_count=target_count, seed=seed)

        with open(out_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for item in sampled:
                # Truncate text to 800 characters
                truncated_text = (item.text or "")[:800]
                writer.writerow(
                    {
                        "item_id": item.id if item.id is not None else "",
                        "source": item.source,
                        "country": item.country,
                        "title": item.title,
                        "text": truncated_text,
                        "url": item.url,
                        "human_relevant": "",  # Must be empty
                        "human_sentiment": "",  # Must be empty
                        "notes": "",  # Must be empty
                    }
                )

        logger.info("Exported %d items for labelling to %s", len(sampled), out_file)
        return out_file
    finally:
        if own_session:
            session.close()


def main():
    parser = argparse.ArgumentParser(description="Export stratified items for labelling.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output CSV file (default: data/labelling/to_label.csv)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=300,
        help="Number of items to export (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable sampling (default: 42)",
    )
    args = parser.parse_args()

    out = export_for_labelling(output_path=args.output, target_count=args.count, seed=args.seed)
    print(f"Exported labelling dataset to: {out}")


if __name__ == "__main__":
    main()
