"""Export a blind relabelling sample from already-labelled items.

Samples 120 items stratified by current human_relevant / human_sentiment
(~10 irrelevant, ~20 negative, ~40 neutral, ~50 positive) with a fixed seed (7),
shuffles them, and writes data/labelling/blind_relabel.csv with NO existing
labels, notes, scores, or model/VADER output.
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

from sqlmodel import Session, select

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db import get_engine
from models import Item, Label

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = BACKEND_DIR.parent / "data" / "labelling" / "blind_relabel.csv"
COLUMNS = [
    "item_id",
    "source",
    "country",
    "title",
    "text",
    "url",
    "human_relevant",
    "human_sentiment",
]


def sample_blind_items(
    labelled_pairs: list[tuple[Item, Label]],
    target_count: int = 120,
    seed: int = 7,
) -> list[Item]:
    """Sample target_count items stratified by existing human_relevant and human_sentiment."""
    if not labelled_pairs or target_count <= 0:
        return []

    rng = random.Random(seed)

    buckets: dict[str, list[Item]] = {
        "irrelevant": [],
        "negative": [],
        "neutral": [],
        "positive": [],
    }

    for item, lbl in labelled_pairs:
        if not lbl.human_relevant:
            buckets["irrelevant"].append(item)
        else:
            sent = (lbl.human_sentiment or "").strip().lower()
            if sent in ("negative", "neutral", "positive"):
                buckets[sent].append(item)

    for k in buckets:
        rng.shuffle(buckets[k])

    # Default targets for 120 items: 10 irrelevant, 20 negative, 40 neutral, 50 positive
    if target_count == 120:
        quotas = {"irrelevant": 10, "negative": 20, "neutral": 40, "positive": 50}
    else:
        irrel_q = round(target_count * (10 / 120))
        neg_q = round(target_count * (20 / 120))
        neu_q = round(target_count * (40 / 120))
        pos_q = max(0, target_count - irrel_q - neg_q - neu_q)
        quotas = {"irrelevant": irrel_q, "negative": neg_q, "neutral": neu_q, "positive": pos_q}

    selected: list[Item] = []
    leftover: list[Item] = []

    for key in ("irrelevant", "negative", "neutral", "positive"):
        q = quotas[key]
        pool = buckets[key]
        take = min(q, len(pool))
        selected.extend(pool[:take])
        leftover.extend(pool[take:])

    shortfall = target_count - len(selected)
    if shortfall > 0 and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[:shortfall])

    rng.shuffle(selected)
    return selected


def export_blind(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    target_count: int = 120,
    seed: int = 7,
    source_labeller: str = "human",
    session: Session | None = None,
) -> Path:
    """Export blind relabelling CSV without any existing labels or predictions."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        labels = list(
            session.exec(select(Label).where(Label.labeller == source_labeller)).all()
        )
        if not labels:
            # Fallback to any non-blind labels if source_labeller not matched
            labels = list(
                session.exec(select(Label).where(Label.labeller != "blind")).all()
            )

        item_ids = [lbl.item_id for lbl in labels]
        items_map: dict[int, Item] = {}
        if item_ids:
            items_map = {
                it.id: it
                for it in session.exec(select(Item).where(Item.id.in_(item_ids))).all()
                if it.id is not None
            }

        pairs = [
            (items_map[lbl.item_id], lbl)
            for lbl in labels
            if lbl.item_id in items_map
        ]

        sampled = sample_blind_items(pairs, target_count=target_count, seed=seed)

        with open(out_file, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for item in sampled:
                writer.writerow(
                    {
                        "item_id": item.id if item.id is not None else "",
                        "source": item.source,
                        "country": item.country,
                        "title": item.title,
                        "text": (item.text or "")[:800],
                        "url": item.url,
                        "human_relevant": "",
                        "human_sentiment": "",
                    }
                )

        logger.info("Exported %d blind relabelling items to %s", len(sampled), out_file)
        return out_file
    finally:
        if own_session:
            session.close()


def main():
    parser = argparse.ArgumentParser(description="Export blind relabelling dataset.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output CSV file (default: data/labelling/blind_relabel.csv)",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=120,
        help="Number of items to export (default: 120)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7)",
    )
    parser.add_argument(
        "--source-labeller",
        default="human",
        help="Labeller name of existing labels to sample from (default: human)",
    )
    args = parser.parse_args()

    out = export_blind(
        output_path=args.output,
        target_count=args.count,
        seed=args.seed,
        source_labeller=args.source_labeller,
    )
    print(f"Exported blind relabelling dataset to: {out}")


if __name__ == "__main__":
    main()
