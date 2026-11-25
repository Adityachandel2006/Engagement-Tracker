"""Validate and import human labels from CSV into the Label table.

Enforces allowed values only:
- human_relevant: 1 or 0 (or True/False)
- human_sentiment: 'positive', 'negative', 'neutral' (or blank if human_relevant is 0/False)

Rejects any other value, reports the first bad row number, and reports how many rows are still blank.
"""

import argparse
import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session, select

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db import get_engine
from models import Label

logger = logging.getLogger(__name__)

ALLOWED_SENTIMENTS = {"positive", "negative", "neutral"}


def parse_boolean(val: str | bool | None, row_idx: int) -> bool:
    """Parse boolean value or raise ValueError if invalid."""
    if isinstance(val, bool):
        return val
    if val is None or str(val).strip() == "":
        raise ValueError(f"Row {row_idx}: 'human_relevant' cannot be empty.")

    cleaned = str(val).strip().lower()
    if cleaned in ("1", "true"):
        return True
    if cleaned in ("0", "false"):
        return False

    raise ValueError(
        f"Row {row_idx}: Invalid value '{val}' for human_relevant. Allowed values: 1 or 0."
    )


def parse_sentiment(val: str | None, relevant: bool, row_idx: int) -> str:
    """Validate allowed sentiment values."""
    cleaned = (val or "").strip().lower()
    if not relevant:
        # If item is not relevant, sentiment is blank or allowed to be neutral
        if cleaned and cleaned not in ALLOWED_SENTIMENTS:
            raise ValueError(
                f"Row {row_idx}: Invalid human_sentiment '{val}'. Allowed values: positive, negative or neutral."
            )
        return cleaned

    if not cleaned:
        raise ValueError(
            f"Row {row_idx}: 'human_sentiment' is required when human_relevant is 1."
        )

    if cleaned not in ALLOWED_SENTIMENTS:
        raise ValueError(
            f"Row {row_idx}: Invalid human_sentiment '{val}'. Allowed values: positive, negative or neutral."
        )

    return cleaned


def import_labels(
    csv_path: Path | str,
    session: Session | None = None,
    labeller: str = "human",
) -> int:
    """Validate allowed values in csv_path and upsert into the Label table.

    Rejects any invalid value, reports the first bad row number, and
    reports how many rows are still blank.
    Returns the count of successfully imported Label records.
    """
    file_path = Path(csv_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Label CSV file not found: {file_path}")

    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        validated_rows = []
        blank_count = 0

        with open(file_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required_cols = {"item_id", "human_relevant", "human_sentiment"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                missing = required_cols - set(reader.fieldnames or [])
                raise ValueError(f"CSV missing required columns: {missing}")

            for row_idx, row in enumerate(reader, start=2):  # line 2 is first data row
                raw_id = row.get("item_id", "").strip()
                if not raw_id:
                    continue  # skip empty id rows

                try:
                    item_id = int(raw_id)
                except ValueError:
                    raise ValueError(f"Row {row_idx}: 'item_id' must be an integer, got '{raw_id}'.")

                raw_relevant = (row.get("human_relevant") or "").strip()
                raw_sentiment = (row.get("human_sentiment") or "").strip()

                # Check if row is blank
                if raw_relevant == "":
                    blank_count += 1
                    continue

                # Validate human_relevant - raises on first bad value
                rel_bool = parse_boolean(raw_relevant, row_idx)

                # Check if sentiment is blank when relevant is 1/true
                if rel_bool and raw_sentiment == "":
                    blank_count += 1
                    continue

                # Validate human_sentiment - raises on first bad value
                sent_str = parse_sentiment(raw_sentiment, rel_bool, row_idx)

                validated_rows.append(
                    {
                        "item_id": item_id,
                        "human_relevant": rel_bool,
                        "human_sentiment": sent_str,
                    }
                )

        now_str = datetime.now(timezone.utc).isoformat()
        imported_count = 0

        for record in validated_rows:
            item_id = record["item_id"]
            existing = session.exec(
                select(Label).where(Label.item_id == item_id, Label.labeller == labeller)
            ).first()
            if existing:
                existing.human_relevant = record["human_relevant"]
                existing.human_sentiment = record["human_sentiment"]
                existing.labeller = labeller
                existing.labelled_at = now_str
                session.add(existing)
            else:
                new_label = Label(
                    item_id=item_id,
                    human_relevant=record["human_relevant"],
                    human_sentiment=record["human_sentiment"],
                    labeller=labeller,
                    labelled_at=now_str,
                )
                session.add(new_label)
            imported_count += 1

        session.commit()

        if blank_count > 0:
            logger.info("%d rows are still blank.", blank_count)
            print(f"{blank_count} rows are still blank.")
        else:
            print("0 rows are still blank.")

        logger.info("Imported %d labels from %s into Label table.", imported_count, file_path)
        return imported_count

    finally:
        if own_session:
            session.close()


def main():
    parser = argparse.ArgumentParser(description="Import and validate human labels from CSV.")
    parser.add_argument("csv_path", type=Path, help="Path to labelled CSV file")
    parser.add_argument("--labeller", default="human", help="Name or ID of labeller")
    args = parser.parse_args()

    count = import_labels(args.csv_path, labeller=args.labeller)
    print(f"Successfully validated and imported {count} labels from {args.csv_path}.")


if __name__ == "__main__":
    main()
