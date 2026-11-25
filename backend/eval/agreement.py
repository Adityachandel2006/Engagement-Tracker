"""Compute inter-annotator agreement between original and blind label sets.

Reads both label sets from the Label table and reports raw agreement,
Cohen's kappa (via sklearn.metrics.cohen_kappa_score), and confusion matrices
for relevance and sentiment.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sqlmodel import Session, select

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db import get_engine
from models import Label

SENTIMENT_CLASSES = ["positive", "negative", "neutral"]


def _safe_kappa(y1: list[Any], y2: list[Any], labels: list[Any]) -> float:
    """Compute Cohen's kappa, returning 1.0 if both sets are identical single-class."""
    if not y1:
        return 0.0
    kappa = float(cohen_kappa_score(y1, y2, labels=labels))
    if math.isnan(kappa):
        return 1.0 if y1 == y2 else 0.0
    return kappa


def compute_agreement(
    session: Session | None = None,
    old_labeller: str = "human",
    blind_labeller: str = "blind",
) -> dict[str, Any]:
    """Compare old_labeller vs blind_labeller labels in the Label table."""
    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        old_rows = session.exec(select(Label).where(Label.labeller == old_labeller)).all()
        blind_rows = session.exec(select(Label).where(Label.labeller == blind_labeller)).all()

        old_map = {r.item_id: r for r in old_rows}
        blind_map = {r.item_id: r for r in blind_rows}

        shared_ids = sorted(set(old_map.keys()) & set(blind_map.keys()))
        if not shared_ids:
            return {
                "paired_count": 0,
                "sentiment_count": 0,
                "relevance": {
                    "raw_agreement": 0.0,
                    "cohen_kappa": 0.0,
                    "confusion_matrix": [[0, 0], [0, 0]],
                    "labels": [True, False],
                },
                "sentiment": {
                    "raw_agreement": 0.0,
                    "cohen_kappa": 0.0,
                    "confusion_matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    "labels": SENTIMENT_CLASSES,
                },
            }

        y_old_rel = [bool(old_map[i].human_relevant) for i in shared_ids]
        y_blind_rel = [bool(blind_map[i].human_relevant) for i in shared_ids]

        rel_raw = sum(a == b for a, b in zip(y_old_rel, y_blind_rel)) / len(shared_ids)
        rel_kappa = _safe_kappa(y_old_rel, y_blind_rel, labels=[True, False])
        rel_cm = confusion_matrix(y_old_rel, y_blind_rel, labels=[True, False]).tolist()

        # Sentiment agreement on items where both sets marked relevant=True and have valid sentiment
        y_old_sent: list[str] = []
        y_blind_sent: list[str] = []
        for i in shared_ids:
            o_lbl = old_map[i]
            b_lbl = blind_map[i]
            if bool(o_lbl.human_relevant) and bool(b_lbl.human_relevant):
                o_s = (o_lbl.human_sentiment or "").strip().lower()
                b_s = (b_lbl.human_sentiment or "").strip().lower()
                if o_s in SENTIMENT_CLASSES and b_s in SENTIMENT_CLASSES:
                    y_old_sent.append(o_s)
                    y_blind_sent.append(b_s)

        if y_old_sent:
            sent_raw = sum(a == b for a, b in zip(y_old_sent, y_blind_sent)) / len(y_old_sent)
            sent_kappa = _safe_kappa(y_old_sent, y_blind_sent, labels=SENTIMENT_CLASSES)
            sent_cm = confusion_matrix(y_old_sent, y_blind_sent, labels=SENTIMENT_CLASSES).tolist()
        else:
            sent_raw = 0.0
            sent_kappa = 0.0
            sent_cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        return {
            "paired_count": len(shared_ids),
            "sentiment_count": len(y_old_sent),
            "relevance": {
                "raw_agreement": rel_raw,
                "cohen_kappa": rel_kappa,
                "confusion_matrix": rel_cm,
                "labels": [True, False],
            },
            "sentiment": {
                "raw_agreement": sent_raw,
                "cohen_kappa": sent_kappa,
                "confusion_matrix": sent_cm,
                "labels": SENTIMENT_CLASSES,
            },
        }
    finally:
        if own_session:
            session.close()


def format_agreement_report(metrics: dict[str, Any]) -> str:
    """Format agreement metrics and confusion matrices as a readable report."""
    lines = [
        f"Paired items: {metrics['paired_count']}",
        "",
        "=== Relevance Agreement ===",
        f"Raw Agreement: {metrics['relevance']['raw_agreement']:.4f}",
        f"Cohen's Kappa: {metrics['relevance']['cohen_kappa']:.4f}",
        "Confusion Matrix (Old rows x Blind cols; order: [True, False]):",
    ]
    rel_labels = ["True", "False"]
    lines.append("          " + "  ".join(f"{lbl:>6}" for lbl in rel_labels))
    for lbl, row in zip(rel_labels, metrics["relevance"]["confusion_matrix"]):
        lines.append(f"{lbl:<8}  " + "  ".join(f"{val:>6}" for val in row))

    lines.extend([
        "",
        f"=== Sentiment Agreement (n={metrics['sentiment_count']}) ===",
        f"Raw Agreement: {metrics['sentiment']['raw_agreement']:.4f}",
        f"Cohen's Kappa: {metrics['sentiment']['cohen_kappa']:.4f}",
        "Confusion Matrix (Old rows x Blind cols; order: [positive, negative, neutral]):",
        "            " + "  ".join(f"{lbl:>8}" for lbl in SENTIMENT_CLASSES),
    ])
    for lbl, row in zip(SENTIMENT_CLASSES, metrics["sentiment"]["confusion_matrix"]):
        lines.append(f"{lbl:<10}  " + "  ".join(f"{val:>8}" for val in row))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute agreement between old and blind labels.")
    parser.add_argument("--old-labeller", default="human", help="Original labeller name (default: human)")
    parser.add_argument("--blind-labeller", default="blind", help="Blind labeller name (default: blind)")
    args = parser.parse_args()

    metrics = compute_agreement(old_labeller=args.old_labeller, blind_labeller=args.blind_labeller)
    print(format_agreement_report(metrics))


if __name__ == "__main__":
    main()
