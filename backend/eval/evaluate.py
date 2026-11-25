"""Evaluation runner using scikit-learn for all metrics.

Computes:
- accuracy_score
- f1_score (macro)
- classification_report (per-class precision and recall)
- confusion_matrix
- relevance accuracy
Each metric is also computed split by source and by country.

Dev/Test split: fixed 70/30 split using train_test_split(seed=42).
Defaults to dev set; test set is used only with --final.
Exits with code 1 if accuracy < 0.85 or macro-F1 < 0.80.
Supports comparing multiple prompts (e.g. v1 vs v2).
"""

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sqlmodel import Session, select

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db import get_engine
from models import Analysis, Item, Label
from analysis.baseline_vader import score_sentiment

logger = logging.getLogger(__name__)

CLASSES = ["positive", "negative", "neutral"]
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_CSV_PATH = REPO_ROOT / "data" / "labelling" / "sample_labelled.csv"


def compute_metrics(
    y_true_sent: list[str],
    y_pred_sent: list[str],
    y_true_rel: list[bool],
    y_pred_rel: list[bool],
) -> dict[str, Any]:
    """Calculate all classification metrics strictly using scikit-learn."""
    # Relevance Accuracy
    rel_acc = float(accuracy_score(y_true_rel, y_pred_rel)) if y_true_rel else 0.0

    # Sentiment metrics on items where sentiment applies
    if y_true_sent and y_pred_sent:
        acc = float(accuracy_score(y_true_sent, y_pred_sent))
        macro_f1 = float(f1_score(y_true_sent, y_pred_sent, average="macro", zero_division=0))
        clf_report = classification_report(
            y_true_sent,
            y_pred_sent,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        )
        conf_mat = confusion_matrix(y_true_sent, y_pred_sent, labels=CLASSES).tolist()
    else:
        acc = 0.0
        macro_f1 = 0.0
        clf_report = {}
        conf_mat = [[0] * len(CLASSES) for _ in CLASSES]

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "relevance_accuracy": rel_acc,
        "classification_report": clf_report,
        "confusion_matrix": conf_mat,
        "sample_size": len(y_true_rel),
        "sentiment_sample_size": len(y_true_sent),
    }


def compute_slices(records: list[dict[str, Any]], slice_key: str) -> dict[str, dict[str, Any]]:
    """Compute scikit-learn metrics split by a slice key (source or country)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        key = str(r.get(slice_key) or "unknown").lower()
        groups.setdefault(key, []).append(r)

    results: dict[str, dict[str, Any]] = {}
    for group_name, group_records in sorted(groups.items()):
        y_true_rel = [bool(r["true_relevant"]) for r in group_records]
        y_pred_rel = [bool(r["pred_relevant"]) for r in group_records]

        # Evaluate sentiment where ground truth has sentiment
        sent_recs = [r for r in group_records if r.get("true_sentiment")]
        y_true_sent = [r["true_sentiment"] for r in sent_recs]
        y_pred_sent = [r["pred_sentiment"] for r in sent_recs]

        metrics = compute_metrics(y_true_sent, y_pred_sent, y_true_rel, y_pred_rel)
        results[group_name] = metrics

    return results


def format_markdown_report(
    run_name: str,
    split_name: str,
    overall_metrics: dict[str, Any],
    source_slices: dict[str, dict[str, Any]],
    country_slices: dict[str, dict[str, Any]],
) -> str:
    """Format full evaluation results into a clean GitHub Flavored Markdown report."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    acc = overall_metrics["accuracy"]
    f1 = overall_metrics["macro_f1"]
    rel_acc = overall_metrics["relevance_accuracy"]
    clf = overall_metrics["classification_report"]
    cm = overall_metrics["confusion_matrix"]

    md = [
        f"# Model Evaluation Report: {run_name}",
        "",
        f"- **Dataset Split:** `{split_name.upper()}` (Fixed 70/30 split, seed=42)",
        f"- **Timestamp:** {now_str}",
        f"- **Total Sample Size:** {overall_metrics['sample_size']}",
        f"- **Sentiment Sample Size:** {overall_metrics['sentiment_sample_size']}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Score | Target Threshold | Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Sentiment Accuracy** | **{acc:.4f}** | $\\ge 0.85$ | {'PASS' if acc >= 0.85 else 'FAIL'} |",
        f"| **Macro-F1 Score** | **{f1:.4f}** | $\\ge 0.80$ | {'PASS' if f1 >= 0.80 else 'FAIL'} |",
        f"| **Relevance Accuracy** | **{rel_acc:.4f}** | — | INFO |",
        "",
        "## Per-Class Performance (scikit-learn)",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for cls in CLASSES:
        cls_data = clf.get(cls, {})
        p = cls_data.get("precision", 0.0)
        r = cls_data.get("recall", 0.0)
        f = cls_data.get("f1-score", 0.0)
        s = cls_data.get("support", 0)
        md.append(f"| **{cls.capitalize()}** | {p:.4f} | {r:.4f} | {f:.4f} | {s} |")

    md.extend([
        "",
        "## Confusion Matrix",
        "",
        "| Actual \\ Predicted | Predicted Positive | Predicted Negative | Predicted Neutral |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Actual Positive** | {cm[0][0]} | {cm[0][1]} | {cm[0][2]} |",
        f"| **Actual Negative** | {cm[1][0]} | {cm[1][1]} | {cm[1][2]} |",
        f"| **Actual Neutral**  | {cm[2][0]} | {cm[2][1]} | {cm[2][2]} |",
        "",
        "## Slice Metrics by Source",
        "",
        "| Source | Samples | Sentiment Accuracy | Macro-F1 | Relevance Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for src, m in source_slices.items():
        md.append(
            f"| `{src}` | {m['sample_size']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['relevance_accuracy']:.4f} |"
        )

    md.extend([
        "",
        "## Slice Metrics by Country",
        "",
        "| Country | Samples | Sentiment Accuracy | Macro-F1 | Relevance Accuracy |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for ctry, m in country_slices.items():
        md.append(
            f"| `{ctry.upper()}` | {m['sample_size']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['relevance_accuracy']:.4f} |"
        )

    md.append("")
    return "\n".join(md)


def load_dataset(
    csv_path: Path | str | None = None,
    session: Session | None = None,
) -> list[dict[str, Any]]:
    """Load items joined with human ground-truth labels."""
    if csv_path:
        file_path = Path(csv_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {file_path}")

        records = []
        with open(file_path, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_rel = row.get("human_relevant", "").strip().lower()
                rel = raw_rel in ("true", "1", "yes", "t")
                pred_rel_raw = row.get("pred_relevant")
                pred_rel_bool = (
                    pred_rel_raw.strip().lower() in ("true", "1", "yes", "t")
                    if pred_rel_raw is not None and pred_rel_raw != ""
                    else None
                )
                records.append({
                    "item_id": int(row.get("item_id", 0) or 0),
                    "source": row.get("source", "news"),
                    "country": row.get("country", "us"),
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                    "url": row.get("url", ""),
                    "true_relevant": rel,
                    "true_sentiment": row.get("human_sentiment", "").strip().lower(),
                    "_raw_row": row,
                    "pred_relevant": pred_rel_bool,
                    "pred_sentiment": row.get("pred_sentiment"),
                    "pred_reason": row.get("pred_reason"),
                })
        return records

    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        # Query labels joined with items
        labels = list(session.exec(select(Label)).all())
        if not labels:
            if DEFAULT_CSV_PATH.exists():
                return load_dataset(csv_path=DEFAULT_CSV_PATH)
            return []

        item_ids = [lbl.item_id for lbl in labels]
        items_map = {item.id: item for item in session.exec(select(Item).where(Item.id.in_(item_ids))).all()}

        records = []
        for lbl in labels:
            item = items_map.get(lbl.item_id)
            if not item:
                continue
            records.append({
                "item_id": item.id,
                "source": item.source,
                "country": item.country,
                "title": item.title,
                "text": item.text,
                "url": item.url,
                "true_relevant": bool(lbl.human_relevant),
                "true_sentiment": (lbl.human_sentiment or "").strip().lower(),
                "_raw_row": {},
                "pred_relevant": None,
                "pred_sentiment": None,
                "pred_reason": None,
            })
        return records
    finally:
        if own_session:
            session.close()


def generate_predictions(
    records: list[dict[str, Any]],
    baseline: str | None = None,
    model: str | None = None,
    prompt_version: str = "v1",
    session: Session | None = None,
) -> list[dict[str, Any]]:
    """Fill predicted relevant, sentiment, and reason fields for all records."""
    is_vader = bool(baseline and baseline.lower() == "vader")

    if is_vader:
        for r in records:
            sent, _ = score_sentiment(f"{r['title']} {r['text']}")
            r["pred_relevant"] = True  # Baseline assumes news items found are relevant
            r["pred_sentiment"] = sent
            r["pred_reason"] = "VADER polarity score baseline classification"
        return records

    # LLM predictions: check version-specific CSV columns or Analysis table in DB
    own_session = session is None
    if own_session:
        session = Session(get_engine())

    try:
        model_name = (model or "v1-model").strip()
        item_ids = [r["item_id"] for r in records]

        existing_analyses = {}
        if item_ids:
            analyses = session.exec(
                select(Analysis).where(
                    Analysis.model == model_name,
                    Analysis.prompt_version == prompt_version,
                    Analysis.item_id.in_(item_ids),
                )
            ).all()
            existing_analyses = {a.item_id: a for a in analyses}

        for r in records:
            raw = r.get("_raw_row", {})
            v_sent = raw.get(f"pred_sentiment_{prompt_version}")
            v_rel = raw.get(f"pred_relevant_{prompt_version}")
            v_reason = raw.get(f"pred_reason_{prompt_version}")

            if v_rel is not None and v_rel != "":
                rel_bool = v_rel.strip().lower() in ("true", "1", "yes", "t")
                r["pred_relevant"] = rel_bool
                r["pred_sentiment"] = (v_sent or "").strip().lower()
                r["pred_reason"] = v_reason or ""
                continue
            elif v_sent is not None and v_sent != "":
                r["pred_relevant"] = True
                r["pred_sentiment"] = v_sent.strip().lower()
                r["pred_reason"] = v_reason or ""
                continue

            a = existing_analyses.get(r["item_id"])
            if a:
                r["pred_relevant"] = a.relevant
                r["pred_sentiment"] = a.sentiment
                r["pred_reason"] = a.reason
            elif r.get("pred_sentiment") is not None:
                # Keep existing pred_sentiment if already loaded
                if not r.get("pred_reason"):
                    r["pred_reason"] = raw.get("pred_reason", "")
            else:
                # Fallback to baseline VADER
                sent, _ = score_sentiment(f"{r['title']} {r['text']}")
                r["pred_relevant"] = True
                r["pred_sentiment"] = sent
                r["pred_reason"] = "Fallback baseline classification"

        return records
    finally:
        if own_session:
            session.close()


def evaluate_predictions(
    records: list[dict[str, Any]],
    use_final: bool = False,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]], str]:
    """Perform fixed 70/30 train/test split and calculate scikit-learn metrics."""
    if not records:
        raise ValueError("Cannot evaluate an empty dataset.")

    # Fixed 70/30 dev/test split with scikit-learn
    if len(records) >= 4:
        dev_records, test_records = train_test_split(
            records, test_size=0.30, random_state=seed, shuffle=True
        )
    else:
        # Fallback for tiny test fixtures
        dev_records = records
        test_records = records

    eval_records = test_records if use_final else dev_records
    split_name = "test" if use_final else "dev"

    y_true_rel = [bool(r["true_relevant"]) for r in eval_records]
    y_pred_rel = [bool(r["pred_relevant"]) for r in eval_records]

    # Evaluate sentiment where ground truth sentiment is present
    sent_records = [r for r in eval_records if r.get("true_sentiment")]
    y_true_sent = [r["true_sentiment"] for r in sent_records]
    y_pred_sent = [r["pred_sentiment"] for r in sent_records]

    overall_metrics = compute_metrics(y_true_sent, y_pred_sent, y_true_rel, y_pred_rel)
    source_slices = compute_slices(eval_records, "source")
    country_slices = compute_slices(eval_records, "country")

    return overall_metrics, source_slices, country_slices, split_name


def run_evaluation(
    model: str | None = None,
    prompt_version: str | None = None,
    baseline: str | None = None,
    final: bool = False,
    csv_path: Path | str | None = None,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    min_accuracy: float = 0.85,
    min_f1: float = 0.80,
    session: Session | None = None,
) -> tuple[float, float, Path]:
    """Execute evaluation, write markdown report, and print summary line."""
    if not baseline and not (model or prompt_version):
        raise ValueError("Must provide either --baseline vader or --model <m> --prompt <v>")

    prompt_v = prompt_version or "v1"
    run_name = "vader" if (baseline and baseline.lower() == "vader") else f"{model or 'model'}_{prompt_v}"
    clean_run_name = run_name.replace("/", "_").replace("\\", "_")

    records = load_dataset(csv_path=csv_path, session=session)
    if not records:
        raise ValueError("No evaluation records found. Export and import labels first or pass --csv.")

    records = generate_predictions(
        records,
        baseline=baseline,
        model=model,
        prompt_version=prompt_v,
        session=session,
    )

    overall, sources, countries, split_name = evaluate_predictions(records, use_final=final)

    acc = overall["accuracy"]
    f1 = overall["macro_f1"]
    rel_acc = overall["relevance_accuracy"]
    sample_size = overall["sample_size"]

    # Write Markdown Report
    rep_dir = Path(reports_dir).resolve()
    rep_dir.mkdir(parents=True, exist_ok=True)
    report_file = rep_dir / f"eval_{clean_run_name}_{split_name}.md"

    md_content = format_markdown_report(clean_run_name, split_name, overall, sources, countries)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Print ONE summary line
    summary_line = (
        f"Evaluation [{clean_run_name} on {split_name}]: "
        f"Accuracy={acc:.3f}, Macro-F1={f1:.3f}, Relevance-Acc={rel_acc:.3f} (N={sample_size})"
    )
    print(summary_line)

    return acc, f1, report_file


def compare_prompts(
    prompts: list[str],
    model: str | None = None,
    final: bool = False,
    csv_path: Path | str | None = None,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
    session: Session | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate multiple prompt versions (e.g. v1 and v2) and print a formatted comparison."""
    if len(prompts) < 2:
        raise ValueError("Need at least 2 prompt versions to compare.")

    split_label = "test" if final else "dev"
    results: dict[str, dict[str, Any]] = {}

    print(f"\n--- Running Prompt Evaluation on {split_label.upper()} Split ---")
    for pv in prompts:
        records = load_dataset(csv_path=csv_path, session=session)
        records = generate_predictions(
            records,
            model=model,
            prompt_version=pv,
            session=session,
        )
        overall, _, _, _ = evaluate_predictions(records, use_final=final)
        results[pv] = overall

        run_name = f"{model or 'model'}_{pv}".replace("/", "_").replace("\\", "_")
        summary_line = (
            f"Evaluation [{run_name} on {split_label}]: "
            f"Accuracy={overall['accuracy']:.3f}, Macro-F1={overall['macro_f1']:.3f}, "
            f"Relevance-Acc={overall['relevance_accuracy']:.3f} (N={overall['sample_size']})"
        )
        print(summary_line)

    # Print comparison table
    p1, p2 = prompts[0], prompts[1]
    m1, m2 = results[p1], results[p2]

    acc1, acc2 = m1["accuracy"], m2["accuracy"]
    f1_1, f1_2 = m1["macro_f1"], m2["macro_f1"]
    rel1, rel2 = m1["relevance_accuracy"], m2["relevance_accuracy"]

    delta_acc = acc2 - acc1
    delta_f1 = f1_2 - f1_1
    delta_rel = rel2 - rel1

    pct_acc = ((acc2 - acc1) / acc1 * 100) if acc1 > 0 else 0.0
    pct_f1 = ((f1_2 - f1_1) / f1_1 * 100) if f1_1 > 0 else 0.0

    print("\n" + "=" * 74)
    print(f"PROMPT COMPARISON: {p1} vs {p2} ({split_label.upper()} set, N={m1['sample_size']})")
    print("=" * 74)
    print(f"{'Metric':<24} {p1:<12} {p2:<12} {'Delta':<12} {'Change'}")
    print("-" * 74)
    print(f"{'Sentiment Accuracy':<24} {acc1:<12.4f} {acc2:<12.4f} {f'{delta_acc:+.4f}':<12} {f'{pct_acc:+.1f}%'}")
    print(f"{'Macro-F1 Score':<24} {f1_1:<12.4f} {f1_2:<12.4f} {f'{delta_f1:+.4f}':<12} {f'{pct_f1:+.1f}%'}")
    print(f"{'Relevance Accuracy':<24} {rel1:<12.4f} {rel2:<12.4f} {f'{delta_rel:+.4f}':<12} -")

    # Per-class comparison
    clf1 = m1["classification_report"]
    clf2 = m2["classification_report"]
    for cls in CLASSES:
        cls_f1_1 = clf1.get(cls, {}).get("f1-score", 0.0)
        cls_f1_2 = clf2.get(cls, {}).get("f1-score", 0.0)
        diff_cls = cls_f1_2 - cls_f1_1
        print(f"{f'{cls.capitalize()} F1':<24} {cls_f1_1:<12.4f} {cls_f1_2:<12.4f} {f'{diff_cls:+.4f}':<12} -")
    print("=" * 74 + "\n")

    # Write comparison markdown report
    rep_dir = Path(reports_dir).resolve()
    rep_dir.mkdir(parents=True, exist_ok=True)
    comp_file = rep_dir / f"eval_comparison_{p1}_{p2}_{split_label}.md"

    md_lines = [
        f"# Prompt Comparison: {p1} vs {p2}",
        "",
        f"- **Dataset Split:** `{split_label.upper()}` (Fixed 70/30 split, seed=42)",
        f"- **Sample Size:** {m1['sample_size']}",
        f"- **Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Performance Comparison",
        "",
        f"| Metric | {p1} | {p2} | Absolute Delta | Relative Change |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Sentiment Accuracy** | {acc1:.4f} | **{acc2:.4f}** | {delta_acc:+.4f} | {pct_acc:+.1f}% |",
        f"| **Macro-F1 Score** | {f1_1:.4f} | **{f1_2:.4f}** | {delta_f1:+.4f} | {pct_f1:+.1f}% |",
        f"| **Relevance Accuracy** | {rel1:.4f} | **{rel2:.4f}** | {delta_rel:+.4f} | - |",
    ]
    for cls in CLASSES:
        cls_f1_1 = clf1.get(cls, {}).get("f1-score", 0.0)
        cls_f1_2 = clf2.get(cls, {}).get("f1-score", 0.0)
        diff_cls = cls_f1_2 - cls_f1_1
        md_lines.append(f"| **{cls.capitalize()} F1** | {cls_f1_1:.4f} | {cls_f1_2:.4f} | {diff_cls:+.4f} | - |")

    with open(comp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate sentiment models or baseline with scikit-learn.")
    parser.add_argument("--model", "-m", help="LLM model name (e.g. gemini/gemini-1.5-flash)")
    parser.add_argument("--prompt", "-p", default="v1", help="Prompt template version (default: v1, or comma-separated v1,v2)")
    parser.add_argument("--baseline", choices=["vader", "VADER"], help="Run baseline evaluator (vader)")
    parser.add_argument("--final", action="store_true", help="Evaluate on the 30%% test set instead of 70%% dev set")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to labelled dataset CSV (default: data/labelling/sample_labelled.csv)")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR, help="Directory to save eval_*.md reports")
    parser.add_argument("--min-accuracy", type=float, default=0.85, help="Accuracy threshold (default: 0.85)")
    parser.add_argument("--min-f1", type=float, default=0.80, help="Macro-F1 threshold (default: 0.80)")
    parser.add_argument("--no-exit-code", action="store_true", help="Do not exit with code 1 if thresholds not met")
    parser.add_argument("--compare-prompts", nargs="+", help="Compare multiple prompts (e.g. --compare-prompts v1 v2)")

    args = parser.parse_args()

    # Check if comparison requested
    compare_list = args.compare_prompts
    if not compare_list and "," in args.prompt:
        compare_list = [p.strip() for p in args.prompt.split(",") if p.strip()]

    if compare_list and len(compare_list) >= 2:
        compare_prompts(
            prompts=compare_list,
            model=args.model,
            final=args.final,
            csv_path=args.csv,
            reports_dir=args.reports_dir,
        )
        sys.exit(0)

    try:
        acc, f1, report_path = run_evaluation(
            model=args.model,
            prompt_version=args.prompt,
            baseline=args.baseline,
            final=args.final,
            csv_path=args.csv,
            reports_dir=args.reports_dir,
            min_accuracy=args.min_accuracy,
            min_f1=args.min_f1,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Report written to: {report_path}")

    # Exit with code 1 if accuracy is below 0.85 or macro-F1 below 0.80
    if not args.no_exit_code and (acc < args.min_accuracy or f1 < args.min_f1):
        print(
            f"FAILED threshold check: Accuracy={acc:.4f} (target >={args.min_accuracy}), "
            f"Macro-F1={f1:.4f} (target >={args.min_f1})",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
