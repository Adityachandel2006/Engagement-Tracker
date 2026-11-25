"""Error analysis tool for model disagreements on the development set.

CRITICAL REQUIREMENT:
- Evaluates dev-set items ONLY (fixed 70/30 dev/test split, seed=42).
- The test set is held out and NEVER touched.
- Disagreements are grouped by (my label -> model label) and by source.
- Writes findings to data/errors.md including text, both labels, and the model's reason.
"""

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split
from sqlmodel import Session

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from eval.evaluate import generate_predictions, load_dataset

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = REPO_ROOT / "data" / "labelling" / "sample_labelled.csv"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "errors.md"


def extract_dev_set_errors(
    records: list[dict[str, Any]],
    seed: int = 42,
    test_size: float = 0.30,
) -> tuple[list[dict[str, Any]], int, int]:
    """Split into dev/test and extract disagreements on the DEV set ONLY.

    The test set is completely held out and never evaluated or inspected.
    """
    if len(records) >= 4:
        dev_records, test_records = train_test_split(
            records, test_size=test_size, random_state=seed, shuffle=True
        )
    else:
        dev_records = records
        test_records = []

    dev_count = len(dev_records)
    test_count = len(test_records)

    errors: list[dict[str, Any]] = []

    for r in dev_records:
        true_rel = bool(r.get("true_relevant", False))
        pred_rel = bool(r.get("pred_relevant", False))
        true_sent = str(r.get("true_sentiment") or "").strip().lower()
        pred_sent = str(r.get("pred_sentiment") or "").strip().lower()

        is_relevance_mismatch = true_rel != pred_rel
        is_sentiment_mismatch = true_rel and (true_sent != pred_sent)

        if is_relevance_mismatch:
            transition = f"relevant={str(true_rel).lower()} -> relevant={str(pred_rel).lower()}"
            err_record = dict(r)
            err_record["transition"] = transition
            errors.append(err_record)
        elif is_sentiment_mismatch:
            transition = f"{true_sent} -> {pred_sent}"
            err_record = dict(r)
            err_record["transition"] = transition
            errors.append(err_record)

    return errors, dev_count, test_count


def group_errors(
    errors: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Group errors by transition (my label -> model label) and by source."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for err in errors:
        transition = err.get("transition", "unknown -> unknown")
        source = str(err.get("source") or "unknown").lower()
        grouped[transition][source].append(err)

    return grouped


def format_errors_markdown(
    grouped_errors: dict[str, dict[str, list[dict[str, Any]]]],
    dev_count: int,
    total_errors: int,
    prompt_version: str,
    model_name: str,
) -> str:
    """Format errors into data/errors.md."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    err_rate = (total_errors / dev_count * 100) if dev_count > 0 else 0.0

    md = [
        "# Error Analysis Report (Development Set)",
        "",
        "> **Note:** This report inspects the **DEV set ONLY** (fixed 70/30 split, seed=42).",
        "> The **TEST set** is held out and was NOT accessed during this analysis.",
        "",
        f"- **Model:** `{model_name}`",
        f"- **Prompt Version:** `{prompt_version}`",
        f"- **Timestamp:** {now_str}",
        f"- **Dev Set Items Analyzed:** {dev_count}",
        f"- **Total Disagreements:** {total_errors}",
        f"- **Error Rate on Dev Set:** {err_rate:.1f}%",
        "",
        "## Summary of Disagreements",
        "",
        "| Transition (My Label → Model Label) | Error Count | Sources Affected |",
        "| :--- | :---: | :--- |",
    ]

    for transition, source_dict in sorted(grouped_errors.items()):
        cnt = sum(len(items) for items in source_dict.values())
        sources = ", ".join(f"`{s}` ({len(items)})" for s, items in sorted(source_dict.items()))
        md.append(f"| `{transition}` | {cnt} | {sources} |")

    md.extend(["", "---", "", "## Detailed Disagreements by Group", ""])

    for transition, source_dict in sorted(grouped_errors.items()):
        md.append(f"### Transition: `{transition}`")
        md.append("")

        for source, items in sorted(source_dict.items()):
            md.append(f"#### Source: `{source}` ({len(items)} items)")
            md.append("")

            for item in items:
                item_id = item.get("item_id", "N/A")
                title = item.get("title", "")
                text = item.get("text", "")
                country = str(item.get("country", "")).upper()
                true_rel = item.get("true_relevant")
                pred_rel = item.get("pred_relevant")
                true_sent = item.get("true_sentiment") or "N/A"
                pred_sent = item.get("pred_sentiment") or "N/A"
                reason = item.get("pred_reason") or "No explicit reason returned by model."

                md.extend([
                    f"**Item ID {item_id}** ({country})",
                    f"- **Title:** {title}",
                    f"- **Text:** \"{text}\"",
                    f"- **My Label (Ground Truth):** `{true_sent}` (Relevant: `{true_rel}`)",
                    f"- **Model Label (Prediction):** `{pred_sent}` (Relevant: `{pred_rel}`)",
                    f"- **Model Reason:** *\"{reason}\"*",
                    "",
                ])

    return "\n".join(md)


def run_error_analysis(
    csv_path: Path | str | None = None,
    prompt_version: str = "v1",
    model: str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    seed: int = 42,
    session: Session | None = None,
) -> tuple[int, Path]:
    """Execute error analysis on the dev set and write data/errors.md."""
    records = load_dataset(csv_path=csv_path, session=session)
    if not records:
        raise ValueError("No records found for error analysis.")

    model_name = (model or "v1-model").strip()
    records = generate_predictions(
        records,
        model=model_name,
        prompt_version=prompt_version,
        session=session,
    )

    errors, dev_count, test_count = extract_dev_set_errors(records, seed=seed)
    grouped = group_errors(errors)

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    md_content = format_errors_markdown(
        grouped,
        dev_count=dev_count,
        total_errors=len(errors),
        prompt_version=prompt_version,
        model_name=model_name,
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(
        f"Error analysis complete [{prompt_version}]: {len(errors)} disagreements found "
        f"across {dev_count} dev items (test set of {test_count} items untouched). "
        f"Written to {out_file}"
    )

    return len(errors), out_file


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dev-set model disagreements and write data/errors.md."
    )
    parser.add_argument("--prompt", "-p", default="v1", help="Prompt version to analyze (default: v1)")
    parser.add_argument("--model", "-m", help="Model name (e.g. gemini/gemini-1.5-flash)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Path to labelled dataset CSV")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output markdown path (default: data/errors.md)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dev/test split (default: 42)")

    args = parser.parse_args()

    run_error_analysis(
        csv_path=args.csv,
        prompt_version=args.prompt,
        model=args.model,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
