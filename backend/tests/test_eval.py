"""Tests for eval package: scikit-learn metrics, stratified export, label import, and evaluation runner."""

import csv
from pathlib import Path
import pytest
from sqlmodel import Session, SQLModel, create_engine, select

from models import Item, Label
from eval.evaluate import compute_metrics, evaluate_predictions, format_markdown_report, run_evaluation
from eval.export_blind import export_blind
from eval.export_for_labelling import export_for_labelling, stratified_sample
from eval.agreement import compute_agreement, format_agreement_report
from eval.import_labels import import_labels, parse_boolean, parse_sentiment


@pytest.fixture()
def db_session():
    """In-memory SQLite session with all tables created."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


# ── 1. Hand-checked metrics test ──────────────────────────────────────────

def test_metrics_on_tiny_hand_checked_example():
    """Verify scikit-learn metrics against hand-calculated values on a tiny example.

    5 items with sentiment:
    1: True: positive, Pred: positive (TP pos)
    2: True: positive, Pred: positive (TP pos)
    3: True: negative, Pred: negative (TP neg)
    4: True: neutral,  Pred: neutral  (TP neu)
    5: True: neutral,  Pred: positive (FP pos, FN neu)

    Summary:
    - 4 correct out of 5 -> Accuracy = 4/5 = 0.80
    - Precision:
        pos: 2/3 ≈ 0.6667
        neg: 1/1 = 1.0000
        neu: 1/1 = 1.0000
    - Recall:
        pos: 2/2 = 1.0000
        neg: 1/1 = 1.0000
        neu: 1/2 = 0.5000
    - F1-scores:
        pos: 2 * (2/3 * 1) / (2/3 + 1) = 4/5 = 0.8000
        neg: 1.0000
        neu: 2 * (1 * 0.5) / (1 + 0.5) = 2/3 ≈ 0.6667
    - Macro-F1 = (0.8 + 1.0 + 0.66667) / 3 = 2.46667 / 3 ≈ 0.8222

    Relevance (6 items):
    1-5: True=True, Pred=True
    6:   True=False, Pred=False
    - Relevance Accuracy = 6/6 = 1.00
    """
    y_true_sent = ["positive", "positive", "negative", "neutral", "neutral"]
    y_pred_sent = ["positive", "positive", "negative", "neutral", "positive"]
    y_true_rel = [True, True, True, True, True, False]
    y_pred_rel = [True, True, True, True, True, False]

    metrics = compute_metrics(y_true_sent, y_pred_sent, y_true_rel, y_pred_rel)

    # Assert exact hand-checked values
    assert pytest.approx(metrics["accuracy"], rel=1e-3) == 0.80
    assert pytest.approx(metrics["macro_f1"], rel=1e-3) == 0.8222
    assert pytest.approx(metrics["relevance_accuracy"], rel=1e-3) == 1.00

    report = metrics["classification_report"]
    assert pytest.approx(report["positive"]["precision"], rel=1e-3) == 2 / 3
    assert pytest.approx(report["positive"]["recall"], rel=1e-3) == 1.0
    assert pytest.approx(report["positive"]["f1-score"], rel=1e-3) == 0.80

    assert pytest.approx(report["negative"]["precision"], rel=1e-3) == 1.0
    assert pytest.approx(report["negative"]["recall"], rel=1e-3) == 1.0
    assert pytest.approx(report["negative"]["f1-score"], rel=1e-3) == 1.0

    assert pytest.approx(report["neutral"]["precision"], rel=1e-3) == 1.0
    assert pytest.approx(report["neutral"]["recall"], rel=1e-3) == 0.5
    assert pytest.approx(report["neutral"]["f1-score"], rel=1e-3) == 2 / 3

    # Check confusion matrix: [[TP_pos, FN_pos->neg, FN_pos->neu], [FP_pos<-neg...]]
    cm = metrics["confusion_matrix"]
    # classes: ["positive", "negative", "neutral"]
    assert cm == [
        [2, 0, 0],  # actual positive: 2 pred positive
        [0, 1, 0],  # actual negative: 1 pred negative
        [1, 0, 1],  # actual neutral:  1 pred positive, 1 pred neutral
    ]


# ── 2. Fixed 70/30 Dev/Test Split test ───────────────────────────────────

def test_dev_test_split_seed_42():
    records = []
    for i in range(20):
        records.append({
            "item_id": i + 1,
            "source": "news" if i % 2 == 0 else "reddit",
            "country": "us" if i % 2 == 0 else "gb",
            "true_relevant": True,
            "pred_relevant": True,
            "true_sentiment": "positive" if i % 3 == 0 else ("negative" if i % 3 == 1 else "neutral"),
            "pred_sentiment": "positive" if i % 3 == 0 else ("negative" if i % 3 == 1 else "neutral"),
        })

    # Default dev split (70% of 20 = 14)
    dev_overall, _, _, dev_name = evaluate_predictions(records, use_final=False, seed=42)
    assert dev_name == "dev"
    assert dev_overall["sample_size"] == 14

    # Final test split (30% of 20 = 6)
    test_overall, _, _, test_name = evaluate_predictions(records, use_final=True, seed=42)
    assert test_name == "test"
    assert test_overall["sample_size"] == 6


# ── 3. Export for labelling tests ────────────────────────────────────────

def test_export_for_labelling_schema_and_blanks(tmp_path: Path, db_session: Session):
    for i in range(1, 31):
        item = Item(
            id=i,
            source="news" if i % 2 == 0 else "reddit",
            source_name="TestSrc",
            title=f"Sample title {i}",
            text=f"Sample text {i}",
            url=f"https://example.com/{i}",
            country="us" if i % 3 == 0 else ("gb" if i % 3 == 1 else "in"),
            product="nike running shoes",
            content_hash=f"hash_{i}",
        )
        db_session.add(item)
    db_session.commit()

    out_file = tmp_path / "to_label.csv"
    export_for_labelling(output_path=out_file, target_count=15, session=db_session)

    assert out_file.exists()
    with open(out_file, encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 15
        for row in reader:
            # Check required columns
            assert "item_id" in row
            assert "source" in row
            assert "country" in row
            assert "title" in row
            assert "text" in row
            assert "url" in row
            assert "notes" in row
            assert len(row["text"]) <= 800
            # Never include predictions; human labels and notes MUST be blank
            assert row["human_relevant"] == ""
            assert row["human_sentiment"] == ""
            assert row["notes"] == ""
            assert "pred_relevant" not in row
            assert "pred_sentiment" not in row


def test_exporter_excludes_labelled_and_few_shots(tmp_path: Path, db_session: Session):
    """Verify exporter excludes already labelled items and few-shots, and includes ~10% irrelevant."""
    # Item 1 already labelled
    item1 = Item(id=1, source="news", country="us", title="Title 1", text="Text 1", url="http://1", product="nike running shoes", content_hash="h1")
    # Item 2 is a few-shot text
    item2 = Item(id=2, source="news", country="us", title="Title 2", text="Ran my first half-marathon in the Pegasus 40 and my feet felt incredible the whole way through. Best cushion ever.", url="http://2", product="nike running shoes", content_hash="h2")
    # Item 3 is relevant
    item3 = Item(id=3, source="news", country="us", title="Nike Pegasus review", text="The Pegasus 40 is a great daily running shoe.", url="http://3", product="nike running shoes", content_hash="h3")
    # Item 4 is irrelevant (passing mention / different topic)
    item4 = Item(id=4, source="news", country="us", title="Local weather report", text="Sunny skies today in the valley with light breeze.", url="http://4", product="nike running shoes", content_hash="h4")
    
    db_session.add(item1)
    db_session.add(item2)
    db_session.add(item3)
    db_session.add(item4)
    db_session.commit()

    # Add Label for item1
    db_session.add(Label(item_id=1, human_relevant=True, human_sentiment="positive", labeller="human"))
    db_session.commit()

    out_file = tmp_path / "to_label.csv"
    export_for_labelling(output_path=out_file, target_count=2, session=db_session, seed=42)

    with open(out_file, encoding="utf-8-sig") as f:
        reader = list(csv.DictReader(f))
        item_ids = [int(r["item_id"]) for r in reader]
        assert 1 not in item_ids  # labelled item excluded
        assert 2 not in item_ids  # few-shot item excluded
        assert 3 in item_ids or 4 in item_ids


# ── 4. Import labels tests ───────────────────────────────────────────────

def test_import_labels_validation_and_storage(tmp_path: Path, db_session: Session):
    # Valid CSV
    csv_file = tmp_path / "labelled.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["item_id", "source", "country", "title", "text", "url", "human_relevant", "human_sentiment"],
        )
        writer.writeheader()
        writer.writerow({
            "item_id": 101,
            "source": "news",
            "country": "us",
            "title": "Good shoe",
            "text": "Great shoes",
            "url": "http://example.com/101",
            "human_relevant": "true",
            "human_sentiment": "positive",
        })
        writer.writerow({
            "item_id": 102,
            "source": "reddit",
            "country": "gb",
            "title": "Stock market news",
            "text": "Stocks are up",
            "url": "http://example.com/102",
            "human_relevant": "false",
            "human_sentiment": "",
        })

    count = import_labels(csv_file, session=db_session, labeller="tester")
    assert count == 2

    # Verify storage in DB
    lbl101 = db_session.exec(select(Label).where(Label.item_id == 101)).first()
    assert lbl101 is not None
    assert lbl101.human_relevant is True
    assert lbl101.human_sentiment == "positive"
    assert lbl101.labeller == "tester"

    lbl102 = db_session.exec(select(Label).where(Label.item_id == 102)).first()
    assert lbl102 is not None
    assert lbl102.human_relevant is False


def test_import_labels_rejects_invalid_sentiment(tmp_path: Path, db_session: Session):
    bad_csv = tmp_path / "bad_sentiment.csv"
    with open(bad_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "human_relevant", "human_sentiment"])
        writer.writeheader()
        writer.writerow({"item_id": 1, "human_relevant": "true", "human_sentiment": "super_happy"})

    with pytest.raises(ValueError) as exc:
        import_labels(bad_csv, session=db_session)
    assert "Invalid human_sentiment" in str(exc.value)


def test_import_labels_rejects_invalid_boolean(tmp_path: Path, db_session: Session):
    bad_csv = tmp_path / "bad_bool.csv"
    with open(bad_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "human_relevant", "human_sentiment"])
        writer.writeheader()
        writer.writerow({"item_id": 1, "human_relevant": "maybe", "human_sentiment": "positive"})

    with pytest.raises(ValueError) as exc:
        import_labels(bad_csv, session=db_session)
    assert "Invalid value 'maybe' for human_relevant" in str(exc.value)


def test_import_labels_reports_blank_rows(tmp_path: Path, db_session: Session, capsys):
    csv_file = tmp_path / "partially_blank.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "human_relevant", "human_sentiment"])
        writer.writeheader()
        writer.writerow({"item_id": 201, "human_relevant": "1", "human_sentiment": "positive"})
        writer.writerow({"item_id": 202, "human_relevant": "", "human_sentiment": ""})
        writer.writerow({"item_id": 203, "human_relevant": "1", "human_sentiment": ""})

    count = import_labels(csv_file, session=db_session)
    assert count == 1
    captured = capsys.readouterr()
    assert "2 rows are still blank" in captured.out


# ── 5. Evaluation runner & Report generation tests ───────────────────────

def test_run_evaluation_generates_report(tmp_path: Path):
    sample_csv = tmp_path / "eval_sample.csv"
    with open(sample_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id", "source", "country", "title", "text", "url",
                "human_relevant", "human_sentiment", "pred_relevant", "pred_sentiment"
            ],
        )
        writer.writeheader()
        for i in range(20):
            writer.writerow({
                "item_id": i + 1,
                "source": "news" if i % 2 == 0 else "reddit",
                "country": "us" if i % 2 == 0 else "in",
                "title": f"Title {i}",
                "text": f"This shoe is {'fantastic and awesome' if i % 2 == 0 else 'terrible and broken'}",
                "url": f"http://example.com/{i}",
                "human_relevant": "true",
                "human_sentiment": "positive" if i % 2 == 0 else "negative",
                "pred_relevant": "true",
                "pred_sentiment": "positive" if i % 2 == 0 else "negative",
            })

    reports_dir = tmp_path / "reports"
    acc, f1, report_file = run_evaluation(
        baseline="vader",
        csv_path=sample_csv,
        reports_dir=reports_dir,
    )

    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "# Model Evaluation Report: vader" in content
    assert "## Confusion Matrix" in content
    assert "## Slice Metrics by Source" in content
    assert "## Slice Metrics by Country" in content
    assert acc >= 0.85
    assert f1 >= 0.80


# ── 6. Error analysis & Prompt comparison tests ──────────────────────────

def test_error_analysis_inspects_dev_set_only():
    """Verify that error analysis inspects the dev set only and never touches the test set."""
    from eval.error_analysis import extract_dev_set_errors, group_errors

    records = []
    # 20 items: dev set gets 14 items, test set gets 6 items
    for i in range(1, 21):
        # Introduce intentional disagreement for even item IDs
        records.append({
            "item_id": i,
            "source": "reddit" if i % 2 == 0 else "news",
            "country": "us",
            "title": f"Item {i}",
            "text": f"Review {i}",
            "true_relevant": True,
            "pred_relevant": True,
            "true_sentiment": "negative" if i % 2 == 0 else "positive",
            "pred_sentiment": "neutral" if i % 2 == 0 else "positive",
            "pred_reason": f"Model reason for item {i}",
        })

    errors, dev_count, test_count = extract_dev_set_errors(records, seed=42, test_size=0.30)

    assert dev_count == 14
    assert test_count == 6  # Test set is 6 items
    assert len(errors) > 0

    # Ensure none of the test set items appear in errors
    from sklearn.model_selection import train_test_split
    _, test_records = train_test_split(records, test_size=0.30, random_state=42, shuffle=True)
    test_ids = {r["item_id"] for r in test_records}

    for err in errors:
        assert err["item_id"] not in test_ids, f"Test item {err['item_id']} leaked into error analysis!"

    # Test grouping by transition and by source
    grouped = group_errors(errors)
    assert "negative -> neutral" in grouped
    assert "reddit" in grouped["negative -> neutral"]


def test_run_error_analysis_writes_file(tmp_path: Path):
    from eval.error_analysis import run_error_analysis

    csv_file = tmp_path / "errors_sample.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id", "source", "country", "title", "text", "url",
                "human_relevant", "human_sentiment", "pred_relevant", "pred_sentiment", "pred_reason"
            ],
        )
        writer.writeheader()
        for i in range(1, 21):
            writer.writerow({
                "item_id": i,
                "source": "reddit" if i % 2 == 0 else "news",
                "country": "us",
                "title": f"Shoe {i}",
                "text": f"Feedback {i}",
                "url": f"http://example.com/{i}",
                "human_relevant": "true",
                "human_sentiment": "negative" if i % 2 == 0 else "positive",
                "pred_relevant": "true",
                "pred_sentiment": "neutral" if i % 2 == 0 else "positive",
                "pred_reason": "Technical wear observation.",
            })

    out_md = tmp_path / "errors.md"
    err_count, file_path = run_error_analysis(
        csv_path=csv_file,
        prompt_version="v1",
        output_path=out_md,
    )

    assert file_path.exists()
    content = file_path.read_text(encoding="utf-8")
    assert "# Error Analysis Report (Development Set)" in content
    assert "TEST set" in content and "held out" in content
    assert "### Transition: `negative -> neutral`" in content
    assert "Model Reason:" in content


def test_compare_prompts_runner(tmp_path: Path):
    from eval.evaluate import compare_prompts

    csv_file = tmp_path / "comp_sample.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_id", "source", "country", "title", "text", "url",
                "human_relevant", "human_sentiment",
                "pred_relevant_v1", "pred_sentiment_v1", "pred_reason_v1",
                "pred_relevant_v2", "pred_sentiment_v2", "pred_reason_v2",
            ],
        )
        writer.writeheader()
        for i in range(1, 21):
            writer.writerow({
                "item_id": i,
                "source": "news",
                "country": "us",
                "title": f"Title {i}",
                "text": f"Text {i}",
                "url": f"http://example.com/{i}",
                "human_relevant": "true",
                "human_sentiment": "positive",
                # v1 has some mistakes, v2 has all correct
                "pred_relevant_v1": "true",
                "pred_sentiment_v1": "neutral" if i % 2 == 0 else "positive",
                "pred_reason_v1": "reason v1",
                "pred_relevant_v2": "true",
                "pred_sentiment_v2": "positive",
                "pred_reason_v2": "reason v2",
            })

    reports_dir = tmp_path / "reports"
    results = compare_prompts(
        prompts=["v1", "v2"],
        csv_path=csv_file,
        reports_dir=reports_dir,
    )

    assert "v1" in results
    assert "v2" in results
    assert results["v2"]["accuracy"] >= results["v1"]["accuracy"]
    comp_md = reports_dir / "eval_comparison_v1_v2_dev.md"
    assert comp_md.exists()


# ── 7. Blind relabelling & agreement tests ───────────────────────────────

def test_export_blind_schema_no_leaks(tmp_path: Path, db_session: Session):
    for i in range(1, 25):
        item = Item(
            id=i,
            source="youtube" if i % 2 == 0 else "news",
            source_name="Src",
            title=f"Title {i}",
            text="A" * 900,  # verify truncation to 800
            url=f"https://example.com/{i}",
            country="us",
            product="nike running shoes",
            content_hash=f"blind_hash_{i}",
        )
        db_session.add(item)
        if i <= 4:
            rel, sent = False, ""
        elif i <= 8:
            rel, sent = True, "negative"
        elif i <= 14:
            rel, sent = True, "neutral"
        else:
            rel, sent = True, "positive"
        db_session.add(
            Label(item_id=i, human_relevant=rel, human_sentiment=sent, labeller="human")
        )
    db_session.commit()

    out_file = tmp_path / "blind_relabel.csv"
    export_blind(output_path=out_file, target_count=12, seed=7, session=db_session)

    assert out_file.exists()
    with open(out_file, "rb") as f:
        assert f.read(3) == b"\xef\xbb\xbf"  # utf-8-sig BOM

    with open(out_file, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        expected_cols = [
            "item_id", "source", "country", "title", "text", "url",
            "human_relevant", "human_sentiment",
        ]
        assert reader.fieldnames == expected_cols
        rows = list(reader)
        assert len(rows) == 12
        for row in rows:
            assert row["human_relevant"] == ""
            assert row["human_sentiment"] == ""
            assert len(row["text"]) == 800
            assert "notes" not in row
            assert "score" not in row
            assert "vader" not in "".join(row.values()).lower()


def test_import_labels_blind_does_not_overwrite_human(tmp_path: Path, db_session: Session):
    csv1 = tmp_path / "human.csv"
    with open(csv1, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "human_relevant", "human_sentiment"])
        w.writeheader()
        w.writerow({"item_id": 1, "human_relevant": "1", "human_sentiment": "positive"})

    csv2 = tmp_path / "blind.csv"
    with open(csv2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "human_relevant", "human_sentiment"])
        w.writeheader()
        w.writerow({"item_id": 1, "human_relevant": "1", "human_sentiment": "neutral"})

    import_labels(csv1, session=db_session, labeller="human")
    import_labels(csv2, session=db_session, labeller="blind")

    all_labels = db_session.exec(select(Label).where(Label.item_id == 1)).all()
    assert len(all_labels) == 2
    by_labeller = {lbl.labeller: lbl for lbl in all_labels}
    assert by_labeller["human"].human_sentiment == "positive"
    assert by_labeller["blind"].human_sentiment == "neutral"


def test_agreement_script_metrics_and_confusion_matrix(db_session: Session):
    # 5 paired items:
    # 1: old=(1, positive), blind=(1, positive)
    # 2: old=(1, negative), blind=(1, negative)
    # 3: old=(1, neutral),  blind=(1, positive)
    # 4: old=(1, neutral),  blind=(1, neutral)
    # 5: old=(0, ""),       blind=(0, "")
    pairs = [
        (1, True, "positive", True, "positive"),
        (2, True, "negative", True, "negative"),
        (3, True, "neutral", True, "positive"),
        (4, True, "neutral", True, "neutral"),
        (5, False, "", False, ""),
    ]
    for item_id, o_rel, o_sent, b_rel, b_sent in pairs:
        db_session.add(Label(item_id=item_id, human_relevant=o_rel, human_sentiment=o_sent, labeller="human"))
        db_session.add(Label(item_id=item_id, human_relevant=b_rel, human_sentiment=b_sent, labeller="blind"))
    db_session.commit()

    metrics = compute_agreement(session=db_session, old_labeller="human", blind_labeller="blind")
    assert metrics["paired_count"] == 5
    assert metrics["sentiment_count"] == 4
    assert metrics["relevance"]["raw_agreement"] == 1.0
    assert pytest.approx(metrics["relevance"]["cohen_kappa"], rel=1e-4) == 1.0
    assert metrics["relevance"]["confusion_matrix"] == [[4, 0], [0, 1]]

    # Sentiment: 3/4 = 0.75 raw agreement; CM order [positive, negative, neutral]
    assert pytest.approx(metrics["sentiment"]["raw_agreement"], rel=1e-4) == 0.75
    assert metrics["sentiment"]["confusion_matrix"] == [
        [1, 0, 0],  # old positive -> 1 blind positive
        [0, 1, 0],  # old negative -> 1 blind negative
        [1, 0, 1],  # old neutral  -> 1 blind positive, 1 blind neutral
    ]
    report = format_agreement_report(metrics)
    assert "Cohen's Kappa" in report
    assert "Confusion Matrix" in report

