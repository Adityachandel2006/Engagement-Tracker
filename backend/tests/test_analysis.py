"""Tests for analysis module: LLM classifier, schemas, prompt, and baseline VADER."""

from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from sqlmodel import Session, SQLModel, create_engine, select

from models import Analysis, Item
from analysis.schemas import BatchAnalysisResult, ItemAnalysisResult
from analysis.classifier import analyze_items, load_prompt_template
from analysis.baseline_vader import clean_text, score_sentiment


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def db_session():
    """In-memory SQLite session with all tables created."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def _create_sample_items(session: Session, count: int = 20) -> list[Item]:
    items = []
    for i in range(1, count + 1):
        item = Item(
            source="news",
            source_name="TechDaily",
            title=f"Nike Shoe Model {i} Released",
            text=f"The new running shoes {i} from Nike offer excellent cushioning and lightweight feel.",
            url=f"https://example.com/item/{i}",
            country="us",
            product="nike running shoes",
            content_hash=Item.make_hash(f"Nike Shoe Model {i} Released", f"Body {i}"),
        )
        session.add(item)
        items.append(item)
    session.commit()
    for item in items:
        session.refresh(item)
    return items


# ── Schema validation tests ──────────────────────────────────────────────

def test_valid_item_analysis_schema():
    valid = ItemAnalysisResult(
        item_id=1,
        relevant=True,
        sentiment="positive",
        score=0.85,
        aspect="comfort",
        reason="Cushioning is great and comfortable.",
    )
    assert valid.item_id == 1
    assert valid.score == 0.85
    assert valid.sentiment == "positive"


def test_invalid_sentiment_fails():
    with pytest.raises(ValidationError):
        ItemAnalysisResult(
            item_id=1,
            relevant=True,
            sentiment="ecstatic",  # invalid literal
            score=0.5,
            aspect="comfort",
            reason="Good shoe.",
        )


def test_invalid_score_fails():
    with pytest.raises(ValidationError):
        ItemAnalysisResult(
            item_id=1,
            relevant=True,
            sentiment="positive",
            score=1.5,  # must be <= 1.0
            aspect="comfort",
            reason="Good shoe.",
        )


def test_reason_exceeding_20_words_fails():
    long_reason = " ".join(["word"] * 25)
    with pytest.raises(ValidationError) as excinfo:
        ItemAnalysisResult(
            item_id=1,
            relevant=True,
            sentiment="positive",
            score=0.5,
            aspect="quality",
            reason=long_reason,
        )
    assert "20 words or fewer" in str(excinfo.value)


# ── Prompt template tests ────────────────────────────────────────────────

def test_load_prompt_v1():
    template = load_prompt_template("v1")
    assert "{product}" in template
    # Verify core prompt rules
    assert "consumer would" in template
    assert "sarcasm" in template
    assert "relevant=false" in template
    assert "PASTE REAL EXAMPLES FROM LABELLED SET BELOW THIS LINE:" in template


def test_load_missing_prompt_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt_template("v999")


# ── Mock LLM & Acceptance tests ──────────────────────────────────────────

def test_analyze_20_items_and_zero_calls_on_second_run(db_session: Session):
    """Acceptance test: 20 sample items are analysed; a second run makes zero model calls."""
    items = _create_sample_items(db_session, count=20)
    assert len(items) == 20

    # Build mock instructor client returning canned batch analyses
    mock_client = MagicMock()

    def fake_create(**kwargs):
        messages = kwargs.get("messages", [])
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        # Extract item IDs from user_content prompt
        canned_analyses = []
        for item in items:
            if f"[Item ID: {item.id}]" in user_content:
                canned_analyses.append(
                    ItemAnalysisResult(
                        item_id=item.id,
                        relevant=True,
                        sentiment="positive",
                        score=0.8,
                        aspect="comfort",
                        reason=f"Item {item.id} has great cushioning and comfort.",
                    )
                )
        return BatchAnalysisResult(analyses=canned_analyses)

    mock_client.chat.completions.create.side_effect = fake_create

    # First run: should call mock client (2 batches of 10 for 20 items)
    first_results = analyze_items(
        items=items,
        product="nike running shoes",
        model="test-mock-model",
        prompt_version="v1",
        session=db_session,
        client=mock_client,
        batch_size=10,
    )

    assert len(first_results) == 20
    assert mock_client.chat.completions.create.call_count == 2
    for res in first_results:
        assert res.model == "test-mock-model"
        assert res.sentiment == "positive"
        assert res.relevant is True

    # Check database persistence
    stored = db_session.exec(
        select(Analysis).where(Analysis.model == "test-mock-model")
    ).all()
    assert len(stored) == 20

    # Reset mock call count
    mock_client.chat.completions.create.reset_mock()

    # Second run on exact same items: MUST MAKE ZERO MODEL CALLS!
    second_results = analyze_items(
        items=items,
        product="nike running shoes",
        model="test-mock-model",
        prompt_version="v1",
        session=db_session,
        client=mock_client,
        batch_size=10,
    )

    assert len(second_results) == 20
    assert mock_client.chat.completions.create.call_count == 0  # ZERO MODEL CALLS!


def test_invalid_response_triggers_validation_error():
    """Verify that an invalid schema response is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        # Invalid aspect and invalid sentiment
        ItemAnalysisResult(
            item_id=99,
            relevant=True,
            sentiment="unknown",
            score=0.0,
            aspect="invalid_aspect",
            reason="testing",
        )


# ── Baseline VADER tests ─────────────────────────────────────────────────

def test_baseline_vader_sentiment():
    pos_label, pos_score = score_sentiment("This product is absolutely wonderful and fantastic!")
    assert pos_label == "positive"
    assert pos_score > 0.5

    neg_label, neg_score = score_sentiment("Terrible product, broke immediately. Very disappointed.")
    assert neg_label == "negative"
    assert neg_score < -0.5

    clean = clean_text("Check this out: https://example.com/shoes! Best EVER???")
    assert "https" not in clean
    assert "best ever" in clean
