"""Tests for SQLModel models and legacy DB functions."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, SQLModel, create_engine

from models import Analysis, FetchLog, Item, Label


@pytest.fixture()
def session():
    """In-memory SQLite session for each test."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


# ── Insert tests ─────────────────────────────────────────────────────────

def test_insert_item(session: Session):
    item = Item(
        source="news",
        source_name="BBC",
        title="Test title",
        text="Test body",
        url="https://example.com/1",
        product="nike shoes",
        content_hash=Item.make_hash("Test title", "Test body"),
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    assert item.id is not None
    assert item.content_hash == Item.make_hash("Test title", "Test body")


def test_insert_analysis(session: Session):
    item = Item(
        source="reddit",
        product="nike shoes",
        content_hash=Item.make_hash("a", "b"),
    )
    session.add(item)
    session.commit()
    session.refresh(item)

    analysis = Analysis(
        item_id=item.id,
        model="gpt-4o",
        prompt_version="v1",
        relevant=True,
        sentiment="positive",
        score=0.8,
        aspect="comfort",
        reason="Users praise comfort",
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    assert analysis.id is not None
    assert analysis.item_id == item.id


def test_insert_label(session: Session):
    item = Item(
        source="youtube",
        product="nike shoes",
        content_hash=Item.make_hash("c", "d"),
    )
    session.add(item)
    session.commit()
    session.refresh(item)

    label = Label(item_id=item.id, human_relevant=True, human_sentiment="positive", labeller="alice")
    session.add(label)
    session.commit()
    session.refresh(label)
    assert label.id is not None


# ── Unique content_hash ──────────────────────────────────────────────────

def test_unique_content_hash(session: Session):
    h = Item.make_hash("duplicate", "text")
    session.add(Item(source="news", product="shoes", content_hash=h))
    session.commit()
    session.add(Item(source="reddit", product="shoes", content_hash=h))
    with pytest.raises(IntegrityError):
        session.commit()


def test_unique_analysis_constraint(session: Session):
    item = Item(source="news", product="shoes", content_hash=Item.make_hash("u", "v"))
    session.add(item)
    session.commit()
    session.refresh(item)

    session.add(Analysis(item_id=item.id, model="gpt-4o", prompt_version="v1"))
    session.commit()
    session.add(Analysis(item_id=item.id, model="gpt-4o", prompt_version="v1"))
    with pytest.raises(IntegrityError):
        session.commit()


# ── FetchLog freshness ───────────────────────────────────────────────────

def test_fetchlog_fresh():
    log = FetchLog(product="shoes", country="US", source="news", items_found=5)
    assert log.is_fresh()


def test_fetchlog_stale():
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    log = FetchLog(product="shoes", country="US", source="news", fetched_at=stale_time, items_found=0)
    assert not log.is_fresh()


def test_fetchlog_fresh_zero_items():
    """A fetch counts as fresh even when it found nothing."""
    log = FetchLog(product="shoes", country="US", source="news", items_found=0)
    assert log.is_fresh()
