"""Tests for collectors — HTTP mocked with respx, Reddit/YouTube mocked."""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from sqlmodel import Session, SQLModel, create_engine, select

from models import FetchLog, Item


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def db_session():
    """In-memory SQLite with all tables."""
    engine = create_engine("sqlite://", echo=False)
    SQLModel.metadata.create_all(engine)
    with Session(engine) as s:
        yield s, engine


# ── NewsAPI ──────────────────────────────────────────────────────────────

@respx.mock
def test_newsapi_returns_items():
    from collectors.news import fetch_newsapi

    payload = {
        "status": "ok",
        "articles": [
            {
                "source": {"name": "BBC"},
                "title": "Nike shoes review",
                "description": "A great pair of shoes",
                "url": "https://bbc.co.uk/a1",
                "publishedAt": "2026-01-01T00:00:00Z",
            }
        ],
    }
    respx.get("https://newsapi.org/v2/everything").mock(
        return_value=httpx.Response(200, json=payload)
    )

    with patch("collectors.news.settings") as mock_settings:
        mock_settings.news_api_key = "test-key"
        client = httpx.Client()
        items = fetch_newsapi("nike shoes", "us", client=client)
        client.close()

    assert len(items) == 1
    assert items[0].source == "news"
    assert items[0].source_name == "BBC"
    assert items[0].title == "Nike shoes review"
    assert items[0].url == "https://bbc.co.uk/a1"
    assert items[0].product == "nike shoes"


@respx.mock
def test_newsapi_api_error_returns_empty():
    from collectors.news import fetch_newsapi

    respx.get("https://newsapi.org/v2/everything").mock(
        return_value=httpx.Response(429, text="rate limited")
    )

    with patch("collectors.news.settings") as mock_settings:
        mock_settings.news_api_key = "test-key"
        client = httpx.Client()
        items = fetch_newsapi("nike shoes", "us", client=client)
        client.close()

    assert items == []


@respx.mock
def test_newsapi_no_key_returns_empty():
    from collectors.news import fetch_newsapi

    with patch("collectors.news.settings") as mock_settings:
        mock_settings.news_api_key = ""
        items = fetch_newsapi("nike shoes", "us")

    assert items == []


# ── GNews ────────────────────────────────────────────────────────────────

@respx.mock
def test_gnews_returns_items():
    from collectors.news import fetch_gnews

    payload = {
        "totalArticles": 1,
        "articles": [
            {
                "title": "Nike launch in India",
                "description": "Nike launches new line",
                "url": "https://example.in/a1",
                "publishedAt": "2026-02-01T00:00:00Z",
                "source": {"name": "Times of India"},
            }
        ],
    }
    respx.get("https://gnews.io/api/v4/search").mock(
        return_value=httpx.Response(200, json=payload)
    )

    with patch("collectors.news.settings") as mock_settings:
        mock_settings.gnews_api_key = "test-key"
        client = httpx.Client()
        items = fetch_gnews("nike shoes", "in", client=client)
        client.close()

    assert len(items) == 1
    assert items[0].country == "in"
    assert items[0].source_name == "Times of India"


# ── Reddit ───────────────────────────────────────────────────────────────

def _mock_submission(title, selftext, permalink, created_utc, comments_data):
    sub = MagicMock()
    sub.title = title
    sub.selftext = selftext
    sub.permalink = permalink
    sub.created_utc = created_utc

    comments = []
    for body, plink, utc in comments_data:
        c = MagicMock()
        c.body = body
        c.permalink = plink
        c.created_utc = utc
        comments.append(c)

    mock_comments = MagicMock()
    mock_comments.__getitem__ = lambda self, s: comments[s] if isinstance(s, int) else comments.__getitem__(s)
    mock_comments.__iter__ = lambda self: iter(comments)
    mock_comments.__len__ = lambda self: len(comments)
    mock_comments.replace_more = MagicMock()
    sub.comments = mock_comments
    return sub


def test_reddit_returns_items():
    from collectors.reddit import fetch_reddit

    submission = _mock_submission(
        "Nike review",
        "Great shoes",
        "/r/all/comments/abc/nike_review/",
        1700000000,
        [("Love them!", "/r/all/comments/abc/nike_review/xyz/", 1700000100)],
    )

    mock_reddit = MagicMock()
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [submission]
    mock_reddit.subreddit.return_value = mock_subreddit

    with patch("collectors.reddit._make_reddit", return_value=mock_reddit):
        items = fetch_reddit("nike shoes", "us")

    # post + 1 comment, from each of: r/AskAnAmerican, r/BuyItForLife, r/all = 6
    assert len(items) == 6
    assert all(i.source == "reddit" for i in items)


def test_reddit_no_creds_returns_empty():
    from collectors.reddit import fetch_reddit

    with patch("collectors.reddit._make_reddit", return_value=None):
        items = fetch_reddit("nike shoes", "us")

    assert items == []


# ── YouTube ──────────────────────────────────────────────────────────────

def test_youtube_returns_items():
    from collectors.youtube import fetch_youtube

    search_resp = {
        "items": [
            {
                "id": {"videoId": "abc123"},
                "snippet": {
                    "title": "Nike review",
                    "description": "Honest review",
                    "channelTitle": "ShoeTube",
                    "publishedAt": "2026-01-01T00:00:00Z",
                },
            }
        ]
    }
    comments_resp = {
        "items": [
            {
                "snippet": {
                    "topLevelComment": {
                        "snippet": {
                            "textDisplay": "Great video!",
                            "authorDisplayName": "User1",
                            "publishedAt": "2026-01-02T00:00:00Z",
                        }
                    }
                }
            }
        ]
    }

    mock_youtube = MagicMock()
    mock_youtube.search().list().execute.return_value = search_resp
    mock_youtube.commentThreads().list().execute.return_value = comments_resp

    with (
        patch("collectors.youtube.settings") as mock_settings,
        patch("collectors.youtube.build", return_value=mock_youtube),
    ):
        mock_settings.youtube_api_key = "test-key"
        items = fetch_youtube("nike shoes", "us")

    assert len(items) == 2  # 1 video + 1 comment
    assert items[0].source == "youtube"
    assert items[0].url == "https://www.youtube.com/watch?v=abc123"


def test_youtube_no_key_returns_empty():
    from collectors.youtube import fetch_youtube

    with patch("collectors.youtube.settings") as mock_settings:
        mock_settings.youtube_api_key = ""
        items = fetch_youtube("nike shoes", "us")

    assert items == []


# ── collect_all freshness ────────────────────────────────────────────────

def test_collect_all_skips_fresh(db_session):
    """A second collect_all within 24 hours makes zero HTTP calls."""
    session, engine = db_session

    # Seed a fresh FetchLog for (shoes, us, news)
    session.add(
        FetchLog(
            product="shoes",
            country="us",
            source="news",
            fetched_at=datetime.now(timezone.utc).isoformat(),
            items_found=5,
        )
    )
    session.commit()

    with (
        patch("collectors.get_engine", return_value=engine),
        patch("collectors.news.fetch_newsapi") as mock_newsapi,
        patch("collectors.news.fetch_gnews") as mock_gnews,
    ):
        from collectors import collect_all

        result = collect_all("shoes", ["us"], sources=["news"])

    # Should skip — no HTTP calls
    mock_newsapi.assert_not_called()
    mock_gnews.assert_not_called()
    assert result == {"us:news": 0}


def test_collect_all_fetches_when_stale(db_session):
    """When FetchLog is > 24h old, collect_all actually calls fetchers."""
    session, engine = db_session

    # Seed a stale FetchLog
    stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    session.add(
        FetchLog(product="shoes", country="us", source="news", fetched_at=stale, items_found=3)
    )
    session.commit()

    item = Item(
        source="news",
        source_name="Test",
        title="Stale test",
        text="Body",
        url="https://example.com",
        product="shoes",
        country="us",
        content_hash=Item.make_hash("Stale test", "Body"),
    )

    with (
        patch("collectors.get_engine", return_value=engine),
        patch("collectors.fetch_newsapi", return_value=[item]) as mock_newsapi,
        patch("collectors.fetch_gnews", return_value=[]) as mock_gnews,
    ):
        # Re-patch the fetcher list so collect_all sees our mocks
        import collectors
        collectors._SOURCE_FETCHERS["news"] = [mock_newsapi, mock_gnews]

        result = collectors.collect_all("shoes", ["us"], sources=["news"])

    mock_newsapi.assert_called_once()
    assert result["us:news"] >= 1
