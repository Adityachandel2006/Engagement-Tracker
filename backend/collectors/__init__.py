"""Collector orchestrator.

collect_all() fans out to every source, deduplicates by content_hash,
tags language, persists Items and FetchLog entries.  It skips any
(product, country, source) combination that already has a fresh FetchLog
entry (< 24 hours old).
"""

import logging
from datetime import datetime, timezone

from langdetect import detect, LangDetectException
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from db import get_engine
from models import FetchLog, Item

from collectors.news import fetch_gnews, fetch_newsapi
from collectors.reddit import fetch_reddit
from collectors.youtube import fetch_youtube
from collectors.trends import fetch_trends

logger = logging.getLogger(__name__)

_SOURCE_FETCHERS: dict[str, list] = {
    "news": [fetch_newsapi, fetch_gnews],
    "reddit": [fetch_reddit],
    "youtube": [fetch_youtube],
    "trends": [fetch_trends],
}

ALL_SOURCES = list(_SOURCE_FETCHERS.keys())


def _is_fresh(session: Session, product: str, country: str, source: str) -> bool:
    """Return True if a FetchLog entry exists within the last 24 hours."""
    stmt = (
        select(FetchLog)
        .where(FetchLog.product == product)
        .where(FetchLog.country == country)
        .where(FetchLog.source == source)
        .order_by(FetchLog.fetched_at.desc())  # type: ignore[union-attr]
        .limit(1)
    )
    log = session.exec(stmt).first()
    return log is not None and log.is_fresh()


def _detect_language(text: str) -> str:
    """Best-effort language detection; default to 'en'."""
    try:
        return detect(text) if text.strip() else "en"
    except LangDetectException:
        return "en"


def collect_all(
    product: str,
    countries: list[str],
    sources: list[str] | None = None,
) -> dict[str, int]:
    """Collect items from all requested sources for each country.

    Returns a dict mapping ``"country:source"`` to the number of NEW items
    stored (0 when the fetch was skipped as fresh).
    """
    if sources is None:
        sources = ALL_SOURCES

    result: dict[str, int] = {}
    engine = get_engine()

    for country in countries:
        for source in sources:
            key = f"{country}:{source}"
            fetchers = _SOURCE_FETCHERS.get(source, [])
            if not fetchers:
                logger.warning("Unknown source %r — skipping", source)
                result[key] = 0
                continue

            with Session(engine) as session:
                if _is_fresh(session, product, country, source):
                    logger.info("Skipping %s (fresh FetchLog)", key)
                    result[key] = 0
                    continue

            # Collect from all fetchers for this source
            raw_items: list[Item] = []
            for fetcher in fetchers:
                try:
                    raw_items.extend(fetcher(product, country))
                except Exception:
                    logger.exception("Fetcher %s failed", fetcher.__name__)

            # Tag language
            for item in raw_items:
                item.language = _detect_language(f"{item.title} {item.text}")

            # Persist, skipping duplicates by content_hash
            stored = 0
            with Session(engine) as session:
                for item in raw_items:
                    try:
                        session.add(item)
                        session.commit()
                        stored += 1
                    except IntegrityError:
                        session.rollback()

                # Always record the fetch (even when 0 items found)
                session.add(
                    FetchLog(
                        product=product,
                        country=country,
                        source=source,
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                        items_found=stored,
                    )
                )
                session.commit()

            result[key] = stored
            logger.info("Stored %d items for %s", stored, key)

    return result
