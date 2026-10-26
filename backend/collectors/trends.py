"""Google Trends collector.

pytrends (the unofficial library) was archived in April 2025 and is blocked
by Google's anti-bot measures.  This module is a placeholder — it logs a
warning and returns an empty list so that the rest of the pipeline is
unaffected.  When a reliable Trends data source becomes available, implement
fetch_trends here.
"""

import logging

from models import Item

logger = logging.getLogger(__name__)


def fetch_trends(product: str, country: str) -> list[Item]:
    """Placeholder — pytrends is broken; skip Trends for now."""
    logger.info("Google Trends collector skipped (pytrends unavailable)")
    return []
