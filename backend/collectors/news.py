"""NewsAPI and GNews collectors. Return lists of Item (unsaved)."""

import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from models import Item
from settings import settings

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(15.0)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.TransportError),
    reraise=True,
)
def _get(client: httpx.Client, url: str, params: dict) -> httpx.Response:
    return client.get(url, params=params, timeout=_TIMEOUT)


def fetch_newsapi(product: str, country: str, client: httpx.Client | None = None) -> list[Item]:
    """Fetch articles from NewsAPI /v2/everything."""
    key = settings.news_api_key
    if not key:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI")
        return []

    params = {
        "q": f'"{product}"',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": key,
    }

    own_client = client is None
    if own_client:
        client = httpx.Client()
    try:
        resp = _get(client, "https://newsapi.org/v2/everything", params)
        if resp.status_code != 200:
            logger.error("NewsAPI %s: %s", resp.status_code, resp.text[:300])
            return []
        articles = resp.json().get("articles", [])
    except Exception:
        logger.exception("NewsAPI request failed for %r / %s", product, country)
        return []
    finally:
        if own_client:
            client.close()

    items: list[Item] = []
    for art in articles:
        title = art.get("title") or ""
        text = art.get("description") or ""
        if not title.strip() and not text.strip():
            continue
        items.append(
            Item(
                source="news",
                source_name=art.get("source", {}).get("name", ""),
                title=title,
                text=text,
                url=art.get("url", ""),
                published_at=art.get("publishedAt", ""),
                country=country,
                product=product,
                content_hash=Item.make_hash(title, text),
            )
        )
    return items


def fetch_gnews(product: str, country: str, client: httpx.Client | None = None) -> list[Item]:
    """Fetch articles from GNews /api/v4/search."""
    key = settings.gnews_api_key
    if not key:
        logger.warning("GNEWS_API_KEY not set — skipping GNews")
        return []

    params = {
        "q": f'"{product}"',
        "apikey": key,
        "lang": "en",
        "country": country,
        "max": 100,
        "sortby": "publishedAt",
    }

    own_client = client is None
    if own_client:
        client = httpx.Client()
    try:
        resp = _get(client, "https://gnews.io/api/v4/search", params)
        if resp.status_code != 200:
            logger.error("GNews %s: %s", resp.status_code, resp.text[:300])
            return []
        articles = resp.json().get("articles", [])
    except Exception:
        logger.exception("GNews request failed for %r / %s", product, country)
        return []
    finally:
        if own_client:
            client.close()

    items: list[Item] = []
    for art in articles:
        title = art.get("title") or ""
        text = art.get("description") or ""
        if not title.strip() and not text.strip():
            continue
        items.append(
            Item(
                source="news",
                source_name=art.get("source", {}).get("name", ""),
                title=title,
                text=text,
                url=art.get("url", ""),
                published_at=art.get("publishedAt", ""),
                country=country,
                product=product,
                content_hash=Item.make_hash(title, text),
            )
        )
    return items
