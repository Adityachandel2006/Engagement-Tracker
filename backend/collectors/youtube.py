"""YouTube collector using google-api-python-client. Returns Item rows."""

import logging

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from models import Item
from settings import settings

logger = logging.getLogger(__name__)

# Reserve some quota for future calls. Each search costs 100, each commentThreads
# costs 1. Default daily quota is 10 000.
_QUOTA_BUDGET = 9500
_SEARCH_COST = 100
_COMMENT_COST = 1


def fetch_youtube(product: str, country: str) -> list[Item]:
    """Search YouTube for '<product> review', collect top-level comments."""
    key = settings.youtube_api_key
    if not key:
        logger.warning("YOUTUBE_API_KEY not set — skipping YouTube")
        return []

    try:
        youtube = build("youtube", "v3", developerKey=key)
    except Exception:
        logger.exception("Failed to build YouTube client")
        return []

    items: list[Item] = []
    quota_used = 0

    # Search for videos
    try:
        search_resp = youtube.search().list(
            q=f"{product} review",
            part="snippet",
            type="video",
            regionCode=country.upper(),
            relevanceLanguage="en",
            maxResults=25,
            order="relevance",
        ).execute()
        quota_used += _SEARCH_COST
    except HttpError:
        logger.exception("YouTube search failed for %r / %s", product, country)
        return []
    except Exception:
        logger.exception("YouTube search unexpected error")
        return []

    for vid in search_resp.get("items", []):
        video_id = vid["id"]["videoId"]
        snippet = vid["snippet"]
        title = snippet.get("title", "")
        description = snippet.get("description", "")
        url = f"https://www.youtube.com/watch?v={video_id}"

        items.append(
            Item(
                source="youtube",
                source_name=snippet.get("channelTitle", ""),
                title=title,
                text=description[:2000],
                url=url,
                published_at=snippet.get("publishedAt", ""),
                country=country,
                product=product,
                content_hash=Item.make_hash(title, description),
            )
        )

        # Fetch top-level comments if quota allows
        if quota_used + _COMMENT_COST > _QUOTA_BUDGET:
            logger.info("YouTube quota budget nearing limit — skipping comments")
            continue

        try:
            comments_resp = youtube.commentThreads().list(
                videoId=video_id,
                part="snippet",
                maxResults=50,
                order="relevance",
                textFormat="plainText",
            ).execute()
            quota_used += _COMMENT_COST
        except HttpError as exc:
            if exc.resp.status in (403, 404):
                logger.debug("Comments disabled/unavailable for %s", video_id)
            else:
                logger.exception("YouTube comments error for %s", video_id)
            continue
        except Exception:
            logger.exception("YouTube comments unexpected error for %s", video_id)
            continue

        for thread in comments_resp.get("items", []):
            comment = thread["snippet"]["topLevelComment"]["snippet"]
            body = comment.get("textDisplay", "")
            if not body.strip():
                continue
            items.append(
                Item(
                    source="youtube",
                    source_name=comment.get("authorDisplayName", ""),
                    title=title,
                    text=body[:2000],
                    url=url,
                    published_at=comment.get("publishedAt", ""),
                    country=country,
                    product=product,
                    content_hash=Item.make_hash(title, body),
                )
            )

    return items
