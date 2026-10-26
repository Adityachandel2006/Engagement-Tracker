"""Reddit collector using PRAW. Returns lists of Item (unsaved)."""

import logging

import praw
from prawcore.exceptions import PrawcoreException

from models import Item
from settings import settings

logger = logging.getLogger(__name__)

# Country → subreddits likely to discuss products in that market.
_COUNTRY_SUBS: dict[str, list[str]] = {
    "us": ["AskAnAmerican", "BuyItForLife"],
    "gb": ["AskUK", "UnitedKingdom"],
    "in": ["india", "IndianGaming"],
    "ca": ["canada", "CanadianHardwareSwap"],
    "au": ["australia", "AustralianMakeup"],
    "de": ["germany", "de"],
}


def _make_reddit() -> praw.Reddit | None:
    cid = settings.reddit_client_id
    sec = settings.reddit_client_secret
    ua = settings.reddit_user_agent or "engagement-tracker/1.0"
    if not cid or not sec:
        logger.warning("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set — skipping Reddit")
        return None
    return praw.Reddit(client_id=cid, client_secret=sec, user_agent=ua)


def fetch_reddit(product: str, country: str) -> list[Item]:
    """Search Reddit for product mentions; collect post + top comments."""
    reddit = _make_reddit()
    if reddit is None:
        return []

    subreddits = _COUNTRY_SUBS.get(country, []) + ["all"]
    items: list[Item] = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.search(f'"{product}"', sort="relevance", limit=25):
                # Post itself
                title = submission.title or ""
                text = submission.selftext or ""
                items.append(
                    Item(
                        source="reddit",
                        source_name=f"r/{sub_name}",
                        title=title,
                        text=text[:2000],
                        url=f"https://reddit.com{submission.permalink}",
                        published_at=str(int(submission.created_utc)),
                        country=country,
                        product=product,
                        content_hash=Item.make_hash(title, text),
                    )
                )
                # Top-level comments (up to 50)
                submission.comment_sort = "top"
                submission.comments.replace_more(limit=0)
                for comment in submission.comments[:50]:
                    body = comment.body or ""
                    if not body.strip():
                        continue
                    items.append(
                        Item(
                            source="reddit",
                            source_name=f"r/{sub_name}",
                            title=title,
                            text=body[:2000],
                            url=f"https://reddit.com{comment.permalink}",
                            published_at=str(int(comment.created_utc)),
                            country=country,
                            product=product,
                            content_hash=Item.make_hash(title, body),
                        )
                    )
        except PrawcoreException:
            logger.exception("Reddit search failed for r/%s product=%r", sub_name, product)
        except Exception:
            logger.exception("Unexpected error searching r/%s", sub_name)

    return items
