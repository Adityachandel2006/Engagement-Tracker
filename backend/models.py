import hashlib
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel, UniqueConstraint


class Item(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source: str = Field(index=True)  # news|reddit|youtube|review
    source_name: str = ""
    title: str = ""
    text: str = ""
    url: str = ""
    published_at: str = ""
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    country: str = ""
    language: str = "en"
    product: str = Field(index=True)
    content_hash: str = Field(unique=True, index=True)

    @staticmethod
    def make_hash(title: str, text: str) -> str:
        normalised = f"{title.strip().lower()}|{text.strip().lower()}"
        return hashlib.sha256(normalised.encode()).hexdigest()


class Analysis(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("item_id", "model", "prompt_version"),)

    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    model: str = ""
    prompt_version: str = ""
    relevant: bool = True
    sentiment: str = ""  # positive|negative|neutral
    score: float = 0.0  # -1..1
    aspect: str = ""
    reason: str = ""
    analysed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FetchLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    product: str = Field(index=True)
    country: str = ""
    source: str = ""
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    items_found: int = 0

    def is_fresh(self, hours: int = 24) -> bool:
        fetched = datetime.fromisoformat(self.fetched_at)
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - fetched
        return age.total_seconds() < hours * 3600


class Label(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    human_relevant: bool | None = None
    human_sentiment: str = ""
    labeller: str = ""
    labelled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
