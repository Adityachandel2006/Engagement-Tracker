import sqlite3
from contextlib import contextmanager
from pathlib import Path

from sqlmodel import SQLModel, Session, create_engine, text

import models  # noqa: F401 – registers tables with SQLModel.metadata

DB_PATH = Path(__file__).parent / "data.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})

# ---------------------------------------------------------------------------
# New-style helpers (SQLModel engine)
# ---------------------------------------------------------------------------

def get_engine():
    return engine


def get_session():
    with Session(engine) as session:
        yield session


# ---------------------------------------------------------------------------
# Legacy articles table – kept so main.py / pipeline.py still work until
# Prompt 11 migrates them to the new Item + Analysis models.
# ---------------------------------------------------------------------------

_LEGACY_SCHEMA = """\
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    topic       TEXT NOT NULL,
    text        TEXT NOT NULL,
    sentiment   TEXT NOT NULL,
    score       REAL NOT NULL,
    created_at  TEXT NOT NULL,
    country     TEXT,
    category    TEXT
);
CREATE INDEX IF NOT EXISTS idx_topic ON articles(topic);
"""


@contextmanager
def _legacy_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    SQLModel.metadata.create_all(engine)
    with _legacy_conn() as conn:
        conn.executescript(_LEGACY_SCHEMA)
        conn.commit()


def insert_articles(rows: list[tuple]):
    with _legacy_conn() as conn:
        conn.executemany(
            "INSERT INTO articles (topic, text, sentiment, score, created_at, country, category) VALUES (?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()


def fetch_articles(topic: str, country: str = None, category: str = None, limit: int = None) -> list:
    with _legacy_conn() as conn:
        query = "SELECT * FROM articles WHERE topic = ?"
        params = [topic.lower()]

        if country:
            query += " AND country = ?"
            params.append(country)
        else:
            query += " AND country IS NULL"

        if category:
            query += " AND category = ?"
            params.append(category)
        else:
            query += " AND category IS NULL"

        query += " ORDER BY created_at DESC" if limit else " ORDER BY created_at"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        cur = conn.execute(query, tuple(params))
        rows = [dict(row) for row in cur.fetchall()]
        return list(reversed(rows)) if limit else rows


def topic_exists(topic: str, country: str = None, category: str = None) -> bool:
    with _legacy_conn() as conn:
        query = "SELECT 1 FROM articles WHERE topic = ?"
        params = [topic.lower()]

        if country:
            query += " AND country = ?"
            params.append(country)
        else:
            query += " AND country IS NULL"

        if category:
            query += " AND category = ?"
            params.append(category)
        else:
            query += " AND category IS NULL"

        query += " LIMIT 1"

        cur = conn.execute(query, tuple(params))
        return cur.fetchone() is not None


def distinct_topics() -> list[str]:
    with _legacy_conn() as conn:
        cur = conn.execute("SELECT DISTINCT topic FROM articles ORDER BY topic")
        return [row["topic"] for row in cur.fetchall()]
