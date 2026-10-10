import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "data.db"

SCHEMA = """
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
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        try:
            conn.execute("ALTER TABLE articles ADD COLUMN country TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE articles ADD COLUMN category TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()

def insert_articles(rows: list[tuple]):
    with get_conn() as conn:
        conn.executemany(
            "INSERT INTO articles (topic, text, sentiment, score, created_at, country, category) VALUES (?,?,?,?,?,?,?)",
            rows
        )
        conn.commit()

def fetch_articles(topic: str, country: str = None, category: str = None, limit: int = None) -> list:
    with get_conn() as conn:
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
    with get_conn() as conn:
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
    with get_conn() as conn:
        cur = conn.execute("SELECT DISTINCT topic FROM articles ORDER BY topic")
        return [row["topic"] for row in cur.fetchall()]
