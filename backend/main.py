from collections import Counter
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from db import init_db, insert_articles, fetch_articles, topic_exists, distinct_topics
from pipeline import fetch_and_process, extract_keywords

app = FastAPI(title="Engagement Tracker API")
COMPARISON_SAMPLE_LIMIT = 30

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/search")
def search(topic: str = Query(..., min_length=1), category: str = Query(None)):
    topic = topic.strip()
    category = category.strip() if category else None

    try:
        # Serve from cache if already fetched
        if not topic_exists(topic, None, category):
            rows = fetch_and_process(topic, None, category)
            if not rows:
                raise HTTPException(status_code=404, detail="No articles found for this topic/category.")
            insert_articles(rows)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    articles = fetch_articles(topic, None, category)
    df = pd.DataFrame(articles)

    sentiment_counts = df["sentiment"].value_counts().to_dict()
    volume_by_date = (
        df.groupby("created_at").size()
        .reset_index(name="count")
        .rename(columns={"created_at": "date"})
        .to_dict(orient="records")
    )
    keywords = extract_keywords(df["text"].tolist())

    return {
        "topic": topic,
        "category": category,
        "total": len(df),
        "sentiment_breakdown": {
            "positive": sentiment_counts.get("positive", 0),
            "negative": sentiment_counts.get("negative", 0),
            "neutral":  sentiment_counts.get("neutral", 0),
        },
        "volume_by_date": volume_by_date,
        "top_keywords": keywords,
        "articles": articles[-10:],  # return last 10 for preview
    }

@app.get("/compare")
def compare(topic: str = Query(..., min_length=1), countries: str = Query(..., min_length=1), category: str = Query(None)):
    topic = topic.strip()
    country_list = [c.strip().lower() for c in countries.split(",")]
    category = category.strip() if category else None
    
    result = {
        "topic": topic,
        "category": category,
        "countries": {}
    }
    
    for country in country_list:
        try:
            if not topic_exists(topic, country, category):
                rows = fetch_and_process(topic, country, category)
                if rows:
                    insert_articles(rows)
        except ValueError as e:
            raise HTTPException(status_code=500, detail=str(e))
                
        articles = fetch_articles(topic, country, category, limit=COMPARISON_SAMPLE_LIMIT)
        df = pd.DataFrame(articles)
        
        if df.empty:
            result["countries"][country] = {
                "total": 0,
                "sample_size": 0,
                "sample_limit": COMPARISON_SAMPLE_LIMIT,
                "coverage": "Limited coverage",
                "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_percentages": {"positive": 0, "negative": 0, "neutral": 0},
                "sentiment_by_date": [],
                "volume_by_date": [],
                "articles": []
            }
        else:
            sentiment_counts = df["sentiment"].value_counts().to_dict()
            volume_by_date = (
                df.groupby("created_at").size()
                .reset_index(name="count")
                .rename(columns={"created_at": "date"})
                .to_dict(orient="records")
            )
            sentiment_by_date = (
                df.groupby("created_at")["score"].mean()
                .round(3)
                .reset_index(name="score")
                .rename(columns={"created_at": "date"})
                .to_dict(orient="records")
            )
            total = len(df)
            result["countries"][country] = {
                "total": total,
                "sample_size": total,
                "sample_limit": COMPARISON_SAMPLE_LIMIT,
                "coverage": "Full sample" if len(df) >= COMPARISON_SAMPLE_LIMIT else "Limited coverage",
                "sentiment_breakdown": {
                    "positive": sentiment_counts.get("positive", 0),
                    "negative": sentiment_counts.get("negative", 0),
                    "neutral":  sentiment_counts.get("neutral", 0),
                },
                "sentiment_percentages": {
                    "positive": round(sentiment_counts.get("positive", 0) / total * 100),
                    "negative": round(sentiment_counts.get("negative", 0) / total * 100),
                    "neutral": round(sentiment_counts.get("neutral", 0) / total * 100),
                },
                "sentiment_by_date": sentiment_by_date,
                "volume_by_date": volume_by_date,
                "articles": articles[-10:]
            }
            
    return result

@app.get("/topics")
def topics():
    return {"topics": distinct_topics()}
