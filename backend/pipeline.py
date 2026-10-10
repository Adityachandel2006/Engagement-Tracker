import os
import re
import requests
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
analyzer = SentimentIntensityAnalyzer()

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for",
    "of","with","is","was","are","were","be","been","has","have",
    "had","will","would","could","should","may","might","it","its",
    "this","that","these","those","he","she","they","we","i","you",
    "s","t","said","says","new","also","one","two","after","before",
    "about","from","into","by","as","up","out","no","not","more"
}

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return " ".join(text.split()).lower()

def score_sentiment(text: str) -> tuple[str, float]:
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        label = "positive"
    elif score <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, round(score, 4)

def extract_keywords(texts: list[str], top_n: int = 10) -> list[dict]:
    all_words = []
    for text in texts:
        all_words.extend([w for w in text.split() if w not in STOPWORDS and len(w) > 2])
    freq = pd.Series(all_words).value_counts().head(top_n)
    return [{"word": word, "count": int(cnt)} for word, cnt in freq.items()]

def fetch_and_process(topic: str, country: str = None, category: str = None) -> list[tuple]:
    url = "https://newsapi.org/v2/everything"
    
    # Append category to the search query if provided
    search_query = f"{topic} AND {category}" if category else topic
    
    params = {
        "q": search_query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY
    }
    
    if country:
        country_domains = {
            "us": "nytimes.com,cnn.com,wsj.com,washingtonpost.com,bloomberg.com",
            "gb": "bbc.co.uk,theguardian.com,telegraph.co.uk,independent.co.uk,ft.com",
            "in": "indiatimes.com,hindustantimes.com,indianexpress.com,thehindu.com,moneycontrol.com",
            "ca": "cbc.ca,theglobeandmail.com,globalnews.ca,torontostar.com,nationalpost.com",
            "au": "abc.net.au,smh.com.au,theage.com.au,theaustralian.com.au,news.com.au"
        }
        if country in country_domains:
            params["domains"] = country_domains[country]

    try:
        resp = requests.get(url, params=params, timeout=10)
        
        # Improve error handling for API limits or errors
        if not resp.ok:
            error_data = resp.json()
            error_msg = error_data.get("message", "Error fetching news data.")
            raise ValueError(f"NewsAPI Error ({resp.status_code}): {error_msg}")
            
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to connect to NewsAPI: {str(e)}")

    articles = resp.json().get("articles", [])
    rows = []

    for art in articles:
        raw = f"{art.get('title','')} {art.get('description','')}"
        if not raw.strip():
            continue
        text = clean_text(raw)
        sentiment, score = score_sentiment(text)
        date = (art.get("publishedAt") or datetime.utcnow().isoformat())[:10]
        # Include category in the returned tuple
        rows.append((topic.lower(), text, sentiment, score, date, country, category))

    return rows
