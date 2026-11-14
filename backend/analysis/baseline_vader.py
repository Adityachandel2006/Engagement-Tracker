"""Baseline VADER sentiment analysis.

Copied from pipeline.py as a baseline for evaluation.
The original in pipeline.py stays until Prompt 11.
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


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
