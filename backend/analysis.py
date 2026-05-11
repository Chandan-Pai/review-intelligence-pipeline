"""Lightweight sentiment helpers (VADER + TextBlob)."""

from __future__ import annotations

from typing import Any

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def vader_scores(text: str) -> dict[str, float]:
    return _analyzer.polarity_scores(text)


def textblob_polarity(text: str) -> float:
    return float(TextBlob(text).sentiment.polarity)


def analyze_text(text: str) -> dict[str, Any]:
    v = vader_scores(text)
    return {
        "vader": v,
        "textblob_polarity": textblob_polarity(text),
        "compound": v.get("compound", 0.0),
    }


def analyze_batch(texts: list[str]) -> list[dict[str, Any]]:
    return [analyze_text(t) for t in texts]
