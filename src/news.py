"""NewsAPI fetch + simple lexicon sentiment + optional Gemini assistant."""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st

POSITIVE_WORDS = {"gain", "surge", "beat", "growth", "rally", "up", "bullish", "strong",
                   "record", "profit", "upgrade", "outperform", "soar"}
NEGATIVE_WORDS = {"loss", "drop", "fall", "decline", "bearish", "down", "weak", "cut",
                   "downgrade", "miss", "plunge", "recession", "sell-off", "crash"}


def get_secret(key: str) -> str:
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def simple_sentiment(text: str) -> float:
    if not text:
        return 0.0
    words = text.lower().split()
    pos = sum(1 for w in words if any(p in w for p in POSITIVE_WORDS))
    neg = sum(1 for w in words if any(n in w for n in NEGATIVE_WORDS))
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


@st.cache_data(show_spinner="Lade Nachrichten ...", ttl=1800)
def fetch_news(query: str, language: str = "en", page_size: int = 10) -> Tuple[pd.DataFrame, str]:
    api_key = get_secret("NEWS_API_KEY")
    if not api_key:
        return pd.DataFrame(), "no_key"
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query, "language": language, "sortBy": "publishedAt",
                    "pageSize": page_size, "apiKey": api_key},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") != "ok":
            return pd.DataFrame(), data.get("message", "error")
        articles = data.get("articles", [])
        df = pd.DataFrame(articles)
        return df, "ok"
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def analyze_news(query: str) -> Tuple[pd.DataFrame, float, str]:
    df, status = fetch_news(query)
    if df.empty:
        return df, 0.0, status
    texts = (df.get("title", "").fillna("") + " " + df.get("description", "").fillna(""))
    scores = texts.apply(simple_sentiment)
    avg = float(scores.mean()) if len(scores) else 0.0
    label = "positiv" if avg > 0.15 else "negativ" if avg < -0.15 else "neutral"
    return df, avg, label


def get_gemini_client():
    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def assistant_answer(prompt: str, context: str) -> str:
    client = get_gemini_client()
    if client is None:
        return ("ℹ️ Kein Gemini-API-Key konfiguriert (`.streamlit/secrets.toml` → `GEMINI_API_KEY`). "
                "Ich kann trotzdem helfen, sobald ein Key hinterlegt ist.")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Kontext:\n{context}\n\nFrage: {prompt}\n\nAntworte auf Deutsch, knapp und faktenbasiert.",
        )
        return response.text or "(keine Antwort erhalten)"
    except Exception as exc:
        text = str(exc)
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            return ("⚠️ Gemini-Freikontingent für heute/diese Minute ausgeschöpft. "
                     "Bitte in Kürze erneut versuchen (Details in der Google-Cloud-Konsole).")
        return "⚠️ Assistent aktuell nicht verfügbar. Bitte später erneut versuchen."
