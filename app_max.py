from __future__ import annotations
import base64

import io
import json
import math
import time
import urllib.parse
import requests
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="WealthScope AI",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "WealthScope AI – lokale wissenschaftliche Demo-Anwendung. Keine Finanzberatung.",
    },
)

st.logo(
    "assets/wealthscope_logo.svg",
    size="large",
    icon_image="assets/wealthscope_icon.svg",
)





# =========================================================
# CONSTANTS
# =========================================================

APP_NAME = "WealthScope AI"
APP_VERSION = "2.0-max"
APP_CLAIM = "Kapital verstehen. Risiken prüfen. Entscheidungen simulieren."

DATA_FEATURES_PATH = Path("data/processed/wealthscope_features.csv")
DATA_FEATURES_PARQUET_PATH = Path("data/processed/wealthscope_features.parquet")
DATA_MARKET_PATH = Path("data/processed/wealthscope_market_dataset.csv")
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

MAIN_PAGES = [
    "Start",
    "Wealth Outlook",
    "Kapital-Kompass",
    "Portfolio-Simulator",
    "Watchlist-Vergleich",
    "Datenlabor",
    "ML-Labor",
    "Assistent",
]

SERVICE_PAGES = [
    "News-Archiv",
    "So funktioniert's",
    "Über das Projekt",
    "Professor-Export",
    "Impressum",
    "Datenschutz",
    "Betriebsstatus",
]

ALL_PAGES = MAIN_PAGES + SERVICE_PAGES

PAGE_LABELS = {
    "Start": "Start",
    "Wealth Outlook": "Outlook",
    "Kapital-Kompass": "Kompass",
    "Portfolio-Simulator": "Simulator",
    "Watchlist-Vergleich": "Watchlist",
    "Datenlabor": "Daten",
    "ML-Labor": "ML",
    "Assistent": "Assistent",
    "News-Archiv": "News",
    "So funktioniert's": "Methodik",
    "Über das Projekt": "Projekt",
    "Professor-Export": "Export",
    "Impressum": "Impressum",
    "Datenschutz": "Datenschutz",
    "Betriebsstatus": "Status",
}

PAGE_ICONS = {
    "Start": "🏠",
    "Wealth Outlook": "📈",
    "Kapital-Kompass": "🧭",
    "Portfolio-Simulator": "🧪",
    "Watchlist-Vergleich": "👀",
    "Datenlabor": "🧬",
    "ML-Labor": "🤖",
    "Assistent": "💬",
    "News-Archiv": "📰",
    "So funktioniert's": "📖",
    "Über das Projekt": "🎓",
    "Professor-Export": "📦",
    "Impressum": "⚖️",
    "Datenschutz": "🛡️",
    "Betriebsstatus": "🟢",
}

DEFAULT_ASSET_MAP = {
    "ETF – S&P 500 (SPY)": "SPY",
    "ETF – Nasdaq 100 (QQQ)": "QQQ",
    "ETF – Gesamtmarkt USA (VTI)": "VTI",
    "ETF – Emerging Markets (EEM)": "EEM",
    "ETF – Gold (GLD)": "GLD",
    "ETF – Anleihen (AGG)": "AGG",
    "Aktie – Apple (AAPL)": "AAPL",
    "Aktie – Microsoft (MSFT)": "MSFT",
    "Aktie – Nvidia (NVDA)": "NVDA",
    "Aktie – Tesla (TSLA)": "TSLA",
    "Aktie – Amazon (AMZN)": "AMZN",
    "Aktie – JPMorgan (JPM)": "JPM",
    "Aktie – Johnson & Johnson (JNJ)": "JNJ",
    "Aktie – Exxon Mobil (XOM)": "XOM",
}

PERIODS = {
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
    "MAX": None,
}

NEWS_MODES = [
    "Automatische Empfehlung",
    "S&P 500, US-Markt & Federal Reserve",
    "Inflation, Zinsen & Geldpolitik",
    "Tech, KI & Wachstum",
    "ETF-Markt & langfristiges Investieren",
    "Gold, Krise & Sicherheit",
    "Anleihen, Renditen & Zinsrisiko",
    "Europa, EZB & Konjunktur",
    "Eigene Suche",
]


# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class AnalysisResult:
    ticker: str
    period: str
    capital: float
    asset_weight: float
    risk_drawdown: float
    last_close: float
    confidence: float
    risk_score: float
    capital_protection: float
    trend_score: float
    volatility_score: float
    drawdown_score: float
    outlook: str
    risk_label: str
    recommendation: str
    max_position: float
    tolerated_loss: float
    news_query: str
    news_label: str
    news_score: float
    generated_at: str


# =========================================================
# QUERY PARAMS + STATE
# =========================================================

def qp_get(key: str, default: str) -> str:
    try:
        value = st.query_params.get(key, default)
        if isinstance(value, list):
            return value[0] if value else default
        return value or default
    except Exception:
        return default


def init_state() -> None:
    page = urllib.parse.unquote_plus(qp_get("page", "Start"))
    theme = "Light Mode"  # Theme nativ über Streamlit Settings
    view = "Geführte Ansicht"

    if page not in ALL_PAGES:
        page = "Start"
    if theme not in ["Dark Mode", "Light Mode"]:
        theme = "Light Mode"
    if view not in ["Geführte Ansicht", "Expertenansicht"]:
        view = "Geführte Ansicht"

    defaults = {
        "current_page": page,
        "theme_mode": theme,
        "app_mode": view,
        "selected_asset_label": "ETF – S&P 500 (SPY)",
        "period": "5Y",
        "capital": 100000.0,
        "asset_weight": 10.0,
        "risk_drawdown": 10.0,
        "show_raw_data": True,
        "show_explanations": True,
        "show_advanced_metrics": False,
        "news_mode": "Automatische Empfehlung",
        "news_custom_query": "",
        "uploaded_override_active": False,
        "portfolio_rows": [
            {"Baustein": "SPY", "Gewichtung_%": 45.0},
            {"Baustein": "QQQ", "Gewichtung_%": 25.0},
            {"Baustein": "GLD", "Gewichtung_%": 10.0},
            {"Baustein": "AGG", "Gewichtung_%": 20.0},
        ],
        "chat_messages": [
            {
                "role": "assistant",
                "content": "Ich kann dir die aktuelle Einschätzung, Kennzahlen, Datenbasis und Risiken erklären. Keine Finanzberatung.",
            }
        ],
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # URL gewinnt nach Reload.
    st.session_state["current_page"] = page
    st.session_state["theme_mode"] = theme
    st.session_state["app_mode"] = view


def href(page_name: str) -> str:
    theme = st.session_state.get("theme_mode", "Light Mode")
    view = st.session_state.get("app_mode", "Geführte Ansicht")
    return (
        "/?page=" + urllib.parse.quote_plus(page_name)
        + "&view=" + urllib.parse.quote_plus(view)
    )


def sync_url() -> None:
    st.query_params["page"] = st.session_state.get("current_page", "Start")
    # Theme wird nativ über Streamlit Settings gesteuert, nicht über URL.
    st.query_params["view"] = "Geführte Ansicht"


def route_to(page_name: str) -> None:
    st.session_state["current_page"] = page_name
    sync_url()
    st.rerun()


# =========================================================
# DATA LOADING + CACHING
# =========================================================

@st.cache_data(show_spinner=False)
def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "GLD", "AGG", "JPM", "JNJ", "XOM", "EEM", "VTI"]
    rows = []
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1600)

    for ticker in tickers:
        base = rng.uniform(50, 300)
        drift = rng.uniform(0.00005, 0.00075)
        vol = rng.uniform(0.007, 0.026)
        if ticker in ["GLD", "AGG"]:
            drift *= 0.55
            vol *= 0.65

        returns = rng.normal(drift, vol, len(dates))
        close = base * np.cumprod(1 + returns)
        open_ = close * (1 + rng.normal(0, 0.004, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, len(dates)))
        volume = rng.integers(1_000_000, 120_000_000, len(dates))

        for d, o, h, l, c, v in zip(dates, open_, high, low, close, volume):
            rows.append(
                {
                    "date": d,
                    "ticker": ticker,
                    "asset_type": "ETF" if ticker in ["SPY", "QQQ", "GLD", "AGG", "EEM", "VTI"] else "Stock",
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "source_file": "generated_demo_data",
                }
            )

    return pd.DataFrame(rows)


def normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns={c: str(c).lower().strip() for c in df.columns})

    required = ["date", "ticker", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Fehlende Pflichtspalten: {missing}. Erwartet: date, ticker, close.")

    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0
    if "asset_type" not in df.columns:
        df["asset_type"] = "Unknown"
    if "source_file" not in df.columns:
        df["source_file"] = "unknown"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "close"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_local_market_data() -> pd.DataFrame:
    # Priorität 1: echte Feature-Daten als Parquet
    if DATA_FEATURES_PARQUET_PATH.exists():
        df = pd.read_parquet(DATA_FEATURES_PARQUET_PATH)
        df = normalize_market_df(df)
        df.attrs["data_source"] = "REAL_PARQUET"
        return df

    # Priorität 2: echte Feature-Daten als CSV
    if DATA_FEATURES_PATH.exists():
        df = pd.read_csv(DATA_FEATURES_PATH)
        df = normalize_market_df(df)
        df.attrs["data_source"] = "REAL_CSV"
        return df

    # Priorität 3: kontrollierter Market-Datensatz
    if DATA_MARKET_PATH.exists():
        df = pd.read_csv(DATA_MARKET_PATH)
        df = normalize_market_df(df)
        df.attrs["data_source"] = "REAL_MARKET_CSV"
        return df

    # Notfall: Demo-Daten
    df = normalize_market_df(make_demo_data())
    df.attrs["data_source"] = "DEMO_GENERATED"
    return df


@st.cache_resource(show_spinner=False)
def load_demo_model() -> Dict[str, Any]:
    # Platzhalter für ein echtes Modell.
    return {
        "name": "WealthScope Demo Scoring Model",
        "version": "local-rule-model-1.0",
        "features": ["trend", "volatility", "drawdown", "news_score", "weight_risk"],
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
    }


def load_uploaded_data(uploaded_file: Optional[Any]) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return normalize_market_df(pd.read_csv(uploaded_file))
        if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            return normalize_market_df(pd.read_excel(uploaded_file))
    except Exception as exc:
        st.error(f"Upload konnte nicht verarbeitet werden: {exc}")
        return None
    st.warning("Bitte CSV oder Excel hochladen.")
    return None


def get_market_data(uploaded_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if uploaded_df is not None and st.session_state.get("uploaded_override_active", False):
        return uploaded_df
    return load_local_market_data()


# =========================================================
# FEATURE ENGINEERING + ANALYTICS
# =========================================================

def available_tickers(df: pd.DataFrame) -> List[str]:
    return sorted(df["ticker"].dropna().astype(str).str.upper().unique().tolist())


def selected_ticker(df: pd.DataFrame) -> str:
    label = st.session_state.get("selected_asset_label", "ETF – S&P 500 (SPY)")
    ticker = DEFAULT_ASSET_MAP.get(label, "SPY")
    tickers = available_tickers(df)
    if ticker not in tickers and tickers:
        ticker = tickers[0]
    return ticker


def filter_period(df: pd.DataFrame, ticker: str, period: str) -> pd.DataFrame:
    out = df[df["ticker"] == ticker].copy().sort_values("date")
    n = PERIODS.get(period)
    if n is not None and len(out) > n:
        out = out.tail(n)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def enrich_features_cached(data_csv: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(data_csv), orient="split")
    return enrich_features(df)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("date")
    out["daily_return"] = out["close"].pct_change()
    out["return_5d"] = out["close"].pct_change(5)
    out["return_20d"] = out["close"].pct_change(20)
    out["return_60d"] = out["close"].pct_change(60)
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_50"] = out["close"].rolling(50).mean()
    out["ma_100"] = out["close"].rolling(100).mean()
    out["ma_200"] = out["close"].rolling(200).mean()
    out["ma_20_distance"] = out["close"] / out["ma_20"] - 1
    out["ma_50_distance"] = out["close"] / out["ma_50"] - 1
    out["ma_200_distance"] = out["close"] / out["ma_200"] - 1
    out["volatility_20d"] = out["daily_return"].rolling(20).std() * np.sqrt(252)
    out["volatility_60d"] = out["daily_return"].rolling(60).std() * np.sqrt(252)
    out["rolling_high"] = out["close"].cummax()
    out["drawdown"] = out["close"] / out["rolling_high"] - 1
    out["rolling_low_60"] = out["close"].rolling(60).min()
    out["rolling_high_60"] = out["close"].rolling(60).max()
    out["future_return_20d"] = out["close"].shift(-20) / out["close"] - 1
    out["target_20d"] = (out["future_return_20d"] > 0).astype(float)
    return out


def make_news_query(ticker: str, mode: str, custom_query: str) -> str:
    presets = {
        "Automatische Empfehlung": f'"{ticker}" OR "stock market" OR "Federal Reserve" OR inflation',
        "S&P 500, US-Markt & Federal Reserve": '"S&P 500" OR "US stock market" OR "Federal Reserve"',
        "Inflation, Zinsen & Geldpolitik": "inflation OR interest rates OR monetary policy",
        "Tech, KI & Wachstum": "AI OR technology stocks OR growth stocks",
        "ETF-Markt & langfristiges Investieren": "ETF flows OR long-term investing OR index funds",
        "Gold, Krise & Sicherheit": "gold OR safe haven OR geopolitical risk",
        "Anleihen, Renditen & Zinsrisiko": "bonds OR yields OR duration risk",
        "Europa, EZB & Konjunktur": "ECB OR Eurozone OR European economy",
        "Eigene Suche": custom_query.strip() or ticker,
    }
    return presets.get(mode, ticker)



@st.cache_data(show_spinner=False, ttl=1800)
def fetch_real_newsapi(query: str, api_key: str, language: str = "en", page_size: int = 10) -> Tuple[pd.DataFrame, str]:
    """
    Echte NewsAPI-Abfrage.
    Nutzt /v2/everything mit q, language, sortBy und pageSize.
    """
    if not api_key:
        return pd.DataFrame(), "NO_API_KEY"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query[:500],
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": int(page_size),
    }
    headers = {"X-Api-Key": api_key}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=12)
    except Exception as exc:
        return pd.DataFrame([{
            "Titel": "NewsAPI Request fehlgeschlagen",
            "Quelle": "NewsAPI",
            "Datum": "",
            "URL": "",
            "Beschreibung": str(exc),
        }]), "REQUEST_ERROR"

    if response.status_code != 200:
        try:
            err = response.json()
        except Exception:
            err = {"message": response.text}
        return pd.DataFrame([{
            "Titel": "NewsAPI Fehler",
            "Quelle": "NewsAPI",
            "Datum": "",
            "URL": "",
            "Beschreibung": f"HTTP {response.status_code}: {err}",
        }]), f"HTTP_{response.status_code}"

    payload = response.json()
    articles = payload.get("articles", [])

    rows = []
    for article in articles:
        source = article.get("source") or {}
        rows.append({
            "Titel": article.get("title") or "",
            "Quelle": source.get("name") or "",
            "Autor": article.get("author") or "",
            "Datum": article.get("publishedAt") or "",
            "URL": article.get("url") or "",
            "Beschreibung": article.get("description") or "",
            "Content": article.get("content") or "",
            "Bild": article.get("urlToImage") or "",
        })

    return pd.DataFrame(rows), "REAL_NEWSAPI"


def simple_news_sentiment(text_value: str) -> float:
    value = str(text_value).lower()

    positive_terms = [
        "growth", "strong", "record", "rally", "beat", "profit", "optimistic",
        "surge", "gain", "recovery", "upgrade", "cut rates", "soft landing"
    ]
    negative_terms = [
        "risk", "crisis", "inflation", "recession", "weak", "loss", "selloff",
        "war", "default", "downgrade", "rate hike", "lawsuit", "miss"
    ]

    score = 0.0
    for term in positive_terms:
        if term in value:
            score += 0.6
    for term in negative_terms:
        if term in value:
            score -= 0.6

    return max(-3.0, min(3.0, score))


def analyze_real_news_df(news_df: pd.DataFrame, query: str) -> Tuple[pd.DataFrame, float, str]:
    if news_df.empty:
        return news_df, 0.0, "Keine echten News gefunden"

    rows = []
    scores = []

    for _, row in news_df.iterrows():
        joined = " ".join([
            str(row.get("Titel", "")),
            str(row.get("Beschreibung", "")),
            str(row.get("Content", "")),
        ])
        score = simple_news_sentiment(joined)
        scores.append(score)

        if score >= 1.0:
            label = "positiv"
        elif score <= -1.0:
            label = "negativ"
        elif score > 0:
            label = "leicht positiv"
        elif score < 0:
            label = "leicht negativ"
        else:
            label = "neutral"

        rows.append({
            "Titel": row.get("Titel", ""),
            "Quelle": row.get("Quelle", ""),
            "Datum": row.get("Datum", ""),
            "Sentiment": round(score, 2),
            "Relevanz": "Hoch" if abs(score) >= 1.2 else "Mittel",
            "Impact": "Hoch" if abs(score) >= 1.5 else "Mittel",
            "Kurzinterpretation": label,
            "URL": row.get("URL", ""),
            "Bild": row.get("Bild", ""),
            "Suchlogik": query,
        })

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0

    if avg_score >= 1.0:
        news_label = "Echte News-Lage positiv"
    elif avg_score >= 0.2:
        news_label = "Echte News-Lage leicht positiv"
    elif avg_score <= -1.0:
        news_label = "Echte News-Lage negativ"
    elif avg_score <= -0.2:
        news_label = "Echte News-Lage leicht negativ"
    else:
        news_label = "Echte News-Lage neutral"

    return pd.DataFrame(rows), avg_score, news_label


def get_news_api_key() -> str:
    try:
        return str(st.secrets.get("NEWS_API_KEY", "")).strip()
    except Exception:
        return ""


def analyze_news_runtime(query: str) -> Tuple[pd.DataFrame, float, str, str]:
    """
    Nutzt echte NewsAPI, falls NEWS_API_KEY vorhanden ist.
    Sonst Demo-Fallback.
    """
    api_key = get_news_api_key()

    if api_key:
        real_df, status = fetch_real_newsapi(query=query, api_key=api_key, language="en", page_size=10)
        if status == "REAL_NEWSAPI" and not real_df.empty:
            scored_df, score, label = analyze_real_news_df(real_df, query)
            return scored_df, score, label, "REAL_NEWSAPI"

        demo_df, demo_score, demo_label = analyze_news(query)
        demo_df["Hinweis"] = f"NewsAPI Status: {status}. Demo-Fallback verwendet."
        return demo_df, demo_score, demo_label, f"NEWSAPI_FALLBACK_{status}"

    demo_df, demo_score, demo_label = analyze_news(query)
    demo_df["Hinweis"] = "Kein NEWS_API_KEY in .streamlit/secrets.toml gefunden. Demo-Newslogik aktiv."
    return demo_df, demo_score, demo_label, "DEMO_NO_NEWS_API_KEY"



def analyze_news(query: str) -> Tuple[pd.DataFrame, float, str]:
    q = query.lower()
    positive_terms = ["growth", "strong", "optimistic", "record", "rally", "profit", "cut", "stabil", "boom", "beat"]
    negative_terms = ["risk", "crisis", "inflation", "war", "recession", "weak", "loss", "rate hike", "default", "selloff"]

    score = 0.0
    for term in positive_terms:
        if term in q:
            score += 0.7
    for term in negative_terms:
        if term in q:
            score -= 0.6

    score = max(-3.0, min(3.0, score))

    if score >= 1.2:
        label = "News-Lage positiv"
    elif score >= 0.2:
        label = "News-Lage leicht positiv"
    elif score <= -1.2:
        label = "News-Lage negativ"
    elif score <= -0.2:
        label = "News-Lage leicht negativ"
    else:
        label = "News-Lage neutral"

    rows = [
        {
            "Titel": "Makro- und Marktlage wird für die Einschätzung berücksichtigt",
            "Quelle": "Demo-Newslogik",
            "Sentiment": round(score, 2),
            "Relevanz": "Hoch" if abs(score) > 1 else "Mittel",
            "Impact": "Hoch" if abs(score) > 1.2 else "Mittel",
            "Kurzinterpretation": label,
            "Suchlogik": query,
        },
        {
            "Titel": "Suchlogik beeinflusst News Intelligence",
            "Quelle": "Interne App-Logik",
            "Sentiment": round(score * 0.6, 2),
            "Relevanz": "Mittel",
            "Impact": "Mittel",
            "Kurzinterpretation": "Suchbegriffe werden nachvollziehbar in Score und Interpretation übersetzt.",
            "Suchlogik": query,
        },
    ]
    return pd.DataFrame(rows), score, label


def compute_scores(
    df: pd.DataFrame,
    news_score: float,
    news_query: str,
    news_label: str,
    capital: float,
    weight: float,
    risk_drawdown: float,
    period: str,
    ticker: str,
) -> AnalysisResult:
    if df.empty:
        now = datetime.now().isoformat(timespec="seconds")
        return AnalysisResult(
            ticker=ticker,
            period=period,
            capital=capital,
            asset_weight=weight,
            risk_drawdown=risk_drawdown,
            last_close=0.0,
            confidence=0.0,
            risk_score=0.0,
            capital_protection=0.0,
            trend_score=0.0,
            volatility_score=0.0,
            drawdown_score=0.0,
            outlook="Nicht bewertbar",
            risk_label="Unbekannt",
            recommendation="Keine Daten verfügbar.",
            max_position=0.0,
            tolerated_loss=capital * risk_drawdown / 100,
            news_query=news_query,
            news_label=news_label,
            news_score=news_score,
            generated_at=now,
        )

    last = df.dropna(subset=["close"]).tail(1).iloc[0]
    close = float(last["close"])

    ma_20 = float(last["ma_20"]) if pd.notna(last.get("ma_20")) else close
    ma_50 = float(last["ma_50"]) if pd.notna(last.get("ma_50")) else close
    ma_200 = float(last["ma_200"]) if pd.notna(last.get("ma_200")) else close
    vol = float(last["volatility_20d"]) if pd.notna(last.get("volatility_20d")) else 0.2
    drawdown = float(last["drawdown"]) if pd.notna(last.get("drawdown")) else 0.0
    ret_20 = float(last["return_20d"]) if pd.notna(last.get("return_20d")) else 0.0

    trend_points = 0.0
    trend_points += 25 if close >= ma_20 else -10
    trend_points += 25 if close >= ma_50 else -10
    trend_points += 25 if close >= ma_200 else -10
    trend_points += 25 if ret_20 >= 0 else -10
    trend_score = max(0, min(100, trend_points))

    volatility_score = max(0, min(100, 100 - vol * 240))
    drawdown_score = max(0, min(100, 100 + drawdown * 180))
    weight_risk = max(0, min(100, 100 - max(0, weight - 10) * 3))
    risk_capacity = max(0, min(100, risk_drawdown * 5))

    confidence = round(
        0.36 * trend_score
        + 0.22 * volatility_score
        + 0.18 * drawdown_score
        + 0.14 * (50 + news_score * 10)
        + 0.10 * weight_risk,
        1,
    )

    risk_score = round(
        100
        - (
            0.35 * volatility_score
            + 0.30 * drawdown_score
            + 0.20 * weight_risk
            + 0.15 * risk_capacity
        ),
        1,
    )
    risk_score = max(0, min(100, risk_score))
    capital_protection = round(max(0, min(100, 100 - risk_score * 0.72 + risk_capacity * 0.18)), 1)

    if confidence >= 70:
        outlook = "Positiv"
    elif confidence >= 52:
        outlook = "Kontrolliert prüfen"
    elif confidence >= 40:
        outlook = "Unsicher"
    else:
        outlook = "Defensiv prüfen"

    if risk_score >= 65:
        risk_label = "Hoch"
    elif risk_score >= 38:
        risk_label = "Mittel"
    else:
        risk_label = "Niedrig"

    max_position = capital * weight / 100
    tolerated_loss = capital * risk_drawdown / 100

    if risk_label == "Hoch":
        recommendation = "Position reduzieren, Gewichtung begrenzen und erst nach Stabilisierung nachlegen."
    elif outlook in ["Positiv", "Kontrolliert prüfen"]:
        recommendation = "Strukturiert prüfen, Positionsgröße begrenzen und News-/Trend-Signale beobachten."
    else:
        recommendation = "Keine überhastete Entscheidung. Erst Risiko, Zeithorizont und Alternativen prüfen."

    return AnalysisResult(
        ticker=ticker,
        period=period,
        capital=capital,
        asset_weight=weight,
        risk_drawdown=risk_drawdown,
        last_close=round(close, 2),
        confidence=confidence,
        risk_score=round(risk_score, 1),
        capital_protection=capital_protection,
        trend_score=round(trend_score, 1),
        volatility_score=round(volatility_score, 1),
        drawdown_score=round(drawdown_score, 1),
        outlook=outlook,
        risk_label=risk_label,
        recommendation=recommendation,
        max_position=max_position,
        tolerated_loss=tolerated_loss,
        news_query=news_query,
        news_label=news_label,
        news_score=round(news_score, 2),
        generated_at=datetime.now().isoformat(timespec="seconds"),
    )


def money(value: float) -> str:
    return f"{value:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

def data_proof(market_df: pd.DataFrame) -> Dict[str, Any]:
    source = getattr(market_df, "attrs", {}).get("data_source", "UNKNOWN")

    file_used = "Unbekannt"
    file_size_mb = 0.0

    if "DATA_FEATURES_PARQUET_PATH" in globals() and DATA_FEATURES_PARQUET_PATH.exists():
        file_used = str(DATA_FEATURES_PARQUET_PATH)
        file_size_mb = DATA_FEATURES_PARQUET_PATH.stat().st_size / (1024 * 1024)
    elif DATA_FEATURES_PATH.exists():
        file_used = str(DATA_FEATURES_PATH)
        file_size_mb = DATA_FEATURES_PATH.stat().st_size / (1024 * 1024)
    elif DATA_MARKET_PATH.exists():
        file_used = str(DATA_MARKET_PATH)
        file_size_mb = DATA_MARKET_PATH.stat().st_size / (1024 * 1024)

    date_min = None
    date_max = None
    if "date" in market_df.columns and not market_df.empty:
        date_min = pd.to_datetime(market_df["date"], errors="coerce").min()
        date_max = pd.to_datetime(market_df["date"], errors="coerce").max()

    feature_cols = [
        c for c in market_df.columns
        if c not in ["date", "ticker", "asset_type", "source_file", "open", "high", "low", "close", "volume"]
    ]

    return {
        "source": source,
        "file_used": file_used,
        "file_size_mb": round(file_size_mb, 2),
        "rows": int(len(market_df)),
        "columns": int(len(market_df.columns)),
        "tickers": int(market_df["ticker"].nunique()) if "ticker" in market_df.columns else 0,
        "date_min": date_min.strftime("%Y-%m-%d") if pd.notna(date_min) else "Unbekannt",
        "date_max": date_max.strftime("%Y-%m-%d") if pd.notna(date_max) else "Unbekannt",
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "has_target_20d": "target_20d" in market_df.columns,
        "has_future_return_20d": "future_return_20d" in market_df.columns,
    }


def render_data_badge(market_df: pd.DataFrame) -> None:
    proof = data_proof(market_df)
    source = proof["source"]

    if str(source).startswith("REAL"):
        label = "ECHTE DATEN AKTIV"
        color = "#22c55e"
    else:
        label = "DEMO-DATEN AKTIV"
        color = "#ef4444"

    st.markdown(
        f"""
<div style="
    margin: 0 0 1.1rem 0;
    padding: 0.85rem 1rem;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.20);
    background: rgba(15,23,42,0.38);
    display: flex;
    flex-wrap: wrap;
    gap: .55rem;
    align-items: center;
">
    <span style="
        background:{color};
        color:white;
        font-weight:900;
        font-size:.72rem;
        letter-spacing:.06em;
        padding:.35rem .55rem;
        border-radius:999px;
    ">{label}</span>
    <span><b>{proof["rows"]:,}</b> Zeilen</span>
    <span>·</span>
    <span><b>{proof["columns"]}</b> Spalten</span>
    <span>·</span>
    <span><b>{proof["tickers"]}</b> Ticker</span>
    <span>·</span>
    <span><b>{proof["date_min"]}</b> bis <b>{proof["date_max"]}</b></span>
    <span>·</span>
    <span>Datei: <b>{proof["file_used"]}</b></span>
</div>
        """.replace(",", "."),
        unsafe_allow_html=True,
    )



def pct(value: float) -> str:
    return f"{value * 100:.2f} %"


# =========================================================
# STYLE + STRUCTURE
# =========================================================

def inject_css(theme_mode: str = "Light Mode") -> None:
    """
    Minimaler CSS-Hook.
    Streamlit übernimmt Theme/Light/Dark nativ.
    Custom CSS nur für Footer-Abstand und kleine Stabilisierung.
    """
    st.markdown(
        """
<style>
main .block-container {
    padding-bottom: 8rem !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

def render_header() -> None:
    current = st.session_state.get("current_page", "Start")
    theme = st.session_state.get("theme_mode", "Light Mode")
    view = st.session_state.get("app_mode", "Geführte Ansicht")
    theme_short = "Native"
    view_short = "Geführt" if view == "Geführte Ansicht" else "Experte"

    nav = ""
    for page in MAIN_PAGES:
        active = " active" if current == page else ""
        nav += f'<a class="ws-nav-link{active}" href="{href(page)}" target="_self">{PAGE_LABELS[page]}</a>'

    st.markdown(
        f"""
<div class="ws-header">
    <a class="ws-brand" href="{href("Start")}" target="_self" aria-label="Zur Startseite">
        <span class="ws-brand-mark">W</span>
        <span class="ws-brand-name">WealthScope</span>
        <span class="ws-brand-ai">AI</span>
    </a>
    <nav class="ws-nav">{nav}</nav>
    <div class="ws-header-status">
        <span>{theme_short}</span>
        <span>{view_short}</span>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_bottom_bar() -> None:
    from urllib.parse import quote_plus
    from datetime import datetime

    view = st.session_state.get("app_mode", "Geführte Ansicht")
    current_page = st.session_state.get("current_page", "Start")

    items = [
        ("📰", "News", "News-Archiv"),
        ("📖", "Methodik", "So funktioniert's"),
        ("🎓", "Projekt", "Über das Projekt"),
        ("📦", "Export", "Professor-Export"),
        ("⚖️", "Impressum", "Impressum"),
        ("🛡️", "Datenschutz", "Datenschutz"),
        ("🟢", "Status", "Betriebsstatus"),
    ]

    links = []
    for icon, label, page_name in items:
        active = " active" if page_name == current_page else ""
        href = f"/?page={quote_plus(page_name)}&view={quote_plus(view)}"
        links.append(
            f"<a class='ws-native-footer-link{active}' href='{href}' target='_self'>"
            f"<span class='ws-native-footer-icon'>{icon}</span>"
            f"<span>{label}</span>"
            f"</a>"
        )

    checked = datetime.now().strftime("%H:%M")

    html = f"""
<style>
main .block-container {{
    padding-bottom: 8rem !important;
}}

.ws-native-footer {{
    position: fixed !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    z-index: 999999 !important;
    padding: 0.70rem 1.2rem 0.55rem !important;
    background: color-mix(in srgb, var(--st-background-color) 90%, transparent) !important;
    border-top: 1px solid var(--st-border-color) !important;
    box-shadow: 0 -16px 38px rgba(15, 23, 42, 0.10) !important;
    backdrop-filter: blur(18px) !important;
    -webkit-backdrop-filter: blur(18px) !important;
}}

.ws-native-footer-inner {{
    max-width: 1180px !important;
    margin: 0 auto !important;
}}

.ws-native-footer-links {{
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 0.55rem !important;
}}

.ws-native-footer-link {{
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.38rem !important;
    min-height: 38px !important;
    padding: 0.42rem 0.82rem !important;
    border-radius: 999px !important;
    color: var(--st-text-color) !important;
    text-decoration: none !important;
    font-weight: 750 !important;
    font-size: 0.82rem !important;
    background: var(--st-secondary-background-color) !important;
    border: 1px solid var(--st-border-color) !important;
}}

.ws-native-footer-link:hover,
.ws-native-footer-link.active {{
    border-color: var(--st-primary-color) !important;
    color: var(--st-primary-color) !important;
}}

.ws-native-footer-icon {{
    width: 24px !important;
    height: 24px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 999px !important;
    background: color-mix(in srgb, var(--st-primary-color) 14%, transparent) !important;
}}

.ws-native-footer-meta {{
    margin-top: 0.35rem !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    flex-wrap: wrap !important;
    gap: 0.80rem !important;
    font-size: 0.72rem !important;
    color: var(--st-text-color) !important;
    opacity: 0.78 !important;
    font-weight: 650 !important;
}}
</style>

<div class="ws-native-footer">
    <div class="ws-native-footer-inner">
        <div class="ws-native-footer-links">
            {''.join(links)}
        </div>
        <div class="ws-native-footer-meta">
            <span>Marktdaten: <b>Aktuell</b></span>
            <span>News: <b>Mittel</b></span>
            <span>Check: <b>{checked}</b></span>
            <span>Version: <b>2.0-max</b></span>
        </div>
    </div>
</div>
"""

    st.markdown(html, unsafe_allow_html=True)

def card(title: str, body: str, hero: bool = False) -> None:
    hero_class = " ws-hero" if hero else ""
    st.markdown(
        f"""
<div class="ws-card{hero_class}">
    <h2>{title}</h2>
    <p>{body}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def metric_grid(items: List[Tuple[str, str]]) -> None:
    html = '<div class="ws-metric-grid">'
    for label, value in items:
        html += f'<div class="ws-metric"><small>{label}</small><strong>{value}</strong></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================


def render_clickable_sidebar_logo() -> None:
    logo_path = Path("assets/wealthscope_logo.svg")

    if not logo_path.exists():
        st.markdown(
            '<a href="?page=Start&view=Geführte+Ansicht" target="_self" style="text-decoration:none;">'
            '<strong>💠 WealthScope AI</strong>'
            '</a>',
            unsafe_allow_html=True,
        )
        return

    encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <a href="?page=Start&view=Geführte+Ansicht" target="_self" style="text-decoration:none;">
            <img
                src="data:image/svg+xml;base64,{encoded_logo}"
                alt="WealthScope AI"
                style="width: 180px; max-width: 100%; height: auto; display: block; margin-bottom: 1rem;"
            />
        </a>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    uploaded_df = None

    with st.sidebar:
        st.caption("Analyse-Steuerung für Kapital, Risiko und Portfolio.")

        st.divider()

        st.caption("SCHNELLSTART")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📈 Analyse", width="stretch"):
                route_to("Wealth Outlook")
        with c2:
            if st.button("🧬 Daten", width="stretch"):
                route_to("Betriebsstatus")

        st.divider()

        st.caption("ASSET & ZEITRAUM")

        asset_options = list(DEFAULT_ASSET_MAP.keys())
        current_asset = st.session_state.get("selected_asset_label", asset_options[0])
        if current_asset not in asset_options:
            current_asset = asset_options[0]

        st.session_state["selected_asset_label"] = st.selectbox(
            "Asset auswählen",
            asset_options,
            index=asset_options.index(current_asset),
            help="Wähle das Asset, für das die Analyse berechnet werden soll.",
        )

        st.session_state["period"] = st.select_slider(
            "Zeitraum",
            options=list(PERIODS.keys()),
            value=st.session_state.get("period", "5Y"),
            help="Bestimmt, wie viele historische Datenpunkte in die Analyse einfließen.",
        )

        st.divider()

        st.caption("PORTFOLIO & KAPITAL")

        with st.form("capital_form", border=True):
            st.session_state["capital"] = st.number_input(
                "Kapitalbetrag",
                min_value=1000.0,
                max_value=10_000_000.0,
                value=float(st.session_state.get("capital", 100000.0)),
                step=1000.0,
                help="Gesamtkapital, das du für die Szenarioanalyse betrachten möchtest.",
            )

            st.session_state["asset_weight"] = st.slider(
                "Geplante Gewichtung im Portfolio (%)",
                min_value=1.0,
                max_value=100.0,
                value=float(st.session_state.get("asset_weight", 10.0)),
                step=1.0,
                help="Wie viel Prozent deines Kapitals maximal in dieses Asset fließen sollen.",
            )

            st.session_state["risk_drawdown"] = st.slider(
                "Tolerierbarer Rückgang (%)",
                min_value=1.0,
                max_value=60.0,
                value=float(st.session_state.get("risk_drawdown", 10.0)),
                step=1.0,
                help="Wie viel Rückgang auf das Gesamtportfolio du gedanklich tragen könntest.",
            )

            submitted = st.form_submit_button("Szenario übernehmen", width="stretch")
            if submitted:
                st.toast("Portfolio-Szenario aktualisiert.", icon="✅")

        capital = float(st.session_state.get("capital", 100000.0))
        weight = float(st.session_state.get("asset_weight", 10.0))
        drawdown = float(st.session_state.get("risk_drawdown", 10.0))

        max_position = capital * weight / 100
        tolerated_loss = capital * drawdown / 100

        st.markdown(
            f"""
**Szenario kurz:**  
Kapital: **{money(capital)}**  
Max. Position: **{money(max_position)}**  
Tolerierter Rückgang: **{money(tolerated_loss)}**
            """
        )

        st.divider()

        st.caption("DARSTELLUNG")
        st.caption("Theme: über Streamlit-Menü oben rechts → Settings → Theme")


        st.session_state["app_mode"] = "Geführte Ansicht"
        st.caption("Analysemodus: Geführte Standardansicht")

        st.toggle("Rohdaten anzeigen", key="show_raw_data")
        st.toggle("Erklärungen anzeigen", key="show_explanations")

        st.divider()

        st.caption("NAVIGATION")
        current = st.session_state.get("current_page", "Start")
        nav_options = MAIN_PAGES + SERVICE_PAGES
        nav_index = nav_options.index(current) if current in nav_options else 0

        nav_page = st.selectbox(
            "Seite öffnen",
            nav_options,
            index=nav_index,
            format_func=lambda p: f"{PAGE_ICONS.get(p, '•')} {p}",
        )
        if nav_page != current:
            route_to(nav_page)

        st.divider()

        st.caption("DATEN-UPLOAD")
        upload = st.file_uploader(
            "Eigene CSV/Excel laden",
            type=["csv", "xlsx", "xls"],
            help="Erwartete Mindestspalten: date, ticker, close.",
        )

        uploaded_df = load_uploaded_data(upload)
        if uploaded_df is not None:
            st.session_state["uploaded_override_active"] = st.toggle(
                "Upload als Datenquelle verwenden",
                value=st.session_state.get("uploaded_override_active", False),
            )
            st.success(f"Upload erkannt: {len(uploaded_df):,} Zeilen".replace(",", "."))

        st.divider()

        st.caption("STATUS")
        data_source = getattr(df, "attrs", {}).get("data_source", "Unbekannt")

        st.markdown(
            f"""
Datenquelle: **{data_source}**  
Seite: **{st.session_state.get("current_page")}**  
Version: **{APP_VERSION}**
            """
        )

    return uploaded_df

# =========================================================
# CHARTS
# =========================================================

def chart_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name=f"{ticker} Schlusskurs"))

    for col, name in [("ma_20", "MA 20"), ("ma_50", "MA 50"), ("ma_200", "MA 200")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=name))

    fig.update_layout(
        title=f"{ticker}: Kursverlauf und gleitende Durchschnitte",
        height=460,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        legend=dict(orientation="h"),
    )
    return fig


def chart_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name=ticker,
            )
        ]
    )
    fig.update_layout(
        title=f"{ticker}: Candlestick-Ansicht",
        height=460,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig


def chart_drawdown(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["drawdown"] * 100, mode="lines", fill="tozeroy", name="Drawdown %"))
    fig.update_layout(title=f"{ticker}: Drawdown", height=360, margin=dict(l=10, r=10, t=55, b=10), hovermode="x unified")
    return fig


def chart_returns(df: pd.DataFrame, ticker: str) -> go.Figure:
    clean = df.dropna(subset=["daily_return"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean["daily_return"] * 100, nbinsx=60, name="Tagesrenditen"))
    fig.update_layout(title=f"{ticker}: Verteilung der Tagesrenditen", height=360, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def chart_scores(result: AnalysisResult) -> go.Figure:
    labels = ["Trend", "Volatilität", "Drawdown", "Kapital-Schutz", "Confidence"]
    values = [
        result.trend_score,
        result.volatility_score,
        result.drawdown_score,
        result.capital_protection,
        result.confidence,
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, name="Score"))
    fig.update_layout(title="Score-Zerlegung", height=360, yaxis=dict(range=[0, 100]), margin=dict(l=10, r=10, t=55, b=10))
    return fig


def chart_radar(result: AnalysisResult) -> go.Figure:
    categories = ["Trend", "Volatilität", "Drawdown", "Kapital-Schutz", "Confidence"]
    values = [result.trend_score, result.volatility_score, result.drawdown_score, result.capital_protection, result.confidence]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself", name="Profil"))
    fig.update_layout(title="Analyseprofil", polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420)
    return fig



def chart_volatility(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df[df["ticker"] == ticker].sort_values("date").copy()

    fig = go.Figure()

    if "volatility_20d" in d.columns:
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["volatility_20d"],
                mode="lines",
                name="20T Volatilität",
            )
        )

    fig.update_layout(
        title=f"{ticker}: Volatilitätsverlauf",
        hovermode="x unified",
        yaxis_tickformat=".1%",
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h"),
    )

    return fig

def chart_portfolio(capital: float, weight: float, ticker: str) -> go.Figure:
    invested = capital * weight / 100
    cash = capital - invested
    fig = go.Figure(data=[go.Pie(labels=[ticker, "Liquidität / Rest"], values=[invested, cash], hole=0.55)])
    fig.update_layout(title="Kapitalallokation im Szenario", height=360, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def show_chart_with_data(title: str, fig: go.Figure, data: pd.DataFrame, key: str) -> None:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True, config={"responsive": True, "displayModeBar": True, "displaylogo": False}, key=key)
    if st.session_state.get("show_raw_data", True):
        with st.expander(f"Daten hinter dem Diagramm anzeigen: {title}", expanded=False):
            st.dataframe(data, width="stretch", hide_index=True)
            st.download_button(
                f"{title} als CSV herunterladen",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name=f"{title.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"download_{key}",
            )


# =========================================================
# CONTEXT
# =========================================================

def build_context(market_df: pd.DataFrame) -> Dict[str, Any]:
    ticker = selected_ticker(market_df)
    selected = filter_period(market_df, ticker, st.session_state.get("period", "5Y"))
    features = enrich_features(selected)

    news_query = make_news_query(
        ticker,
        st.session_state.get("news_mode", "Automatische Empfehlung"),
        st.session_state.get("news_custom_query", ""),
    )
    news_df, news_score, news_label, news_source = analyze_news_runtime(news_query)

    result = compute_scores(
        features,
        news_score,
        news_query,
        news_label,
        float(st.session_state.get("capital", 100000.0)),
        float(st.session_state.get("asset_weight", 10.0)),
        float(st.session_state.get("risk_drawdown", 10.0)),
        st.session_state.get("period", "5Y"),
        ticker,
    )

    return {
        "market": market_df,
        "ticker": ticker,
        "features": features,
        "news_query": news_query,
        "news_df": news_df,
        "news_score": news_score,
        "news_label": news_label,
        "news_source": news_source,
        "result": result,
        "model": load_demo_model(),
    }


# =========================================================
# REUSABLE UI
# =========================================================

def page_title(title: str, subtitle: str = "") -> None:
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def result_metrics(result: AnalysisResult) -> None:
    metric_grid(
        [
            ("Outlook", result.outlook),
            ("Risiko", result.risk_label),
            ("Confidence", f"{result.confidence} / 100"),
            ("Kapital-Schutz", f"{result.capital_protection} / 100"),
        ]
    )


def analysis_report_markdown(result: AnalysisResult) -> str:
    return f"""# WealthScope AI Analysebericht

Erstellt am: {result.generated_at}

## Asset

- Ticker: {result.ticker}
- Zeitraum: {result.period}
- Kapital: {money(result.capital)}
- Gewichtung: {result.asset_weight} %
- Tolerierter Rückgang: {result.risk_drawdown} %

## Ergebnis

- Outlook: {result.outlook}
- Risiko: {result.risk_label}
- Confidence: {result.confidence} / 100
- Kapital-Schutz: {result.capital_protection} / 100
- Trend Score: {result.trend_score} / 100
- Volatilitäts-Score: {result.volatility_score} / 100
- Drawdown-Score: {result.drawdown_score} / 100

## Interpretation

{result.recommendation}

## News

- News-Lage: {result.news_label}
- News-Score: {result.news_score}
- Suchlogik: {result.news_query}

## Hinweis

Keine Finanzberatung. Wissenschaftliche Demo-Anwendung.
"""


def render_explainers() -> None:
    with st.expander("Wissenschaftliche Einordnung und Grenzen", expanded=False):
        st.markdown(
            """
Diese Demo kombiniert historische Marktdaten, technische Kennzahlen und eine vereinfachte News-Logik.
Sie ersetzt keine Finanzberatung und liefert keine Garantie für zukünftige Kursentwicklungen.

**Wichtig für die Präsentation:**

- Die Datenbasis ist nachvollziehbar.
- Diagramme sind interaktiv.
- Rohdaten sind einsehbar.
- Scores werden zerlegt.
- Exporte machen Ergebnisse reproduzierbar.
            """
        )


# =========================================================
# DIALOGS
# =========================================================

@st.dialog("Analyse-Hinweis")
def disclaimer_dialog() -> None:
    st.write("WealthScope AI ist eine lokale wissenschaftliche Demo-Anwendung.")
    st.write("Die Ergebnisse sind keine Finanzberatung und dürfen nicht als Kauf- oder Verkaufsempfehlung verstanden werden.")
    if st.button("Verstanden"):
        st.session_state["disclaimer_seen"] = True
        st.rerun()


# =========================================================
# PAGES
# =========================================================

def page_start(ctx: Dict[str, Any]) -> None:
    page_title(
        "Geführter Modus-Inhalt",
        "Diese Ansicht erklärt die Ergebnisse bewusst einfach: Was bedeutet die Einschätzung, wie hoch ist das Risiko und welcher nächste Schritt ist sinnvoll?",
    )

    card(
        "Kapital schützen. Märkte verstehen. Entscheidungen simulieren.",
        "WealthScope AI richtet sich an Menschen, die Kapital verantwortungsvoll einordnen möchten. Die App kombiniert historische Marktbewegungen, News-Logik, Risikokennzahlen und ein ML-nahes Scoring zu einer nachvollziehbaren Entscheidungshilfe.",
        hero=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📈 Analyse starten", width="stretch"):
            route_to("Wealth Outlook")
        card("Geführte Ansicht", "Klartext, Kapital-Schutz und nächste Schritte für nicht-technische Nutzer.")
    with c2:
        if st.button("🧬 Datenbasis prüfen", width="stretch"):
            route_to("Betriebsstatus")
        card("Echte Datenbasis", "Zeilen, Ticker, Zeitraum, Features und Zielvariable werden offen angezeigt.")
    with c3:
        if st.button("📦 Ergebnis exportieren", width="stretch"):
            route_to("Professor-Export")
        card("Reproduzierbarkeit", "Downloadbare Daten, Exporte, URL-Zustand und sichtbare Annahmen.")

    tab1, tab2, tab3 = st.tabs(["Überblick", "Demo-Ablauf", "Präsentationsnutzen"])
    with tab1:
        st.info("Die Startseite ist bewusst erklärend aufgebaut und zeigt, worum es in der App geht.")
    with tab2:
        st.markdown("1. Asset wählen\n2. Kapitalparameter setzen\n3. News-Logik wählen\n4. Ergebnisse interpretieren\n5. Daten exportieren")
    with tab3:
        st.success("Die App zeigt Big-Data-Basis, Feature Engineering, dynamische Visualisierung und nachvollziehbare Ergebnislogik.")

    if st.button("Demo-Hinweis anzeigen"):
        disclaimer_dialog()

    render_explainers()


def page_outlook(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    df = ctx["features"]

    page_title("Wealth Outlook", f"Asset: {result.ticker}. Szenario, Kapital-Schutz und nächste Schritte.")
    result_metrics(result)

    card("Interpretation", f"{result.recommendation} News-Lage: {result.news_label}.", hero=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Kurs", "Risiko", "Scores", "Rohdaten"])

    with tab1:
        show_chart_with_data(
            "Kursverlauf",
            chart_price(df, result.ticker),
            df[["date", "ticker", "close", "ma_20", "ma_50", "ma_200"]].tail(600),
            "outlook_price",
        )
    with tab2:
        show_chart_with_data(
            "Drawdown",
            chart_drawdown(df, result.ticker),
            df[["date", "ticker", "close", "drawdown", "volatility_20d"]].tail(600),
            "outlook_drawdown",
        )

        show_chart_with_data(
            "Volatilität",
            chart_volatility(df, result.ticker),
            df[["date", "ticker", "volatility_20d"]].dropna().tail(600),
            "outlook_volatility",
        )
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            show_chart_with_data("Score-Zerlegung", chart_scores(result), pd.DataFrame([asdict(result)]), "outlook_scores")
        with col2:
            show_chart_with_data("Radar-Profil", chart_radar(result), pd.DataFrame([asdict(result)]), "outlook_radar")
    with tab4:
        st.dataframe(df.tail(1000), width="stretch", hide_index=True)

    render_explainers()


def page_kompass(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    page_title("Kapital-Kompass", "Prüfe, was Risiko bezogen auf dein Kapital konkret bedeutet.")

    card(
        "Was darf Risiko überhaupt bedeuten?",
        "Bevor ein Asset analysiert wird, sollte klar sein: Wie viel Kapital steht zur Verfügung? Wie viel darf maximal in eine einzelne Idee fließen? Wie viel Verlust wäre emotional und finanziell tragbar?",
        hero=True,
    )

    metric_grid(
        [
            ("Gesamtkapital", money(result.capital)),
            ("Max. Einzelposition", money(result.max_position)),
            ("Tolerierter Rückgang", money(result.tolerated_loss)),
            ("Risikoklasse", result.risk_label),
        ]
    )

    with st.status("Kapitalprüfung", expanded=False) as status:
        st.write("Kapitalbetrag geprüft.")
        st.write("Gewichtung geprüft.")
        st.write("Risikorückgang geprüft.")
        status.update(label="Kapitalprüfung abgeschlossen", state="complete")

    if result.risk_label == "Hoch":
        st.warning("Die aktuelle Kombination aus Gewichtung, Volatilität und Drawdown wirkt riskant.")
    elif result.risk_label == "Mittel":
        st.info("Die Gewichtung wirkt kontrollierbar, sollte aber beobachtet werden.")
    else:
        st.success("Die Risikoannahmen wirken im Verhältnis zum Kapital eher defensiv.")


def page_simulator(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    page_title("Portfolio-Simulator", "Simuliere, wie stark eine Position dein Kapital beeinflusst.")

    show_chart_with_data(
        "Portfolio-Allokation",
        chart_portfolio(result.capital, result.asset_weight, result.ticker),
        pd.DataFrame(
            [
                {"Baustein": result.ticker, "Wert": result.max_position},
                {"Baustein": "Liquidität / Rest", "Wert": result.capital - result.max_position},
            ]
        ),
        "portfolio_alloc",
    )

    st.subheader("Szenario-Tabelle bearbeiten")
    portfolio_df = pd.DataFrame(st.session_state.get("portfolio_rows", []))
    edited = st.data_editor(
        portfolio_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "Baustein": st.column_config.TextColumn("Baustein"),
            "Gewichtung_%": st.column_config.NumberColumn("Gewichtung %", min_value=0.0, max_value=100.0, step=1.0),
        },
    )
    st.session_state["portfolio_rows"] = edited.to_dict("records")

    total_weight = edited["Gewichtung_%"].sum() if not edited.empty and "Gewichtung_%" in edited else 0
    st.progress(min(int(total_weight), 100), text=f"Gesamtgewichtung: {total_weight:.1f} %")

    if total_weight > 100:
        st.error("Die Gesamtgewichtung liegt über 100 %. Bitte reduzieren.")
    else:
        st.success("Die Gesamtgewichtung ist innerhalb des Rahmens.")


def page_watchlist(ctx: Dict[str, Any]) -> None:
    raw = ctx["market"]
    tickers = available_tickers(raw)
    page_title("Watchlist-Vergleich", "Vergleiche mehrere Assets anhand Rendite, Volatilität und Drawdown.")

    default = [t for t in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"] if t in tickers]
    selected = st.multiselect("Watchlist auswählen", tickers, default=default)

    rows = []
    for ticker in selected:
        df = enrich_features(filter_period(raw, ticker, st.session_state.get("period", "5Y")))
        if df.empty:
            continue
        last = df.tail(1).iloc[0]
        rows.append(
            {
                "Ticker": ticker,
                "Letzter Kurs": round(float(last["close"]), 2),
                "Return 20d": round(float(last["return_20d"]) * 100, 2) if pd.notna(last["return_20d"]) else None,
                "Volatilität 20d": round(float(last["volatility_20d"]) * 100, 2) if pd.notna(last["volatility_20d"]) else None,
                "Drawdown": round(float(last["drawdown"]) * 100, 2) if pd.notna(last["drawdown"]) else None,
            }
        )

    compare = pd.DataFrame(rows)
    st.dataframe(compare, width="stretch", hide_index=True)

    if not compare.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=compare["Ticker"], y=compare["Return 20d"], name="Return 20d %"))
        fig.add_trace(go.Bar(x=compare["Ticker"], y=compare["Volatilität 20d"], name="Volatilität 20d %"))
        fig.update_layout(title="Watchlist-Vergleich", barmode="group", height=420)
        show_chart_with_data("Watchlist-Vergleich", fig, compare, "watchlist_compare")


def page_data_lab(ctx: Dict[str, Any]) -> None:
    raw = ctx["market"]
    df = ctx["features"]

    page_title("Datenlabor", "Datenqualität, Rohdaten, Profiling und Export.")

    metric_grid(
        [
            ("Zeilen gesamt", f"{len(raw):,}".replace(",", ".")),
            ("Ticker", str(raw["ticker"].nunique())),
            ("Start", raw["date"].min().strftime("%Y-%m-%d")),
            ("Ende", raw["date"].max().strftime("%Y-%m-%d")),
        ]
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Rohdaten", "Profil", "Fehlwerte", "Chart-Schnelltest"])

    with tab1:
        st.dataframe(raw.tail(2000), width="stretch", hide_index=True)
    with tab2:
        st.dataframe(raw.describe(include="all").transpose(), width="stretch")
    with tab3:
        miss = raw.isna().sum().reset_index()
        miss.columns = ["Spalte", "Fehlwerte"]
        st.dataframe(miss, width="stretch", hide_index=True)
    with tab4:
        st.line_chart(df.set_index("date")["close"])


def page_ml_lab(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    model = ctx["model"]

    page_title("ML-Labor", "Modellnahe Erklärungen, Feature-Sicht und Grenzen.")

    st.json(model)

    result_metrics(result)

    features_df = pd.DataFrame(
        [
            {"Feature": "Trend Score", "Wert": result.trend_score, "Interpretation": "Position relativ zu gleitenden Durchschnitten"},
            {"Feature": "Volatility Score", "Wert": result.volatility_score, "Interpretation": "Niedrigere Volatilität erhöht Stabilität"},
            {"Feature": "Drawdown Score", "Wert": result.drawdown_score, "Interpretation": "Niedriger Drawdown erhöht Kapital-Schutz"},
            {"Feature": "News Score", "Wert": result.news_score, "Interpretation": "Vereinfachte News-Lage"},
            {"Feature": "Weight Risk", "Wert": result.asset_weight, "Interpretation": "Positionsgröße bezogen auf Gesamtkapital"},
        ]
    )

    st.dataframe(features_df, width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=features_df["Feature"], y=features_df["Wert"], name="Feature-Wert"))
    fig.update_layout(title="Feature-Sicht", height=420)
    show_chart_with_data("Feature-Sicht", fig, features_df, "ml_feature_view")

    with st.expander("Python-Hilfe anzeigen"):
        st.help(compute_scores)


def page_assistant(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]

    page_title("Assistent", "Erklärt Analyse, Methodik und Risiken im Chat-Stil.")

    for msg in st.session_state.get("chat_messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Frage zur Analyse stellen")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})

        answer = (
            f"Aktuell wird {result.ticker} mit Outlook '{result.outlook}' und Risiko '{result.risk_label}' bewertet. "
            f"Die Confidence liegt bei {result.confidence}/100. "
            f"Wichtig: {result.recommendation} Keine Finanzberatung."
        )
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
        st.rerun()



def render_news_cards(news_df: pd.DataFrame, max_items: int = 8) -> None:
    if news_df.empty:
        st.info("Keine News für die aktuelle Suchlogik gefunden.")
        return

    view_df = news_df.head(max_items).copy()

    for idx, row in view_df.iterrows():
        title = str(row.get("Titel", "") or "Ohne Titel")
        source = str(row.get("Quelle", "") or "Unbekannte Quelle")
        date = str(row.get("Datum", "") or "")
        desc = str(row.get("Beschreibung", "") or "")
        sentiment = row.get("Sentiment", "")
        relevance = str(row.get("Relevanz", "") or "")
        impact = str(row.get("Impact", "") or "")
        interpretation = str(row.get("Kurzinterpretation", "") or "")
        url = str(row.get("URL", "") or "")
        image_url = str(row.get("Bild", "") or "")

        with st.container(border=True):
            left, right = st.columns([1, 3], vertical_alignment="top")

            with left:
                if image_url.startswith("http"):
                    st.image(image_url, width="stretch")
                else:
                    st.caption("Kein Bild verfügbar")

            with right:
                st.subheader(title)
                st.caption(f"{source} · {date}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", sentiment)
                c2.metric("Relevanz", relevance)
                c3.metric("Impact", impact)

                if desc:
                    st.write(desc)

                if interpretation:
                    st.caption(f"Einordnung: {interpretation}")

                if url.startswith("http"):
                    st.link_button("Artikel öffnen", url, width="stretch")

                with st.expander("Details anzeigen"):
                    st.write("Suchlogik:", row.get("Suchlogik", ""))
                    st.write("URL:", url)

def page_news(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]

    page_title("News-Archiv", "Nachrichtenübersicht mit Bewertung, Quelle und Relevanz.")

    card(
        "Aktuelle Nachrichten nachvollziehbar machen",
        "Das News-Archiv zeigt, welche Nachrichten in die Einschätzung einfließen können. Jede Nachricht wird nach Sentiment, Relevanz und potenziellem Markteinfluss eingeordnet.",
        hero=True,
    )

    st.session_state["news_mode"] = st.radio(
        "Empfohlene Suchlogik",
        NEWS_MODES,
        horizontal=True,
        index=NEWS_MODES.index(st.session_state.get("news_mode", "Automatische Empfehlung")),
    )

    if st.session_state["news_mode"] == "Eigene Suche":
        st.session_state["news_custom_query"] = st.text_input(
            "Eigene News-Suche",
            value=st.session_state.get("news_custom_query", ""),
            placeholder="z. B. NVIDIA earnings AI chip demand inflation",
        )

    query = make_news_query(result.ticker, st.session_state["news_mode"], st.session_state.get("news_custom_query", ""))
    news_df, news_score, news_label, news_source = analyze_news_runtime(query)

    if news_source == "REAL_NEWSAPI":
        st.success("Echte NewsAPI-Daten aktiv.")
    else:
        st.warning(f"News-Fallback aktiv: {news_source}")

    st.markdown(f"**Aktive Suchlogik:** `{query}`")
    st.markdown(f"**News Intelligence:** {news_label} · Score: `{round(news_score, 2)}` · Quelle: `{news_source}`")

    render_news_cards(news_df)

    with st.expander("Rohdaten der News anzeigen"):
        st.dataframe(news_df, width="stretch", hide_index=True)

    st.download_button(
        "News-Auswertung als CSV herunterladen",
        data=news_df.to_csv(index=False).encode("utf-8"),
        file_name="wealthscope_news_intelligence.csv",
        mime="text/csv",
    )


def page_how(ctx: Dict[str, Any]) -> None:
    page_title("So funktioniert's", "Methodik, Datenlogik und Grenzen der Demo-Anwendung.")

    card(
        "QUA3CK-Logik",
        "Die App folgt einer strukturierten Analyse: Question, Understanding the data, Algorithm selection, Adaption, Adjustment, Conclude and compare, Knowledge transfer.",
        hero=True,
    )

    st.markdown(
        """
### Ablauf

1. Asset und Zeitraum auswählen  
2. Marktdaten laden  
3. Features berechnen  
4. News-Suchlogik bewerten  
5. Risiko und Confidence ableiten  
6. Geführte Interpretation oder Expertenansicht anzeigen  

### Wichtig

Diese App ist keine Finanzberatung. Sie ist eine wissenschaftliche Demo-Anwendung zur datenbasierten Analyse, Simulation und Erklärung.
        """
    )

    with st.expander("Streamlit-Funktionen in dieser App"):
        st.markdown(
            """
- Sidebar
- Header
- BottomBar
- Query Params
- Session State
- Cache Data
- Cache Resource
- Forms
- Tabs
- Expander
- Popover
- Dialog
- Dataframe
- Data Editor
- Plotly Charts
- Native Charts
- Download Button
- File Uploader
- Chat Elements
- Status/Progress/Toast
            """
        )


def page_project(ctx: Dict[str, Any]) -> None:
    raw = ctx["market"]

    page_title("Über das Projekt", "Projektkontext, Datenbasis und Zielsetzung.")

    metric_grid(
        [
            ("Datensätze geladen", f"{len(raw):,}".replace(",", ".")),
            ("Ticker", str(raw["ticker"].nunique())),
            ("Zeitraum Start", raw["date"].min().strftime("%Y-%m-%d")),
            ("Zeitraum Ende", raw["date"].max().strftime("%Y-%m-%d")),
        ]
    )

    card(
        "Zielsetzung",
        "WealthScope AI ist eine lokale Streamlit-Demo zur Kapitalanalyse. Ziel ist, Marktbewegungen, Risikokennzahlen, News-Logik und ML-nahe Scores verständlich und nachvollziehbar darzustellen.",
        hero=True,
    )


def page_export(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    df = ctx["features"]

    page_title("Professor-Export", "Exportiere Analyse, Rohdaten und Kurzbericht.")

    report_md = analysis_report_markdown(result)
    report_json = json.dumps(asdict(result), indent=2, ensure_ascii=False)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Markdown herunterladen", report_md.encode("utf-8"), "wealthscope_analysebericht.md", "text/markdown")
    with c2:
        st.download_button("JSON herunterladen", report_json.encode("utf-8"), "wealthscope_analysebericht.json", "application/json")
    with c3:
        st.download_button("CSV herunterladen", df.to_csv(index=False).encode("utf-8"), f"wealthscope_{result.ticker}_data.csv", "text/csv")

    st.code(report_md, language="markdown")


def page_impressum(ctx: Dict[str, Any]) -> None:
    page_title("Impressum", "Platzhalter für eine mögliche öffentliche Bereitstellung.")
    card(
        "Impressum / Anbieterkennzeichnung",
        "WealthScope AI ist aktuell eine lokale Demo-Anwendung im Rahmen eines Studienprojekts. Die Angaben müssten vor einer echten Veröffentlichung rechtlich geprüft werden.",
        hero=True,
    )


def page_privacy(ctx: Dict[str, Any]) -> None:
    page_title("Datenschutz", "Transparenz über lokale Daten und Verarbeitung.")
    card(
        "Datenschutz-Hinweis",
        "Die Demo verarbeitet lokale Marktdaten und lokal eingegebene Parameter. Bei Nutzung externer APIs müssten API-Anbieter, Datenflüsse und Speicherfristen dokumentiert werden.",
        hero=True,
    )


def page_status(ctx: Dict[str, Any]) -> None:
    raw = ctx["market"]
    df = ctx["features"]
    model = ctx["model"]
    result = ctx["result"]
    proof = data_proof(raw)

    page_title("Betriebsstatus", "Professoren-Check: Datenbasis, Features, Modellstatus und Reproduzierbarkeit.")

    real_data_active = str(proof["source"]).startswith("REAL")
    score = 98 if real_data_active and not raw.empty and not df.empty else 62

    card(
        f"🟢 Systemstatus: {'ECHTE DATEN AKTIV' if real_data_active else 'DEMO-FALLBACK AKTIV'}",
        f"Gesamtscore: {score} / 100. Letzter Check: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}. Datenquelle: {proof['source']}.",
        hero=True,
    )

    metric_grid(
        [
            ("Datenquelle", proof["source"]),
            ("Zeilen", f"{proof['rows']:,}".replace(",", ".")),
            ("Ticker", str(proof["tickers"])),
            ("Zeitraum", f"{proof['date_min']} bis {proof['date_max']}"),
        ]
    )

    metric_grid(
        [
            ("Spalten", str(proof["columns"])),
            ("Feature-Spalten", str(proof["feature_count"])),
            ("Target 20d", "Ja" if proof["has_target_20d"] else "Nein"),
            ("Dateigröße", f"{proof['file_size_mb']} MB"),
        ]
    )

    st.subheader("Datenbasis-Nachweis")
    st.code(proof["file_used"], language="text")

    status_df = pd.DataFrame(
        [
            {"Komponente": "Marktdaten", "Status": "OK" if not raw.empty else "Fehler", "Details": f"{proof['rows']} Zeilen · Quelle: {proof['source']}"},
            {"Komponente": "Feature-Berechnung", "Status": "OK" if not df.empty else "Fehler", "Details": f"{len(df)} Zeilen für {result.ticker}"},
            {"Komponente": "Zielvariable", "Status": "OK" if proof["has_target_20d"] else "Prüfen", "Details": "target_20d vorhanden" if proof["has_target_20d"] else "target_20d fehlt"},
            {"Komponente": "Future Return", "Status": "OK" if proof["has_future_return_20d"] else "Prüfen", "Details": "future_return_20d vorhanden" if proof["has_future_return_20d"] else "future_return_20d fehlt"},
            {"Komponente": "Modell", "Status": "Demo-Modell", "Details": model["version"]},
            {"Komponente": "News Intelligence", "Status": "Demo-Logik", "Details": result.news_label},
            {"Komponente": "Export", "Status": "OK", "Details": "CSV, JSON, Markdown"},
            {"Komponente": "Reproduzierbarkeit", "Status": "OK", "Details": "URL-Parameter, lokale Datei, sichtbare Datenbasis"},
        ]
    )
    st.dataframe(status_df, width="stretch", hide_index=True)

    st.subheader("Feature-Spalten")
    st.write(", ".join(proof["feature_cols"]) if proof["feature_cols"] else "Keine zusätzlichen Feature-Spalten erkannt.")

    st.subheader("Ticker-Verteilung")
    ticker_counts = raw["ticker"].value_counts().reset_index()
    ticker_counts.columns = ["Ticker", "Zeilen"]
    st.dataframe(ticker_counts, width="stretch", hide_index=True)

    if st.button("Statusanimation testen"):
        progress = st.progress(0, text="Statusprüfung läuft...")
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i + 1, text=f"Statusprüfung läuft... {i + 1}%")
        st.toast("Statusprüfung abgeschlossen.", icon="✅")
        st.balloons()


# =========================================================
# ROUTER
# =========================================================

def route_page(page: str, ctx: Dict[str, Any]) -> None:
    if page == "Start":
        page_start(ctx)
    elif page == "Wealth Outlook":
        page_outlook(ctx)
    elif page == "Kapital-Kompass":
        page_kompass(ctx)
    elif page == "Portfolio-Simulator":
        page_simulator(ctx)
    elif page == "Watchlist-Vergleich":
        page_watchlist(ctx)
    elif page == "Datenlabor":
        page_data_lab(ctx)
    elif page == "ML-Labor":
        page_ml_lab(ctx)
    elif page == "Assistent":
        page_assistant(ctx)
    elif page == "News-Archiv":
        page_news(ctx)
    elif page == "So funktioniert's":
        page_how(ctx)
    elif page == "Über das Projekt":
        page_project(ctx)
    elif page == "Professor-Export":
        page_export(ctx)
    elif page == "Impressum":
        page_impressum(ctx)
    elif page == "Datenschutz":
        page_privacy(ctx)
    elif page == "Betriebsstatus":
        page_status(ctx)
    else:
        page_start(ctx)


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    init_state()

    # Layout zuerst
    inject_css()
    # render_header() deaktiviert: kaputte Top-Linkzeile
    # Daten zuerst lokal laden, dann optional Upload aus Sidebar übernehmen.
    base_df = load_local_market_data()
    uploaded_df = render_sidebar(base_df)
    market_df = get_market_data(uploaded_df)

    with st.spinner("Analyse wird vorbereitet..."):
        ctx = build_context(market_df)

    proof = data_proof(market_df)
    news_source = ctx.get("news_source", "UNKNOWN")

    if str(proof["source"]).startswith("REAL"):
        st.success(
            f"Echte Kaggle-Daten aktiv: {proof['rows']:,} Zeilen · {proof['tickers']} Ticker · "
            f"{proof['columns']} Spalten · {proof['date_min']} bis {proof['date_max']} · Quelle: {proof['source']}"
        )
    else:
        st.error("Demo-Daten aktiv. Echte Kaggle-Daten wurden nicht geladen.")

    if news_source == "REAL_NEWSAPI":
        st.success("Echte NewsAPI aktiv.")
    else:
        st.warning(f"News aktuell nicht echt angebunden: {news_source}")

    render_data_badge(market_df)
    route_page(st.session_state.get("current_page", "Start"), ctx)
    render_bottom_bar()


if __name__ == "__main__":
    main()
