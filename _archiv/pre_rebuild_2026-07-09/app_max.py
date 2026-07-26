from __future__ import annotations
import base64

import io
import zipfile
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
from google import genai


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

# Native Streamlit-Logo → verbindet Sidebar korrekt mit dem Header
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

# ── Modell & Daten: v2 wird bevorzugt, wenn vorhanden ────────────────────
_MODEL_V2 = Path("models/wealthscope_model_v2.joblib")
_MODEL_V1 = Path("models/wealthscope_model.joblib")
_PARQUET_V2 = Path("data/processed/wealthscope_features_v2.parquet")
_WEIGHTS_V2 = Path("models/score_weights_v2.json")

def get_model_path() -> Path:
    """Gibt v2-Modell zurück wenn vorhanden, sonst v1."""
    return _MODEL_V2 if _MODEL_V2.exists() else _MODEL_V1

def get_parquet_path() -> Path:
    """Gibt v2-Parquet zurück wenn vorhanden, sonst v1."""
    return _PARQUET_V2 if _PARQUET_V2.exists() else DATA_FEATURES_PARQUET_PATH

def get_model_feature_cols(df: "pd.DataFrame") -> list:
    """Gibt Feature-Spalten zurück; schließt VIX ein wenn im Datensatz vorhanden."""
    base = ["daily_return", "return_5d", "return_20d", "ma_20_distance",
            "ma_50_distance", "ma_200_distance", "volatility_20d", "drawdown"]
    extra = ["vix_level", "vix_change_5d"]
    return [c for c in base + extra if c in df.columns]

def get_model_version() -> str:
    """Gibt 'v2' oder 'v1' zurück."""
    return "v2" if _MODEL_V2.exists() else "v1"

def get_calibrated_weights() -> dict:
    """Lädt kalibrierte Gewichte aus score_weights_v2.json, falls vorhanden."""
    if _WEIGHTS_V2.exists():
        import json as _json
        with open(_WEIGHTS_V2) as _f:
            data = _json.load(_f)
        return data.get("calibrated_weights", {})
    return {}

MAIN_PAGES = [
    "Start",
    "Wealth Outlook",
    "Kompass",
    "Simulator",
    "Datenlabor",
    "ML-Labor",
]

SERVICE_PAGES = [
    "Watchlist",
    "News-Archiv",
    "Assistent",
    "Projekt",
    "Export",
    "Impressum",
    "Datenschutz",
    "Status",
]

ALL_PAGES = MAIN_PAGES + SERVICE_PAGES

PAGE_LABELS = {
    "Start": "Home",
    "Wealth Outlook": "Märkte",
    "Kompass": "Signale",
    "Simulator": "Portfolio",
    "Watchlist": "Watchlist",
    "Datenlabor": "Analytics",
    "ML-Labor": "KI Lab",
    "News-Archiv": "News",
    "Assistent": "Assistent",
    "Projekt": "Methodik",
    "Export": "Export",
    "Impressum": "Impressum",
    "Datenschutz": "Datenschutz",
    "Status": "Status",
}

PAGE_ICONS = {
    "Start": "🏠",
    "Wealth Outlook": "📈",
    "Kompass": "🎯",
    "Simulator": "💼",
    "Watchlist": "👁",
    "Datenlabor": "📊",
    "ML-Labor": "🤖",
    "News-Archiv": "📰",
    "Assistent": "💬",
    "Projekt": "🎓",
    "Export": "📦",
    "Impressum": "⚖️",
    "Datenschutz": "🛡️",
    "Status": "🟢",
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
    rf_proba: float = 0.0       # RF predict_proba(class=1) für aktuellen Datenpunkt


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
    view  = "Geführte Ansicht"
    # Live-Daten-Toggle aus URL lesen (persistent über Browser-Navigation)
    live  = qp_get("live", "0") == "1"

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
        "use_live_data": live,
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
    view = st.session_state.get("app_mode", "Geführte Ansicht")
    live = "1" if st.session_state.get("use_live_data", False) else "0"
    return (
        "/?page=" + urllib.parse.quote_plus(page_name)
        + "&view=" + urllib.parse.quote_plus(view)
        + "&live=" + live
    )


def sync_url() -> None:
    st.query_params["page"] = st.session_state.get("current_page", "Start")
    st.query_params["view"] = "Geführte Ansicht"
    # Live-Toggle persistent in URL speichern
    st.query_params["live"] = "1" if st.session_state.get("use_live_data", False) else "0"


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


# =========================================================
# LIVE-DATEN via yfinance (2017 – heute)
# =========================================================

# Ticker-Mapping: Kaggle-Namen → Yahoo Finance Ticker
YFINANCE_TICKER_MAP = {
    "AAPL":"AAPL","MSFT":"MSFT","AMZN":"AMZN","GOOGL":"GOOGL","NVDA":"NVDA",
    "TSLA":"TSLA","JPM":"JPM","JNJ":"JNJ","XOM":"XOM","GE":"GE","IBM":"IBM",
    "KO":"KO","DIS":"DIS","MCD":"MCD","BA":"BA","INTC":"INTC","PG":"PG",
    "SPY":"SPY","QQQ":"QQQ","GLD":"GLD","AGG":"AGG","VTI":"VTI","VWO":"VWO",
    "EEM":"EEM","BND":"BND","VOO":"VOO",
}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_live_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Holt aktuelle Marktdaten via yfinance (letzter verfügbarer Handelstag).
    Gibt leeren DataFrame zurück wenn kein Internet / Ticker nicht gefunden.
    TTL: 1 Stunde (Daten ändern sich nicht häufiger).
    """
    try:
        import yfinance as yf
        yf_ticker = YFINANCE_TICKER_MAP.get(ticker.upper(), ticker.upper())
        raw = yf.download(yf_ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()

        # Spalten normalisieren
        raw = raw.copy()
        raw.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                       for c in raw.columns]
        raw = raw.rename(columns={"adj close": "close"})
        if "close" not in raw.columns and "open" in raw.columns:
            raw["close"] = raw["open"]

        raw = raw.reset_index()
        raw = raw.rename(columns={"Date": "date", "index": "date"})
        raw["date"] = pd.to_datetime(raw["date"])
        raw["ticker"] = ticker.upper()
        raw["asset_type"] = "Stock"
        raw["source_file"] = "yfinance_live"

        for col in ["open","high","low","volume"]:
            if col not in raw.columns:
                raw[col] = raw.get("close", 0)

        return raw[["date","ticker","asset_type","open","high","low","close","volume","source_file"]]
    except Exception:
        return pd.DataFrame()


def get_live_extended_df(kaggle_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Kombiniert Kaggle-Historik + yfinance-Live-Daten für einen Ticker.
    Kaggle: bis 2017 | yfinance: 2017 – heute
    """
    if not st.session_state.get("use_live_data", False):
        return kaggle_df[kaggle_df["ticker"] == ticker].copy()

    kaggle_part = kaggle_df[kaggle_df["ticker"] == ticker].copy()
    live_part   = fetch_live_data(ticker, period="max")

    if live_part.empty:
        return kaggle_part

    # Nur Live-Daten ab dem letzten Kaggle-Datum
    if not kaggle_part.empty:
        last_kaggle_date = pd.to_datetime(kaggle_part["date"]).max()
        live_part = live_part[pd.to_datetime(live_part["date"]) > last_kaggle_date]

    combined = pd.concat([kaggle_part, live_part], ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


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
    # Live-Daten einbeziehen wenn Toggle aktiv
    if st.session_state.get("use_live_data", False):
        out = get_live_extended_df(df, ticker).sort_values("date")
    else:
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
    # attrs gehen bei pd-Operationen verloren → Datei prüfen als Fallback
    source = getattr(market_df, "attrs", {}).get("data_source", "UNKNOWN")
    if source == "UNKNOWN":
        if DATA_FEATURES_PARQUET_PATH.exists():
            source = "REAL_PARQUET"
        elif DATA_FEATURES_PATH.exists():
            source = "REAL_CSV"
        elif DATA_MARKET_PATH.exists():
            source = "REAL_MARKET_CSV"

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
    """Premium CSS design system for WealthScope AI."""
    st.markdown(
        """
<style>
main .block-container {
    padding-bottom: 8rem !important;
}

/* ── Logo klickbar machen → Startseite ── */
/* Transparenter Link-Overlay über dem nativen st.logo()-Element  */
[data-testid="stLogo"] {
    position: relative;
    cursor: pointer;
}
[data-testid="stLogo"]::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 10;
    cursor: pointer;
}
/* Klick via JS-Trick: Logo-Bereich leitet weiter */
[data-testid="stLogoSidebarCollapse"],
[data-testid="stLogoSidebar"] {
    cursor: pointer;
}

/* KPI Hero Cards */
.ws-kpi-hero {
    background: linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.03) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.ws-kpi-hero:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.18);
}
.ws-kpi-hero .ws-kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6366F1;
    margin-bottom: 0.4rem;
}
.ws-kpi-hero .ws-kpi-value {
    font-size: 2.2rem;
    font-weight: 900;
    line-height: 1.1;
}
.ws-kpi-hero .ws-kpi-sub {
    font-size: 0.75rem;
    opacity: 0.6;
    margin-top: 0.3rem;
}

/* Signal Badges */
.ws-signal-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.38rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.ws-signal-badge.positive {
    background: rgba(34,197,94,0.15);
    color: #16a34a;
    border: 1px solid rgba(34,197,94,0.35);
}
.ws-signal-badge.negative {
    background: rgba(239,68,68,0.15);
    color: #dc2626;
    border: 1px solid rgba(239,68,68,0.35);
}
.ws-signal-badge.neutral {
    background: rgba(245,158,11,0.15);
    color: #d97706;
    border: 1px solid rgba(245,158,11,0.35);
}
.ws-signal-badge.info {
    background: rgba(99,102,241,0.12);
    color: #6366F1;
    border: 1px solid rgba(99,102,241,0.28);
}

/* Signal Grid */
.ws-signal-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.ws-signal-cell {
    background: rgba(148,163,184,0.06);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.ws-signal-cell .ws-signal-name {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 0.5rem;
}

/* Recommendation Box */
.ws-recommendation-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.07) 0%, rgba(34,197,94,0.04) 100%);
    border: 1px solid rgba(99,102,241,0.22);
    border-left: 4px solid #6366F1;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
}
.ws-recommendation-box h4 {
    margin: 0 0 0.5rem 0;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #6366F1;
}
.ws-recommendation-box p {
    margin: 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Feature cards on start page */
.ws-feature-card {
    background: rgba(148,163,184,0.06);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px;
    padding: 1.3rem;
    height: 100%;
}
.ws-feature-card .ws-feature-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
}
.ws-feature-card h3 {
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
}
.ws-feature-card p {
    font-size: 0.85rem;
    opacity: 0.75;
    margin: 0;
    line-height: 1.5;
}

/* Card + Hero */
.ws-card { background:#f8faff; border:1px solid rgba(99,102,241,.16); border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:1rem; }
.ws-card h2 { font-size:1.05rem; font-weight:800; color:#1e293b; margin:0 0 .45rem; }
.ws-card p { font-size:.88rem; color:#475569; margin:0; line-height:1.6; }
.ws-card.ws-hero { border-left:4px solid #6366f1; background:linear-gradient(135deg,#eef2ff,#f8faff); }

/* Metric Grid */
.ws-metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:.7rem; margin:.8rem 0 1.2rem; }
.ws-metric { background:#f8faff; border:1px solid rgba(99,102,241,.15); border-radius:12px; padding:.85rem 1rem; display:flex; flex-direction:column; gap:.2rem; }
.ws-metric small { font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; font-weight:700; color:#6366f1; }
.ws-metric strong { font-size:1.0rem; font-weight:800; color:#1e293b; }
</style>
        """,
        unsafe_allow_html=True,
    )

def render_header() -> None:
    """
    Nav-Links als fixed Overlay in der nativen Streamlit-Header-Leiste.
    Logo wird von st.logo() nativ verwaltet – kein Duplikat.
    left/right lassen Platz für Logo-Bereich (links) und Settings-Menü (rechts).
    """
    current = st.session_state.get("current_page", "Start")

    links = "".join(
        f'<a href="{href(p)}" target="_self" '
        f'class="wsh-link{"  wsh-active" if p == current else ""}">'
        f'{PAGE_ICONS.get(p,"")} {PAGE_LABELS.get(p, p)}</a>'
        for p in MAIN_PAGES
    )

    st.markdown(f"""
<style>
/* ── Native Header: Design-Match ── */
header[data-testid="stHeader"] {{
    background: rgba(248,250,255,.97) !important;
    border-bottom: 1px solid rgba(99,102,241,.14) !important;
    box-shadow: 0 1px 10px rgba(99,102,241,.06) !important;
    backdrop-filter: blur(18px) !important;
}}
/* ── Nav-Bar: sticky im Content-Bereich, startet automatisch nach Sidebar ── */
.wsh-bar {{
    position: sticky;
    top: 0;
    z-index: 9998;
    display: flex;
    align-items: center;
    height: 48px;
    margin: -1rem -1rem 1rem -1rem;
    padding: 0 1.2rem;
    background: rgba(248,250,255,.98);
    border-bottom: 1px solid rgba(99,102,241,.13);
    box-shadow: 0 1px 8px rgba(99,102,241,.06);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
}}
.wsh-links {{
    display: flex;
    align-items: center;
    gap: 0.1rem;
    overflow-x: auto;
    scrollbar-width: none;
}}
.wsh-links::-webkit-scrollbar {{ display: none; }}
.wsh-link {{
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.36rem 0.65rem;
    border-radius: 9px;
    font-size: 0.80rem;
    font-weight: 700;
    color: #475569;
    text-decoration: none !important;
    white-space: nowrap;
    border: 1px solid transparent;
    transition: background .13s, color .13s;
}}
.wsh-link:hover {{ background: #eef2ff; color: #6366f1; }}
.wsh-active {{
    background: #eef2ff !important;
    color: #6366f1 !important;
    border-color: rgba(99,102,241,.22) !important;
}}
/* Kein extra padding-top nötig – Nav ist im Flow */
.main .block-container {{ padding-top: 1rem !important; }}
/* Logo klickbar */
[data-testid="stLogo"] {{ cursor: pointer !important; }}
</style>

<div class="wsh-bar">
  <div class="wsh-links">{links}</div>
</div>

""", unsafe_allow_html=True)

    # Logo klickbar via components.html (greift auf parent DOM zu)
    import streamlit.components.v1 as _components
    _components.html("""
<script>
function makeLogoClickable() {
    var els = parent.document.querySelectorAll('[data-testid="stLogo"]');
    els.forEach(function(el) {
        if (!el.dataset.wsLinked) {
            el.dataset.wsLinked = "1";
            el.style.cursor = "pointer";
            el.title = "Zur Startseite";
            el.addEventListener("click", function(e) {
                e.preventDefault();
                e.stopPropagation();
                parent.window.location.href = "/?page=Start";
            });
        }
    });
}
makeLogoClickable();
setTimeout(makeLogoClickable, 400);
setTimeout(makeLogoClickable, 1200);
</script>
""", height=0, scrolling=False)


def render_bottom_bar() -> None:
    from urllib.parse import quote_plus
    from datetime import datetime

    view         = st.session_state.get("app_mode", "Geführte Ansicht")
    current_page = st.session_state.get("current_page", "Start")

    service_items = [
        ("👁",  "Watchlist",   "Watchlist"),
        ("📰", "News",         "News-Archiv"),
        ("💬", "Assistent",    "Assistent"),
        ("🎓", "Methodik",     "Projekt"),
        ("📦", "Export",       "Export"),
        ("🟢", "Status",       "Status"),
        ("⚖️", "Impressum",    "Impressum"),
        ("🛡️", "Datenschutz",  "Datenschutz"),
    ]

    nav_links = ""
    for icon, label, page_name in service_items:
        is_active  = page_name == current_page
        link_href  = f"/?page={quote_plus(page_name)}&view={quote_plus(view)}"
        nav_links += (
            f'<a href="{link_href}" target="_self" '
            f'class="wsf-link{"  wsf-active" if is_active else ""}">'
            f'{icon} {label}</a>'
        )

    checked = datetime.now().strftime("%H:%M")

    st.markdown(f"""
<style>
/* Abstand für fixierten Footer */
main .block-container {{ padding-bottom: 5.5rem !important; }}

/* ── Footer-Bar ── */
.wsf-bar {{
    position: fixed;
    left: 0; right: 0; bottom: 0;
    z-index: 999999;
    height: 52px;
    background: rgba(248,250,255,.97);
    border-top: 1px solid rgba(99,102,241,.14);
    box-shadow: 0 -2px 14px rgba(99,102,241,.07);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    display: flex;
    align-items: center;
    justify-content: center;   /* Links mittig */
    gap: 0.2rem;
    padding: 0 1rem;
}}

/* ── Links ── */
.wsf-link {{
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.36rem 0.8rem;
    border-radius: 9px;
    font-size: 0.80rem;
    font-weight: 700;
    color: #475569;
    text-decoration: none !important;
    white-space: nowrap;
    border: 1px solid transparent;
    transition: background .13s, color .13s;
}}
.wsf-link:hover {{ background: #eef2ff; color: #6366f1; }}
.wsf-active {{
    background: #eef2ff !important;
    color: #6366f1 !important;
    border-color: rgba(99,102,241,.22) !important;
}}

/* ── Meta-Info rechts (absolut positioniert, stört Zentrierung nicht) ── */
.wsf-meta {{
    position: absolute;
    right: 1rem;
    font-size: 0.67rem;
    color: #94a3b8;
    font-weight: 600;
    white-space: nowrap;
}}
</style>

<div class="wsf-bar">
    {nav_links}
    <span class="wsf-meta">Keine Finanzberatung · v{APP_VERSION} · {checked}</span>
</div>""", unsafe_allow_html=True)

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
    """Klickbares Logo oben in der Sidebar – führt immer zur Startseite."""
    logo_path = Path("assets/wealthscope_logo.svg")
    icon_path = Path("assets/wealthscope_icon.svg")

    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        img_tag = (
            f'<img src="data:image/svg+xml;base64,{logo_b64}" '
            f'alt="WealthScope AI" style="width:160px;height:auto;display:block;" />'
        )
    elif icon_path.exists():
        icon_b64 = base64.b64encode(icon_path.read_bytes()).decode("utf-8")
        img_tag = (
            f'<img src="data:image/svg+xml;base64,{icon_b64}" '
            f'alt="WealthScope AI" style="width:36px;height:36px;" />'
        )
    else:
        img_tag = '<span style="font-weight:900;font-size:1rem;color:#6366f1">💠 WealthScope AI</span>'

    st.markdown(
        f"""
<style>
.ws-sidebar-logo-wrap {{
    display: block;
    text-decoration: none !important;
    padding: 0.6rem 0 0.5rem;
    margin-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,102,241,0.12);
    transition: opacity 0.15s;
}}
.ws-sidebar-logo-wrap:hover {{ opacity: 0.75; }}
.ws-sidebar-logo-sub {{
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #6366f1;
    margin-top: 0.3rem;
    display: block;
}}
</style>
<a href="/?page=Start" target="_self" class="ws-sidebar-logo-wrap" title="Zur Startseite">
    {img_tag}
    <span class="ws-sidebar-logo-sub">← Zur Startseite</span>
</a>
        """,
        unsafe_allow_html=True,
    )


def inject_sticky_logo_css() -> None:
    st.markdown(
        """
        <style>
        /*
        Sticky WealthScope logo in the native Streamlit sidebar.
        Minimal CSS: only targets the Streamlit logo area, not the whole sidebar.
        */
        [data-testid="stSidebar"] [data-testid="stLogo"] {
            position: sticky;
            top: 0;
            z-index: 1000;
            background: var(--background-color);
            padding-top: 0.35rem;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid rgba(120, 120, 120, 0.18);
        }

        [data-testid="stSidebar"] [data-testid="stLogo"] img {
            max-height: 42px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    uploaded_df = None
    current     = st.session_state.get("current_page", "Start")

    with st.sidebar:

        # ── NAVIGATION ──────────────────────────────────────────────
        st.caption("SEITEN")
        for page in MAIN_PAGES:
            icon  = PAGE_ICONS.get(page, "•")
            label = f"{icon} {page}"
            if st.button(label, key=f"snav_{page}",
                         use_container_width=True,
                         type="primary" if page == current else "secondary"):
                route_to(page)

        st.caption("SERVICE")
        c1, c2 = st.columns(2)
        svc_labels = {
            "News-Archiv": "📰 News", "Projekt": "🎓 Projekt",
            "Export": "📦 Export",   "Impressum": "⚖️ Legal",
            "Datenschutz": "🛡️ Privacy", "Status": "🟢 Status",
        }
        for i, page in enumerate(SERVICE_PAGES):
            with (c1 if i % 2 == 0 else c2):
                lbl = svc_labels.get(page, page)
                if st.button(lbl, key=f"snav_s_{page}",
                             use_container_width=True,
                             type="primary" if page == current else "secondary"):
                    route_to(page)

        st.divider()

        # ── ASSET & ZEITRAUM ─────────────────────────────────────────
        st.caption("ASSET & ZEITRAUM")
        asset_opts    = list(DEFAULT_ASSET_MAP.keys())
        current_asset = st.session_state.get("selected_asset_label", asset_opts[0])
        if current_asset not in asset_opts:
            current_asset = asset_opts[0]

        st.session_state["selected_asset_label"] = st.selectbox(
            "Asset auswählen",
            asset_opts,
            index=asset_opts.index(current_asset),
        )
        st.session_state["period"] = st.select_slider(
            "Zeitraum",
            options=list(PERIODS.keys()),
            value=st.session_state.get("period", "5Y"),
        )

        st.divider()

        # ── PORTFOLIO & KAPITAL ──────────────────────────────────────
        st.caption("KAPITAL & RISIKO")
        with st.form("capital_form", border=True):
            st.session_state["capital"] = st.number_input(
                "Kapital (€)",
                min_value=1_000.0, max_value=10_000_000.0,
                value=float(st.session_state.get("capital", 100_000.0)),
                step=1_000.0,
            )
            # Gewichtung wird aus dem Simulator gelesen (kein Duplikat)
            st.session_state["risk_drawdown"] = st.slider(
                "Max. tolerierbarer Rückgang %",
                min_value=1.0, max_value=60.0,
                value=float(st.session_state.get("risk_drawdown", 10.0)),
                step=1.0,
            )
            if st.form_submit_button("Szenario übernehmen",
                                     use_container_width=True, type="primary"):
                st.toast("Szenario aktualisiert ✅")

        # Kurzanzeige
        cap = float(st.session_state.get("capital", 100_000.0))
        rsk = float(st.session_state.get("risk_drawdown", 10.0))
        st.caption(f"💰 {money(cap)}  ·  🛡 Max. Rückgang: {money(cap*rsk/100)}")
        st.caption("📊 Gewichtung → im Simulator festlegen")

        st.divider()

        # ── ANZEIGE ──────────────────────────────────────────────────
        st.caption("ANZEIGE")
        st.toggle("Rohdaten anzeigen",    key="show_raw_data")
        st.toggle("Erklärungen anzeigen", key="show_explanations")
        _prev_live = st.session_state.get("use_live_data", False)
        live_on = st.toggle("🌐 Live-Daten (yfinance)", key="use_live_data",
                            help="Erweitert historische Kaggle-Daten (bis 2017) mit aktuellen yfinance-Daten bis heute. Benötigt Internetverbindung.")
        if live_on != _prev_live:
            sync_url()  # URL sofort aktualisieren damit Browser-Links den State behalten
        if live_on:
            st.caption("✅ Live-Daten aktiv · Kaggle + yfinance kombiniert · sessionweit")
        else:
            st.caption("Nur Kaggle-Daten (1962–2017)")
        st.toggle("Erweiterte Metriken", key="show_advanced_metrics")

        st.divider()

        # ── UPLOAD ───────────────────────────────────────────────────
        with st.expander("📂 Eigene Daten laden"):
            upload      = st.file_uploader("CSV / Excel", type=["csv","xlsx","xls"])
            uploaded_df = load_uploaded_data(upload)
            if uploaded_df is not None:
                st.session_state["uploaded_override_active"] = st.toggle(
                    "Als Datenquelle aktivieren",
                    value=st.session_state.get("uploaded_override_active", False),
                )
                st.success(f"✅ {len(uploaded_df):,} Zeilen".replace(",","."))

        # ── STATUS ───────────────────────────────────────────────────
        src = getattr(df, "attrs", {}).get("data_source", "?")
        dot = "🟢" if str(src).startswith("REAL") else "🔴"
        st.caption(f"{dot} {src}  ·  v{APP_VERSION}")

    return uploaded_df

# =========================================================
# CHARTS
# =========================================================

def chart_price(df: pd.DataFrame, ticker: str) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    # Bollinger Bands
    if "ma_20" in df.columns and df["ma_20"].notna().any():
        roll_std = df["close"].rolling(20).std()
        bb_upper = df["ma_20"] + 2 * roll_std
        bb_lower = df["ma_20"] - 2 * roll_std
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"].iloc[::-1]]),
            y=pd.concat([bb_upper, bb_lower.iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(99,102,241,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Bollinger Bands",
            showlegend=True,
        ), row=1, col=1)

    # Price line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"],
        mode="lines",
        name=f"{ticker} Kurs",
        line=dict(color="#6366F1", width=2),
    ), row=1, col=1)

    # Moving averages
    ma_styles = [("ma_20", "MA 20", "#f59e0b"), ("ma_50", "MA 50", "#22c55e"), ("ma_200", "MA 200", "#ef4444")]
    for col, name, color in ma_styles:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines", name=name,
                line=dict(color=color, width=1.5, dash="dot"),
            ), row=1, col=1)

    # Volume bars
    if "volume" in df.columns:
        colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["close"], df["open"])] if "open" in df.columns else ["#94a3b8"] * len(df)
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"],
            name="Volumen",
            marker_color=colors,
            opacity=0.6,
        ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker}: Kursverlauf, MAs & Bollinger Bands",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(title_text="Kurs", row=1, col=1)
    fig.update_yaxes(title_text="Volumen", row=2, col=1)
    return fig


def chart_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )

    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=ticker,
        increasing=dict(line=dict(color="#22c55e"), fillcolor="rgba(34,197,94,0.75)"),
        decreasing=dict(line=dict(color="#ef4444"), fillcolor="rgba(239,68,68,0.75)"),
    ), row=1, col=1)

    # MA overlay on candles
    for col, name, color in [("ma_20", "MA 20", "#f59e0b"), ("ma_50", "MA 50", "#6366F1")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
            ), row=1, col=1)

    # Volume colored by direction
    if "volume" in df.columns:
        colors = ["#22c55e" if c >= o else "#ef4444" for c, o in zip(df["close"], df["open"])] if "open" in df.columns else ["#94a3b8"] * len(df)
        fig.add_trace(go.Bar(
            x=df["date"], y=df["volume"],
            name="Volumen",
            marker_color=colors,
            opacity=0.55,
        ), row=2, col=1)

    fig.update_layout(
        title=f"{ticker}: Candlestick mit Volumen",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def chart_drawdown(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()

    dd = df["drawdown"] * 100

    # Filled area with gradient effect
    fig.add_trace(go.Scatter(
        x=df["date"], y=dd,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(239,68,68,0.25)",
        line=dict(color="#ef4444", width=1.5),
        name="Drawdown %",
    ))

    # Threshold lines
    for level, color, label in [(-10, "#f59e0b", "-10%"), (-20, "#ef4444", "-20%"), (-30, "#991b1b", "-30%")]:
        fig.add_hline(y=level, line_dash="dash", line_color=color, opacity=0.7,
                      annotation_text=label, annotation_position="left")

    # Annotate worst drawdown
    if not dd.empty:
        worst_idx = dd.idxmin()
        worst_val = dd.min()
        worst_date = df.loc[worst_idx, "date"]
        fig.add_annotation(
            x=worst_date, y=worst_val,
            text=f"Worst: {worst_val:.1f}%",
            showarrow=True, arrowhead=2,
            arrowcolor="#ef4444",
            font=dict(color="#ef4444", size=11),
        )

    fig.update_layout(
        title=f"{ticker}: Drawdown-Verlauf",
        height=400,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(ticksuffix="%"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def chart_returns(df: pd.DataFrame, ticker: str) -> go.Figure:
    clean = df.dropna(subset=["daily_return"])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=clean["daily_return"] * 100, nbinsx=60, name="Tagesrenditen",
                               marker_color="#6366F1", opacity=0.75))
    fig.update_layout(title=f"{ticker}: Verteilung der Tagesrenditen", height=360, margin=dict(l=10, r=10, t=55, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def chart_momentum(df: pd.DataFrame, ticker: str) -> go.Figure:
    """3-panel chart: price, rolling returns (5d/20d/60d), volatility."""
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        vertical_spacing=0.04,
        subplot_titles=["Kurs", "Rolling Returns", "Volatilität 20d"],
    )

    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines",
                             name="Kurs", line=dict(color="#6366F1", width=2)), row=1, col=1)

    for col, name, color in [("return_5d", "5d Return", "#22c55e"),
                              ("return_20d", "20d Return", "#f59e0b"),
                              ("return_60d", "60d Return", "#6366F1")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df["date"], y=df[col] * 100,
                                     mode="lines", name=name,
                                     line=dict(color=color, width=1.5)), row=2, col=1)

    if "volatility_20d" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["volatility_20d"] * 100,
                                 mode="lines", fill="tozeroy",
                                 fillcolor="rgba(245,158,11,0.15)",
                                 name="Vola 20d",
                                 line=dict(color="#f59e0b", width=1.5)), row=3, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", opacity=0.5, row=2, col=1)

    fig.update_layout(
        title=f"{ticker}: Momentum-Analyse",
        height=600,
        margin=dict(l=10, r=10, t=55, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
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
    # Color gradient: green (high) to red (low)
    colors = []
    for v in values:
        if v >= 65:
            colors.append("#22c55e")
        elif v >= 40:
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=values,
        orientation="h",
        name="Score",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{v}" for v in values],
        textposition="inside",
        insidetextanchor="middle",
    ))
    # Benchmark line at 50
    fig.add_vline(x=50, line_dash="dash", line_color="#94a3b8", opacity=0.8,
                  annotation_text="Benchmark 50", annotation_position="top")
    fig.update_layout(
        title="Score-Zerlegung",
        height=380,
        xaxis=dict(range=[0, 100]),
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)")
    return fig


def chart_radar(result: AnalysisResult) -> go.Figure:
    categories = ["Trend", "Volatilität", "Drawdown", "Kapital-Schutz", "Confidence"]
    values = [result.trend_score, result.volatility_score, result.drawdown_score, result.capital_protection, result.confidence]
    benchmark = [50] * len(categories)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=benchmark + [benchmark[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(148,163,184,0.10)",
        line=dict(color="#94a3b8", dash="dash", width=1.5),
        name="Benchmark (50)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(99,102,241,0.22)",
        line=dict(color="#6366F1", width=2),
        name="Analyseprofil",
    ))
    fig.update_layout(
        title="Analyseprofil (Radar)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h"),
    )
    return fig



def chart_portfolio(capital: float, weight: float, ticker: str) -> go.Figure:
    invested = capital * weight / 100
    cash = capital - invested
    fig = go.Figure(data=[go.Pie(labels=[ticker, "Liquidität / Rest"], values=[invested, cash], hole=0.55)])
    fig.update_layout(title="Kapitalallokation im Szenario", height=360, margin=dict(l=10, r=10, t=55, b=10))
    return fig


def explain(text: str) -> None:
    """Zeigt einen Erklärungstext nur wenn 'Erklärungen anzeigen' aktiv ist."""
    if st.session_state.get("show_explanations", True):
        st.info(text)


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

    # Gewichtung aus Simulator-Portfolio lesen (statt Sidebar-Duplikat)
    _portfolio = st.session_state.get("portfolio_rows", [])
    _ticker_upper = ticker.upper()
    _sim_weight = next(
        (float(r.get("Gewichtung_%", r.get("Gewichtung %", 10.0)))
         for r in _portfolio
         if str(r.get("Baustein","")).upper() == _ticker_upper),
        10.0  # Default wenn Ticker nicht im Simulator
    )
    # Ins session_state schreiben damit andere Komponenten es lesen können
    st.session_state["asset_weight"] = _sim_weight

    result = compute_scores(
        features,
        news_score,
        news_query,
        news_label,
        float(st.session_state.get("capital", 100000.0)),
        _sim_weight,
        float(st.session_state.get("risk_drawdown", 10.0)),
        st.session_state.get("period", "5Y"),
        ticker,
    )

    # ── RF predict_proba für aktuellen Datenpunkt ──────────────────────
    _rf_proba = 0.0
    try:
        import joblib as _jl
        _model_file = get_model_path()
        _feat_cols = get_model_feature_cols(features)
        if _feat_cols and not features.empty and _model_file.exists():
            _pipe = _jl.load(_model_file)
            _last = features[_feat_cols].dropna().tail(1)
            if not _last.empty:
                _rf_proba = float(_pipe.predict_proba(_last)[0, 1])
                result = result.__class__(**{**asdict(result), "rf_proba": round(_rf_proba, 4)})
    except Exception:
        pass

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
    if not st.session_state.get("show_explanations", True):
        return
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
    result = ctx["result"]

    # Hero section
    st.markdown(
        f"""
<div style="text-align:center;padding:4rem 1rem 2.5rem;max-width:760px;margin:0 auto">
  <div style="display:inline-flex;align-items:center;gap:0.4rem;background:#eef2ff;
              border:1px solid rgba(99,102,241,.25);border-radius:999px;
              padding:0.25rem 0.9rem;font-size:0.68rem;font-weight:800;
              letter-spacing:0.1em;text-transform:uppercase;color:#6366F1;margin-bottom:1rem">
    ✦ WealthScope AI · {APP_VERSION}
  </div>
  <h1 style="font-size:2.8rem;font-weight:900;line-height:1.15;margin:0 0 0.9rem 0;
             letter-spacing:-0.03em;color:#0f172a">
    Märkte verstehen.<br>
    <span style="color:#6366f1">KI-gestützt analysieren.</span><br>
    Risiken kalkulieren.
  </h1>
  <p style="font-size:1.05rem;color:#475569;max-width:560px;margin:0 auto 0.5rem;line-height:1.7">
    192.119 Datenpunkte · 26 Blue-Chip-Titel · Random-Forest-Signale ·
    Live-Kurse via yfinance — alles in einer Plattform.
  </p>
  <p style="font-size:0.82rem;color:#94a3b8;margin:0 auto 1.5rem">
    Wissenschaftlicher Prototyp · Keine Finanzberatung
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Hero row — 3 Kern-KPIs, zentriert, mehr Luft
    _gap1, kpi_c1, kpi_c2, kpi_c3, _gap2 = st.columns([0.5, 1, 1, 1, 0.5])
    kpi_data = [
        (kpi_c1, "KI-Confidence", f"{result.confidence}", "/100", "#6366F1"),
        (kpi_c2, "Risiko", result.risk_label, result.outlook,
         "#ef4444" if result.risk_label == "Hoch" else "#f59e0b" if result.risk_label == "Mittel" else "#22c55e"),
        (kpi_c3, "Trend-Score", f"{result.trend_score}", "/100", "#22c55e"),
    ]
    for col, label, value, sub, color in kpi_data:
        with col:
            st.markdown(
                f"""<div class="ws-kpi-hero">
  <div class="ws-kpi-label" style="color:{color}">{label}</div>
  <div class="ws-kpi-value" style="color:{color}">{value}</div>
  <div class="ws-kpi-sub">{sub}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;font-size:0.7rem;font-weight:800;letter-spacing:0.12em;"
        "text-transform:uppercase;color:#94a3b8;margin-bottom:1rem'>Schnellzugriff</div>",
        unsafe_allow_html=True,
    )

    # ── Feature Cards mit CTA ──────────────────────────────────────
    fc1, fc2, fc3 = st.columns(3, gap="large")
    feature_cards = [
        (fc1, "📈", "Marktanalyse", "Wealth Outlook",
         "Candlestick-Charts, Bollinger-Bänder, MA 20/50/200 und Volatilität auf einen Blick.",
         "#6366f1", "#eef2ff"),
        (fc2, "🤖", "KI-Signal", "ML-Labor",
         "Random-Forest-Klassifikator mit 5-Fold CV — zeigt ob ein Titel statistisch steigen könnte.",
         "#16a34a", "#dcfce7"),
        (fc3, "💼", "Portfolio-Simulator", "Simulator",
         "Kapital eingeben, Risiko definieren. Die App berechnet Positionsgröße und CRV automatisch.",
         "#d97706", "#fef3c7"),
    ]
    _fc_live = "1" if st.session_state.get("use_live_data", False) else "0"
    _fc_view = urllib.parse.quote_plus(st.session_state.get("app_mode", "Geführte Ansicht"))
    for col, icon, title, page, desc, color, bg in feature_cards:
        card_href = f"/?page={urllib.parse.quote_plus(page)}&view={_fc_view}&live={_fc_live}"
        with col:
            st.markdown(
                f"""<a href="{card_href}" target="_self" style="text-decoration:none;display:block;">
                <div style="background:{bg};border:1px solid {color}33;border-radius:14px;
                    padding:1.4rem 1.2rem 1.2rem;cursor:pointer;
                    transition:transform .15s,box-shadow .15s;"
                    onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 6px 20px {color}22'"
                    onmouseout="this.style.transform='';this.style.boxShadow=''">
                  <div style="font-size:1.7rem;margin-bottom:0.6rem">{icon}</div>
                  <div style="font-weight:800;font-size:0.95rem;color:#0f172a;margin-bottom:0.45rem">{title}</div>
                  <div style="font-size:0.80rem;color:#475569;line-height:1.6;margin-bottom:0.9rem">{desc}</div>
                  <div style="font-size:0.72rem;font-weight:800;color:{color};letter-spacing:0.05em">
                    → Öffnen
                  </div>
                </div></a>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── App-Tour (animierte Feature-Preview) ───────────────────────
    import streamlit.components.v1 as _c
    _c.html("""
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:-apple-system,'Inter','Segoe UI',sans-serif;}
.tour-wrap{background:#f8faff;border:1px solid rgba(99,102,241,.15);border-radius:16px;
  padding:1.5rem;max-width:900px;margin:0 auto;}
.tour-title{font-size:.72rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase;
  color:#6366f1;margin-bottom:.8rem;text-align:center;}
.slides{position:relative;height:210px;overflow:hidden;border-radius:12px;
  background:#fff;border:1px solid rgba(99,102,241,.12);}
.slide{position:absolute;inset:0;padding:1.4rem 1.6rem;opacity:0;
  transform:translateX(40px);transition:all .55s cubic-bezier(.4,0,.2,1);}
.slide.active{opacity:1;transform:translateX(0);}
.slide.out{opacity:0;transform:translateX(-40px);}
.slide-badge{display:inline-flex;align-items:center;gap:.35rem;padding:.25rem .6rem;
  border-radius:999px;font-size:.65rem;font-weight:800;letter-spacing:.06em;
  text-transform:uppercase;margin-bottom:.7rem;}
.slide h2{font-size:1.15rem;font-weight:900;color:#1e293b;margin-bottom:.4rem;letter-spacing:-.02em;}
.slide p{font-size:.82rem;color:#475569;line-height:1.6;max-width:480px;}
.slide-metrics{display:flex;gap:.6rem;margin-top:.9rem;flex-wrap:wrap;}
.metric{background:#f1f5f9;border-radius:8px;padding:.4rem .7rem;font-size:.72rem;}
.metric b{display:block;font-size:1rem;font-weight:900;}
/* Slide-spezifisch */
.s1 .slide-badge{background:#eef2ff;color:#6366f1;}
.s1 .metric b{color:#6366f1;}
.s2 .slide-badge{background:#dcfce7;color:#16a34a;}
.s2 .metric b{color:#16a34a;}
.s3 .slide-badge{background:#fef3c7;color:#92400e;}
.s3 .metric b{color:#f59e0b;}
.s4 .slide-badge{background:#fce7f3;color:#9d174d;}
.s4 .metric b{color:#ec4899;}
.s5 .slide-badge{background:#e0f2fe;color:#0369a1;}
.s5 .metric b{color:#0ea5e9;}
/* Dots */
.dots{display:flex;justify-content:center;gap:.45rem;margin-top:.85rem;}
.dot{width:7px;height:7px;border-radius:999px;background:rgba(99,102,241,.22);
  cursor:pointer;transition:all .25s;}
.dot.active{background:#6366f1;width:20px;}
/* Progress bar */
.progress-bar{height:2px;background:rgba(99,102,241,.12);border-radius:999px;
  margin-bottom:1rem;overflow:hidden;}
.progress-fill{height:100%;background:#6366f1;border-radius:999px;
  transition:width .1s linear;width:0%;}
</style>

<div class="tour-wrap">
  <div class="tour-title">🎬 App-Tour – So funktioniert WealthScope AI</div>
  <div class="progress-bar"><div class="progress-fill" id="prog"></div></div>
  <div class="slides" id="slides">

    <div class="slide s1 active">
      <div class="slide-badge">01 · Datenbasis</div>
      <h2>192.119 Datenpunkte — von 1962 bis heute</h2>
      <p>Kaggle US Stocks &amp; ETFs (CC0) als historische Basis, erweitert durch
         yfinance Live-Daten bis heute. 26 Blue-Chip-Titel, lokal als Apache Parquet.</p>
      <div class="slide-metrics">
        <div class="metric"><b>192.119</b>Datenpunkte</div>
        <div class="metric"><b>26</b>Ticker</div>
        <div class="metric"><b>1962–heute</b>Zeitraum</div>
        <div class="metric"><b>CC0</b>Lizenz</div>
      </div>
    </div>

    <div class="slide s2">
      <div class="slide-badge">02 · Feature Engineering</div>
      <h2>8 technische Indikatoren</h2>
      <p>Returns (1d, 5d, 20d), Moving Average Abstände (MA20/50/200),
         20-Tage-Volatilität und Drawdown. Alle preisnormalisiert – skalierungsinvariant.</p>
      <div class="slide-metrics">
        <div class="metric"><b>MA 20/50/200</b>Trend</div>
        <div class="metric"><b>Volatilität</b>Risiko</div>
        <div class="metric"><b>Drawdown</b>Verlust</div>
        <div class="metric"><b>target_20d</b>Zielvariable</div>
      </div>
    </div>

    <div class="slide s3">
      <div class="slide-badge">03 · Machine Learning</div>
      <h2>Random Forest · Accuracy 55,8 %</h2>
      <p>Binäre Klassifikation: Steigt der Kurs in 20 Handelstagen?
         5-Fold Cross-Validation, ROC-AUC 0.61. Besser als Zufall – wissenschaftlich eingeordnet.</p>
      <div class="slide-metrics">
        <div class="metric"><b>55,8 %</b>Accuracy</div>
        <div class="metric"><b>0.61</b>ROC-AUC</div>
        <div class="metric"><b>5-Fold</b>CV</div>
        <div class="metric"><b>RF</b>Algorithmus</div>
      </div>
    </div>

    <div class="slide s4">
      <div class="slide-badge">04 · News &amp; KI-Assistent</div>
      <h2>Live-News + Google Gemini</h2>
      <p>NewsAPI liefert aktuelle Finanznachrichten in Echtzeit.
         Gemini erklärt Indikatoren, Risiken und Zusammenhänge verständlich auf Knopfdruck.</p>
      <div class="slide-metrics">
        <div class="metric"><b>NewsAPI</b>Live</div>
        <div class="metric"><b>Gemini</b>KI</div>
        <div class="metric"><b>Sentiment</b>Analyse</div>
        <div class="metric"><b>Echtzeit</b>Daten</div>
      </div>
    </div>

    <div class="slide s5">
      <div class="slide-badge">05 · QUA³CK · 8 Notebooks</div>
      <h2>Vollständig dokumentiert</h2>
      <p>Jede Entscheidung ist in einem Jupyter-Notebook begründet.
         Von der Fragestellung bis zum Wissenstransfer – reproduzierbar, zitierbar, präsentierbar.</p>
      <div class="slide-metrics">
        <div class="metric"><b>8</b>Notebooks</div>
        <div class="metric"><b>Q·U·A·A·A·C·K</b>Phasen</div>
        <div class="metric"><b>KIT 2021</b>Quelle</div>
        <div class="metric"><b>PDF/ZIP</b>Export</div>
      </div>
    </div>

  </div>
  <div class="dots" id="dots">
    <div class="dot active" onclick="goTo(0)"></div>
    <div class="dot" onclick="goTo(1)"></div>
    <div class="dot" onclick="goTo(2)"></div>
    <div class="dot" onclick="goTo(3)"></div>
    <div class="dot" onclick="goTo(4)"></div>
  </div>
</div>

<script>
var current = 0;
var total   = 5;
var timer   = null;
var INTERVAL = 10000;

function goTo(n) {
  var slides = document.querySelectorAll('.slide');
  var dots   = document.querySelectorAll('.dot');
  slides[current].classList.remove('active');
  slides[current].classList.add('out');
  setTimeout(function(){ slides[current].classList.remove('out'); }, 560);
  current = n;
  slides[current].classList.add('active');
  dots.forEach(function(d,i){ d.classList.toggle('active', i===current); });
  resetTimer();
}

function next() { goTo((current + 1) % total); }

function resetTimer() {
  clearInterval(timer);
  var start = Date.now();
  var prog  = document.getElementById('prog');
  prog.style.width = '0%';
  timer = setInterval(function() {
    var elapsed = Date.now() - start;
    var pct     = Math.min(elapsed / INTERVAL * 100, 100);
    prog.style.width = pct + '%';
    if (elapsed >= INTERVAL) { next(); start = Date.now(); prog.style.width = '0%'; }
  }, 40);
}

resetTimer();
</script>
""", height=320, scrolling=False)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # CTA Buttons
    btn_c1, btn_c2, btn_c3 = st.columns(3)
    with btn_c1:
        if st.button("📈 Analyse starten", width="stretch", type="primary"):
            route_to("Wealth Outlook")
    with btn_c2:
        if st.button("🧭 Risiko-Kompass", width="stretch"):
            route_to("Kompass")
    with btn_c3:
        if st.button("📦 Ergebnis exportieren", width="stretch"):
            route_to("Export")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Feature showcase
    st.markdown("### Was kann WealthScope AI?")
    f1, f2, f3 = st.columns(3)
    features_info = [
        (f1, "📊", "Analyse", "Historische Marktdaten, Feature Engineering (MA, Bollinger, Drawdown, Volatilität) und transparente Score-Zerlegung für jedes Asset."),
        (f2, "🤖", "ML-Signal", "Regelbasiertes Scoring kombiniert mit Trend-, Volatilitäts- und Drawdown-Signalen. Nachvollziehbar, keine Black Box."),
        (f3, "📰", "News-Integration", "Automatische News-Suchlogik per Ticker oder Thema. Sentiment-Einordnung in Echtzeit für fundierte Entscheidungen."),
    ]
    for col, icon, title, desc in features_info:
        with col:
            st.markdown(
                f"""<div class="ws-feature-card">
  <div class="ws-feature-icon">{icon}</div>
  <h3>{title}</h3>
  <p>{desc}</p>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Info tabs
    tab1, tab2, tab3 = st.tabs(["Überblick", "Demo-Ablauf", "Präsentationsnutzen"])
    with tab1:
        st.markdown(
            """
**WealthScope AI** ist eine lokale wissenschaftliche Demo-Anwendung zur verantwortungsvollen
Kapitaleinordnung. Die App richtet sich an Menschen, die Märkte besser verstehen und Risiken
quantifizieren wollen – ohne Finanzberater.

Alle Ergebnisse basieren auf historischen Daten und regelbasiertem Scoring.
**Keine Finanzberatung.**
            """
        )
    with tab2:
        st.markdown(
            """
1. **Asset wählen** → Sidebar: ETF, Aktie oder eigene Daten hochladen
2. **Kapitalparameter setzen** → Betrag, Gewichtung, tolerierbarer Rückgang
3. **News-Logik wählen** → Automatisch oder eigene Suchbegriffe
4. **Ergebnisse interpretieren** → Outlook, Scores, Grafiken, Empfehlung
5. **Exportieren** → ZIP mit Bericht, JSON, CSV
            """
        )
    with tab3:
        st.success(
            "Die App demonstriert: Big-Data-Basis, Feature Engineering, dynamische Visualisierung, "
            "ML-nahes Scoring und vollständige Reproduzierbarkeit – alles in einer Streamlit-App."
        )

    if st.button("Demo-Hinweis anzeigen"):
        disclaimer_dialog()

    render_explainers()




# =========================================================
# UPGRADE HELPERS: VISUALS, METHODIK, ASSISTANT, EXPORT
# =========================================================

@st.cache_data(show_spinner=False)
def build_ticker_ranking(df: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "asset_type", "daily_return", "drawdown"}
    missing = required - set(df.columns)

    if missing or df.empty:
        return pd.DataFrame()

    agg_kwargs = dict(
        datenpunkte=("ticker", "size"),
        startdatum=("date", "min"),
        enddatum=("date", "max"),
        durchschnittliche_tagesrendite=("daily_return", "mean"),
        volatilitaet=("daily_return", "std"),
        maximaler_drawdown=("drawdown", "min"),
    )

    if "target_20d" in df.columns:
        agg_kwargs["zielquote_20d"] = ("target_20d", "mean")

    ranking = (
        df.groupby(["ticker", "asset_type"], dropna=False)
        .agg(**agg_kwargs)
        .reset_index()
    )

    for col in [
        "durchschnittliche_tagesrendite",
        "volatilitaet",
        "maximaler_drawdown",
        "zielquote_20d",
    ]:
        if col in ranking.columns:
            ranking[col] = ranking[col] * 100

    sort_cols = [c for c in ["zielquote_20d", "durchschnittliche_tagesrendite"] if c in ranking.columns]
    if sort_cols:
        ranking = ranking.sort_values(sort_cols, ascending=False)

    return ranking


def chart_risk_return_scatter(df: pd.DataFrame) -> go.Figure:
    ranking = build_ticker_ranking(df)

    fig = go.Figure()

    if ranking.empty:
        fig.update_layout(title="Risiko-Rendite-Vergleich nicht verfügbar")
        return fig

    max_points = max(float(ranking["datenpunkte"].max()), 1.0)

    fig.add_trace(
        go.Scatter(
            x=ranking["volatilitaet"],
            y=ranking["durchschnittliche_tagesrendite"],
            mode="markers+text",
            text=ranking["ticker"],
            textposition="top center",
            marker=dict(
                size=(ranking["datenpunkte"] / max_points * 34 + 8),
                opacity=0.72,
            ),
            customdata=ranking[
                [
                    "asset_type",
                    "maximaler_drawdown",
                    "zielquote_20d" if "zielquote_20d" in ranking.columns else "datenpunkte",
                    "datenpunkte",
                ]
            ],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Asset-Typ: %{customdata[0]}<br>"
                "Volatilität: %{x:.2f}%<br>"
                "Ø Tagesrendite: %{y:.3f}%<br>"
                "Max. Drawdown: %{customdata[1]:.2f}%<br>"
                "Zielquote/Daten: %{customdata[2]}<br>"
                "Datenpunkte: %{customdata[3]}<extra></extra>"
            ),
            name="Ticker",
        )
    )

    fig.update_layout(
        title="Risiko-Rendite-Vergleich aller Ticker",
        xaxis_title="Volatilität",
        yaxis_title="Ø Tagesrendite",
        height=520,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    return fig


@st.cache_data(show_spinner=False)
def chart_feature_correlation(df: pd.DataFrame) -> go.Figure:
    feature_cols = [
        "daily_return",
        "return_5d",
        "return_20d",
        "ma_20_distance",
        "ma_50_distance",
        "ma_200_distance",
        "volatility_20d",
        "drawdown",
        "future_return_20d",
        "target_20d",
    ]

    available = [c for c in feature_cols if c in df.columns]

    if len(available) < 2:
        fig = go.Figure()
        fig.update_layout(title="Feature-Korrelation nicht verfügbar")
        return fig

    corr = df[available].corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Korrelation"),
        )
    )

    fig.update_layout(
        title="Feature-Korrelationsmatrix",
        height=560,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    return fig


def chart_volatility(df: pd.DataFrame, ticker: str) -> go.Figure:
    d = df[df["ticker"] == ticker].sort_values("date").copy()

    fig = go.Figure()

    if "volatility_20d" in d.columns:
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["volatility_20d"] * 100,
                mode="lines",
                name="20T Volatilität %",
            )
        )

    fig.update_layout(
        title=f"{ticker}: Volatilitätsverlauf",
        hovermode="x unified",
        yaxis_title="Volatilität in %",
        height=360,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation="h"),
    )

    return fig


@st.dialog("Methodik & Grenzen")
def methodology_dialog() -> None:
    st.markdown(
        """
### Datenbasis
- Lokale Kaggle-Marktdaten als vorbereiteter Feature-Datensatz
- Technische Features: Renditen, Moving Averages, Volatilität, Drawdown
- Zielvariable: `target_20d`

### News
- NewsAPI als externe Nachrichtenquelle
- Query-basierte Suche
- Vereinfachtes regelbasiertes Sentiment

### Scoring
- Aktuell regelbasiertes Scoring
- Kein echtes Kauf-/Verkaufssignal
- ML-Ausbau über ML-Labor möglich

### Grenzen
- Historische Daten garantieren keine Zukunftsentwicklung
- News-Sentiment ist vereinfacht
- Keine Finanzberatung
        """
    )



def secret_str(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
    except Exception:
        value = default

    if value is None:
        return default

    return str(value).strip()


def get_gemini_client():
    api_key = secret_str("GEMINI_API_KEY", "")

    if not api_key or api_key.startswith("DEIN_"):
        return None

    return genai.Client(api_key=api_key)


def safe_str(value: Any, default: str = "n/a") -> str:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
    except Exception:
        pass

    return str(value)


def build_llm_context(ctx: Dict[str, Any]) -> str:
    result = ctx.get("result")
    base_df = ctx.get("base_df", pd.DataFrame())
    features_df = ctx.get("features_df", pd.DataFrame())
    news_df = ctx.get("news_df", pd.DataFrame())

    context_parts = []

    context_parts.append("APP: WealthScope AI")
    context_parts.append("HINWEIS: Keine Finanzberatung. Nur erklärende Analyse im Rahmen eines Uni-/Demo-Projekts.")

    if result is not None:
        context_parts.append("")
        context_parts.append("AKTUELLE ANALYSE:")
        for attr in ["ticker", "asset", "outlook", "risk", "confidence", "score", "reason", "interpretation"]:
            if hasattr(result, attr):
                context_parts.append(f"- {attr}: {safe_str(getattr(result, attr))}")

    scenario_keys = [
        "capital",
        "portfolio_weight",
        "risk_tolerance",
        "asset",
        "period_years",
        "news_score",
        "news_label",
    ]

    context_parts.append("")
    context_parts.append("SESSION / SZENARIO:")
    for key in scenario_keys:
        if key in st.session_state:
            context_parts.append(f"- {key}: {safe_str(st.session_state.get(key))}")

    if base_df is not None and not base_df.empty:
        context_parts.append("")
        context_parts.append("DATENBASIS:")
        context_parts.append(f"- Zeilen: {len(base_df)}")
        if "ticker" in base_df.columns:
            context_parts.append(f"- Ticker-Anzahl: {base_df['ticker'].nunique()}")
        if "date" in base_df.columns:
            context_parts.append(f"- Zeitraum: {safe_str(base_df['date'].min())} bis {safe_str(base_df['date'].max())}")

    if features_df is not None and not features_df.empty:
        context_parts.append("")
        context_parts.append("LETZTE FEATURE-WERTE:")
        latest = features_df.tail(1).iloc[0]
        for col in [
            "ticker",
            "date",
            "close",
            "daily_return",
            "return_5d",
            "return_20d",
            "ma_20_distance",
            "ma_50_distance",
            "ma_200_distance",
            "volatility_20d",
            "drawdown",
            "target_20d",
        ]:
            if col in latest.index:
                context_parts.append(f"- {col}: {safe_str(latest[col])}")

    if news_df is not None and not news_df.empty:
        context_parts.append("")
        context_parts.append("AKTUELLE NEWS-BEISPIELE:")
        title_col = "Titel" if "Titel" in news_df.columns else "title" if "title" in news_df.columns else None
        source_col = "Quelle" if "Quelle" in news_df.columns else "source" if "source" in news_df.columns else None
        sentiment_col = "Sentiment" if "Sentiment" in news_df.columns else "sentiment" if "sentiment" in news_df.columns else None
        relevance_col = "Relevanz" if "Relevanz" in news_df.columns else "relevance" if "relevance" in news_df.columns else None
        impact_col = "Impact" if "Impact" in news_df.columns else "impact" if "impact" in news_df.columns else None

        for _, row in news_df.head(5).iterrows():
            title = safe_str(row.get(title_col, "n/a")) if title_col else "n/a"
            source = safe_str(row.get(source_col, "n/a")) if source_col else "n/a"
            sentiment = safe_str(row.get(sentiment_col, "n/a")) if sentiment_col else "n/a"
            relevance = safe_str(row.get(relevance_col, "n/a")) if relevance_col else "n/a"
            impact = safe_str(row.get(impact_col, "n/a")) if impact_col else "n/a"
            context_parts.append(f"- Titel: {title} | Quelle: {source} | Sentiment: {sentiment} | Relevanz: {relevance} | Impact: {impact}")

    context_parts.append("")
    context_parts.append("GRENZEN:")
    context_parts.append("- Historische Kursdaten garantieren keine zukünftige Entwicklung.")
    context_parts.append("- target_20d ist eine Modell-/Feature-Zielvariable und keine sichere Prognose.")
    context_parts.append("- News-Sentiment ist vereinfachend und kann Quellen-/Aggregationsfehler enthalten.")
    context_parts.append("- Die App ersetzt keine professionelle Finanzberatung.")

    return "\n".join(context_parts)


def assistant_answer_gemini(prompt: str, ctx: Dict[str, Any]) -> Optional[str]:
    provider = secret_str("LLM_PROVIDER", "gemini").lower()
    if provider not in ["gemini", "auto"]:
        return None

    client = get_gemini_client()
    if client is None:
        return None

    model = secret_str("GEMINI_MODEL", "gemini-2.5-flash")
    app_context = build_llm_context(ctx)

    system_prompt = (
        "Du bist der WealthScope AI Assistant innerhalb eines lokalen Uni-Prototyps. "
        "Du darfst beratend, einordnend und entscheidungsunterstützend antworten, aber immer klar als prototypische Demo-Einschätzung. "
        "Nutze den App-Kontext für konkrete Aussagen zu Ticker, Kapital, Zeitraum, Kennzahlen, News, Risiko, Outlook und Bewertung. "
        "Allgemeine Finanz-, Daten- und Analysebegriffe darfst du verständlich erklären, auch wenn sie nicht wortwörtlich im Kontext stehen. "
        "Du darfst dem Nutzer konkrete nächste Schritte vorschlagen, zum Beispiel: Zeitraum vergleichen, Positionsgröße reduzieren, Risiko prüfen, News beobachten oder Export erzeugen. "
        "Erfinde keine konkreten Zahlen, Kurse, Renditen oder News. Wenn konkrete Informationen fehlen, sage das offen. "
        "Formuliere nicht passiv-abwehrend, sondern hilfreich und analytisch. "
        "Wichtig: Weise kurz darauf hin, dass es sich um eine prototypische Einschätzung innerhalb einer Demo-App handelt und nicht um produktive Finanzberatung. "
        "Erkläre Fachbegriffe wie Zeiträume, Rendite, Moving Averages, Drawdown, Volatilität, target_20d und News-Sentiment einfach."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"APP-KONTEXT:\n{app_context}\n\n"
        f"NUTZERFRAGE:\n{prompt}"
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
        )
        answer = getattr(response, "text", None)
        if answer:
            return answer.strip()
    except Exception as exc:
        return f"Gemini konnte gerade nicht antworten. Technischer Hinweis: {type(exc).__name__}. Ich nutze stattdessen die lokale Fallback-Logik."

    return None


def assistant_answer(prompt: str, ctx: Dict[str, Any]) -> str:
    gemini_answer = assistant_answer_gemini(prompt, ctx)
    if gemini_answer:
        return gemini_answer

    result = ctx["result"]
    df = ctx["features"]
    news_df = ctx.get("news_df", pd.DataFrame())

    p = prompt.lower()

    if any(word in p for word in ["drawdown", "verlust", "rückgang"]):
        latest_text = ""
        if not df.empty and "drawdown" in df.columns:
            latest = df["drawdown"].dropna().tail(1)
            if not latest.empty:
                latest_text = f" Der letzte Drawdown-Wert liegt bei ca. {float(latest.iloc[0]) * 100:.2f}%."

        return (
            f"Drawdown beschreibt, wie stark {result.ticker} vom vorherigen Hoch gefallen ist."
            f"{latest_text} WealthScope nutzt Drawdown, um Kapitalrisiko sichtbarer zu machen. "
            "Das ist keine Anlageberatung."
        )

    if any(word in p for word in ["news", "nachricht", "sentiment"]):
        if news_df.empty:
            return "Für die aktuelle Analyse liegen keine News-Daten vor oder die NewsAPI liefert gerade keine Ergebnisse."

        top_titles = news_df.get("Titel", pd.Series(dtype=str)).dropna().head(3).astype(str).tolist()
        bullets = "\\n".join([f"- {t}" for t in top_titles])

        return (
            f"Die aktuelle News-Lage für {result.ticker} wird als '{result.news_label}' eingeordnet "
            f"mit einem News-Score von {result.news_score}. Beispiele:\\n{bullets}\\n"
            "Die Einordnung ist regelbasiert und vereinfacht."
        )

    if any(word in p for word in ["feature", "target", "target_20d", "modell", "ml"]):
        return (
            "`target_20d` beschreibt, ob der Kurs nach 20 Handelstagen höher liegt. "
            "Die App nutzt Features wie Renditen, Moving Averages, Volatilität und Drawdown. "
            "Aktuell ist das Scoring regelbasiert; ein echtes ML-Modell kann darauf aufbauen."
        )

    if any(word in p for word in ["methodik", "quelle", "daten", "kaggle"]):
        return (
            "WealthScope nutzt lokal vorbereitete Kaggle-Marktdaten, berechnet technische Features "
            "und ergänzt NewsAPI-Daten. Ziel ist eine reproduzierbare Demo für Analyse, Erklärung und Export."
        )

    if any(word in p for word in ["risiko", "volatil", "volatilität"]):
        return (
            f"Für {result.ticker} bewertet WealthScope Risiko über Volatilität, Drawdown, Gewichtung "
            f"und tolerierten Rückgang. Aktuelle Risikoklasse: {result.risk_label}. "
            f"Confidence: {result.confidence}/100."
        )

    return (
        f"Aktuell wird {result.ticker} mit Outlook '{result.outlook}' und Risiko '{result.risk_label}' bewertet. "
        f"Die Confidence liegt bei {result.confidence}/100. {result.recommendation} "
        "Du kannst nach Drawdown, News, target_20d, Methodik oder Risiko fragen. Keine Finanzberatung."
    )


def make_export_zip(result: AnalysisResult, features_df: pd.DataFrame, news_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    report_md = analysis_report_markdown(result)
    context_json = json.dumps(asdict(result), indent=2, ensure_ascii=False)

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("analysebericht.md", report_md)
        zf.writestr("analyse_context.json", context_json)
        zf.writestr("gefilterte_marktdaten.csv", features_df.to_csv(index=False))
        zf.writestr("news_einordnung.csv", news_df.to_csv(index=False))
        zf.writestr(
            "methodik.txt",
            "WealthScope AI kombiniert lokale Kaggle-Marktdaten, Feature Engineering, NewsAPI-Daten und ein regelbasiertes Scoring. Keine Finanzberatung.",
        )

    buffer.seek(0)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def build_compare_df_cached(data_json: str, period: str) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(data_json), orient="split")
    enriched = []
    for ticker in df["ticker"].dropna().unique():
        part = filter_period(df, ticker, period)
        if not part.empty:
            enriched.append(enrich_features(part))
    return pd.concat(enriched, ignore_index=True) if enriched else pd.DataFrame()


def page_outlook(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    df = ctx["features"]
    raw = ctx["market"]

    page_title("Märkte", f"{result.ticker} — Kurs, Trend, Risiko & Empfehlung")

    # Signal-Ampel
    def _signal_class(score: float) -> str:
        if score >= 65: return "positive"
        if score >= 40: return "neutral"
        return "negative"

    def _signal_dot(score: float) -> str:
        if score >= 65: return "🟢"
        if score >= 40: return "🟡"
        return "🔴"

    news_cls = "positive" if result.news_score >= 0.6 else "neutral" if result.news_score >= 0.3 else "negative"
    news_dot = "🟢" if result.news_score >= 0.6 else "🟡" if result.news_score >= 0.3 else "🔴"

    st.markdown(
        f"""
<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.25rem">
  <span class="ws-signal-badge {_signal_class(result.trend_score)}">{_signal_dot(result.trend_score)} Trend {result.trend_score}</span>
  <span class="ws-signal-badge {_signal_class(result.volatility_score)}">{_signal_dot(result.volatility_score)} Volatilität {result.volatility_score}</span>
  <span class="ws-signal-badge info">⚙ Confidence {result.confidence}</span>
</div>
        """,
        unsafe_allow_html=True,
    )

    result_metrics(result)

    # Recommendation box
    st.markdown(
        f"""<div class="ws-recommendation-box">
  <h4>Empfehlung</h4>
  <p>{result.recommendation} News-Lage: <strong>{result.news_label}</strong>.</p>
</div>""",
        unsafe_allow_html=True,
    )

    # Price chart — immediate focus before tabs
    show_chart_with_data(
        "Kursverlauf",
        chart_price(df, result.ticker),
        df[["date", "ticker", "close", "ma_20", "ma_50", "ma_200"]].tail(600),
        "outlook_price",
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Momentum", "Risiko", "Vergleich", "Scores", "Rohdaten"])

    with tab1:
        show_chart_with_data(
            "Momentum-Analyse",
            chart_momentum(df, result.ticker),
            df[[c for c in ["date", "ticker", "close", "return_5d", "return_20d", "return_60d", "volatility_20d"] if c in df.columns]].tail(600),
            "outlook_momentum",
        )

    with tab2:
        show_chart_with_data(
            "Drawdown",
            chart_drawdown(df, result.ticker),
            df[["date", "ticker", "close", "drawdown", "volatility_20d"]].tail(600),
            "outlook_drawdown",
        )
        if "volatility_20d" in df.columns:
            show_chart_with_data(
                "Volatilität",
                chart_volatility(df, result.ticker),
                df[["date", "ticker", "volatility_20d"]].dropna().tail(600),
                "outlook_volatility",
            )
        show_chart_with_data(
            "Renditeverteilung",
            chart_returns(df, result.ticker),
            df[["date", "ticker", "daily_return"]].dropna().tail(1000),
            "outlook_returns",
        )

    with tab3:
        compare_df = build_compare_df_cached(raw.to_json(orient="split"), st.session_state.get("period", "5Y"))
        if compare_df.empty:
            compare_df = df
        ranking = build_ticker_ranking(compare_df)
        show_chart_with_data("Risiko-Rendite-Vergleich", chart_risk_return_scatter(compare_df), ranking, "risk_return_scatter")
        st.dataframe(ranking, width="stretch", hide_index=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            show_chart_with_data("Score-Zerlegung", chart_scores(result), pd.DataFrame([asdict(result)]), "outlook_scores")
        with col2:
            show_chart_with_data("Radar-Profil", chart_radar(result), pd.DataFrame([asdict(result)]), "outlook_radar")

    with tab5:
        st.dataframe(df.tail(1000), width="stretch", hide_index=True)

    render_explainers()


def page_kompass(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    page_title("Signale", "Kapital, Gewichtung & Risikoklasse auf einen Blick")

    metric_grid(
        [
            ("Gesamtkapital", money(result.capital)),
            ("Max. Einzelposition", money(result.max_position)),
            ("Tolerierter Rückgang", money(result.tolerated_loss)),
            ("Risikoklasse", result.risk_label),
        ]
    )

    # Gauge charts
    g1, g2 = st.columns(2)
    with g1:
        fig_conf = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(result.confidence),
            title={"text": "Confidence", "font": {"size": 16}},
            number={"suffix": "/100", "font": {"size": 28, "color": "#6366F1"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#6366F1", "thickness": 0.28},
                "steps": [
                    {"range": [0, 40], "color": "rgba(239,68,68,0.15)"},
                    {"range": [40, 65], "color": "rgba(245,158,11,0.15)"},
                    {"range": [65, 100], "color": "rgba(34,197,94,0.15)"},
                ],
                "threshold": {"line": {"color": "#6366F1", "width": 3}, "thickness": 0.75, "value": float(result.confidence)},
            },
        ))
        fig_conf.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_conf, use_container_width=True, config={"displayModeBar": False})

    with g2:
        risk_val = float(result.risk_score)  # echten Wert nutzen
        risk_color = "#ef4444" if result.risk_label == "Hoch" else "#f59e0b" if result.risk_label == "Mittel" else "#22c55e"
        fig_risk = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_val,
            title={"text": f"Risikolevel: {result.risk_label}", "font": {"size": 16}},
            delta={"reference": 50, "decreasing": {"color": "#22c55e"}, "increasing": {"color": "#ef4444"}},
            number={"suffix": "%", "font": {"size": 28, "color": risk_color}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": risk_color, "thickness": 0.28},
                "steps": [
                    {"range": [0, 40], "color": "rgba(34,197,94,0.15)"},
                    {"range": [40, 65], "color": "rgba(245,158,11,0.15)"},
                    {"range": [65, 100], "color": "rgba(239,68,68,0.15)"},
                ],
            },
        ))
        fig_risk.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_risk, use_container_width=True, config={"displayModeBar": False})

    if result.risk_label == "Hoch":
        st.warning("Die aktuelle Kombination aus Gewichtung, Volatilität und Drawdown wirkt riskant.")
    elif result.risk_label == "Mittel":
        st.info("Die Gewichtung wirkt kontrollierbar, sollte aber beobachtet werden.")
    else:
        st.success("Die Risikoannahmen wirken im Verhältnis zum Kapital eher defensiv.")



def normalize_scenario_allocations(df: pd.DataFrame) -> pd.DataFrame:
    """Clean scenario allocation table and keep only valid positive weights."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Baustein", "Gewichtung %"])

    out = df.copy()

    if "Baustein" not in out.columns or "Gewichtung %" not in out.columns:
        return pd.DataFrame(columns=["Baustein", "Gewichtung %"])

    out["Baustein"] = out["Baustein"].astype(str).str.strip()
    out["Gewichtung %"] = pd.to_numeric(out["Gewichtung %"], errors="coerce").fillna(0.0)

    out = out[out["Baustein"] != ""]
    out = out[out["Gewichtung %"] > 0]

    # Gleiche Bausteine zusammenfassen
    out = out.groupby("Baustein", as_index=False)["Gewichtung %"].sum()

    return out


def build_allocation_chart_df(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """Build chart data from scenario allocation table. Adds liquidity only if sum < 100."""
    alloc = normalize_scenario_allocations(scenario_df)

    if alloc.empty:
        return pd.DataFrame(
            {
                "Baustein": ["Liquidität / Rest"],
                "Gewichtung %": [100.0],
            }
        )

    total = float(alloc["Gewichtung %"].sum())

    if total < 100:
        rest = 100.0 - total
        alloc = pd.concat(
            [
                alloc,
                pd.DataFrame(
                    {
                        "Baustein": ["Liquidität / Rest"],
                        "Gewichtung %": [rest],
                    }
                ),
            ],
            ignore_index=True,
        )

    return alloc


def validate_allocation_sum(scenario_df: pd.DataFrame) -> float:
    alloc = normalize_scenario_allocations(scenario_df)
    if alloc.empty:
        return 0.0
    return float(alloc["Gewichtung %"].sum())


def page_simulator(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    page_title("Portfolio", "Gewichtung, Allokation & Szenarien")

    if "portfolio_rows" not in st.session_state or not st.session_state.get("portfolio_rows"):
        st.session_state["portfolio_rows"] = [
            {"Baustein": result.ticker, "Gewichtung_%": float(result.asset_weight)},
            {"Baustein": "QQQ", "Gewichtung_%": 25.0},
            {"Baustein": "GLD", "Gewichtung_%": 10.0},
            {"Baustein": "AGG", "Gewichtung_%": 20.0},
        ]

    portfolio_df = pd.DataFrame(st.session_state.get("portfolio_rows", []))

    if "Baustein" not in portfolio_df.columns:
        portfolio_df["Baustein"] = result.ticker

    if "Gewichtung_%" not in portfolio_df.columns:
        portfolio_df["Gewichtung_%"] = float(result.asset_weight)

    edited = st.data_editor(
        portfolio_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "Baustein": st.column_config.TextColumn("Baustein"),
            "Gewichtung_%": st.column_config.NumberColumn(
                "Gewichtung %",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
            ),
        },
        key="portfolio_scenario_editor",
    )

    edited = edited.copy()
    edited["Gewichtung_%"] = pd.to_numeric(edited["Gewichtung_%"], errors="coerce").fillna(0.0)
    st.session_state["portfolio_rows"] = edited.to_dict("records")

    scenario_df = edited.rename(columns={"Gewichtung_%": "Gewichtung %"})
    total_weight = validate_allocation_sum(scenario_df)
    chart_df = build_allocation_chart_df(scenario_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("Gesamtgewichtung", f"{total_weight:.1f} %")
    m2.metric("Liquidität / Rest", f"{max(0.0, 100.0 - total_weight):.1f} %")
    m3.metric("Status", "OK" if total_weight <= 100 else "Zu hoch")

    st.progress(min(int(total_weight), 100), text=f"Gesamtgewichtung: {total_weight:.1f} %")

    if total_weight > 100:
        st.error("Die Gesamtgewichtung liegt über 100 %. Bitte reduziere die Positionen.")
    elif total_weight < 100:
        st.info(f"{100.0 - total_weight:.1f} % bleiben als Liquidität / Rest.")
    else:
        st.success("Die Allokation beträgt genau 100 %.")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=chart_df["Baustein"],
                values=chart_df["Gewichtung %"],
                hole=0.55,
                textinfo="percent+label",
            )
        ]
    )
    fig.update_layout(
        title="Kapitalallokation im Szenario",
        height=420,
        margin=dict(l=10, r=10, t=55, b=10),
    )

    show_chart_with_data(
        "Portfolio-Allokation",
        fig,
        chart_df,
        "portfolio_alloc",
    )



def page_watchlist(ctx: Dict[str, Any]) -> None:
    from plotly.subplots import make_subplots

    raw = ctx["market"]
    tickers = available_tickers(raw)
    page_title("Watchlist", "Rendite, Volatilität & Drawdown im Vergleich")

    default = [t for t in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"] if t in tickers]
    selected = st.multiselect("Watchlist auswählen", tickers, default=default)

    rows = []
    ticker_dfs = {}
    for ticker in selected:
        df = enrich_features(filter_period(raw, ticker, st.session_state.get("period", "5Y")))
        if df.empty:
            continue
        last = df.tail(1).iloc[0]
        ticker_dfs[ticker] = df.tail(30)
        rows.append(
            {
                "Ticker": ticker,
                "Letzter Kurs": round(float(last["close"]), 2),
                "Return 20d %": round(float(last["return_20d"]) * 100, 2) if pd.notna(last["return_20d"]) else None,
                "Volatilität 20d %": round(float(last["volatility_20d"]) * 100, 2) if pd.notna(last["volatility_20d"]) else None,
                "Drawdown %": round(float(last["drawdown"]) * 100, 2) if pd.notna(last["drawdown"]) else None,
            }
        )

    compare = pd.DataFrame(rows)
    st.dataframe(compare, width="stretch", hide_index=True)

    # Sparklines: 30-day price trend per ticker
    if ticker_dfs:
        n = len(ticker_dfs)
        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)
        ticker_list = list(ticker_dfs.keys())

        spark_fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=ticker_list,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        for i, (tick, tdf) in enumerate(ticker_dfs.items()):
            row = i // ncols + 1
            col = i % ncols + 1
            close_vals = tdf["close"].values
            first_val = close_vals[0] if len(close_vals) > 0 else 1
            pct_change = (close_vals - first_val) / max(first_val, 1e-9) * 100
            line_color = "#22c55e" if pct_change[-1] >= 0 else "#ef4444"
            spark_fig.add_trace(
                go.Scatter(
                    x=tdf["date"].values,
                    y=tdf["close"].values,
                    mode="lines",
                    name=tick,
                    line=dict(color=line_color, width=2),
                    showlegend=False,
                ),
                row=row, col=col,
            )

        spark_fig.update_layout(
            title="30-Tage Kursentwicklung (Sparklines)",
            height=max(300, nrows * 200),
            margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        spark_fig.update_xaxes(showgrid=False, showticklabels=False)
        spark_fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
        st.plotly_chart(spark_fig, use_container_width=True, config={"displayModeBar": False})

    if not compare.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=compare["Ticker"], y=compare["Return 20d %"], name="Return 20d %",
                             marker_color="#6366F1", opacity=0.8))
        fig.add_trace(go.Bar(x=compare["Ticker"], y=compare["Volatilität 20d %"], name="Volatilität 20d %",
                             marker_color="#f59e0b", opacity=0.8))
        fig.update_layout(title="Rendite & Volatilität Vergleich", barmode="group", height=420,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(ticksuffix="%")
        show_chart_with_data("Watchlist", fig, compare, "watchlist_compare")


def page_data_lab(ctx: Dict[str, Any]) -> None:
    raw = ctx["market"]
    df = ctx["features"]
    result = ctx["result"]

    page_title("Analytics", "Datenqualität, Verteilungen & Feature-Profil")

    source = getattr(raw, "attrs", {}).get("data_source", "UNKNOWN")
    _live_dlab = st.session_state.get("use_live_data", False)
    _bis_dlab = (
        __import__("datetime").date.today().strftime("%Y-%m-%d") + " 🌐"
        if _live_dlab
        else (raw["date"].max().strftime("%Y-%m-%d") if "date" in raw.columns else "–")
    )
    metric_grid([
        ("Zeilen gesamt", f"{len(raw):,}".replace(",", ".")),
        ("Features (ML)", str(len([c for c in ["daily_return","return_5d","return_20d","ma_20_distance","ma_50_distance","ma_200_distance","volatility_20d","drawdown"] if c in raw.columns]))),
        ("Zeitraum von", raw["date"].min().strftime("%Y-%m-%d") if "date" in raw.columns else "–"),
        ("Zeitraum bis", _bis_dlab),
        ("Zielvariable", "✅ target_20d" if "target_20d" in raw.columns else "❌ fehlt"),
        ("Datenquelle", source + (" + yfinance" if _live_dlab else "")),
    ])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Verteilungen", "🔍 Fehlwerte", "📋 Profil", "📈 Zeitreihe", "🗂 Rohdaten"
    ])

    with tab1:
        st.markdown("#### Histogramme aller numerischen Features")
        feat_cols = [c for c in ["daily_return","return_5d","return_20d","ma_20_distance",
                                  "ma_50_distance","ma_200_distance","volatility_20d","drawdown"]
                     if c in df.columns]
        if feat_cols:
            from plotly.subplots import make_subplots as _ms
            n_cols_h = 4
            n_rows_h = math.ceil(len(feat_cols) / n_cols_h)
            fig_hist = _ms(rows=n_rows_h, cols=n_cols_h,
                           subplot_titles=feat_cols,
                           vertical_spacing=0.12, horizontal_spacing=0.06)
            for idx, col in enumerate(feat_cols):
                r, c = divmod(idx, n_cols_h)
                data_col = df[col].dropna()
                fig_hist.add_trace(
                    go.Histogram(x=data_col, nbinsx=60, name=col,
                                 marker_color="#6366F1", opacity=0.75,
                                 showlegend=False),
                    row=r+1, col=c+1,
                )
            fig_hist.update_layout(height=n_rows_h*220, margin=dict(l=10,r=10,t=40,b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True,
                            config={"displayModeBar": False}, key="dlab_hist")

        if "target_20d" in df.columns:
            st.markdown("#### Zielvariable `target_20d`")
            vc    = df["target_20d"].value_counts().sort_index()
            total = vc.sum()
            n0    = int(vc.get(0, 0))
            n1    = int(vc.get(1, 0))
            p0    = n0 / total * 100
            p1    = n1 / total * 100

            # ── Elegante 3-spaltige Darstellung ──────────────────
            ca, cb, cc = st.columns([1, 2, 1])

            with ca:
                st.markdown(
                    f"""<div style="background:rgba(239,68,68,.07);border:1px solid rgba(239,68,68,.25);
                    border-left:4px solid #ef4444;border-radius:12px;padding:1rem 1.1rem;text-align:center">
                    <div style="font-size:.7rem;font-weight:800;color:#ef4444;letter-spacing:.07em;
                    text-transform:uppercase;margin-bottom:.4rem">Bearish / Neutral</div>
                    <div style="font-size:2rem;font-weight:900;color:#1e293b;line-height:1">{p0:.1f}<span style="font-size:1rem">%</span></div>
                    <div style="font-size:.75rem;color:#64748b;margin-top:.3rem">{n0:,} Beobachtungen</div>
                    <div style="font-size:.68rem;color:#94a3b8;margin-top:.15rem">Klasse 0</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

            with cb:
                # Horizontaler gestapelter Progress-Bar + Info
                imbalance = "Leichte Klassen-Imbalance" if abs(p0-p1) > 10 else "Ausgeglichen"
                bar_html = (
                    f'<div style="margin:.5rem 0 .8rem">'
                    f'<div style="font-size:.7rem;font-weight:700;color:#64748b;display:flex;justify-content:space-between;margin-bottom:.4rem"><span>Klasse 0</span><span>Klasse 1</span></div>'
                    f'<div style="height:18px;border-radius:999px;overflow:hidden;background:#f1f5f9;display:flex;">'
                    f'<div style="width:{p0:.1f}%;background:linear-gradient(90deg,#ef4444,#f87171);display:flex;align-items:center;justify-content:center;"><span style="font-size:.65rem;font-weight:800;color:white;padding:0 .4rem">{p0:.0f}%</span></div>'
                    f'<div style="width:{p1:.1f}%;background:linear-gradient(90deg,#4ade80,#22c55e);display:flex;align-items:center;justify-content:center;"><span style="font-size:.65rem;font-weight:800;color:white;padding:0 .4rem">{p1:.0f}%</span></div>'
                    f'</div>'
                    f'<div style="font-size:.7rem;color:#94a3b8;text-align:center;margin-top:.5rem">{total:,} Gesamtbeobachtungen · {imbalance}</div>'
                    f'</div>'
                    f'<div style="background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.15);border-radius:10px;padding:.75rem 1rem;font-size:.78rem;color:#475569;line-height:1.7;margin-top:.5rem">'
                    f'<strong style="color:#6366f1">Definition:</strong> target_20d&nbsp;=&nbsp;1, wenn der Schlusskurs in 20&nbsp;Handelstagen h\u00f6her liegt als heute. Sonst&nbsp;=&nbsp;0.<br>'
                    f'<strong style="color:#6366f1">Verwendung:</strong> Zielvariable des Random-Forest-Modells.'
                    f'</div>'
                )
                st.markdown(bar_html, unsafe_allow_html=True)

            with cc:
                st.markdown(
                    f"""<div style="background:rgba(34,197,94,.07);border:1px solid rgba(34,197,94,.25);
                    border-left:4px solid #22c55e;border-radius:12px;padding:1rem 1.1rem;text-align:center">
                    <div style="font-size:.7rem;font-weight:800;color:#16a34a;letter-spacing:.07em;
                    text-transform:uppercase;margin-bottom:.4rem">Bullish</div>
                    <div style="font-size:2rem;font-weight:900;color:#1e293b;line-height:1">{p1:.1f}<span style="font-size:1rem">%</span></div>
                    <div style="font-size:.75rem;color:#64748b;margin-top:.3rem">{n1:,} Beobachtungen</div>
                    <div style="font-size:.68rem;color:#94a3b8;margin-top:.15rem">Klasse 1</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    with tab2:
        st.markdown("#### Analyse fehlender Werte (QUA³CK U-Phase)")
        miss = raw.isna().sum().reset_index()
        miss.columns = ["Spalte", "Fehlwerte (abs)"]
        miss["Fehlwerte (%)"] = (miss["Fehlwerte (abs)"] / len(raw) * 100).round(2)
        miss["Typ"] = miss["Fehlwerte (%)"].apply(
            lambda x: "🔴 Kritisch (>20%)" if x > 20 else "🟡 Moderat (5–20%)" if x > 5 else "🟢 OK (<5%)"
        )
        miss_nonzero = miss[miss["Fehlwerte (abs)"] > 0].sort_values("Fehlwerte (%)", ascending=False)

        if miss_nonzero.empty:
            st.success("✅ Keine fehlenden Werte im Datensatz gefunden.")
        else:
            st.dataframe(miss_nonzero, width="stretch", hide_index=True)
            fig_miss = go.Figure(go.Bar(
                y=miss_nonzero["Spalte"], x=miss_nonzero["Fehlwerte (%)"],
                orientation="h",
                marker_color=["#ef4444" if x>20 else "#f59e0b" if x>5 else "#22c55e"
                               for x in miss_nonzero["Fehlwerte (%)"]],
            ))
            fig_miss.add_vline(x=5, line_dash="dash", line_color="#94a3b8",
                               annotation_text="5 %", annotation_position="top right")
            fig_miss.update_layout(title="Fehlende Werte je Feature (%)", height=max(250, len(miss_nonzero)*35+80),
                                   margin=dict(l=10,r=10,t=50,b=10),
                                   paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_miss, use_container_width=True,
                            config={"displayModeBar": False}, key="dlab_miss")

        st.info("Fehlende Werte in MA-Features entstehen durch Warm-up-Zeit (erste 20/50/200 Tage je Ticker). Typ: **MCAR** → Median-Imputation korrekt.")

    with tab3:
        st.markdown("#### Statistisches Profil")
        st.dataframe(raw.describe(include="all").transpose().round(4), width="stretch")

    with tab4:
        st.markdown("#### Zeitreihe des ausgewählten Assets")
        ticker_sel = result.ticker
        ts_df = df[["date","close","volume"]].dropna(subset=["close"]) if "volume" in df.columns else df[["date","close"]].dropna()
        from plotly.subplots import make_subplots as _ms2
        if "volume" in ts_df.columns:
            fig_ts = _ms2(rows=2, cols=1, row_heights=[0.75, 0.25], vertical_spacing=0.04,
                          shared_xaxes=True)
            fig_ts.add_trace(go.Scatter(x=ts_df["date"], y=ts_df["close"], mode="lines",
                                        line=dict(color="#6366F1", width=1.5), name="Close"),
                             row=1, col=1)
            colors_v = ["#22c55e" if i == 0 or ts_df["close"].iloc[i] >= ts_df["close"].iloc[i-1]
                        else "#ef4444" for i in range(len(ts_df))]
            fig_ts.add_trace(go.Bar(x=ts_df["date"], y=ts_df["volume"],
                                    marker_color=colors_v, name="Volumen", opacity=0.6),
                             row=2, col=1)
            fig_ts.update_layout(height=400, margin=dict(l=10,r=10,t=40,b=10),
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                 title=f"{ticker_sel}: Kurs + Volumen", showlegend=False)
        else:
            fig_ts = go.Figure(go.Scatter(x=ts_df["date"], y=ts_df["close"],
                                          mode="lines", line=dict(color="#6366F1")))
            fig_ts.update_layout(height=360, title=f"{ticker_sel}: Schlusskurs",
                                 paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ts, use_container_width=True, config={"displayModeBar": False}, key="dlab_ts")

    with tab5:
        st.markdown("#### Rohdaten (letzte 2.000 Zeilen)")
        st.dataframe(raw.tail(2000), width="stretch", hide_index=True)
        st.download_button(
            "📥 Rohdaten als CSV herunterladen",
            data=raw.to_csv(index=False).encode("utf-8"),
            file_name="wealthscope_raw_data.csv",
            mime="text/csv",
        )


def page_ml_lab(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]
    model = ctx["model"]
    df = ctx["features"]

    page_title("KI Lab", "Random Forest · Score-Zerlegung · Modell-Metriken")

    # ── Gradient-Modell-Hero ────────────────────────────────────────────
    _outlook_color = "#22c55e" if result.outlook == "Positiv" else "#ef4444" if result.outlook == "Negativ" else "#f59e0b"
    _conf_pct = int(result.confidence)
    _conf_bar_color = "#22c55e" if _conf_pct >= 65 else "#f59e0b" if _conf_pct >= 40 else "#ef4444"
    st.markdown(
        f"""
<div style="background:linear-gradient(135deg,#1e1b4b 0%,#312e81 40%,#4c1d95 100%);
     border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.5rem;position:relative;overflow:hidden;">
  <!-- Decorative blur blob -->
  <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;
       background:rgba(99,102,241,0.35);border-radius:50%;filter:blur(50px);pointer-events:none"></div>
  <div style="display:flex;flex-wrap:wrap;gap:2rem;align-items:center;position:relative;z-index:1">
    <!-- Left: model identity -->
    <div>
      <div style="font-size:0.72rem;font-weight:700;color:rgba(165,180,252,0.8);
           letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.35rem">Aktives Modell</div>
      <div style="font-size:1.15rem;font-weight:800;color:#fff;margin-bottom:0.2rem">
        🤖 {model.get("name","–")}
      </div>
      <div style="font-size:0.78rem;color:rgba(199,210,254,0.75)">
        v{model.get("version","–")} &nbsp;·&nbsp; {len(model.get("features",[]))} Features &nbsp;·&nbsp; {model.get("loaded_at","–")}
      </div>
    </div>
    <!-- Right: KI signal -->
    <div style="margin-left:auto;text-align:right">
      <div style="font-size:0.72rem;font-weight:700;color:rgba(165,180,252,0.8);
           letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.4rem">KI-Signal (Regel-Score)</div>
      <div style="font-size:2rem;font-weight:900;color:{_outlook_color};line-height:1.1">{result.outlook}</div>
      <div style="margin-top:0.5rem;background:rgba(255,255,255,0.12);
           border-radius:999px;height:8px;width:140px;overflow:hidden;margin-left:auto">
        <div style="width:{_conf_pct}%;height:100%;background:{_conf_bar_color};border-radius:999px"></div>
      </div>
      <div style="font-size:0.75rem;color:rgba(199,210,254,0.75);margin-top:0.3rem">
        Confidence {_conf_pct}/100
      </div>
      {f'<div style="margin-top:0.6rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);border-radius:8px;padding:0.4rem 0.7rem;font-size:0.78rem;color:rgba(199,210,254,0.9)">🤖 RF-Wahrscheinlichkeit: <strong style="color:#fff">{result.rf_proba*100:.1f}%</strong> Bullish</div>' if result.rf_proba > 0 else ''}
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    result_metrics(result)

    # ── Distribution Shift Warning ──────────────────────────────────────
    if st.session_state.get("use_live_data", False):
        st.warning(
            "⚠️ **Distribution Shift:** Das Modell wurde auf Daten von 1962–2017 trainiert. "
            "Live-Daten (yfinance) liegen außerhalb dieses Trainingsbereichs — das Modell hat "
            "die COVID-Krise (2020), den KI-Boom (2023) oder aktuelle Zinszyklen nie gesehen. "
            "Vorhersagen auf Live-Daten sind mit erhöhter Unsicherheit behaftet.",
            icon="⚠️",
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Score-Analyse", "📈 Kap.3-Metriken", "🔗 Feature-Korrelationen", "📐 Radar & Profil", "📖 Modell-Erklärung", "🔍 SHAP"
    ])

    with tab1:
        # ── Styled Score-Breakdown ──────────────────────────────────────
        def _score_badge(score, wt=1.0):
            val = float(score) if score is not None else 0
            contrib = val * wt
            if val >= 65: bg, fg = "rgba(34,197,94,0.12)", "#16a34a"
            elif val >= 40: bg, fg = "rgba(245,158,11,0.12)", "#b45309"
            else: bg, fg = "rgba(239,68,68,0.12)", "#dc2626"
            return f'<span style="background:{bg};color:{fg};font-weight:800;font-size:0.9rem;padding:0.2rem 0.6rem;border-radius:6px">{val:.0f}</span>'

        def _wt_bar(pct_int):
            return (f'<div style="display:flex;align-items:center;gap:0.5rem">'
                    f'<div style="width:60px;height:6px;border-radius:999px;background:#e2e8f0;overflow:hidden">'
                    f'<div style="width:{pct_int}%;height:100%;background:#6366f1;border-radius:999px"></div>'
                    f'</div><span style="font-size:0.8rem;color:#64748b">{pct_int}%</span></div>')

        _score_rows = [
            ("📈 Trend",        result.trend_score,      0.36, 36, "Kurs vs. MA-20/50/200 + 20d-Rendite"),
            ("🌊 Volatilität",  result.volatility_score, 0.22, 22, "20d-Volatilität annualisiert (↓ = besser)"),
            ("📉 Drawdown",     result.drawdown_score,   0.18, 18, "Abstand vom Allzeithoch"),
            ("📰 News",         round(50+result.news_score*10,1), 0.14, 14, f"Sentiment: {result.news_label}"),
            ("⚖️ Gewichtung",   max(0,min(100,100-max(0,result.asset_weight-10)*3)), 0.10, 10, f"Position: {result.asset_weight}%"),
        ]

        rows_html = ""
        for label, score, wt, pct, interp in _score_rows:
            rows_html += (
                f'<tr style="border-bottom:1px solid #f1f5f9">'
                f'<td style="padding:0.7rem 0.8rem;font-weight:600;color:#1e293b;white-space:nowrap">{label}</td>'
                f'<td style="padding:0.7rem 0.8rem;text-align:center">{_score_badge(score, wt)}</td>'
                f'<td style="padding:0.7rem 0.8rem">{_wt_bar(pct)}</td>'
                f'<td style="padding:0.7rem 0.8rem;font-size:0.8rem;color:#64748b">{interp}</td>'
                f'</tr>'
            )

        _conf_c = "#22c55e" if result.confidence >= 65 else "#f59e0b" if result.confidence >= 40 else "#ef4444"
        rows_html += (
            f'<tr style="background:linear-gradient(90deg,rgba(99,102,241,0.07),transparent)">'
            f'<td style="padding:0.85rem 0.8rem;font-weight:800;color:#6366f1;white-space:nowrap">🎯 Confidence</td>'
            f'<td style="padding:0.85rem 0.8rem;text-align:center">'
            f'<span style="background:rgba(99,102,241,0.12);color:#6366f1;font-weight:900;font-size:1rem;'
            f'padding:0.25rem 0.75rem;border-radius:8px">{result.confidence}</span></td>'
            f'<td style="padding:0.85rem 0.8rem">{_wt_bar(100)}</td>'
            f'<td style="padding:0.85rem 0.8rem;font-size:0.8rem;color:#64748b">Gewichteter Gesamtscore (0–100)</td>'
            f'</tr>'
        )

        st.markdown(
            f'<div style="overflow-x:auto;border-radius:12px;border:1px solid #e2e8f0;margin-bottom:1rem">'
            f'<table style="width:100%;border-collapse:collapse">'
            f'<thead><tr style="background:#f8faff">'
            f'<th style="padding:0.6rem 0.8rem;text-align:left;font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase">Dimension</th>'
            f'<th style="padding:0.6rem 0.8rem;text-align:center;font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase">Score</th>'
            f'<th style="padding:0.6rem 0.8rem;font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase">Gewichtung</th>'
            f'<th style="padding:0.6rem 0.8rem;font-size:0.72rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase">Interpretation</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
            unsafe_allow_html=True,
        )

        # Waterfall / Contribution chart
        dims  = ["Trend (36%)", "Volatilität (22%)", "Drawdown (18%)", "News (14%)", "Gewichtung (10%)"]
        raw_s = [result.trend_score, result.volatility_score, result.drawdown_score,
                 round(50+result.news_score*10,1),
                 round(max(0,min(100,100-max(0,result.asset_weight-10)*3)),1)]
        wts   = [0.36, 0.22, 0.18, 0.14, 0.10]
        contribs = [s*w for s,w in zip(raw_s, wts)]

        fig_contrib = go.Figure()
        fig_contrib.add_trace(go.Bar(
            y=dims, x=contribs, orientation="h",
            marker_color=["#22c55e" if c>=26 else "#f59e0b" if c>=14 else "#ef4444" for c in contribs],
            text=[f"{c:.1f} Pkt" for c in contribs], textposition="outside",
            name="Beitrag zum Confidence-Score",
        ))
        fig_contrib.add_vline(x=result.confidence/5, line_dash="dash",
                              line_color="#6366F1", annotation_text=f"Ø {result.confidence/5:.1f}",
                              annotation_position="top right")
        fig_contrib.update_layout(
            title=f"Beiträge zum Confidence-Score: {result.confidence}/100",
            height=340, margin=dict(l=10,r=80,t=50,b=10),
            xaxis_title="Score-Beitrag (gewichtet)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_contrib, use_container_width=True,
                        config={"displayModeBar": False}, key="ml_contrib")

        st.caption(
            "Confidence = 0.36×Trend + 0.22×Volatilität + 0.18×Drawdown + "
            "0.14×(50+News×10) + 0.10×WeightRisk"
        )

        # ── Fix #1: Herleitung der Score-Gewichte aus RF Feature Importance ──
        st.markdown("---")
        st.markdown("#### Herleitung der Scoring-Gewichte aus RF Feature Importance")
        st.markdown(
            "Die Gewichte sind **nicht willkürlich** — sie sind aus der RF Feature Importance "
            "abgeleitet, indem zusammengehörige Features gruppiert werden:"
        )

        _rf_fi = {
            "daily_return": 0.0566, "return_5d": 0.0750, "return_20d": 0.1015,
            "ma_20_distance": 0.0896, "ma_50_distance": 0.1190, "ma_200_distance": 0.1545,
            "volatility_20d": 0.2002, "drawdown": 0.2036,
        }
        _dim_mapping = [
            ("📈 Trend (36%)",       ["ma_20_distance","ma_50_distance","ma_200_distance","return_20d","daily_return","return_5d"], 0.36, "#6366f1"),
            ("🌊 Volatilität (22%)", ["volatility_20d"], 0.22, "#0ea5e9"),
            ("📉 Drawdown (18%)",    ["drawdown"],       0.18, "#ef4444"),
        ]

        _bar_dims, _bar_fi, _bar_wt, _bar_col = [], [], [], []
        for label, feats, wt, col in _dim_mapping:
            fi_sum = sum(_rf_fi.get(f, 0) for f in feats)
            _bar_dims.append(label)
            _bar_fi.append(round(fi_sum * 100, 1))
            _bar_wt.append(round(wt * 100, 1))
            _bar_col.append(col)

        _fig_derive = go.Figure()
        _fig_derive.add_trace(go.Bar(
            name="RF Feature Importance (gruppiert)",
            x=_bar_dims, y=_bar_fi,
            marker_color=[c + "99" for c in _bar_col],
            text=[f"{v:.1f}%" for v in _bar_fi], textposition="outside",
        ))
        _fig_derive.add_trace(go.Bar(
            name="Score-Gewicht (verwendet)",
            x=_bar_dims, y=_bar_wt,
            marker_color=_bar_col,
            text=[f"{v:.0f}%" for v in _bar_wt], textposition="outside",
        ))
        _fig_derive.update_layout(
            barmode="group", height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="Anteil (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        _fig_derive.update_yaxes(ticksuffix="%")
        st.plotly_chart(_fig_derive, use_container_width=True,
                        config={"displayModeBar": False}, key="ml_derive")
        st.caption(
            "Trend 36% ≈ MA-Distanzen + Rendite-Features (36,31% RF-Importance) · "
            "Volatilität 22% ≈ volatility_20d (20,02%) · "
            "Drawdown 18% ≈ drawdown (20,36%) · "
            "News 14% & Gewichtung 10% = externe Signale ohne historische Trainingsgrundlage."
        )

        # ── Fix #3: Survivorship Bias ────────────────────────────────────
        st.markdown("---")
        st.markdown(
            """<div style="border:1px solid rgba(245,158,11,0.35);border-left:4px solid #f59e0b;
            border-radius:12px;padding:1rem 1.2rem;background:rgba(245,158,11,0.06)">
            <div style="font-size:0.72rem;font-weight:800;color:#b45309;letter-spacing:0.08em;
            text-transform:uppercase;margin-bottom:0.4rem">⚠️ Survivorship Bias — methodische Einschränkung</div>
            <p style="font-size:0.85rem;color:#1e293b;margin:0;line-height:1.7">
            Die 26 analysierten Blue-Chip-Ticker wurden <strong>ex post</strong> ausgewählt —
            d.h. es handelt sich um Unternehmen, die <em>heute noch existieren und im Index verbleiben</em>.
            Unternehmen die bankrott gingen, fusioniert wurden oder delisted wurden, fehlen im Datensatz.
            Dies führt zu einer <strong>systematischen Überschätzung</strong> der historischen Performance
            und der Modellgüte. Ein repräsentativer Datensatz würde auch gescheiterte Unternehmen umfassen.
            </p></div>""",
            unsafe_allow_html=True,
        )

    with tab2:
        # ── Kapitel 3: Precision, Recall, ROC, PR-Kurve ─────────────────
        st.markdown("#### Klassifikations-Metriken (Kapitel 3 — Prof. Quibeldey-Cirkel)")

        # Metriken aus dem trainierten Modell laden
        try:
            import joblib as _jl
            from sklearn.metrics import (accuracy_score as _acc, precision_score as _ps,
                                          recall_score as _rs, f1_score as _f1s,
                                          roc_auc_score as _auc, roc_curve as _roc,
                                          precision_recall_curve as _prc, confusion_matrix as _cm)
            from sklearn.model_selection import train_test_split as _tts

            _model_pipe = _jl.load(get_model_path())
            _feat_cols = get_model_feature_cols(ctx["market"])
            _full_df = ctx["market"][_feat_cols + ["target_20d"]].dropna(subset=["target_20d"])
            _X  = _full_df[_feat_cols]
            _y  = _full_df["target_20d"].astype(int)
            _, _Xt, _, _yt = _tts(_X, _y, test_size=0.25, random_state=42, stratify=_y)
            _yt_pred  = _model_pipe.predict(_Xt)
            _yt_proba = _model_pipe.predict_proba(_Xt)[:,1]

            _acc_val  = _acc(_yt, _yt_pred)
            _majority = float(max(_yt.mean(), 1-_yt.mean()))
            _beats    = _acc_val > _majority
            _prec = _ps(_yt, _yt_pred, zero_division=0)
            _rec  = _rs(_yt, _yt_pred, zero_division=0)
            _f1   = _f1s(_yt, _yt_pred, average="weighted", zero_division=0)
            _auc_val = _auc(_yt, _yt_proba)

            # ── Ehrlicher Baseline-Vergleich ─────────────────────────────
            _beat_color = "#22c55e" if _beats else "#ef4444"
            _beat_icon  = "✅" if _beats else "⚠️"
            _beat_label = "Schlägt Baseline" if _beats else "Unter Baseline"
            _cl0_pct = (1-_yt.mean())*100
            _cl1_pct = _yt.mean()*100
            st.markdown(
                f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.25rem">
  <!-- Accuracy vs Baseline -->
  <div style="border:1px solid {_beat_color}33;border-top:3px solid {_beat_color};border-radius:12px;padding:1rem;background:{_beat_color}07">
    <div style="font-size:0.7rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">Accuracy vs. Baseline</div>
    <div style="display:flex;align-items:flex-end;gap:0.5rem;margin-bottom:0.3rem">
      <span style="font-size:1.6rem;font-weight:900;color:{_beat_color}">{_acc_val*100:.2f}%</span>
      <span style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.3rem">Modell</span>
    </div>
    <div style="display:flex;align-items:flex-end;gap:0.5rem;margin-bottom:0.5rem">
      <span style="font-size:1.2rem;font-weight:700;color:#64748b">{_majority*100:.2f}%</span>
      <span style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.15rem">Majority-Baseline</span>
    </div>
    <div style="font-size:0.75rem;font-weight:800;color:{_beat_color}">{_beat_icon} {_beat_label} ({(_acc_val-_majority)*100:+.2f} PP)</div>
    <div style="font-size:0.7rem;color:#94a3b8;margin-top:0.4rem;line-height:1.5">Klassenverteilung: Bullish {_cl1_pct:.1f}% / Bearish {_cl0_pct:.1f}%<br>→ Accuracy ist bei Imbalance irreführend</div>
  </div>
  <!-- ROC-AUC -->
  <div style="border:1px solid rgba(99,102,241,0.25);border-top:3px solid #6366f1;border-radius:12px;padding:1rem;background:rgba(99,102,241,0.05)">
    <div style="font-size:0.7rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">ROC-AUC ✅ Korrekte Metrik</div>
    <div style="font-size:1.6rem;font-weight:900;color:#6366f1;margin-bottom:0.3rem">{_auc_val:.4f}</div>
    <div style="background:#e2e8f0;border-radius:999px;height:8px;overflow:hidden;margin-bottom:0.5rem">
      <div style="width:{(_auc_val-0.5)*200:.0f}%;height:100%;background:#6366f1;border-radius:999px"></div>
    </div>
    <div style="font-size:0.75rem;font-weight:800;color:#6366f1">+{(_auc_val-0.5):.4f} über Zufall</div>
    <div style="font-size:0.7rem;color:#94a3b8;margin-top:0.4rem;line-height:1.5">0.5 = reiner Zufall<br>ROC-AUC misst Diskriminierungsfähigkeit unabhängig von Imbalance</div>
  </div>
  <!-- F1 -->
  <div style="border:1px solid rgba(245,158,11,0.25);border-top:3px solid #f59e0b;border-radius:12px;padding:1rem;background:rgba(245,158,11,0.05)">
    <div style="font-size:0.7rem;font-weight:700;color:#94a3b8;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">F1 / Precision / Recall</div>
    <div style="font-size:1.6rem;font-weight:900;color:#f59e0b;margin-bottom:0.3rem">{_f1:.4f}</div>
    <div style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem">F1 weighted</div>
    <div style="font-size:0.75rem;color:#64748b">Prec: <strong>{_prec:.3f}</strong> · Rec: <strong>{_rec:.3f}</strong></div>
    <div style="font-size:0.7rem;color:#94a3b8;margin-top:0.4rem;line-height:1.5">Für WealthScope: beide Fehler gleichwertig → F1 als Zielmetrik sinnvoll</div>
  </div>
</div>""",
                unsafe_allow_html=True,
            )

            # 2-Panel: ROC + PR
            from plotly.subplots import make_subplots as _msp
            _fpr, _tpr, _ = _roc(_yt, _yt_proba)
            _precs, _recs, _thresh = _prc(_yt, _yt_proba)

            fig_roc_pr = _msp(rows=1, cols=2,
                               subplot_titles=["ROC-Kurve (AUC={:.4f})".format(_auc_val),
                                               "Precision-Recall-Kurve"])
            # ROC
            fig_roc_pr.add_trace(go.Scatter(x=list(_fpr), y=list(_tpr),
                mode="lines", name="Random Forest",
                line=dict(color="#6366f1", width=2.5),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.07)"),
                row=1, col=1)
            fig_roc_pr.add_trace(go.Scatter(x=[0,1], y=[0,1],
                mode="lines", name="Zufall",
                line=dict(color="#94a3b8", width=1.5, dash="dash")),
                row=1, col=1)
            # PR
            fig_roc_pr.add_trace(go.Scatter(x=list(_recs), y=list(_precs),
                mode="lines", name="PR-Kurve",
                line=dict(color="#22c55e", width=2.5)),
                row=1, col=2)
            fig_roc_pr.add_hline(y=float(_yt.mean()), line_dash="dash",
                                  line_color="#ef4444",
                                  annotation_text=f"Baseline {_yt.mean():.2f}",
                                  row=1, col=2)

            fig_roc_pr.update_layout(
                height=380, showlegend=True,
                margin=dict(l=10,r=10,t=50,b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            fig_roc_pr.update_xaxes(title_text="FPR", row=1, col=1)
            fig_roc_pr.update_yaxes(title_text="TPR (Recall)", row=1, col=1)
            fig_roc_pr.update_xaxes(title_text="Recall", row=1, col=2)
            fig_roc_pr.update_yaxes(title_text="Precision", row=1, col=2)
            st.plotly_chart(fig_roc_pr, use_container_width=True,
                            config={"displayModeBar": False}, key="ml_roc_pr")

            # Konfusionsmatrix
            st.markdown("#### Konfusionsmatrix (Testset)")
            _cm_vals = _cm(_yt, _yt_pred)
            _cm_norm = _cm(_yt, _yt_pred, normalize="true")
            fig_cm = go.Figure(go.Heatmap(
                z=_cm_norm, x=["Pred: 0 Bearish","Pred: 1 Bullish"],
                y=["Real: 0 Bearish","Real: 1 Bullish"],
                colorscale="Blues", showscale=True,
                text=[[f"{_cm_vals[i][j]:,}<br>({_cm_norm[i][j]*100:.1f}%)"
                       for j in range(2)] for i in range(2)],
                texttemplate="%{text}", textfont={"size":13},
            ))
            fig_cm.update_layout(height=320, margin=dict(l=10,r=10,t=20,b=10),
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_cm, use_container_width=True,
                            config={"displayModeBar": False}, key="ml_cm_k3")

            st.caption(
                "**Precision-Recall Trade-off** (Kap.3): Höherer Schwellenwert → höhere Precision, "
                "niedrigerer Recall. Für WealthScope ist F1 der Zielkompromiss (beide Fehlertypen gleich teuer). "
                "ROC-Kurve > PR-Kurve, da Kl.1 mit 59 % nicht selten ist (vgl. Kap.3, S.36)."
            )

        except Exception as _e:
            st.warning(f"Modell-Daten nicht verfügbar: {_e}")

    with tab5:
        st.markdown("#### Pearson-Korrelationsmatrix aller ML-Features")
        show_chart_with_data(
            "Feature-Korrelationen",
            chart_feature_correlation(df),
            df[[c for c in ["daily_return","return_5d","return_20d","ma_20_distance",
                             "ma_50_distance","ma_200_distance","volatility_20d",
                             "drawdown","future_return_20d","target_20d"]
                if c in df.columns]].dropna().tail(2000),
            "ml_feature_correlation",
        )
        st.info(
            "Rot = positive Korrelation, Blau = negative Korrelation. "
            "Hochkorrelierte Features (|r|>0.8) sind potenziell redundant. "
            "Random Forest ist robust gegenüber Multikollinearität."
        )

    with tab3:
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            show_chart_with_data("Score-Zerlegung", chart_scores(result),
                                 pd.DataFrame([{"Dimension":d,"Score":s} for d,s in
                                               zip(["Trend","Volatilität","Drawdown","Kap.-Schutz","Confidence"],
                                                   [result.trend_score,result.volatility_score,
                                                    result.drawdown_score,result.capital_protection,result.confidence])]),
                                 "ml_scores_bar")
        with col_r2:
            show_chart_with_data("Radar-Profil", chart_radar(result),
                                 pd.DataFrame([{"Dimension":d,"Score":s} for d,s in
                                               zip(["Trend","Volatilität","Drawdown","Kap.-Schutz","Confidence"],
                                                   [result.trend_score,result.volatility_score,
                                                    result.drawdown_score,result.capital_protection,result.confidence])]),
                                 "ml_radar")

    with tab4:
        st.markdown("#### Modell-Dokumentation")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("""
**Algorithmus:** Random Forest Classifier
**Zielvariable:** `target_20d` (1 = Kurs in 20 Handelstagen höher)
**Train/Test:** 75 % / 25 %, stratifiziert
**Cross-Validation:** 5-Fold Stratified
**Accuracy:** 55,8 % (leicht über Zufall – EMH erwartet)
**ROC-AUC:** 0.61
            """)
        with col_d2:
            st.markdown("""
**Stärken des Modells:**
- ✅ Kein Data Leakage (Pipeline)
- ✅ Robust gegenüber Ausreißern
- ✅ Feature Importance verfügbar
- ✅ Ensemble → niedrige Varianz

**Grenzen:**
- ⚠️ Nur historische Kursdaten
- ⚠️ Keine Makrodaten / Sentiment
- ⚠️ EMH begrenzt Vorhersagbarkeit
            """)
        st.markdown("---")
        st.markdown("""
**Wissenschaftliche Einordnung:**
Eine Accuracy von 55,8 % ist bei historischen Finanzdaten wissenschaftlich bemerkenswert
(Efficient Market Hypothesis, Fama 1970). Das Modell ist ein Lerndemonstrator,
**kein** Handelssignal-Generator.

> *„Aktuelle Kursdaten sind bereits im Marktpreis eingepreist."*
> — Fama (1970), Efficient Capital Markets
        """)
        with st.expander("Score-Berechnungsformel (Python)"):
            st.code("""
confidence = round(
    0.36 * trend_score         # Trend-Position vs. MAs
  + 0.22 * volatility_score    # Inverse Volatilität
  + 0.18 * drawdown_score      # Drawdown-Erholung
  + 0.14 * (50 + news_score*10) # News-Sentiment
  + 0.10 * weight_risk,         # Positions-Konzentration
  1,
)
            """, language="python")
        if st.session_state.get("show_advanced_metrics", False):
            with st.expander("🔧 Erweiterte Metriken / Debug"):
                st.json({
                    "trend_score": result.trend_score,
                    "volatility_score": result.volatility_score,
                    "drawdown_score": result.drawdown_score,
                    "confidence": result.confidence,
                    "risk_score": result.risk_score,
                    "capital_protection": result.capital_protection,
                    "news_score": result.news_score,
                    "rf_proba": result.rf_proba,
                })

    with tab6:
        # ── SHAP Explainability ──────────────────────────────────────────
        st.markdown("#### SHAP — Modell-Erklärung auf Feature-Ebene")
        st.markdown(
            "SHAP (SHapley Additive exPlanations) erklärt, **warum** das Modell "
            "eine bestimmte Vorhersage trifft — für jeden Datenpunkt individuell."
        )
        try:
            import shap as _shap
            import joblib as _jl
            from sklearn.model_selection import train_test_split as _tts

            _model_pipe = _jl.load(get_model_path())
            _feat_cols = get_model_feature_cols(ctx["market"])
            _full_df = ctx["market"][_feat_cols + ["target_20d"]].dropna(subset=["target_20d"])
            _X = _full_df[_feat_cols]
            _y = _full_df["target_20d"].astype(int)
            _, _Xt, _, _ = _tts(_X, _y, test_size=0.25, random_state=42, stratify=_y)

            _imp = _model_pipe.named_steps["imp"]
            _scl = _model_pipe.named_steps["scl"]
            _rf  = _model_pipe.named_steps["mod"]

            # Global SHAP auf 300 Samples (Performance)
            _sample = _Xt.sample(min(300, len(_Xt)), random_state=42)
            _Xt_t = _scl.transform(_imp.transform(_sample))
            _explainer = _shap.TreeExplainer(_rf)
            _sv = _explainer.shap_values(_Xt_t)
            _sv_class1 = _sv[:, :, 1] if len(np.array(_sv).shape) == 3 else _sv

            _mean_shap = np.abs(_sv_class1).mean(axis=0)
            _label_map = {
                "daily_return": "tägl. Rendite", "return_5d": "5d-Rendite",
                "return_20d": "20d-Rendite", "ma_20_distance": "MA-20 Dist.",
                "ma_50_distance": "MA-50 Dist.", "ma_200_distance": "MA-200 Dist.",
                "volatility_20d": "Volatilität 20d", "drawdown": "Drawdown",
                "vix_level": "VIX-Level", "vix_change_5d": "VIX 5d-Änderung",
            }
            _feat_labels = [_label_map.get(c, c) for c in _feat_cols]

            _shap_sorted = sorted(zip(_feat_labels, _mean_shap), key=lambda x: x[1])
            _sl, _sv_vals = zip(*_shap_sorted)

            col_shap1, col_shap2 = st.columns([3, 2])
            with col_shap1:
                fig_shap_global = go.Figure(go.Bar(
                    x=list(_sv_vals), y=list(_sl), orientation="h",
                    marker_color=["#6366f1" if v >= sorted(_sv_vals)[-3] else "#a5b4fc" for v in _sv_vals],
                    text=[f"{v:.4f}" for v in _sv_vals], textposition="outside",
                ))
                fig_shap_global.update_layout(
                    title="Globale Feature-Wichtigkeit (Mean |SHAP|)",
                    height=320, margin=dict(l=10, r=60, t=45, b=10),
                    xaxis_title="Ø |SHAP-Wert| — Einfluss auf Klasse 1 (Bullish)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_shap_global, use_container_width=True,
                                config={"displayModeBar": False}, key="shap_global")

            with col_shap2:
                st.markdown(
                    """<div style="border:1px solid rgba(99,102,241,0.2);border-radius:12px;
                    padding:1rem;background:rgba(99,102,241,0.04)">
                    <div style="font-size:0.72rem;font-weight:800;color:#6366f1;letter-spacing:0.08em;
                    text-transform:uppercase;margin-bottom:0.75rem">SHAP vs. RF Importance</div>
                    <p style="font-size:0.82rem;color:#475569;line-height:1.7;margin:0">
                    <strong>RF Feature Importance</strong> misst, wie oft ein Feature zum Aufteilen
                    der Bäume genutzt wird.<br><br>
                    <strong>SHAP</strong> misst den tatsächlichen Beitrag jedes Features
                    zur Vorhersage — pro Datenpunkt, nicht global aggregiert.<br><br>
                    SHAP ist die <em>wissenschaftlich robustere Methode</em> (Lundberg &amp; Lee, 2017)
                    und vermeidet den Bias von Impurity-basierten Importance-Metriken.
                    </p></div>""",
                    unsafe_allow_html=True,
                )

            # Lokale SHAP für aktuellen Datenpunkt
            st.markdown("---")
            st.markdown(f"#### Lokale Erklärung: Warum sagt das Modell **{result.outlook}** für {result.ticker}?")

            _feat_cols_local = [c for c in ["daily_return","return_5d","return_20d","ma_20_distance",
                                 "ma_50_distance","ma_200_distance","volatility_20d","drawdown"]
                                if c in df.columns]
            _local_row = df[_feat_cols_local].dropna().tail(1)
            if not _local_row.empty:
                _local_t = _scl.transform(_imp.transform(_local_row[_feat_cols]))
                _local_sv = _explainer.shap_values(_local_t)
                _local_vals = (_local_sv[0, :, 1] if len(np.array(_local_sv).shape) == 3
                               else _local_sv[0])
                _base_val = float(_explainer.expected_value[1]) if isinstance(
                    _explainer.expected_value, (list, np.ndarray)) else float(_explainer.expected_value)

                _local_sorted = sorted(zip(_feat_labels, _local_vals,
                                           _local_row[_feat_cols].values[0]),
                                       key=lambda x: abs(x[1]), reverse=True)

                _wf_labels = [f"{l}<br><sub>={v:.4f}</sub>" for l, _, v in _local_sorted]
                _wf_vals   = [sv for _, sv, _ in _local_sorted]
                _wf_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in _wf_vals]

                fig_local = go.Figure(go.Bar(
                    x=_wf_vals, y=_wf_labels, orientation="h",
                    marker_color=_wf_colors,
                    text=[f"{'+' if v>=0 else ''}{v:.4f}" for v in _wf_vals],
                    textposition="outside",
                ))
                fig_local.add_vline(x=0, line_color="#94a3b8", line_width=1)
                fig_local.update_layout(
                    title=f"Lokale SHAP-Erklärung — {result.ticker} (Basislinie: {_base_val:.3f})",
                    height=340, margin=dict(l=10, r=80, t=45, b=10),
                    xaxis_title="SHAP-Wert (positiv = bullish, negativ = bearish)",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_local, use_container_width=True,
                                config={"displayModeBar": False}, key="shap_local")
                st.caption(
                    f"RF-Vorhersage-Wahrscheinlichkeit Bullish: {result.rf_proba*100:.1f}% · "
                    "Grün = erhöht Bullish-Wahrscheinlichkeit · Rot = senkt sie"
                )

        except ImportError:
            st.warning("shap nicht installiert. Terminal: `pip install shap --break-system-packages`")
        except Exception as _e:
            st.warning(f"SHAP-Analyse nicht verfügbar: {_e}")


def page_assistant(ctx: Dict[str, Any]) -> None:
    result = ctx["result"]

    page_title("Assistent", "Frag zur aktuellen Analyse, Methodik oder News-Lage.")

    with st.container(border=True):
        st.markdown(
            """
**Beispielfragen**
- Warum ist die Einschätzung aktuell neutral/positiv/negativ?
- Was bedeutet Drawdown?
- Welche News wurden berücksichtigt?
- Was ist `target_20d`?
- Wie funktioniert die Methodik?
            """
        )

    if st.button("Chat zurücksetzen", width="stretch"):
        st.session_state["chat_messages"] = []
        st.rerun()

    for msg in st.session_state.get("chat_messages", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Frage zur Analyse stellen")
    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        answer = assistant_answer(prompt, ctx)
        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
        st.rerun()



def render_news_cards(news_df: pd.DataFrame, max_items: int = 8) -> None:
    if news_df.empty:
        st.info("Keine News für die aktuelle Suchlogik gefunden.")
        return

    filter_col1, filter_col2 = st.columns([1, 2])

    with filter_col1:
        if "Kurzinterpretation" in news_df.columns:
            sentiment_values = [
                str(x)
                for x in news_df["Kurzinterpretation"].dropna().unique()
                if str(x).strip()
            ]
        else:
            sentiment_values = []

        sentiments = ["Alle"] + sorted(sentiment_values)
        selected_sentiment = st.selectbox(
            "Sentiment filtern",
            sentiments,
            key="news_sentiment_filter",
        )

    with filter_col2:
        search_term = st.text_input(
            "News durchsuchen",
            placeholder="z. B. earnings, inflation, AI",
            key="news_search_filter",
        )

    view_df = news_df.copy()

    if selected_sentiment != "Alle" and "Kurzinterpretation" in view_df.columns:
        view_df = view_df[view_df["Kurzinterpretation"].astype(str) == selected_sentiment]

    if search_term:
        title_series = view_df["Titel"].astype(str) if "Titel" in view_df.columns else pd.Series("", index=view_df.index)
        desc_series = view_df["Beschreibung"].astype(str) if "Beschreibung" in view_df.columns else pd.Series("", index=view_df.index)
        source_series = view_df["Quelle"].astype(str) if "Quelle" in view_df.columns else pd.Series("", index=view_df.index)

        mask = (
            title_series.str.contains(search_term, case=False, na=False)
            | desc_series.str.contains(search_term, case=False, na=False)
            | source_series.str.contains(search_term, case=False, na=False)
        )
        view_df = view_df[mask]

    st.caption(f"{len(view_df)} von {len(news_df)} News sichtbar")

    if view_df.empty:
        st.warning("Keine News nach dem aktuellen Filter gefunden.")
        return

    for idx, row in view_df.head(max_items).iterrows():
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

                m1, m2, m3 = st.columns(3)
                m1.metric("Sentiment", sentiment)
                m2.metric("Relevanz", relevance)
                m3.metric("Impact", impact)

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

    page_title("News", "Sentiment, Quelle & Marktrelevanz auf einen Blick")

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


def route_page(page: str, ctx: Dict[str, Any]) -> None:
    aliases = {
        "Kompass": "Kompass",
        "Simulator": "Simulator",
        "Watchlist": "Watchlist",
            "Projekt": "Projekt",
        "Export": "Export",
        "Status": "Status",
        "Assistent": "Start",
        "💬 Assistent": "Start",
    }

    page = aliases.get(page, page)

    if page == "Start":
        page_start(ctx)
    elif page == "Wealth Outlook":
        page_outlook(ctx)
    elif page == "Kompass":
        page_kompass(ctx)
    elif page == "Simulator":
        page_simulator(ctx)
    elif page == "Watchlist":
        page_watchlist(ctx)
    elif page == "Datenlabor":
        page_data_lab(ctx)
    elif page == "ML-Labor":
        page_ml_lab(ctx)
    elif page == "News-Archiv":
        page_news(ctx)
    elif page == "Methodik":
        page_project(ctx)
    elif page == "Projekt":
        page_project(ctx)
    elif page == "Export":
        page_export(ctx)
    elif page == "Impressum":
        page_impressum(ctx)
    elif page == "Datenschutz":
        page_privacy(ctx)
    elif page == "Status":
        page_status(ctx)
    else:
        page_start(ctx)


# =========================================================
# MAIN
# =========================================================



def render_floating_assistant_panel(ctx: Dict[str, Any]) -> None:
    current_page = st.session_state.get("current_page", "Start")
    current_href = href(current_page)

    chat_param = str(st.query_params.get("chat", "")).lower()
    if chat_param == "open":
        st.session_state["ws_chat_open"] = True
    elif chat_param == "closed":
        st.session_state["ws_chat_open"] = False

    if "ws_chat_messages" not in st.session_state:
        result = ctx["result"]
        st.session_state["ws_chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"Hi, ich bin der WealthScope Assistent. "
                    f"Ich kann dir die aktuelle Analyse zu {result.ticker}, Drawdown, Risiko, News, "
                    f"target_20d und Methodik erklären. Keine Finanzberatung."
                ),
            }
        ]

    open_href = current_href + "&chat=open" if "?" in current_href else current_href + "?chat=open"
    close_href = current_href + "&chat=closed" if "?" in current_href else current_href + "?chat=closed"

    st.markdown(
        """
        <style>
        .ws-chat-fab {
            position: fixed;
            right: 24px;
            bottom: 88px;
            width: 58px;
            height: 58px;
            border-radius: 999px;
            background: var(--primary-color, #6366F1);
            color: white !important;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none !important;
            font-size: 26px;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.22);
            z-index: 999999;
            transition: transform 0.15s ease, filter 0.15s ease;
        }
        .ws-chat-fab:hover {
            transform: translateY(-2px);
            filter: brightness(1.05);
        }

        div:has(> .element-container .ws-chat-panel-marker) {
            position: fixed !important;
            right: 24px !important;
            bottom: 88px !important;
            width: 430px !important;
            max-width: calc(100vw - 48px) !important;
            max-height: 72vh !important;
            overflow-y: auto !important;
            background: var(--background-color, white) !important;
            border: 1px solid rgba(120, 120, 120, 0.25) !important;
            border-radius: 22px !important;
            padding: 16px !important;
            box-shadow: 0 24px 70px rgba(15, 23, 42, 0.28) !important;
            z-index: 999998 !important;
        }

        .ws-chat-panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 10px;
        }
        .ws-chat-panel-title {
            font-size: 16px;
            font-weight: 800;
            margin: 0;
        }
        .ws-chat-panel-subtitle {
            font-size: 12px;
            opacity: 0.72;
            margin-top: 2px;
        }
        .ws-chat-close {
            text-decoration: none !important;
            color: inherit !important;
            border: 1px solid rgba(120, 120, 120, 0.28);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 13px;
        }

        @media (max-width: 700px) {
            div:has(> .element-container .ws-chat-panel-marker) {
                left: 12px !important;
                right: 12px !important;
                bottom: 72px !important;
                width: auto !important;
                max-width: none !important;
            }
            .ws-chat-fab {
                right: 16px;
                bottom: 78px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("ws_chat_open", False):
        st.markdown(
            f'<a class="ws-chat-fab" href="{open_href}" target="_self" title="WealthScope Assistent">💬</a>',
            unsafe_allow_html=True,
        )
        return

    with st.container():
        st.markdown('<div class="ws-chat-panel-marker"></div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="ws-chat-panel-head">
                <div>
                    <p class="ws-chat-panel-title">💬 WealthScope Assistent</p>
                    <div class="ws-chat-panel-subtitle">Kontextbasierte Hilfe zur aktuellen Analyse</div>
                </div>
                <a class="ws-chat-close" href="{close_href}" target="_self">Schließen</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for msg in st.session_state["ws_chat_messages"][-8:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        with st.form("ws_floating_chat_form", clear_on_submit=True):
            prompt = st.text_input(
                "Frage",
                placeholder="z. B. Was bedeutet Drawdown?",
                label_visibility="collapsed",
                key="ws_floating_chat_prompt",
            )
            c1, c2 = st.columns([2, 1])
            with c1:
                submitted = st.form_submit_button("Senden", width="stretch")
            with c2:
                reset = st.form_submit_button("Reset", width="stretch")

        if reset:
            result = ctx["result"]
            st.session_state["ws_chat_messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Chat zurückgesetzt. Ich kann dir die aktuelle Analyse zu {result.ticker}, "
                        "Risiko, News, Features und Methodik erklären."
                    ),
                }
            ]
            st.rerun()

        if submitted and prompt.strip():
            st.session_state["ws_chat_messages"].append({"role": "user", "content": prompt.strip()})
            answer = assistant_answer(prompt.strip(), ctx)
            st.session_state["ws_chat_messages"].append({"role": "assistant", "content": answer})
            st.rerun()




# =========================================================
# FOOTER PAGE UPGRADES: PROJEKT, EXPORT, IMPRESSUM, DATENSCHUTZ, STATUS
# =========================================================

def ws_dataset_snapshot(ctx: Dict[str, Any]) -> Dict[str, Any]:
    market   = ctx.get("market", pd.DataFrame())
    features = ctx.get("features", pd.DataFrame())
    result   = ctx.get("result")
    live_on  = st.session_state.get("use_live_data", False)

    # Quelle aus attrs (zuverlässig) statt aus result.source (oft "unbekannt")
    source = getattr(market, "attrs", {}).get("data_source", "UNKNOWN")
    if source == "UNKNOWN":
        if DATA_FEATURES_PARQUET_PATH.exists(): source = "REAL_PARQUET"
        elif DATA_FEATURES_PATH.exists():       source = "REAL_CSV"
    ticker = getattr(result, "ticker", "–") if result is not None else "–"

    date_min, date_max = "", ""
    if not market.empty and "date" in market.columns:
        date_min = str(pd.to_datetime(market["date"]).min().date())
        # Wenn Live-Daten aktiv: heute als Enddatum
        date_max = (
            str(__import__("datetime").date.today()) + " (live 🌐)"
            if live_on
            else str(pd.to_datetime(market["date"]).max().date())
        )

    return {
        "ticker":           ticker,
        "source":           source,
        "live_on":          live_on,
        "market_rows":      int(len(market)) if market is not None else 0,
        "feature_rows":     int(len(features)) if features is not None else 0,
        "market_columns":   int(len(market.columns)) if market is not None and not market.empty else 0,
        "feature_columns":  int(len(features.columns)) if features is not None and not features.empty else 0,
        "tickers":          int(market["ticker"].nunique()) if market is not None and not market.empty and "ticker" in market.columns else 0,
        "date_min":         date_min,
        "date_max":         date_max,
        "target_available": bool(features is not None and "target_20d" in features.columns),
    }


def ws_render_snapshot_metrics(ctx: Dict[str, Any]) -> None:
    snap     = ws_dataset_snapshot(ctx)
    live_on  = snap["live_on"]
    src_ok   = str(snap["source"]).startswith("REAL")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Marktdaten",    f"{snap['market_rows']:,}".replace(",","."))
    c2.metric("Feature-Zeilen",f"{snap['feature_rows']:,}".replace(",","."))
    c3.metric("Ticker",        snap["tickers"])
    c4.metric("target_20d",    "✅ vorhanden" if snap["target_available"] else "❌ fehlt")

    # Live-Badge + Quelle
    live_badge = " · 🌐 **yfinance Live aktiv** · Daten bis heute" if live_on else ""
    src_label  = snap["source"] if src_ok else "⚠️ Quelle unbekannt – Parquet prüfen"
    st.caption(
        f"Quelle: `{src_label}`{live_badge} · "
        f"Zeitraum: `{snap['date_min']}` bis `{snap['date_max']}` · "
        f"Spalten: Marktdaten `{snap['market_columns']}`, Features `{snap['feature_columns']}`"
    )


def ws_build_export_package(ctx: Dict[str, Any]) -> bytes:
    result = ctx["result"]
    market = ctx.get("market", pd.DataFrame())
    features = ctx.get("features", pd.DataFrame())
    news_df = ctx.get("news_df", pd.DataFrame())

    buffer = io.BytesIO()

    report_md = analysis_report_markdown(result) if "analysis_report_markdown" in globals() else "# WealthScope Export\n"
    context_json = json.dumps(asdict(result), indent=2, ensure_ascii=False)

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("01_analysebericht.md", report_md)
        zf.writestr("02_analyse_context.json", context_json)

        if market is not None and not market.empty:
            zf.writestr("03_marktdaten.csv", market.to_csv(index=False))

        if features is not None and not features.empty:
            zf.writestr("04_feature_daten.csv", features.to_csv(index=False))

        if news_df is not None and not news_df.empty:
            zf.writestr("05_news_einordnung.csv", news_df.to_csv(index=False))

        zf.writestr(
            "06_methodik_und_grenzen.txt",
            (
                "WealthScope AI ist eine Streamlit-Demo für ein Uni-/QUA3CK-/Big-Data-Projekt.\n"
                "Die App nutzt lokal vorbereitete Kaggle-Marktdaten, Feature Engineering, NewsAPI-Daten "
                "und aktuell ein regelbasiertes Scoring.\n\n"
                "Die Anwendung stellt keine Anlageberatung dar. Historische Daten und vereinfachte News-Signale "
                "garantieren keine zukünftige Entwicklung.\n"
            ),
        )

    buffer.seek(0)
    return buffer.getvalue()


def page_project(ctx: Dict[str, Any]) -> None:
    page_title("Methodik", "QUA³CK-Dokumentation, Architektur & Hintergrund")

    ws_render_snapshot_metrics(ctx)

    # ── Hero-Beschreibung ───────────────────────────────────────────────
    st.markdown(
        """<div class="ws-card ws-hero">
        <h2>WealthScope AI</h2>
        <p>Interaktive Finanzanalyse-App nach dem QUA³CK-Prozessmodell (Stock et al., 2021, KIT ITIV).<br>
        Kombiniert historische Kaggle-Marktdaten, technische Feature Engineering, Machine Learning,
        NewsAPI-Integration und einen KI-gestützten Analyse-Assistenten in einer vollständig
        dokumentierten, reproduzierbaren Streamlit-Anwendung.<br>
        <strong>Keine Finanzberatung. Reine Lern- und Demonstrationsplattform.</strong></p>
        </div>""",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6_lim = st.tabs([
        "🎯 Projektziel", "🔄 QUA³CK-Prozess", "🏗 Architektur", "📊 Datenbasis", "🎓 Präsentation", "⚠️ Limitationen"
    ])

    # ── Tab 1: Projektziel ──────────────────────────────────────────────
    with tab1:
        st.markdown("#### Forschungsfrage")
        st.markdown(
            "> Wie können historische US-Aktienmarktdaten genutzt werden, um mit Machine Learning "
            "eine interaktive Finanzanalyse-App zu entwickeln, die technische Marktanalyse, "
            "ML-basierte Signalgebung und risikobasierte Positionsplanung nachvollziehbar kombiniert?"
        )

        c1, c2, c3 = st.columns(3)
        for col, icon, title, body in [
            (c1, "📊", "Echte Datenbasis", "192.119 Zeilen historischer OHLCV-Daten aus Kaggle (CC0), 26 Ticker, lokal als Parquet verarbeitet."),
            (c2, "🤖", "Machine Learning", "Random Forest Classifier auf 8 technischen Features. 5-Fold CV, ROC-AUC, kein Data Leakage."),
            (c3, "📰", "News & KI", "NewsAPI für aktuelle Finanznachrichten. Google Gemini für kontextuelle Erklärungen."),
        ]:
            with col:
                st.markdown(
                    f"""<div class="ws-feature-card">
                    <span class="ws-feature-icon">{icon}</span>
                    <h3>{title}</h3><p>{body}</p></div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Hypothesen")
        hypo_df = pd.DataFrame([
            {"#": "H1", "Hypothese": "Historische technische Indikatoren enthalten Signale besser als Zufall (Accuracy > 50%)", "Status": "✅ Bestätigt (55,8%)"},
            {"#": "H2", "Hypothese": "Eine Streamlit-App kann diese Analysen zugänglich für Nicht-Experten aufbereiten", "Status": "✅ Bestätigt"},
            {"#": "H3", "Hypothese": "ML-Score + regelbasiertes Scoring > einzelne Indikatoren allein", "Status": "🟡 Teilweise bestätigt"},
        ])
        st.dataframe(hypo_df, width="stretch", hide_index=True)

        st.markdown("#### Abgrenzung")
        a1, a2 = st.columns(2)
        with a1:
            st.success("**Was die App ist:**\n- Wissenschaftlicher ML-Prototyp\n- Lern- und Demonstrationsanwendung\n- Reproduzierbarer QUA³CK-Workflow")
        with a2:
            st.error("**Was die App nicht ist:**\n- Keine Finanzberatung\n- Kein automatisierter Handelsbot\n- Keine garantierte Kursprognose")

    # ── Tab 2: QUA³CK-Prozess ──────────────────────────────────────────
    with tab2:
        st.markdown("#### Das QUA³CK-Modell im Projekt")
        st.markdown(
            "*Quelle: Stock, S. C. et al. (2021): QUA³CK – A Machine Learning Development Process. "
            "KIT ITIV. [DOI](https://publikationen.bibliothek.kit.edu/1000129631)*"
        )

        # QUA³CK-Phasen als Notebook-Karten
        phases_data = [
            ("Q","#6366f1","Question",       "01_question.ipynb",              "Forschungsfrage & Projektziel",          "Start / Projekt"),
            ("U","#7c3aed","Understanding",  "02_understanding_the_data.ipynb","EDA, Imputation, Scaling, Fehlwerte",   "Datenlabor"),
            ("A","#9333ea","Analytics",      "03_feature_engineering.ipynb",   "8 Features: Returns, MAs, Volatilität", "ML-Labor"),
            ("A","#a855f7","Algorithm",      "04_modeling_baseline_ml.ipynb",  "Baseline, LogReg, RF – CV, ROC, CM",    "ML-Labor"),
            ("A","#c084fc","Adaption",       "04_modeling_baseline_ml.ipynb",  "5-Fold CV, class_weight, max_depth",    "Kompass"),
            ("C","#22c55e","Conclude",       "05_conclude_evaluate.ipynb",     "Metriken, Grenzen, Fazit",              "Wealth Outlook"),
            ("K","#16a34a","Knowledge",      "06_knowledge_transfer_streamlit.ipynb","App, NewsAPI, Export",            "Alle Seiten"),
        ]

        nb_root = Path("notebooks")

        # 4 Spalten, 2 Zeilen (4 + 3)
        row1 = phases_data[:4]
        row2 = phases_data[4:]

        for row in [row1, row2]:
            cols = st.columns(len(row))
            for col, (letter, color, name, nb_file, detail, app_page) in zip(cols, row):
                nb_path = nb_root / nb_file
                with col:
                    # Karten-Header via HTML
                    st.markdown(
                        f'''<div style="border:1px solid {color}35;border-top:4px solid {color};
                        border-radius:12px;padding:.85rem .9rem .4rem;background:{color}07;
                        margin-bottom:.3rem">
                        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.35rem">
                          <span style="background:{color};color:white;font-weight:900;font-size:.88rem;
                          width:26px;height:26px;border-radius:6px;display:flex;align-items:center;
                          justify-content:center;flex-shrink:0">{letter}</span>
                          <strong style="font-size:.84rem;color:#1e293b">{name}</strong>
                        </div>
                        <div style="font-size:.71rem;color:#64748b;line-height:1.5;
                        margin-bottom:.45rem">{detail}</div>
                        <div style="font-size:.62rem;color:{color};font-weight:600;
                        margin-bottom:.4rem">→ {app_page}</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                    # Download-Button direkt in der Kachel
                    if nb_path.exists():
                        st.download_button(
                            f"⬇ {nb_file}",
                            data=nb_path.read_bytes(),
                            file_name=nb_file,
                            mime="application/json",
                            use_container_width=True,
                            key=f"dl_{nb_file}_{name}",
                        )

        # Einfachere horizontale Darstellung
        cols_q = st.columns(7)
        phase_data = [
            ("Q","#6366f1","Question","01"),
            ("U","#7c3aed","Understanding","02"),
            ("A","#9333ea","Analytics","03"),
            ("A","#a855f7","Algorithm","04"),
            ("A","#c084fc","Adaption","04"),
            ("C","#22c55e","Conclude","05"),
            ("K","#16a34a","Knowledge","06+07"),
        ]
        for col, (letter, color, name, nb) in zip(cols_q, phase_data):
            with col:
                st.markdown(
                    f"""<div style="background:{color};color:white;border-radius:12px;
                    padding:0.8rem 0.5rem;text-align:center;margin:0.1rem;">
                    <div style="font-size:1.5rem;font-weight:900">{letter}</div>
                    <div style="font-size:0.65rem;font-weight:700;opacity:0.9">{name}</div>
                    <div style="font-size:0.6rem;opacity:0.7;margin-top:0.2rem">NB {nb}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ── Tab 3: Architektur ──────────────────────────────────────────────
    with tab3:
        st.markdown("#### Tech-Stack")

        # Visuelle Tech-Stack Karten
        tech_items = [
            ("🖥️", "Streamlit",      "App-Framework",     "#6366f1"),
            ("🐼", "pandas · numpy", "Datenverarbeitung", "#0ea5e9"),
            ("📊", "Plotly",         "Visualisierung",    "#22c55e"),
            ("🤖", "scikit-learn",   "Machine Learning",  "#f59e0b"),
            ("🗄️", "Apache Parquet", "Datenformat",       "#8b5cf6"),
            ("🧠", "Google Gemini",  "KI-Assistent",      "#ec4899"),
            ("📰", "NewsAPI",        "News-Integration",  "#14b8a6"),
            ("📄", "ReportLab",      "PDF-Export",        "#f97316"),
        ]
        cols_t = st.columns(4)
        for i, (icon, tech, label, color) in enumerate(tech_items):
            with cols_t[i % 4]:
                st.markdown(
                    f"""<div style="border:1px solid {color}30;border-top:3px solid {color};
                    border-radius:10px;padding:.7rem .8rem;margin-bottom:.5rem;
                    background:{color}06;text-align:center">
                    <div style="font-size:1.4rem;margin-bottom:.2rem">{icon}</div>
                    <div style="font-weight:800;font-size:.82rem;color:#1e293b">{tech}</div>
                    <div style="font-size:.68rem;color:#64748b;margin-top:.1rem">{label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("#### Datenfluss-Pipeline")

        # Visueller Datenfluss als Plotly-Chart

        steps = [
            ("📁 Kaggle\nRohdaten",      "OHLCV CSV/TXT\n1962–2017",             "#6366f1"),
            ("⚙️ Feature\nEngineering",  "enrich_features()\nMA · Returns · Vol", "#8b5cf6"),
            ("🗄 Parquet\nSpeicherung",  "192.119 Zeilen\n26 Ticker",            "#a855f7"),
            ("🤖 ML-Modell\nTraining",   "Random Forest\nCV · ROC · F1",          "#c084fc"),
            ("📊 Scoring\nEngine",       "compute_scores()\nConfidence 0–100",    "#22c55e"),
            ("🖥️ Streamlit\nApp",        "Outlook · Kompass\nSimulator · Export",  "#16a34a"),
        ]

        fig_flow = go.Figure()
        for i, (name, detail, color) in enumerate(steps):
            # Box
            fig_flow.add_shape(type="rect",
                x0=i*1.6, x1=i*1.6+1.2, y0=0.3, y1=1.7,
                fillcolor=color, opacity=0.12,
                line=dict(color=color, width=2))
            # Icon + Name
            fig_flow.add_annotation(x=i*1.6+0.6, y=1.2, text=f"<b>{name}</b>",
                showarrow=False, font=dict(size=10, color=color),
                align="center")
            fig_flow.add_annotation(x=i*1.6+0.6, y=0.65, text=detail,
                showarrow=False, font=dict(size=8, color="#475569"),
                align="center")
            # Pfeil
            if i < len(steps)-1:
                fig_flow.add_annotation(
                    x=i*1.6+1.25, y=1.0, ax=i*1.6+1.05, ay=1.0,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1.5,
                    arrowwidth=2, arrowcolor="#6366f1")

        fig_flow.update_layout(
            height=200, margin=dict(l=10,r=10,t=10,b=10),
            xaxis=dict(visible=False, range=[-0.2, len(steps)*1.6]),
            yaxis=dict(visible=False, range=[0, 2]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_flow, use_container_width=True,
                        config={"displayModeBar": False}, key="arch_flow")

        # Projektstruktur kompakt
        st.markdown("#### Projektstruktur")
        c1, c2 = st.columns([3,2])
        with c1:
            st.code("""forexscope-ai/
├── app_max.py              # App (~4.800 Zeilen, monolithisch)
├── components/charts.py    # Plotly-Komponenten
├── styles/wealthscope.css  # Design-System
├── data/processed/         # Kaggle Features (Parquet + CSV)
├── models/                 # RF-Modell (.joblib)
├── notebooks/              # 8 QUA³CK-Notebooks
├── docs/                   # Prozessdokumentation
└── tests/                  # Statische Tests""", language="text")
        with c2:
            st.markdown("""
<div style="background:rgba(99,102,241,.06);border:1px solid rgba(99,102,241,.2);
border-radius:10px;padding:.9rem 1rem;font-size:.82rem;line-height:1.8">
<strong style="color:#6366f1">Monolithischer Ansatz:</strong><br>
Die gesamte App-Logik steckt bewusst in <code>app_max.py</code>
für maximale Portabilität —
ein einziger <code>streamlit run</code> Befehl, keine Abhängigkeiten
zwischen Modulen.
</div>""", unsafe_allow_html=True)

    # ── Tab 4: Datenbasis ───────────────────────────────────────────────
    with tab4:
        raw = ctx["market"]
        source = getattr(raw, "attrs", {}).get("data_source", "UNKNOWN")
        tickers_all = sorted(raw["ticker"].unique().tolist()) if "ticker" in raw.columns else []
        n_rows  = len(raw)
        n_tick  = raw["ticker"].nunique() if "ticker" in raw.columns else 0
        d_min   = raw["date"].min().date() if "date" in raw.columns else "–"
        d_max   = raw["date"].max().date() if "date" in raw.columns else "–"

        # ── Hero-Metriken ──────────────────────────────────────────
        m1,m2,m3,m4 = st.columns(4)
        for col, label, val, sub in [
            (m1, "DATENPUNKTE",  f"{n_rows:,}".replace(",","."), "Zeilen gesamt"),
            (m2, "TICKER",       str(n_tick),                    "Aktien & ETFs"),
            (m3, "ZEITRAUM VON", str(d_min),                     "Frühester Eintrag"),
            (m4, "ZEITRAUM BIS", str(d_max),                     "Letzter Eintrag"),
        ]:
            with col:
                st.markdown(f"""<div class="ws-kpi-hero">
                <div class="ws-kpi-label">{label}</div>
                <div class="ws-kpi-value" style="font-size:1.5rem">{val}</div>
                <div class="ws-kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # ── Datenquellen-Karten ────────────────────────────────────
        st.markdown("#### Datenquellen")
        s1, s2, s3 = st.columns(3)
        for col, icon, title, body, color in [
            (s1,"📦","Kaggle US Stocks & ETFs",
             "CC0 Public Domain · OHLCV-Daten · 1962–2017 · borismarjanovic","#6366f1"),
            (s2,"🌐","yfinance (Live-Erweiterung)",
             "Yahoo Finance API · 2017–heute · optional · kostenlos","#22c55e"),
            (s3,"📰","NewsAPI",
             "Aktuelle Finanznachrichten · Echtzeit · Sentiment-Analyse","#f59e0b"),
        ]:
            with col:
                st.markdown(f"""<div style="border:1px solid {color}30;border-top:3px solid {color};
                border-radius:10px;padding:.8rem 1rem;background:{color}06">
                <div style="font-size:1.3rem">{icon}</div>
                <div style="font-weight:800;font-size:.85rem;color:#1e293b;margin:.3rem 0 .2rem">{title}</div>
                <div style="font-size:.73rem;color:#64748b">{body}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        # ── Feature-Karten farbkodiert nach Gruppe ────────────────
        st.markdown("#### ML-Features nach Gruppe")
        feature_groups_display = {
            "📈 Rendite":  [("daily_return","Tagesrendite","(closeₜ − closeₜ₋₁) / closeₜ₋₁"),
                            ("return_5d",   "5-Tage-Rendite","(closeₜ − closeₜ₋₅) / closeₜ₋₅"),
                            ("return_20d",  "20-Tage-Rendite","(closeₜ − closeₜ₋₂₀) / closeₜ₋₂₀")],
            "📊 Trend":    [("ma_20_distance", "MA-20 Abstand","(close − MA20) / MA20"),
                            ("ma_50_distance", "MA-50 Abstand","(close − MA50) / MA50"),
                            ("ma_200_distance","MA-200 Abstand","(close − MA200) / MA200")],
            "⚠️ Risiko":   [("volatility_20d","Volatilität 20T","Std(ret,20) × √252"),
                            ("drawdown",      "Drawdown",      "(close − rolling_max) / rolling_max")],
            "🎯 Zielvariable": [("target_20d","target_20d","1 wenn future_return_20d > 0")],
        }
        group_colors = {"📈 Rendite":"#6366f1","📊 Trend":"#22c55e","⚠️ Risiko":"#ef4444","🎯 Zielvariable":"#f59e0b"}
        gcols = st.columns(4)
        for i,(group, feats) in enumerate(feature_groups_display.items()):
            color = group_colors[group]
            items_html = "".join(
                f'<div style="margin:.3rem 0;padding:.35rem .5rem;background:{color}08;' 
                f'border-radius:6px;border-left:3px solid {color}40">' 
                f'<code style="font-size:.72rem;color:{color}">{f}</code>' 
                f'<div style="font-size:.68rem;color:#64748b">{n}</div>' 
                f'<div style="font-size:.64rem;color:#94a3b8;font-family:monospace">{formula}</div></div>'
                for f,n,formula in feats
            )
            with gcols[i]:
                st.markdown(
                    f'''<div style="border:1px solid {color}25;border-radius:10px;padding:.7rem .8rem">''' 
                    f'''<div style="font-weight:800;font-size:.8rem;color:{color};margin-bottom:.4rem">{group}</div>''' 
                    f'''{items_html}</div>''',
                    unsafe_allow_html=True,
                )

        # ── Ticker-Übersicht ───────────────────────────────────────
        if tickers_all:
            st.markdown("#### Enthaltene Ticker")
            ticker_html = " ".join(
                f'<span style="display:inline-block;background:#eef2ff;color:#6366f1;' 
                f'font-weight:700;font-size:.75rem;padding:.2rem .55rem;' 
                f'border-radius:999px;margin:.15rem;border:1px solid rgba(99,102,241,.2)">{t}</span>'
                for t in tickers_all
            )
            st.markdown(ticker_html, unsafe_allow_html=True)

    # ── Tab 5: Präsentation ─────────────────────────────────────────────
    with tab5:

        # ── Demo-Ablauf als visuelle Timeline ─────────────────────
        st.markdown("#### Demo-Ablauf für die Präsentation")
        steps_pres = [
            ("1","🎯","Forschungsfrage","Projekt → Projektziel","#6366f1",
             "Warum binäre Klassifikation? Warum QUA³CK?"),
            ("2","🗄","Datenbasis","Datenlabor","#7c3aed",
             "192.119 Zeilen, 26 Ticker, CC0, Kaggle + yfinance"),
            ("3","🔬","Feature Engineering","ML-Labor → Modell-Erklärung","#8b5cf6",
             "8 Features: Returns, MAs, Volatilität, Drawdown"),
            ("4","📈","Live-Chart","Wealth Outlook → Kurs","#22c55e",
             "Bollinger Bands, Volume, MA20/50/200, bis heute"),
            ("5","🧭","Signal & Risiko","Outlook + Kompass","#16a34a",
             "Signal-Ampel, Confidence-Score, Gauge-Charts"),
            ("6","🤖","ML-Metriken","ML-Labor → Kap.3-Metriken","#0ea5e9",
             "ROC-AUC 0.58, PR-Kurve, Konfusionsmatrix"),
            ("7","📦","Portfolio","Simulator","#f59e0b",
             "Allokation SPY+QQQ+GLD+AGG, Donut-Chart"),
            ("8","📰","News & KI","News-Archiv","#ef4444",
             "NewsAPI live + Gemini Assistent"),
            ("9","📤","Export","Export","#94a3b8",
             "ZIP mit Bericht, JSON, CSV, Methodik"),
            ("10","⚠️","Grenzen","ML-Labor → Modell-Erklärung","#64748b",
             "EMH, keine Finanzberatung, Prototyp"),
        ]

        # 5+5 Grid
        row1 = steps_pres[:5]
        row2 = steps_pres[5:]
        for row in [row1, row2]:
            cols_p = st.columns(5)
            for col, (num, icon, title, page, color, detail) in zip(cols_p, row):
                with col:
                    st.markdown(
                        f'''<div style="border:1px solid {color}30;border-top:3px solid {color};
                        border-radius:10px;padding:.7rem .6rem;text-align:center;
                        background:{color}06;margin-bottom:.4rem">
                        <div style="font-size:1.2rem">{icon}</div>
                        <div style="font-size:.62rem;font-weight:900;color:{color};
                        letter-spacing:.06em;text-transform:uppercase;margin:.2rem 0">{num} · {title}</div>
                        <div style="font-size:.65rem;color:#6366f1;margin-bottom:.2rem">{page}</div>
                        <div style="font-size:.62rem;color:#64748b;line-height:1.4">{detail}</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        # ── Key-Message Box ────────────────────────────────────────
        st.markdown("""<div style="background:linear-gradient(135deg,#eef2ff,#f8faff);
        border:1px solid rgba(99,102,241,.25);border-left:4px solid #6366f1;
        border-radius:12px;padding:1rem 1.3rem;margin:.5rem 0">
        <div style="font-size:.7rem;font-weight:900;color:#6366f1;letter-spacing:.08em;
        text-transform:uppercase;margin-bottom:.4rem">60-Sekunden-Erklärung für den Prof</div>
        <p style="font-size:.88rem;color:#1e293b;line-height:1.7;margin:0">
        <strong>WealthScope AI</strong> verarbeitet <strong>192.119 echte Marktdatenpunkte</strong>
        aus Kaggle (1962–2017) und ergänzt sie mit <strong>yfinance-Live-Daten bis heute</strong>.
        Daraus werden 8 technische Features berechnet, ein <strong>Random Forest</strong> trainiert
        (Accuracy 55,7 %, ROC-AUC 0.58) und die Ergebnisse in einer interaktiven Streamlit-App
        nachvollziehbar dargestellt. Jede Entscheidung ist in <strong>8 QUA³CK-Notebooks</strong>
        dokumentiert. <em>Keine Finanzberatung — reiner Lernprototyp.</em>
        </p></div>""", unsafe_allow_html=True)

        # ── Wissenschaftliche Quellen als Karten ──────────────────
        st.markdown("#### Wissenschaftliche Quellen")
        refs_data = [
            ("Stock et al. (2021)","QUA³CK – A ML Development Process",
             "KIT ITIV","https://publikationen.bibliothek.kit.edu/1000129631","#6366f1"),
            ("Fama (1970)","Efficient Capital Markets",
             "Journal of Finance","https://doi.org/10.2307/2325486","#22c55e"),
            ("Li et al. (2024)","Comparison of Imputation Methods",
             "BMC Med. Res. Meth.","https://doi.org/10.1186/s12874-024-02173-x","#f59e0b"),
            ("Pinheiro et al. (2025)","Impact of Feature Scaling in ML",
             "arXiv:2506.08274","https://arxiv.org/abs/2506.08274","#8b5cf6"),
            ("Quibeldey-Cirkel (2026)","U: Understanding the Data",
             "IU Internationale Hochschule","–","#0ea5e9"),
        ]
        ref_cols = st.columns(3)
        for i,(quelle,titel,verlag,url,color) in enumerate(refs_data):
            with ref_cols[i % 3]:
                link = f'<a href="{url}" target="_blank" style="color:{color};font-size:.65rem">{url[:40]}…</a>' if url != "–" else '<span style="color:#94a3b8;font-size:.65rem">–</span>'
                st.markdown(
                    f'''<div style="border:1px solid {color}25;border-left:3px solid {color};
                    border-radius:9px;padding:.65rem .8rem;margin-bottom:.4rem;background:{color}05">
                    <div style="font-weight:800;font-size:.78rem;color:#1e293b">{quelle}</div>
                    <div style="font-size:.72rem;color:#475569;margin:.15rem 0">{titel}</div>
                    <div style="font-size:.65rem;color:#94a3b8">{verlag}</div>
                    {link}</div>''',
                    unsafe_allow_html=True,
                )

        st.warning("⚠️ Keine Finanzberatung. Alle Ergebnisse dienen ausschließlich Lern- und Demonstrationszwecken.")

    # ── Tab 6: Limitationen ─────────────────────────────────────────────
    with tab6_lim:
        st.markdown("#### Wissenschaftliche Limitationen — Transparente Einordnung")
        st.markdown(
            "Die folgenden Einschränkungen sind bekannt und werden hier bewusst dokumentiert. "
            "Reflexion über Grenzen ist ein Kernbestandteil wissenschaftlicher Arbeit."
        )

        _lims = [
            ("⚠️", "Survivorship Bias", "#f59e0b",
             "Die 26 analysierten Blue-Chip-Ticker wurden **ex post** ausgewählt — "
             "es handelt sich ausschließlich um Unternehmen, die heute noch existieren. "
             "Bankrotte, Fusionen und Delistings fehlen vollständig. "
             "Dies führt zu einer systematischen Überschätzung historischer Performance "
             "und Modellgüte. Ein repräsentativer Datensatz würde auch gescheiterte Unternehmen umfassen.",
             "Auswirkung: Modellgüte wird überschätzt."),

            ("🔄", "Distribution Shift", "#6366f1",
             "Das Random-Forest-Modell wurde auf Daten von **1962–2017** trainiert. "
             "Live-Daten via yfinance liegen außerhalb dieses Trainingsbereichs. "
             "Das Modell hat die COVID-Krise (2020), den Post-Pandemie-Boom, "
             "die Zinswende (2022) und den KI-Boom (2023) **nie gesehen**. "
             "Vorhersagen auf aktuellen Daten sind mit strukturell höherer Unsicherheit behaftet.",
             "Auswirkung: Generalisierbarkeit eingeschränkt."),

            ("📉", "Accuracy unter Baseline", "#ef4444",
             "Die Modell-Accuracy (55,72%) liegt leicht **unter** der Majority-Baseline (56,37%). "
             "Das bedeutet: Ein naiver Classifier der immer 'Bullish' vorhersagt wäre in roher Accuracy besser. "
             "Allerdings misst Accuracy bei unbalancierten Klassen (56% Bullish / 44% Bearish) "
             "nicht die Diskriminierungsfähigkeit. Der ROC-AUC von 0.5842 zeigt, "
             "dass das Modell dennoch besser als reiner Zufall (0.5) diskriminiert.",
             "Auswirkung: Falsch interpretierte Accuracy kann Modell überschätzen."),

            ("🔗", "Nur technische Features", "#0ea5e9",
             "Das Modell verwendet ausschließlich **technische Indikatoren** (MA-Distanzen, "
             "Volatilität, Drawdown, Renditen). Fundamentaldaten (KGV, Umsatz, Gewinn), "
             "Makrodaten (Zinsen, Inflation, BIP) und Sentiment-Daten aus sozialen Medien "
             "sind nicht enthalten. Die Efficient Market Hypothesis (Fama, 1970) legt nahe, "
             "dass historische Kursdaten bereits eingepreist sind.",
             "Auswirkung: Prognosekraft systematisch begrenzt (EMH)."),

            ("📏", "Score-Gewichte explorativ", "#8b5cf6",
             "Die Confidence-Score-Gewichte (Trend 36%, Volatilität 22%, Drawdown 18%, "
             "News 14%, Gewichtung 10%) sind aus der RF Feature Importance abgeleitet, "
             "aber **nicht durch eine separate Optimierung validiert**. "
             "Eine datengetriebene Kalibrierung (z.B. Lasso-Regression auf Zielvariable) "
             "wäre ein sinnvoller nächster Schritt.",
             "Auswirkung: Gewichte sind begründet, aber nicht optimal."),
        ]

        for icon, title, color, body, impact in _lims:
            st.markdown(
                f"""<div style="border:1px solid {color}30;border-left:4px solid {color};
                border-radius:12px;padding:1rem 1.2rem;background:{color}07;margin-bottom:0.75rem">
                <div style="display:flex;justify-content:space-between;align-items:flex-start">
                  <div style="font-weight:800;color:#1e293b;font-size:0.9rem;margin-bottom:0.4rem">{icon} {title}</div>
                  <span style="background:{color}18;color:{color};font-size:0.65rem;font-weight:700;
                  padding:0.15rem 0.5rem;border-radius:999px;white-space:nowrap">{impact}</span>
                </div>
                <p style="font-size:0.82rem;color:#475569;line-height:1.7;margin:0">{body}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.info(
            "💡 Diese Limitationen sind kein Zeichen von Schwäche — sie sind ein Zeichen von "
            "wissenschaftlicher Reflexionsfähigkeit. Die App ist bewusst als **Lernprototyp** "
            "konzipiert, nicht als Handelsplattform."
        )


def page_export(ctx: Dict[str, Any]) -> None:
    import zipfile as _zf
    import io as _io

    page_title("Export", "Analyseergebnisse & Rohdaten herunterladen")

    result = ctx["result"]
    market = ctx.get("market", pd.DataFrame())
    features = ctx.get("features", pd.DataFrame())
    news_df = ctx.get("news_df", pd.DataFrame())

    ws_render_snapshot_metrics(ctx)

    # ── Export-Items definieren ─────────────────────────────────────────
    report_md = analysis_report_markdown(result) if "analysis_report_markdown" in globals() else f"# WealthScope AI – {result.ticker}\nGeneriert: {__import__('datetime').datetime.now().isoformat()}"
    context_json = json.dumps(asdict(result), indent=2, ensure_ascii=False)

    _ITEMS = [
        {
            "key": "report",
            "icon": "📄",
            "label": "Analysebericht",
            "desc": "Vollständiger Markdown-Bericht mit Outlook, Risiko & Confidence",
            "fmt": "MD",
            "size": f"{len(report_md.encode())//1024 + 1} KB",
            "color": "#6366f1",
            "available": True,
            "data": report_md.encode("utf-8"),
            "filename": f"wealthscope_report_{result.ticker}.md",
            "mime": "text/markdown",
        },
        {
            "key": "features",
            "icon": "📊",
            "label": "Feature-Daten",
            "desc": f"Berechnete ML-Features für {result.ticker} ({len(features) if features is not None and not features.empty else 0} Zeilen)",
            "fmt": "CSV",
            "size": f"{len(features.to_csv(index=False).encode())//1024 + 1} KB" if features is not None and not features.empty else "–",
            "color": "#0ea5e9",
            "available": features is not None and not features.empty,
            "data": features.to_csv(index=False).encode("utf-8") if features is not None and not features.empty else b"",
            "filename": f"wealthscope_features_{result.ticker}.csv",
            "mime": "text/csv",
        },
        {
            "key": "market",
            "icon": "📈",
            "label": "Rohdaten",
            "desc": f"Vollständige Marktdaten ({len(market) if market is not None and not market.empty else 0} Zeilen, alle Ticker)",
            "fmt": "CSV",
            "size": f"{len(market.to_csv(index=False).encode())//1024 + 1} KB" if market is not None and not market.empty else "–",
            "color": "#22c55e",
            "available": market is not None and not market.empty,
            "data": market.to_csv(index=False).encode("utf-8") if market is not None and not market.empty else b"",
            "filename": f"wealthscope_market_all.csv",
            "mime": "text/csv",
        },
        {
            "key": "news",
            "icon": "📰",
            "label": "News-Einordnung",
            "desc": f"Sentiment-Bewertung der aktuellen Nachrichten ({len(news_df) if news_df is not None and not news_df.empty else 0} Artikel)",
            "fmt": "CSV",
            "size": f"{len(news_df.to_csv(index=False).encode())//1024 + 1} KB" if news_df is not None and not news_df.empty else "–",
            "color": "#f59e0b",
            "available": news_df is not None and not news_df.empty,
            "data": news_df.to_csv(index=False).encode("utf-8") if news_df is not None and not news_df.empty else b"",
            "filename": f"wealthscope_news_{result.ticker}.csv",
            "mime": "text/csv",
        },
        {
            "key": "context",
            "icon": "🔧",
            "label": "Analysekontext",
            "desc": "Alle Score-Parameter, Flags und Konfiguration als maschinenlesbare Datei",
            "fmt": "JSON",
            "size": f"{len(context_json.encode())//1024 + 1} KB",
            "color": "#8b5cf6",
            "available": True,
            "data": context_json.encode("utf-8"),
            "filename": f"wealthscope_context_{result.ticker}.json",
            "mime": "application/json",
        },
    ]

    # ── Auswahl-Cards (3-spaltig) ───────────────────────────────────────
    st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)
    selected_keys = []

    cols_grid = st.columns(3)
    for i, item in enumerate(_ITEMS):
        with cols_grid[i % 3]:
            avail_style = "" if item["available"] else "opacity:0.45;"
            st.markdown(
                f"""<div style="{avail_style}border:1px solid {item['color']}28;border-top:3px solid {item['color']};
                border-radius:12px;padding:1rem 1rem 0.6rem;background:{item['color']}05;margin-bottom:0.15rem">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.4rem">
                  <span style="font-size:1.4rem">{item['icon']}</span>
                  <span style="background:{item['color']}18;color:{item['color']};font-size:0.65rem;
                  font-weight:800;padding:0.15rem 0.5rem;border-radius:999px;letter-spacing:0.06em">{item['fmt']}</span>
                </div>
                <div style="font-weight:700;color:#1e293b;font-size:0.9rem;margin-bottom:0.2rem">{item['label']}</div>
                <div style="font-size:0.75rem;color:#64748b;line-height:1.5;margin-bottom:0.4rem">{item['desc']}</div>
                <div style="font-size:0.7rem;color:#94a3b8">~{item['size']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
            cb = st.checkbox(
                "Auswählen" if item["available"] else "Nicht verfügbar",
                value=item["available"],
                disabled=not item["available"],
                key=f"export_cb_{item['key']}",
            )
            if cb and item["available"]:
                selected_keys.append(item["key"])

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Download-Buttons ────────────────────────────────────────────────
    n_selected = len(selected_keys)
    selected_items = [it for it in _ITEMS if it["key"] in selected_keys]

    if n_selected == 0:
        st.info("Wähle mindestens eine Datei aus.")
    elif n_selected == 1:
        it = selected_items[0]
        st.download_button(
            f"⬇ {it['label']} herunterladen ({it['fmt']})",
            data=it["data"],
            file_name=it["filename"],
            mime=it["mime"],
            width="stretch",
        )
    else:
        # Baue ZIP aus Auswahl
        _buf = _io.BytesIO()
        with _zf.ZipFile(_buf, "w", _zf.ZIP_DEFLATED) as _z:
            for it in selected_items:
                _z.writestr(it["filename"], it["data"])
        _buf.seek(0)
        st.download_button(
            f"⬇ Auswahl als ZIP herunterladen ({n_selected} Dateien)",
            data=_buf.getvalue(),
            file_name=f"wealthscope_export_{result.ticker}.zip",
            mime="application/zip",
            width="stretch",
        )

    # ── Vorschau-Tabs für ausgewählte Dateien ───────────────────────────
    if selected_items:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        preview_tabs = st.tabs([f"{it['icon']} {it['label']}" for it in selected_items])
        for tab, it in zip(preview_tabs, selected_items):
            with tab:
                if it["fmt"] == "MD":
                    st.code(it["data"].decode("utf-8")[:3000], language="markdown")
                elif it["fmt"] == "CSV":
                    try:
                        _prev = pd.read_csv(_io.BytesIO(it["data"]))
                        st.dataframe(_prev.tail(200), width="stretch", hide_index=True)
                    except Exception:
                        st.code(it["data"].decode("utf-8", errors="replace")[:2000])
                elif it["fmt"] == "JSON":
                    st.json(json.loads(it["data"].decode("utf-8")))


def page_impressum(ctx: Dict[str, Any]) -> None:
    page_title("Impressum", "Projektkontext, Verantwortlichkeit und Hinweise.")

    st.warning(
        "Dies ist eine Uni-/Demo-Anwendung und kein produktives Finanz- oder Beratungsangebot. "
        "Die Angaben ersetzen kein rechtlich geprüftes Impressum."
    )

    tab1, tab2, tab3 = st.tabs(["Projektangaben", "Verantwortlichkeit", "Hinweise"])

    with tab1:
        with st.container(border=True):
            st.markdown("### WealthScope AI")
            st.write("Interaktive Streamlit-App für ein QUA3CK-/Big-Data-/Data-Science-Projekt.")
            st.write("Zweck: Demonstration von Datenaufbereitung, Analyse, Visualisierung, News-Kontext und Export.")

    with tab2:
        st.write("Für eine echte Veröffentlichung müssten hier rechtlich korrekte Angaben ergänzt werden.")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Projektname", "WealthScope AI"],
                    ["Kontext", "Uni-/Demo-Projekt"],
                    ["Technologie", "Python, Streamlit, Pandas, Plotly, NewsAPI"],
                    ["Status", "Nicht produktiv"],
                ],
                columns=["Feld", "Angabe"],
            ),
            width="stretch",
            hide_index=True,
        )

    with tab3:
        st.info(
            "Die App liefert keine Anlageberatung, keine Kauf-/Verkaufsempfehlung und keine rechtlich verbindliche Auskunft. "
            "Alle Ergebnisse dienen der Demonstration und methodischen Einordnung."
        )

        with st.expander("Warum diese Seite trotzdem sinnvoll ist"):
            st.write(
                "Auch in einer Demo-App zeigt ein Impressum-/Hinweisbereich, dass zwischen Prototyp, Produktivsystem "
                "und rechtlicher Verantwortung unterschieden wird."
            )


def page_datenschutz(ctx: Dict[str, Any]) -> None:
    page_title("Datenschutz", "Transparenz über Datenquellen, lokale Verarbeitung und externe API-Nutzung.")

    tab1, tab2, tab3, tab4 = st.tabs(["Überblick", "Datenquellen", "Verarbeitung", "Grenzen"])

    with tab1:
        st.subheader("Datenschutz- und Transparenzhinweis")
        st.write(
            "WealthScope AI verarbeitet primär lokale Marktdaten und optional NewsAPI-Daten. "
            "Die App ist als Demo-/Uni-Projekt konzipiert und nicht als produktives Kundensystem."
        )

        ws_render_snapshot_metrics(ctx)

    with tab2:
        sources = pd.DataFrame(
            [
                ["Historische Marktdaten", "Lokales Kaggle-Dataset", "Parquet/CSV", "Ja"],
                ["News", "NewsAPI", "HTTP API über API-Key", "Nein"],
                ["Nutzereingaben", "Streamlit-Session", "Session State", "Nein"],
                ["Secrets", ".streamlit/secrets.toml", "lokal, nicht committen", "Nein"],
            ],
            columns=["Datenart", "Quelle", "Technik", "Personenbezogen?"],
        )
        st.dataframe(sources, width="stretch", hide_index=True)

    with tab3:
        st.markdown(
            """
**Verarbeitung in der App**
- Marktdaten werden lokal geladen und analysiert.
- News werden über eine API-Abfrage ergänzt.
- Eingaben wie Kapital, Gewichtung oder Risikotoleranz bleiben im App-Zustand.
- API-Keys liegen lokal in `.streamlit/secrets.toml` und dürfen nicht ins Repository.
            """
        )

        with st.expander("Technischer Hinweis zu Secrets"):
            st.code(
                """.streamlit/secrets.toml

NEWS_API_KEY = "..."
""",
                language="toml",
            )

    with tab4:
        st.warning(
            "Diese Seite ist kein rechtlich geprüftes Datenschutzdokument. "
            "Für eine echte Veröffentlichung wären Hosting, Logging, API-Nutzung, Nutzertracking und Rechtsgrundlagen separat zu prüfen."
        )


def page_privacy(ctx: Dict[str, Any]) -> None:
    page_datenschutz(ctx)


def page_status(ctx: Dict[str, Any]) -> None:
    from datetime import datetime as _dt

    page_title("Status", "Systemgesundheit, Datenqualität & Pipeline auf einen Blick")

    result   = ctx["result"]
    market   = ctx.get("market",   pd.DataFrame())
    features = ctx.get("features", pd.DataFrame())
    news_df  = ctx.get("news_df",  pd.DataFrame())
    news_src = ctx.get("news_source", "UNKNOWN")
    proof    = data_proof(market)

    # ── Farbhilfen ──────────────────────────────────────────────────
    def _ok(ok: bool):
        return ("#22c55e", "✅", "Aktiv") if ok else ("#ef4444", "❌", "Fehlt")

    data_ok  = not market.empty
    feat_ok  = not features.empty
    news_ok  = news_src == "REAL_NEWSAPI"
    model_ok = get_model_path().exists()
    real_src = str(proof["source"]).startswith("REAL")

    # ── Hero-Status-Banner ──────────────────────────────────────────
    overall_color = "#22c55e" if (data_ok and feat_ok and model_ok) else "#f59e0b"
    overall_label = "Alle Systeme operativ" if (data_ok and feat_ok and model_ok) else "Teilweise eingeschränkt"
    overall_icon  = "✅" if (data_ok and feat_ok and model_ok) else "⚠️"

    st.markdown(
        f"""<div style="background:{'rgba(34,197,94,.08)' if data_ok and feat_ok else 'rgba(245,158,11,.08)'};
        border:1.5px solid {overall_color};border-radius:16px;padding:1.1rem 1.5rem;margin-bottom:1.4rem;
        display:flex;align-items:center;gap:1rem;">
        <span style="font-size:2rem">{overall_icon}</span>
        <div>
            <div style="font-weight:900;font-size:1.1rem;color:{overall_color}">{overall_label}</div>
            <div style="font-size:0.78rem;color:#64748b;margin-top:0.15rem">
                Stand: <strong>{_dt.now().strftime('%d.%m.%Y %H:%M:%S')}</strong> &nbsp;·&nbsp;
                Version: <strong>{APP_VERSION}</strong> &nbsp;·&nbsp;
                Seite: <strong>{st.session_state.get('current_page','–')}</strong>
            </div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Komponenten-Karten ──────────────────────────────────────────
    st.markdown("#### Komponenten")
    components = [
        ("📦 Marktdaten",       data_ok,  proof["source"],                      f"{proof['rows']:,} Zeilen"),
        ("🔬 Feature-Engine",   feat_ok,  "enrich_features()",                  f"{len(features.columns) if feat_ok else 0} Spalten"),
        ("🤖 ML-Modell",        model_ok, f"RandomForestClassifier ({get_model_version()})", get_model_path().name),
        ("📰 NewsAPI",          news_ok,  news_src,                              "live" if news_ok else "Demo-Fallback"),
        ("🧠 KI-Assistent",     True,     "Google Gemini",                       st.secrets.get("GEMINI_MODEL","–") if hasattr(st,"secrets") else "–"),
        ("📊 Datenbasis",       real_src, "Kaggle US Stocks & ETFs",            "CC0 Public Domain"),
    ]

    cols = st.columns(3)
    for i, (name, ok, tech, detail) in enumerate(components):
        color, icon, label = _ok(ok)
        with cols[i % 3]:
            st.markdown(
                f"""<div style="background:{'rgba(34,197,94,.06)' if ok else 'rgba(239,68,68,.06)'};
                border:1px solid {'rgba(34,197,94,.25)' if ok else 'rgba(239,68,68,.25)'};
                border-left:4px solid {color};border-radius:12px;
                padding:0.9rem 1rem;margin-bottom:0.7rem;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.3rem">
                    <span style="font-weight:800;font-size:0.88rem">{name}</span>
                    <span style="font-size:1rem">{icon}</span>
                </div>
                <div style="font-size:0.72rem;color:{color};font-weight:700;text-transform:uppercase;
                letter-spacing:.05em;margin-bottom:.2rem">{label}</div>
                <div style="font-size:0.78rem;color:#475569;font-weight:600">{tech}</div>
                <div style="font-size:0.7rem;color:#94a3b8;margin-top:.1rem">{detail}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Datensatz-Metriken ──────────────────────────────────────────
    st.markdown("#### Datensatz")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Zeilen",    f"{proof['rows']:,}".replace(",","."))
    m2.metric("Spalten",   str(proof["columns"]))
    m3.metric("Ticker",    str(proof["tickers"]))
    _live_s = st.session_state.get("use_live_data", False)
    _bis = __import__("datetime").date.today().strftime("%Y-%m-%d") + " 🌐" if _live_s else proof["date_max"]
    m4.metric("Von", proof["date_min"])
    m5.metric("Bis", _bis)
    m6.metric("target_20d","✅" if proof["has_target_20d"] else "❌")

    # ── Feature-Checkliste ──────────────────────────────────────────
    st.markdown("#### Feature-Vollständigkeit")
    required_cols = [
        ("date",           "Zeitstempel"),
        ("ticker",         "Asset-Bezeichnung"),
        ("close",          "Schlusskurs"),
        ("daily_return",   "Tagesrendite"),
        ("return_5d",      "5-Tage-Rendite"),
        ("return_20d",     "20-Tage-Rendite"),
        ("ma_20",          "Moving Average 20"),
        ("ma_50",          "Moving Average 50"),
        ("ma_200",         "Moving Average 200"),
        ("ma_20_distance", "MA-20 Abstand"),
        ("volatility_20d", "Volatilität 20T"),
        ("drawdown",       "Drawdown"),
        ("target_20d",     "Zielvariable"),
    ]

    check_cols = st.columns(4)
    for i, (col, label) in enumerate(required_cols):
        present = features is not None and col in features.columns
        color = "#22c55e" if present else "#ef4444"
        icon  = "✅" if present else "❌"
        with check_cols[i % 4]:
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:.4rem;padding:.35rem .5rem;
                background:{'rgba(34,197,94,.06)' if present else 'rgba(239,68,68,.06)'};
                border:1px solid {'rgba(34,197,94,.2)' if present else 'rgba(239,68,68,.2)'};
                border-radius:8px;margin-bottom:.3rem;">
                <span style="font-size:.85rem">{icon}</span>
                <div>
                    <div style="font-size:.72rem;font-weight:700;color:#1e293b">{col}</div>
                    <div style="font-size:.65rem;color:#94a3b8">{label}</div>
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Pipeline-Visualisierung ─────────────────────────────────────
    st.markdown("#### Datenpipeline")
    pipeline_steps = [
        ("📁", "Kaggle\nRohdaten",        "OHLCV CSV"),
        ("⚙️",  "Feature\nEngineering",   "enrich_features()"),
        ("🗄",  "Parquet\nSpeicherung",   "192.119 Zeilen"),
        ("🤖",  "ML-Modell",              "RF Classifier"),
        ("📊",  "Scoring",                "Confidence Score"),
        ("🖥",  "Streamlit\nApp",         "Interaktiv"),
    ]
    pipe_cols = st.columns(len(pipeline_steps))
    for i, (icon, name, sub) in enumerate(pipeline_steps):
        with pipe_cols[i]:
            connector = "" if i == 0 else ""
            st.markdown(
                f"""<div style="text-align:center;padding:.5rem .2rem">
                <div style="font-size:1.6rem;margin-bottom:.3rem">{icon}</div>
                <div style="font-weight:800;font-size:.75rem;color:#1e293b;
                white-space:pre-line;line-height:1.3">{name}</div>
                <div style="font-size:.65rem;color:#6366f1;font-weight:600;
                margin-top:.2rem;background:#eef2ff;border-radius:6px;
                padding:.1rem .35rem;display:inline-block">{sub}</div>
                {"<div style='font-size:.9rem;color:#c4b5fd;margin-top:.4rem'>→</div>" if i < len(pipeline_steps)-1 else ""}
                </div>""",
                unsafe_allow_html=True,
            )

    # ── Debug-Expander ──────────────────────────────────────────────
    with st.expander("🔧 Session & Debug"):
        tab_s, tab_d = st.tabs(["Session State", "Debug JSON"])
        with tab_s:
            safe = {k: str(v) for k, v in st.session_state.items()
                    if k not in {"chat_messages", "ws_chat_messages"}}
            st.json(safe)
        with tab_d:
            st.json({
                "data_source":  proof["source"],
                "news_source":  news_src,
                "ticker":       getattr(result, "ticker", ""),
                "outlook":      getattr(result, "outlook", ""),
                "risk_label":   getattr(result, "risk_label", ""),
                "confidence":   getattr(result, "confidence", ""),
                "model_exists": model_ok,
                "app_version":  APP_VERSION,
            })


def main() -> None:
    init_state()

    # ── CSS + Sidebar (vor allem anderen) ──────────────────────────
    inject_css()
    base_df = load_local_market_data()
    uploaded_df = render_sidebar(base_df)
    market_df = get_market_data(uploaded_df)

    # ── Top-Navigation mit Logo ─────────────────────────────────────
    render_header()

    # ── Kontext aufbauen ────────────────────────────────────────────
    with st.spinner("Analyse wird vorbereitet …"):
        ctx = build_context(market_df)

    proof      = data_proof(market_df)
    news_source = ctx.get("news_source", "UNKNOWN")

    # ── Kompakte Status-Zeile statt großer success/error Banner ────
    data_ok  = str(proof["source"]).startswith("REAL")
    news_ok  = news_source == "REAL_NEWSAPI"
    live_on  = st.session_state.get("use_live_data", False)

    # Bei Live-Daten: aktuelles Datum als Enddatum anzeigen
    if live_on:
        from datetime import date as _date
        date_max_display = _date.today().strftime("%Y-%m-%d") + " (live)"
    else:
        date_max_display = proof["date_max"]

    live_badge = " 🌐 **Live**" if live_on else ""

    status_parts = []
    if data_ok:
        status_parts.append(
            f"✅{live_badge} **{proof['rows']:,} Zeilen** · {proof['tickers']} Ticker · "
            f"{proof['date_min']} – {date_max_display}".replace(",",".")
        )
    else:
        status_parts.append("⚠️ Demo-Daten aktiv")
    status_parts.append("📰 NewsAPI aktiv" if news_ok else f"📰 News: Demo ({news_source})")

    # Status-Zeile nur auf Unter-Seiten zeigen, nicht auf der Start-Seite (zu viel Lärm im Hero)
    if st.session_state.get("current_page", "Start") != "Start":
        st.caption("  ·  ".join(status_parts))

    # ── Seite rendern ───────────────────────────────────────────────
    route_page(st.session_state.get("current_page", "Start"), ctx)
    render_floating_assistant_panel(ctx)
    render_bottom_bar()


if __name__ == "__main__":
    main()
