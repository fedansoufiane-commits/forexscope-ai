"""Data loading & feature engineering.

Two data sources:
  1. Kaggle-derived parquet (1962-2017, 26 tickers) — the model's training
     distribution. Always available, no network required.
  2. Live yfinance data (optional) — same feature formulas applied so the
     model can score current prices, with an explicit distribution-shift
     warning shown in the UI (the model never saw 2018+ regimes).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.config import MARKET_PARQUET, TICKERS


@st.cache_data(show_spinner="Lade historische Marktdaten ...")
def load_market_data() -> pd.DataFrame:
    df = pd.read_parquet(MARKET_PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def compute_features(close: pd.Series) -> pd.DataFrame:
    """Reproduce the exact feature formulas used in scripts/train_and_diagnose.py."""
    daily_return = close.pct_change()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()
    rolling_high = close.cummax()
    return pd.DataFrame({
        "close": close,
        "daily_return": daily_return,
        "return_5d": close.pct_change(5),
        "return_20d": close.pct_change(20),
        "ma_20": ma_20, "ma_50": ma_50, "ma_200": ma_200,
        "ma_20_distance": close / ma_20 - 1,
        "ma_50_distance": close / ma_50 - 1,
        "ma_200_distance": close / ma_200 - 1,
        "volatility_20d": daily_return.rolling(20).std() * np.sqrt(252),
        "rolling_high": rolling_high,
        "drawdown": close / rolling_high - 1,
    })


@st.cache_data(show_spinner="Lade Live-Kurse ...", ttl=900)
def fetch_live_data(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index().rename(columns={"Date": "date", "Close": "close",
                                                  "Open": "open", "High": "high",
                                                  "Low": "low", "Volume": "volume"})
        feats = compute_features(raw["close"])
        out = pd.concat([raw[["date", "open", "high", "low", "volume"]], feats], axis=1)
        out["ticker"] = ticker
        return out
    except Exception:
        return None


def available_tickers(df: pd.DataFrame) -> list[str]:
    seen = df["ticker"].dropna().unique().tolist()
    ordered = [t for t in TICKERS if t in seen]
    extra = [t for t in seen if t not in TICKERS]
    return ordered + sorted(extra)


def filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)


def filter_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    if df.empty or period == "Alle":
        return df
    days = {"1M": 30, "3M": 90, "6M": 180, "1J": 365, "3J": 365 * 3, "5J": 365 * 5}.get(period)
    if not days:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].reset_index(drop=True)


def load_uploaded_csv(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower() for c in df.columns]
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "ticker" not in df.columns:
            df["ticker"] = "UPLOAD"
        return df
    except Exception:
        return None
