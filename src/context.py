"""Assembles the per-run context (selected ticker/period/data/analysis) that
every page reads from `st.session_state` — the single place pages pull data
from, so each page module stays a thin render function."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src import data as data_mod
from src import news as news_mod
from src.config import APP_VERSION
from src.diagnostics import load_diagnostics
from src.model import compute_scores, load_model


def get_context() -> Dict[str, Any]:
    market_df = data_mod.load_market_data()
    tickers = data_mod.available_tickers(market_df)

    ticker = st.session_state.get("ticker", "SPY")
    if ticker not in tickers and tickers:
        ticker = tickers[0]
        st.session_state["ticker"] = ticker

    if st.session_state.get("use_live_data") and st.session_state.get("_uploaded_df") is None:
        live = data_mod.fetch_live_data(ticker)
        ticker_df = live if live is not None else data_mod.filter_ticker(market_df, ticker)
    elif st.session_state.get("_uploaded_df") is not None:
        ticker_df = st.session_state["_uploaded_df"]
    else:
        ticker_df = data_mod.filter_ticker(market_df, ticker)

    period = st.session_state.get("period", "1J")
    period_df = data_mod.filter_period(ticker_df, period)

    news_query = ticker
    news_df, news_score, news_label = pd.DataFrame(), 0.0, "neutral"
    if st.session_state.get("enable_news", True):
        try:
            news_df, news_score, news_label = news_mod.analyze_news(news_query)
        except Exception:
            pass

    result = compute_scores(
        period_df, ticker,
        news_score=news_score, news_label=news_label,
        asset_weight=st.session_state.get("asset_weight", 15),
    )

    # The model version is the app version that trained it, read from the
    # artifact itself — a literal here would silently drift from the .joblib.
    model_info = {"name": "RandomForestClassifier", "version": APP_VERSION}
    try:
        model_info["version"] = load_diagnostics().get("app_version", APP_VERSION)
    except Exception:
        pass
    try:
        model_info["loaded"] = load_model() is not None
    except Exception:
        model_info["loaded"] = False

    return {
        "market": market_df,
        "tickers": tickers,
        "ticker": ticker,
        "ticker_df": ticker_df,
        "period_df": period_df,
        "news": news_df,
        "result": result,
        "model": model_info,
        "theme_mode": st.session_state.get("theme_mode", "Hell"),
    }
