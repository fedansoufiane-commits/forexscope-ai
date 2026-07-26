"""ML model loading + the interpretable rule-based scoring engine.

Two distinct signals are surfaced throughout the app, and it matters that
they are never conflated:
  - `rf_proba`   the trained RandomForest's P(price higher in 20 trading
                 days) — the actual ML prediction, evaluated in depth on
                 the ML-Insights page (confusion matrix, ROC, learning curve).
  - `confidence` a transparent, hand-weighted 0-100 composite of trend /
                 volatility / drawdown / news / position-sizing, used for
                 the guided "Outlook" narrative. Weights are derived from
                 grouped RF feature importance (see docs/qua3ck_process.md)
                 — not fitted, so they stay human-auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import joblib
import pandas as pd
import streamlit as st

from src.config import MODEL_FEATURES, MODEL_PATH

SCORE_WEIGHTS = {"trend": 0.36, "volatility": 0.22, "drawdown": 0.18, "news": 0.14, "position": 0.10}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@dataclass
class AnalysisResult:
    ticker: str = "-"
    price: float = 0.0
    trend_score: float = 50.0
    volatility_score: float = 50.0
    drawdown_score: float = 50.0
    news_score: float = 0.0
    news_label: str = "neutral"
    asset_weight: float = 10.0
    confidence: float = 0.0
    outlook: str = "Neutral"
    rf_proba: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def score_trend(row: pd.Series) -> float:
    d20, d50, d200 = row.get("ma_20_distance", 0), row.get("ma_50_distance", 0), row.get("ma_200_distance", 0)
    raw = 0.4 * d20 + 0.35 * d50 + 0.25 * d200
    return _clip(50 + raw * 250)


def score_volatility(row: pd.Series) -> float:
    vol = row.get("volatility_20d", 0.2) or 0.2
    return _clip(100 - (vol / 0.6) * 100)


def score_drawdown(row: pd.Series) -> float:
    dd = row.get("drawdown", 0.0) or 0.0
    return _clip(100 + dd * 200)


def compute_scores(df: pd.DataFrame, ticker: str, news_score: float = 0.0,
                    news_label: str = "neutral", asset_weight: float = 10.0) -> AnalysisResult:
    if df.empty:
        return AnalysisResult(ticker=ticker)
    row = df.dropna(subset=[c for c in MODEL_FEATURES if c in df.columns]).tail(1)
    if row.empty:
        row = df.tail(1)
    row = row.iloc[0]

    trend = score_trend(row)
    vol = score_volatility(row)
    dd = score_drawdown(row)
    weight_risk = _clip(100 - max(0.0, asset_weight - 10) * 3)
    news_component = _clip(50 + news_score * 10)

    confidence = round(
        SCORE_WEIGHTS["trend"] * trend
        + SCORE_WEIGHTS["volatility"] * vol
        + SCORE_WEIGHTS["drawdown"] * dd
        + SCORE_WEIGHTS["news"] * news_component
        + SCORE_WEIGHTS["position"] * weight_risk,
        1,
    )

    rf_proba = 0.0
    try:
        model = load_model()
        feat_row = row[MODEL_FEATURES].to_frame().T
        rf_proba = float(model.predict_proba(feat_row)[0, 1])
    except Exception:
        pass

    outlook = "Positiv" if confidence >= 60 else "Negativ" if confidence <= 40 else "Neutral"

    return AnalysisResult(
        ticker=ticker,
        price=float(row.get("close", 0.0)),
        trend_score=round(trend, 1),
        volatility_score=round(vol, 1),
        drawdown_score=round(dd, 1),
        news_score=news_score,
        news_label=news_label,
        asset_weight=asset_weight,
        confidence=confidence,
        outlook=outlook,
        rf_proba=rf_proba,
        raw=row.to_dict(),
    )
