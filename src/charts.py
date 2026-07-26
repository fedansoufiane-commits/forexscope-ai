"""General-purpose Plotly chart builders (price, risk, portfolio, profile)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.model import AnalysisResult
from src.theme import THEMES, apply_chart_theme


def _t(mode: str) -> dict:
    return THEMES.get(mode, THEMES["Hell"])


def chart_price(df: pd.DataFrame, ticker: str, mode: str) -> go.Figure:
    t = _t(mode)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name=ticker,
                              line=dict(width=2, color=t["primary"])))
    for ma, color in [("ma_20", t["neutral"]), ("ma_50", t["positive"]), ("ma_200", t["negative"])]:
        if ma in df.columns:
            fig.add_trace(go.Scatter(x=df["date"], y=df[ma], mode="lines", name=ma.upper(),
                                      line=dict(width=1.2, dash="dot", color=color)))
    fig.update_layout(title=f"{ticker} — Kursverlauf & gleitende Durchschnitte", height=380)
    return apply_chart_theme(fig, mode, height=380)


def chart_candlestick(df: pd.DataFrame, ticker: str, mode: str) -> go.Figure:
    t = _t(mode)
    fig = go.Figure(go.Candlestick(
        x=df["date"], open=df.get("open", df["close"]), high=df.get("high", df["close"]),
        low=df.get("low", df["close"]), close=df["close"],
        increasing_line_color=t["positive"], decreasing_line_color=t["negative"],
    ))
    fig.update_layout(title=f"{ticker} — Candlestick", height=380, xaxis_rangeslider_visible=False)
    return apply_chart_theme(fig, mode, height=380)


def chart_drawdown(df: pd.DataFrame, ticker: str, mode: str) -> go.Figure:
    t = _t(mode)
    fig = go.Figure(go.Scatter(x=df["date"], y=df["drawdown"] * 100, mode="lines", fill="tozeroy",
                                line=dict(color=t["negative"], width=1.5), fillcolor=t["negative-soft"],
                                name="Drawdown"))
    fig.update_layout(title=f"{ticker} — Drawdown vom Allzeithoch", height=280, yaxis_title="%")
    return apply_chart_theme(fig, mode, height=280)


def chart_radar(result: AnalysisResult, mode: str) -> go.Figure:
    t = _t(mode)
    cats = ["Trend", "Volatilität", "Drawdown", "News", "Confidence"]
    vals = [result.trend_score, result.volatility_score, result.drawdown_score,
            50 + result.news_score * 10, result.confidence]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself",
                                     fillcolor=t["primary-soft"], line=dict(color=t["primary"])))
    fig.update_layout(title="Score-Profil", polar=dict(radialaxis=dict(range=[0, 100])), height=380)
    return apply_chart_theme(fig, mode, height=380)


def chart_gauge(result: AnalysisResult, mode: str) -> go.Figure:
    t = _t(mode)
    color = t["positive"] if result.confidence >= 60 else t["neutral"] if result.confidence >= 40 else t["negative"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=result.confidence,
        title={"text": "Confidence"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 40], "color": t["negative-soft"]},
                         {"range": [40, 60], "color": t["neutral-soft"]},
                         {"range": [60, 100], "color": t["positive-soft"]}]},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return apply_chart_theme(fig, mode, height=260)


def chart_risk_return_scatter(ranking: pd.DataFrame, mode: str) -> go.Figure:
    t = _t(mode)
    fig = go.Figure(go.Scatter(
        x=ranking["volatility_20d"], y=ranking["return_20d"], mode="markers+text",
        text=ranking["ticker"], textposition="top center",
        marker=dict(size=12, color=ranking["confidence"],
                    colorscale=[[0, t["negative"]], [0.5, t["neutral"]], [1, t["positive"]]],
                    cmin=0, cmax=100, showscale=True, colorbar=dict(title="Confidence")),
    ))
    fig.update_layout(title="Risiko/Rendite — Watchlist", xaxis_title="Volatilität (20d, ann.)",
                       yaxis_title="20d-Rendite", height=420)
    return apply_chart_theme(fig, mode, height=420)


def chart_portfolio_allocation(labels: list[str], values: list[float], mode: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45))
    fig.update_layout(title="Portfolio-Allokation", height=360)
    return apply_chart_theme(fig, mode, height=360)
