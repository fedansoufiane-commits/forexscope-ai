from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.config import FEATURE_LABELS, MODEL_FEATURES
from src.context import get_context
from src.theme import apply_chart_theme
from src.ui import disclaimer_footer, kpi_grid, page_header, section_title


def render() -> None:
    ctx = get_context()
    market_df, mode = ctx["market"], ctx["theme_mode"]

    page_header("Datenlabor", "Explorative Datenanalyse (EDA) über den kompletten Trainingsdatensatz")

    kpi_grid([
        ("Zeilen", f"{len(market_df):,}", "1962–2017", "layers"),
        ("Ticker", str(market_df["ticker"].nunique()), "US Blue Chips & ETFs", "briefcase"),
        ("Fehlwerte", f"{market_df[MODEL_FEATURES].isna().mean().mean()*100:.1f}%", "im Schnitt (MA-Warmup)", "warning"),
        ("Klassenbalance", f"{market_df['target_20d'].mean()*100:.1f}%", "Bullish (target_20d=1)", "grid"),
    ])

    section_title("Verteilung eines Features", "chart")
    feature = st.selectbox("Feature", MODEL_FEATURES, format_func=lambda f: FEATURE_LABELS.get(f, f))
    sample = market_df[[feature, "ticker"]].dropna().sample(min(20000, len(market_df)), random_state=42)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(sample, x=feature, nbins=60, marginal="box")
        fig.update_layout(title=f"Verteilung: {FEATURE_LABELS.get(feature, feature)}", height=380)
        st.plotly_chart(apply_chart_theme(fig, mode, 380), config={"displayModeBar": False})
    with c2:
        fig2 = px.box(sample, x="ticker", y=feature, points=False)
        fig2.update_layout(title=f"Nach Ticker: {FEATURE_LABELS.get(feature, feature)}", height=380)
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(apply_chart_theme(fig2, mode, 380), config={"displayModeBar": False})

    section_title("Fehlwerte je Spalte", "warning")
    missing = (market_df[MODEL_FEATURES].isna().mean() * 100).round(2).sort_values(ascending=False)
    st.bar_chart(missing)
    st.caption(
        "Fehlwerte entstehen durch Warmup-Perioden gleitender Durchschnitte (MA-200 braucht 200 Handelstage "
        "Vorlauf) — MCAR (missing completely at random), nicht systematisch verzerrt."
    )

    disclaimer_footer()
