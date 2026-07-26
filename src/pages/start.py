from __future__ import annotations

import streamlit as st

from src.charts import chart_gauge, chart_price
from src.context import get_context
from src.icons import icon
from src.ui import badge, card, disclaimer_footer, kpi_grid, page_header


def render() -> None:
    ctx = get_context()
    result, mode = ctx["result"], ctx["theme_mode"]

    page_header("Willkommen bei WealthScope AI",
                "Interaktive Finanzanalyse auf Basis von Machine Learning — Uni-Projekt, IU Internationale Hochschule.")

    outlook_kind = "positive" if result.outlook == "Positiv" else "negative" if result.outlook == "Negativ" else "neutral"
    card(
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem">'
        f'<div><div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:var(--ws-text-muted)">Aktuelles Signal</div>'
        f'<div style="font-family:var(--ws-font-display);font-size:2rem;font-weight:600">{ctx["ticker"]} — {result.outlook}</div></div>'
        f'{badge(f"Confidence {result.confidence}/100", outlook_kind)}'
        f'</div>',
        variant="ws-hero",
    )

    kpi_grid([
        ("Kurs", f"{result.price:,.2f}", ctx["ticker"], "trend-up"),
        ("Trend-Score", f"{result.trend_score:.0f}", "MA-Position", "chart"),
        ("Volatilität-Score", f"{result.volatility_score:.0f}", "20d annualisiert", "activity"),
        ("RF-Wahrscheinlichkeit", f"{result.rf_proba*100:.1f}%", "Bullish (20d)", "target"),
    ])

    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(chart_price(ctx["period_df"], ctx["ticker"], mode),
                         config={"displayModeBar": False})
    with col2:
        st.plotly_chart(chart_gauge(result, mode),
                         config={"displayModeBar": False})

    st.markdown("#### Was kann diese App?")
    c1, c2, c3, c4 = st.columns(4)
    for col, icon_name, title, desc in [
        (c1, "chart", "Marktanalyse", "Kurscharts, Candlesticks, Drawdown, gleitende Durchschnitte."),
        (c2, "flask", "ML-Insights", "Korrelationsmatrix, Konfusionsmatrix, Lernkurve, ROC/PR, SHAP."),
        (c3, "compass", "Kapital-Kompass", "Risikoeinschätzung & Positionsgrößen-Empfehlung."),
        (c4, "briefcase", "Simulator", "Portfolio-Szenarien und Kapitalplanung."),
    ]:
        with col:
            card(f'<div style="color:var(--ws-primary);margin-bottom:0.4rem">{icon(icon_name, 22)}</div>'
                 f'<b>{title}</b><br>'
                 f'<span style="font-size:0.82rem;color:var(--ws-text-muted)">{desc}</span>')

    disclaimer_footer()
