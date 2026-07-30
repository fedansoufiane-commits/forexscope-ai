from __future__ import annotations

import streamlit as st

from src.context import get_context
from src.ui import badge, card, disclaimer_footer, kpi_grid, page_header


def render() -> None:
    ctx = get_context()
    result = ctx["result"]

    page_header("Kapital-Kompass", "Risikoeinschätzung & Positionsgrößen-Empfehlung")

    vol = result.volatility_score
    risk_level = "Niedrig" if vol >= 65 else "Hoch" if vol <= 35 else "Mittel"
    risk_kind = "positive" if risk_level == "Niedrig" else "negative" if risk_level == "Hoch" else "neutral"

    capital = st.number_input("Verfügbares Kapital (€)", min_value=100.0, value=1000.0, step=100.0)
    max_pct = 25 if risk_level == "Niedrig" else 10 if risk_level == "Hoch" else 15
    suggested = round(capital * max_pct / 100, 2)

    kpi_grid([
        ("Risikoklasse", risk_level, f"Volatilitäts-Score {vol:.0f}/100", "activity"),
        ("Empfohlene Positionsgröße", f"{max_pct}%", f"≈ {suggested:,.2f} €", "briefcase"),
        ("Confidence", f"{result.confidence:.0f}/100", result.outlook, "target"),
        ("Stop-Loss (Richtwert)", "1.5× ATR", "≈ 2–4% vom Kurs", "trend-down"),
    ])

    card(
        f'<b>Einschätzung:</b> Bei {badge(risk_level, risk_kind)} eingeschätzter Volatilität und '
        f'Confidence {result.confidence:.0f}/100 wäre eine Positionsgröße von etwa <b>{max_pct}%</b> '
        f'des verfügbaren Kapitals ({suggested:,.2f} €) eine konservative Ausgangsbasis. '
        f'Dies ist eine regelbasierte Heuristik, keine individuelle Anlageberatung.',
        variant="ws-accent",
    )

    st.markdown("#### Positionsgrößen-Rechner")
    c1, c2 = st.columns(2)
    with c1:
        risk_pct = st.slider("Max. Risiko pro Trade (%)", 0.5, 5.0, 2.0, 0.5)
        stop_pct = st.slider("Stop-Loss-Distanz (%)", 0.5, 10.0, 3.0, 0.5)
    with c2:
        risk_amount = capital * risk_pct / 100
        position_size = risk_amount / (stop_pct / 100) if stop_pct else 0
        st.metric("Risikobetrag", f"{risk_amount:,.2f} €")
        st.metric("Maximale Positionsgröße", f"{min(position_size, capital):,.2f} €")

    disclaimer_footer()
