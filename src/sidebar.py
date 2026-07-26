"""Global sidebar controls shared by every page (ticker, period, theme, data source)."""
from __future__ import annotations

import streamlit as st

from src import data as data_mod
from src.config import APP_NAME, APP_TAGLINE
from src.icons import icon
from src.theme import THEMES


def render_global_sidebar() -> None:
    market_df = data_mod.load_market_data()
    tickers = data_mod.available_tickers(market_df)

    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.1rem">'
            f'<span style="color:var(--ws-primary)">{icon("activity", 20)}</span>'
            f'<span style="font-family:var(--ws-font-display);font-size:1.25rem;font-weight:600">{APP_NAME}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(APP_TAGLINE)
        st.divider()

        st.session_state["ticker"] = st.selectbox(
            "Ticker", tickers,
            index=tickers.index(st.session_state.get("ticker", tickers[0])) if st.session_state.get("ticker") in tickers else 0,
        )
        st.session_state["period"] = st.select_slider(
            "Zeitraum", options=["1M", "3M", "6M", "1J", "3J", "5J", "Alle"],
            value=st.session_state.get("period", "1J"),
        )
        st.session_state["use_live_data"] = st.toggle(
            "Live-Kurse (yfinance)", value=st.session_state.get("use_live_data", False),
            help="Ohne: historische Kaggle-Daten (Trainingsverteilung des Modells). Mit: aktuelle Kurse, aber außerhalb der Trainingsverteilung.",
        )
        st.session_state["asset_weight"] = st.slider(
            "Positionsgröße (% Portfolio)", 1, 100, st.session_state.get("asset_weight", 15),
        )

        st.divider()
        st.session_state["theme_mode"] = st.radio("Darstellung", list(THEMES.keys()),
                                                    index=list(THEMES.keys()).index(st.session_state.get("theme_mode", "Hell")),
                                                    horizontal=True)
        st.session_state["app_mode"] = st.radio("Modus", ["Geführte Ansicht", "Expertenansicht"],
                                                  index=["Geführte Ansicht", "Expertenansicht"].index(
                                                      st.session_state.get("app_mode", "Geführte Ansicht")),
                                                  horizontal=True)
        st.session_state["enable_news"] = st.checkbox("News-Sentiment einbeziehen",
                                                        value=st.session_state.get("enable_news", True))

        st.divider()
        st.caption("⚠️ Keine Finanzberatung — Lernprojekt (IU).")
