"""WealthScope AI — Streamlit entrypoint.

Modular rebuild: this file only wires together config, theme, global sidebar
controls and page routing via `st.navigation`. All actual logic lives in
`src/`. See src/pages/ for one module per page.
"""
from __future__ import annotations

import streamlit as st

from src.config import APP_NAME
from src.pages import (
    assistant_page, data_lab, export_page, kompass, legal,
    market, methodology, ml_insights, news_page, project,
    simulator, start, status, watchlist,
)
from src.sidebar import render_global_sidebar
from src.state import init_state
from src.theme import inject_base_css, inject_theme_vars

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
inject_base_css()

pages = {
    "Analyse": [
        st.Page(start.render, title="Start", icon=":material/home:", url_path="start", default=True),
        st.Page(market.render, title="Marktanalyse", icon=":material/candlestick_chart:", url_path="marktanalyse"),
        st.Page(ml_insights.render, title="ML-Insights", icon=":material/query_stats:", url_path="ml-insights"),
        st.Page(kompass.render, title="Kapital-Kompass", icon=":material/explore:", url_path="kompass"),
        st.Page(simulator.render, title="Portfolio-Simulator", icon=":material/account_balance_wallet:", url_path="simulator"),
        st.Page(watchlist.render, title="Watchlist", icon=":material/visibility:", url_path="watchlist"),
        st.Page(data_lab.render, title="Datenlabor", icon=":material/science:", url_path="datenlabor"),
    ],
    "Service": [
        st.Page(news_page.render, title="News-Archiv", icon=":material/newspaper:", url_path="news"),
        st.Page(assistant_page.render, title="KI-Assistent", icon=":material/smart_toy:", url_path="assistent"),
        st.Page(methodology.render, title="Methodik (QUA³CK)", icon=":material/menu_book:", url_path="methodik"),
        st.Page(project.render, title="Projekt", icon=":material/info:", url_path="projekt"),
        st.Page(export_page.render, title="Export", icon=":material/download:", url_path="export"),
    ],
    "Rechtliches": [
        st.Page(legal.render_impressum, title="Impressum", icon=":material/balance:", url_path="impressum"),
        st.Page(legal.render_datenschutz, title="Datenschutz", icon=":material/lock:", url_path="datenschutz"),
        st.Page(status.render, title="Status", icon=":material/monitor_heart:", url_path="status"),
    ],
}

nav = st.navigation(pages, position="sidebar")
render_global_sidebar()
inject_theme_vars(st.session_state.get("theme_mode", "Hell"))
nav.run()
