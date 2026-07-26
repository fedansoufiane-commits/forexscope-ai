from __future__ import annotations

import streamlit as st

from src.config import APP_NAME
from src.ui import card, disclaimer_footer, page_header, section_title


def render() -> None:
    page_header(APP_NAME, "Projektinformationen")

    card(
        "<b>Bearbeiter:</b> Soufiane Fedan · soufiane.fedan@solvvision.de<br>"
        "<b>Modul:</b> Data Analytics und Big Data (DSDABD072501) · IU Internationale Hochschule<br>"
        "<b>Tutor:</b> Klaus Quibeldey-Cirkel · Prüfungsform: Klausur<br>"
        "<b>Prozessmodell:</b> QUA³CK (Stock et al., 2021, KIT ITIV)<br>"
        "<b>Datenbasis:</b> Kaggle US Stocks &amp; ETFs (Boris Marjanovic), CC0-Lizenz, 192.119 Zeilen, 26 Ticker"
    )

    section_title("Architektur", "layers")
    st.code("""
app.py                 Streamlit-Entrypoint (st.navigation)
src/
  config.py            Pfade, Konstanten, Ticker-/Feature-Listen
  theme.py, icons.py, ui.py   Design-System, eigenes Icon-Set, UI-Bausteine
  data.py                Laden & Feature Engineering (Kaggle + live yfinance)
  model.py                Rule-based Scoring-Engine + Modell-Loader
  diagnostics.py           Korrelation, Konfusionsmatrix, ROC/PR, Lernkurve, SHAP
  charts.py                 Kurs-/Risiko-/Portfolio-Charts
  news.py                    NewsAPI + Sentiment + Gemini-Assistent
  export.py                   PDF/CSV/ZIP-Export
  pages/*.py                    Eine Datei pro Seite
scripts/train_and_diagnose.py  Einmaliges Training + Diagnostik-Cache (JSON)
    """, language="text")

    disclaimer_footer()
