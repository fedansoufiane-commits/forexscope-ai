from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import APP_VERSION, DIAGNOSTICS_PATH, LEARNING_CURVE_PATH, MARKET_PARQUET, MODEL_PATH
from src.news import get_secret
from src.ui import badge, kpi_grid, page_header, section_title


def _check(path: Path) -> str:
    return badge("OK", "positive") if path.exists() else badge("fehlt", "negative")


def render() -> None:
    page_header("Betriebsstatus", "Systemdiagnose — Daten, Modell, API-Keys")

    section_title("Artefakte", "layers")
    for label, path in [
        ("Marktdaten (Parquet)", MARKET_PARQUET),
        ("Trainiertes Modell", MODEL_PATH),
        ("Diagnostics-Cache", DIAGNOSTICS_PATH),
        ("Lernkurven-Cache", LEARNING_CURVE_PATH),
    ]:
        st.markdown(f"{label}: {_check(path)}", unsafe_allow_html=True)

    section_title("API-Keys (optional)", "lock")
    for label, key in [("NewsAPI", "NEWS_API_KEY"), ("Gemini", "GEMINI_API_KEY")]:
        present = bool(get_secret(key))
        st.markdown(f"{label}: {badge('gesetzt', 'positive') if present else badge('nicht gesetzt', 'neutral')}",
                    unsafe_allow_html=True)

    section_title("Laufzeit", "activity")
    import sklearn
    import streamlit as st_mod
    kpi_grid([
        ("WealthScope", APP_VERSION, "Release", "check"),
        ("Streamlit", st_mod.__version__, "", "activity"),
        ("scikit-learn", sklearn.__version__, "", "flask"),
    ])
