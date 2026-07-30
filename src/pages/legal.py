from __future__ import annotations

import streamlit as st

from src.ui import page_header


def render_impressum() -> None:
    page_header("Impressum", "")
    st.markdown("""
**Angaben gemäß § 5 TMG** (Studienprojekt, nicht-kommerziell)

Soufiane Fedan
IU Internationale Hochschule
Kontakt: fedan.soufiane@gmail.com

Dieses Projekt ist eine akademische Studienarbeit im Rahmen des Moduls
Data Analytics und Big Data (DSDABD072501) und stellt kein kommerzielles Angebot dar.
    """)


def render_datenschutz() -> None:
    page_header("Datenschutz", "")
    st.markdown("""
**Datenverarbeitung in dieser App**

- Es werden keine personenbezogenen Nutzerdaten dauerhaft gespeichert.
- Hochgeladene CSV-Dateien werden nur in der laufenden Streamlit-Session gehalten.
- Optionale externe Anfragen: NewsAPI (Nachrichtentext), Google Gemini (Analyse-Kontext für den Assistenten) —
  jeweils nur bei aktivierter Funktion und vorhandenem API-Key.
- Keine Cookies zu Tracking-/Werbezwecken.

Diese App dient ausschließlich Lernzwecken und verarbeitet keine sensiblen personenbezogenen Daten.
    """)
