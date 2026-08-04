# WealthScope AI — Next Level Plan

## Status: Stufe 1 abgeschlossen

Die ursprüngliche 5-Stufen-Liste (Stabilität, Performance, UX, Wissenschaftlichkeit,
Präsentation) ist mit dem Rebuild vom 2026-07-09 vollständig umgesetzt:

| Stufe | Umgesetzt als |
|---|---|
| Stabilität | `tests/test_app_static.py`, `scripts/validate_app.py`, modulare `src/`-Struktur statt Monolith |
| Performance | `st.cache_data`/`st.cache_resource` durchgängig, Diagnostik/Lernkurve vorberechnet (`scripts/train_and_diagnose.py` → JSON-Cache statt Live-Fit) |
| UX | News-Karten, KI-Assistent (Chat), Methodik-Seite, ZIP/PDF-Export, Status-Seite, eigenes Icon-System statt Emoji |
| Wissenschaftlichkeit | Konfusionsmatrix, ROC/PR, **Lernkurve (Bias/Variance-Diagnose)**, Korrelationsmatrix, RF-Feature-Importance, SHAP, EMH-Einordnung |
| Präsentation | Projekt-Seite, Methodik-/QUA³CK-Seite, exportierbarer Markdown/CSV/ZIP/PDF-Bericht |

## In Version 1.0 abgeschlossen

### A. Modell-Robustheit
- Purged Out-of-Time-Holdout statt zufälligem 75/25-Split
- Vier expandierende Walk-forward-Folds
- Fünf Modellgenerationen auf identischen Zeitfenstern
- Model Card und versionierte Diagnostik-Artefakte

## Nächste sinnvolle Ausbaustufen (nach 1.0)

### B. Datenbasis erweitern
- VIX / Makro-Features (vor dem 1.0-Rebuild prototypisch erprobt, aber nicht in
  die aktuelle App übernommen)
- Sentiment aus strukturierten Quellen statt Lexikon-basiertem NewsAPI-Scoring

### C. Deployment
- Streamlit Community Cloud oder Docker-Image für öffentliche Demo-Verfügbarkeit
- Secrets-Handling für den Cloud-Fall dokumentieren (aktuell nur lokale `secrets.toml`)

### D. Tests vertiefen
- Snapshot-Tests für die Diagnostik-Werte (Regression erkennen, falls sich
  Metriken nach einem Retraining unerwartet stark verschieben)
- Playwright/Streamlit-AppTest-Suite für die interaktiven Seiten (aktuell nur
  statische Struktur-Checks)
