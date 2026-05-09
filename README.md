# WealthScope AI

WealthScope AI ist eine interaktive Streamlit-App für ein Uni-/QUA3CK-/Big-Data-Projekt.  
Die App kombiniert historische Markt-/ETF-Daten, Feature Engineering, NewsAPI-Daten, regelbasiertes Scoring, Visualisierungen, Exportfunktionen und einen kontextbasierten Analyse-Assistenten.

## Ziel

Das Projekt soll zeigen, wie eine datenbasierte Finanzanalyse-App nachvollziehbar, reproduzierbar und interaktiv umgesetzt werden kann.

Die App stellt keine Anlageberatung dar.

## Kernfunktionen

- lokale Kaggle-Marktdaten als vorbereiteter Feature-Datensatz
- 92.075 Feature-Zeilen, 23 Spalten, 18 Ticker
- technische Features wie Renditen, Moving Averages, Volatilität und Drawdown
- Zielvariable `target_20d`
- NewsAPI-Integration
- regelbasiertes Scoring
- interaktive Charts mit Plotly und Streamlit
- News-Karten
- Exportfunktionen
- Methodik- und Statusseiten
- QUA3CK-Notebooks zur wissenschaftlichen Dokumentation

## Projektstruktur

```text
app_max.py                         # Streamlit-Hauptapp
.streamlit/config.toml             # Streamlit-Theme-Konfiguration
assets/                            # Logo-Dateien
data/processed/                    # vorbereiteter Feature-Datensatz
notebooks/                         # QUA3CK-Notebooks
scripts/                           # Hilfsskripte und Validierung
tests/                             # statische Tests
docs/                              # Projekt- und Ausbau-Dokumentation
