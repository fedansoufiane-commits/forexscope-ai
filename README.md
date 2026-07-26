# WealthScope AI

**Interaktive Finanzanalyse-App auf Basis von Machine Learning**
Uni-Projekt | IU Internationale Hochschule | Modul: Data Analytics und Big Data (DSDABD072501)
Bearbeiter: Soufiane Fedan | soufiane.fedan@solvvision.de | Tutor: Klaus Quibeldey-Cirkel

> ⚠️ **Disclaimer:** Dieses Projekt dient ausschließlich Lernzwecken. Keine Finanzberatung.
> Alle Analysen sind Demonstrationen. Keine Haftung für Handelsentscheidungen.

---

## Über das Projekt

WealthScope AI ist eine interaktive Streamlit-App, die historische US-Aktien- und ETF-Daten
mit technischen Indikatoren, Machine-Learning-Signalen und einer NewsAPI-Integration verbindet.
Das Projekt folgt dem **QUA³CK-Prozessmodell** (Stock et al., 2021, KIT ITIV) und ist
vollständig durch Jupyter-Notebooks wissenschaftlich dokumentiert.

Die App wurde vollständig modular neu aufgebaut (siehe „Architektur" unten): statt eines
5.600-Zeilen-Monolithen (`app_max.py`, archiviert unter `_archiv/pre_rebuild_2026-07-09/`)
gibt es jetzt ein schlankes `app.py` mit `st.navigation` und ein `src/`-Package pro Verantwortlichkeit.

---

## QUA³CK-Prozess

```
Q → Question          01_question.ipynb
U → Understanding     02_understanding_the_data.ipynb    ← Datenphase (EDA, Imputation, Scaling)
A → Analytics         03_feature_engineering.ipynb
A → Algorithm         04_modeling_baseline_ml.ipynb      ← ML-Modelle, CV, ROC, CM
A → Adaption          04_modeling_baseline_ml.ipynb
C → Conclude          05_conclude_evaluate.ipynb
K → Knowledge         06_knowledge_transfer_streamlit.ipynb
                      07_newsapi_assistant_export.ipynb
```

Interaktiv erklärt in der App unter **Methodik (QUA³CK)**.

---

## Kernfunktionen der App

| Feature | Beschreibung |
|---|---|
| **Marktanalyse** | Kurscharts, Candlestick, gleitende Durchschnitte, Drawdown |
| **ML-Insights** | Score-Zerlegung, **Korrelationsmatrix (Heatmap)**, **Konfusionsmatrix + ROC/PR**, **Lernkurve (Bias/Variance)**, Feature Importance, SHAP |
| **Kapital-Kompass** | Risikoeinschätzung & Positionsgrößen-Empfehlung |
| **Portfolio-Simulator** | Kapitalplanung, Allokations-Editor, Konzentrationsmaß (HHI) |
| **Watchlist** | Ranking aller 26 Ticker nach Confidence-Score, Risiko/Rendite-Karte |
| **Datenlabor** | Explorative Datenanalyse über den vollständigen Trainingsdatensatz |
| **News & Assistent** | NewsAPI-Sentiment + Gemini-Chat zur aktuellen Analyse |
| **Export** | Markdown/CSV/ZIP/PDF-Bericht |

---

## Datenbasis

| Attribut | Wert |
|---|---|
| **Quelle** | Kaggle – US Stocks & ETFs (Boris Marjanovic) |
| **Lizenz** | CC0 Public Domain |
| **Zeilen** | 192.119 Feature-Datenpunkte |
| **Spalten** | 27 (siehe `data/processed/wealthscope_features.parquet`) |
| **Ticker** | 26 Aktien und ETFs |
| **Zeitraum** | ca. 1962–2017 |
| **Format** | Apache Parquet + CSV (lokal) |

---

## ML-Modell & Diagnostik

| Aspekt | Detail |
|---|---|
| **Algorithmus** | Random Forest Classifier (Pipeline: Imputer → StandardScaler → RF) |
| **Zielvariable** | `target_20d`: Kurs in 20 Handelstagen höher? (0/1) |
| **Features (8)** | daily_return, return_5d, return_20d, ma_*_distance, volatility_20d, drawdown |
| **Train/Test** | 75/25, stratifiziert, `random_state=42` |
| **Accuracy** | ~55,3 % (leicht über Zufallsniveau – wissenschaftlich erwartet, EMH) |
| **ROC-AUC** | ~0.59 |
| **Cross-Validation** | 5-Fold, stratifiziert |
| **Diagnostik** | Konfusionsmatrix, ROC/PR-Kurven, **Lernkurve** (`sklearn.model_selection.learning_curve`), SHAP — alle einmalig vorberechnet, siehe unten |

Alle Metriken/Kurven werden **nicht** bei jedem App-Rerun neu gefittet (das wäre bei
`learning_curve()` — ~40 RF-Fits — viel zu langsam), sondern einmalig von
`scripts/train_and_diagnose.py` berechnet und in `models/diagnostics.json` /
`models/learning_curve.json` gecacht. Nach Änderungen an Features, Hyperparametern
oder Daten einfach neu ausführen:

```bash
python3 scripts/train_and_diagnose.py
```

---

## Architektur

```
app.py                          Streamlit-Entrypoint (st.navigation, Sidebar, Theme)
src/
  config.py                     Pfade, Konstanten, Ticker-/Feature-Listen
  state.py                      Session-State-Defaults
  context.py                    Baut den Analyse-Kontext pro Rerun (Daten, Scores, News)
  sidebar.py                    Globale Sidebar-Controls (Ticker, Zeitraum, Theme, Modus)
  theme.py                      Design-System: CSS-Variablen (Hell/Dunkel) + Plotly-Theme
  ui.py                         Wiederverwendbare UI-Bausteine (Card, KPI-Grid, Badge)
  data.py                       Laden & Feature Engineering (Kaggle-Parquet + live yfinance)
  model.py                      Regelbasierte Scoring-Engine + Modell-Loader
  diagnostics.py                Korrelation, Konfusionsmatrix, ROC/PR, Lernkurve, SHAP
  charts.py                     Kurs-/Risiko-/Portfolio-Charts (Plotly)
  news.py                       NewsAPI + Lexikon-Sentiment + Gemini-Assistent
  export.py                     Markdown/CSV/ZIP/PDF-Export
  pages/                        Eine Datei pro Seite (start, market, ml_insights, ...)
scripts/
  train_and_diagnose.py         Einmaliges Training + Diagnostik-Cache (siehe oben)
  validate_app.py               Struktur-Check (Compile, Seiten, url_path, Artefakte)
  rebuild_wealthscope_notebooks.py  Generiert die QUA³CK-Notebooks
tests/
  test_app_static.py            pytest-Suite für die obigen Invarianten
notebooks/                       8 QUA³CK-Notebooks (wissenschaftliche Dokumentation)
data/, models/                   Datensatz, trainiertes Modell, Diagnostik-Caches
_archiv/pre_rebuild_2026-07-09/  Der alte app_max.py-Monolith (Referenz, nicht aktiv)
```

---

## Setup & Ausführen

```bash
# 1. Repository klonen / Ordner öffnen
cd "Big Data and Data Analytics"

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# 3. Modell + Diagnostik-Caches einmalig erzeugen (falls nicht vorhanden)
python3 scripts/train_and_diagnose.py

# 4. API-Keys konfigurieren (optional, für NewsAPI + Gemini)
# .streamlit/secrets.toml:
# NEWS_API_KEY = "dein_key"
# GEMINI_API_KEY = "dein_key"

# 5. App starten
streamlit run app.py
# oder: Startbefehle/start_app.command
```

### Tests & Validierung

```bash
python3 -m pytest tests/
python3 scripts/validate_app.py
```

### Notebooks ausführen

```bash
jupyter lab
# oder: Startbefehle/start_jupyter.command
```

---

## Technischer Stack

| Komponente | Technologie |
|---|---|
| App-Framework | Streamlit (`st.navigation`, Multipage) |
| Datenverarbeitung | pandas, numpy |
| Visualisierung | Plotly, matplotlib, seaborn |
| Machine Learning | scikit-learn, SHAP |
| Datenformat | Apache Parquet (pyarrow) |
| KI-Assistent | Google Gemini (google-genai) |
| News | NewsAPI (requests) |
| Export | ReportLab (PDF) |
| Tests | pytest |

---

## Wissenschaftliche Quellen

- **Stock et al. (2021):** QUA³CK – A Machine Learning Development Process. KIT ITIV.
  https://publikationen.bibliothek.kit.edu/1000129631
- **Fama (1970):** Efficient Capital Markets. Journal of Finance, 25(2), 383–417.
- **Li et al. (2024):** Comparison of Imputation Methods. BMC Medical Research Methodology.
  https://doi.org/10.1186/s12874-024-02173-x
- **Pinheiro et al. (2025):** The Impact of Feature Scaling in ML. arXiv:2506.08274.
- **Lundberg & Lee (2017):** A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.
- **Quibeldey-Cirkel (2026):** U: Understanding the Data. IU Internationale Hochschule.
