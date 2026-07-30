# WealthScope AI 1.0

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
| **ML-Insights** | Historischer Modellvergleich, purged Out-of-Time-Test, Korrelation, Konfusionsmatrix, ROC/PR, Lernkurve, Feature Importance, SHAP |
| **Kapital-Kompass** | Risikoeinschätzung & Positionsgrößen-Empfehlung |
| **Portfolio-Simulator** | Kapitalplanung, Allokations-Editor, Konzentrationsmaß (HHI) |
| **Watchlist** | Ranking aller 26 Ticker nach Confidence-Score, Risiko/Rendite-Karte |
| **Datenlabor** | Explorative Datenanalyse über den vollständigen Trainingsdatensatz |
| **News & Assistent** | NewsAPI-Sentiment + Gemini-Chat zur aktuellen Analyse |
| **Export** | Markdown/CSV/ZIP/PDF-Bericht |
| **Lernstudio** | Acht Kursfragen, direktes Feedback und arsnova.eu-kompatibler JSON-Export |

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
| **Vergleich** | Dummy → Logistic Regression → Decision Tree → Linear SVM → Random Forest |
| **Demonstrator** | Random Forest Classifier (Pipeline: Median-Imputer → RF; StandardScaler nur bei Logistic Regression und Linear SVM, Bäume sind skaleninvariant) |
| **Zielvariable** | `target_20d`: Kurs in 20 Handelstagen höher? (0/1) |
| **Features (8)** | daily_return, return_5d, return_20d, ma_*_distance, volatility_20d, drawdown |
| **Train/Test** | Älteste 80 % / neueste 20 % nach Datum |
| **Leakage-Schutz** | 20 Handelstage Purge passend zu `target_20d` |
| **Cross-Validation** | 4 expandierende Walk-forward-Folds |
| **Diagnostik** | Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, PR, Lernkurve, Laufzeit, Importance und SHAP |

### Das Ergebnis — und warum es niedrig ist

| Kennzahl | Random Forest | Einordnung |
|---|---|---|
| Accuracy | 51,9 % | **unter** der Mehrheitsbaseline von 59,1 % |
| Balanced Accuracy | 51,3 % | knapp über Zufall (50 %) |
| ROC-AUC (Out-of-Time) | 0,519 | Zufallsniveau ist 0,5 |
| ROC-AUC (4 Walk-forward-Folds) | 0,514 | stabil schwach, kein Ausrutscher |

**H1 ist damit falsifiziert, nicht ungeprüft.** Das ist das zentrale Ergebnis des
Projekts: Rein kursbasierte technische Indikatoren tragen fast keine verwertbare
Information — gemessen an 190.527 Beobachtungen mit elf Jahren unangetastetem
Testzeitraum. Eine nachgerechnete statt zitierte Bestätigung der
Effizienzmarkthypothese (Fama 1970).

Die Lernkurve zeigt dabei **hohe Varianz** (Train-AUC 0,637 gegen Validierung
0,525, Gap 0,111), also Overfitting. Behebbar ist das hier aber nicht: Über die
2,3-fache Datenmenge bewegt sich die Validierung um −0,005, und über sieben
Regularisierungsstufen bleibt der Test-AUC in einer Spanne von 0,007. Was
überangepasst wird, ist Rauschen — kein untersampeltes Signal.

Alle Metriken/Kurven werden **nicht** bei jedem App-Rerun neu gefittet, sondern
einmalig von `scripts/train_and_diagnose.py` berechnet. Die erzeugten Artefakte
werden für ein reproduzierbares Deployment versioniert. Nach Änderungen an
Features, Hyperparametern oder Daten:

```bash
python3 scripts/train_and_diagnose.py
```

Ein schwaches Ergebnis ist nur dann ein Befund, wenn die naheliegenden
Gegenerklärungen ausgeschlossen sind. `scripts/validation_experiments.py`
misst beide und schreibt `models/validation_experiments.json`:

- **Kapazitäts-Sweep** — sieben Regularisierungsstufen von unbeschränkt
  (Train-AUC 1,000) bis stark gestutzt (0,543). Der Test-AUC bewegt sich dabei
  nur um 0,007. Die Trainings-Test-Lücke entsteht ausschließlich auf der
  Trainingsseite; Kapazität ist nicht der Engpass.
- **Split-Vergleich** — dasselbe Modell auf denselben Daten, nur die Aufteilung
  variiert: purged Out-of-Time 0,519 · ohne Sperrzone 0,524 · naiver
  Zufalls-Split 0,581. Gemessen am Zufallsniveau 0,5 erscheint das Signal beim
  naiven Split 4,2-mal größer.

```bash
python3 scripts/validation_experiments.py
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
  diagnostics.py                Modellvergleich, Korrelation, ROC/PR, Lernkurve, SHAP
  quiz.py                       Lernfragen + arsnova.eu-Export
  charts.py                     Kurs-/Risiko-/Portfolio-Charts (Plotly)
  news.py                       NewsAPI + Lexikon-Sentiment + Gemini-Assistent
  export.py                     Markdown/CSV/ZIP/PDF-Export
  pages/                        Eine Datei pro Seite (start, market, ml_insights, ...)
scripts/
  train_and_diagnose.py         Purged Out-of-Time-Benchmark + Artefakte
  validation_experiments.py     Falsifikationsexperimente: Kapazitäts-Sweep + Split-Vergleich
  build_wealthscope_report.py   Erzeugt die fünfseitige Ausarbeitung (.docx)
  validate_app.py               Struktur-Check (Compile, Seiten, url_path, Artefakte)
  rebuild_wealthscope_notebooks.py  Generiert die QUA³CK-Notebooks
tests/
  test_app_static.py            pytest-Suite für App, Artefakte und Quizformat
notebooks/                       8 QUA³CK-Notebooks (wissenschaftliche Dokumentation)
data/, models/                   Datensatz, trainiertes Modell, Diagnostik-Caches
_archiv/pre_rebuild_2026-07-09/  Der alte app_max.py-Monolith (Referenz, nicht aktiv)
```

Der vollständige Stand unmittelbar vor 1.0 ist zusätzlich im Commit `834ff98`
auf `codex/backup-status-quo-2026-07-26` gesichert.

---

## Setup & Ausführen

```bash
# 1. Repository klonen / Ordner öffnen
cd "Big Data and Data Analytics"

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# 3. Modell + Diagnostik-Caches einmalig erzeugen (falls nicht vorhanden)
python3 scripts/train_and_diagnose.py
python3 scripts/validation_experiments.py

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
- **Breiman (2001):** Random Forests. Machine Learning, 45, 5–32.
- **Quibeldey-Cirkel (2026):** Kursmaterialien zu QUA³CK, Datenverständnis,
  Klassifikation, Modelltraining, SVM, Entscheidungsbäumen und Random Forests.

Weitere Details: [`docs/model_card.md`](docs/model_card.md) und
[`docs/lecture_alignment.md`](docs/lecture_alignment.md).
