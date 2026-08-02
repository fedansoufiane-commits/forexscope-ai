# QUA³CK-Prozessmodell – WealthScope AI

**Hochschule:** IU Internationale Hochschule  
**Dozent:** Prof. Dr. Klaus Quibeldey-Cirkel  
**Projekt:** WealthScope AI  
**Bearbeiter:** Soufiane Fedan

> Quelle des Prozessmodells: Stock, S. C. et al. (2021): „QUA³CK – A Machine Learning Development Process".  
> KIT, Institut für Technik der Informationsverarbeitung (ITIV).  
> https://publikationen.bibliothek.kit.edu/1000129631

---

## Das QUA³CK-Modell im Überblick

```
┌─────────────────────────────────────────────────────────┐
│                    QUA³CK-Prozess                       │
├──────┬──────────────────────────────────────────────────┤
│  Q   │  Question         – Fragestellung definieren     │
│  U   │  Understanding    – Daten verstehen & aufbereiten│
│  A   │  Analytics        – Analyse & Feature Engineering│
│  A   │  Algorithm        – ML-Algorithmus auswählen     │
│  A   │  Adaption         – Modell optimieren & validieren│
│  C   │  Conclude         – Ergebnisse bewerten          │
│  K   │  Knowledge        – Wissen kommunizieren & nutzen│
└──────┴──────────────────────────────────────────────────┘
```

---

## Q – Question / Fragestellung

### Leitfrage

> „Wie können historische US-Aktienmarktdaten genutzt werden, um mit Machine Learning  
> eine interaktive Finanzanalyse-App zu entwickeln, die technische Marktanalyse,  
> ML-basierte Signalgebung und risikobasierte Positionsplanung nachvollziehbar kombiniert?"

### Zielgruppe

- Studierende im Bereich Data Science / Finance
- Privatanleger mit Interesse an datengetriebenen Ansätzen
- Demonstrationsplattform für akademische Präsentationen

### Abgrenzung

WealthScope AI ist ein **wissenschaftlicher Prototyp** – kein Handelsbot, keine Anlageberatung.

### Notebook

→ `01_question.ipynb`

---

## U – Understanding the Data / Datenphase

### Datenquellen

| Quelle | Art | Verwendung |
|---|---|---|
| Kaggle US Stocks & ETFs | CSV/TXT, strukturiert | Historische OHLCV-Daten |
| yfinance (optional) | API, strukturiert | Live-Kurse in der App |
| NewsAPI | JSON, semi-strukturiert | Finanznachrichten |

### Durchgeführte Analysen

**Datenqualität:**
- Datensatz-Profil: 192.119 Zeilen × 27 Spalten, 26 Ticker
- Fehlende Werte: Identifizierung, Kategorisierung (MCAR/MAR/MNAR), Quantifizierung

**Explorative Datenanalyse (EDA):**
- Histogramme mit KDE für alle 8 ML-Features
- Boxplots zur Ausreißer-Identifikation (IQR-Methode)
- Scatterplots: Feature-Zusammenhänge mit Zielvariable
- Pearson- und Spearman-Korrelationsmatrix

**Fehlende Werte – Imputationsstrategie:**
- Typ: MCAR (Moving Average Warm-up, erste 20–200 Handelstage je Ticker)
- Vergleich: Mean vs. Median vs. KNN (k=5)
- Entscheidung: **Median-Imputation** in sklearn-Pipeline
- Begründung: MCAR-Daten + Robustheit gegenüber Ausreißern

**Feature Scaling:**
- StandardScaler (Z-Score): μ=0, σ=1 – gewählt für ML-Pipeline
- MinMaxScaler: [0,1] – verglichen
- Data-Leakage-Prävention: `fit()` nur auf Trainingsdaten

### Notebook

→ `02_understanding_the_data.ipynb`

### Wissenschaftliche Belege

- Li et al. (2024): 5–10 % fehlende Werte → bis zu 15 % Genauigkeitsverlust. BMC.
- Pinheiro et al. (2025): Feature Scaling → bis zu 25 % Performance-Verbesserung. arXiv.

---

## A – Analytics / Feature Engineering

### Features

| Feature | Gruppe | Formel |
|---|---|---|
| `daily_return` | Rendite | (close_t − close_{t-1}) / close_{t-1} |
| `return_5d` | Rendite | (close_t − close_{t-5}) / close_{t-5} |
| `return_20d` | Rendite | (close_t − close_{t-20}) / close_{t-20} |
| `ma_20_distance` | Trend | (close − MA20) / MA20 |
| `ma_50_distance` | Trend | (close − MA50) / MA50 |
| `ma_200_distance` | Trend | (close − MA200) / MA200 |
| `volatility_20d` | Risiko | Std(daily_return, Fenster=20) |
| `drawdown` | Risiko | (close − rolling_max) / rolling_max |

### Zielvariable

`target_20d`: Binär – steigt der Kurs in 20 Handelstagen? (1 = ja, 0 = nein)

### Notebook

→ `03_feature_engineering.ipynb`

---

## A – Algorithm / Modellauswahl

### Verglichene Modelle

| Modell | Typ | Begründung |
|---|---|---|
| DummyClassifier | Baseline | Untergrenze definieren |
| Logistische Regression | Parametrisch, linear | Interpretierbar, schnell |
| Entscheidungsbaum | Regelbasiert, nichtlinear | White-Box-Vergleich |
| Linear SVM | Maximum-Margin-Verfahren | Historischer SVM-Vergleich |
| Random Forest | Ensemble, nicht-parametrisch | Robust, non-linear, kein Scaling nötig |

### Entscheidung

**Random Forest** als finales Modell:
- Robustheit gegenüber Ausreißern und Korrelationen
- Kein striktes Scaling erforderlich
- Feature Importance direkt verfügbar
- Ensemble-Methode mit niedrigerer Varianz

### Notebook

→ `04_modeling_baseline_ml.ipynb`

---

## A – Adaption / Modell-Optimierung

### Maßnahmen

- `class_weight='balanced'`: Ausgleich der leichten Klassenimbalance
- `max_depth=8`: Regularisierung gegen Overfitting
- `n_estimators=200`: Stabilere Schätzungen
- **4 expandierende Walk-forward-Folds**: zeitlich belastbare Performance-Schätzung
- Neueste 20 % der Handelstage als unangetasteter Out-of-Time-Test
- 20 Handelstage Sperrzone vor jedem Validierungsfenster passend zu `target_20d`

### Pipeline-Design (kein Data Leakage)

Der Scaler ist modellabhängig: Logistische Regression und Linear SVM brauchen ihn,
Entscheidungsbaum und Random Forest nicht — Schwellenwert-Splits sind
skaleninvariant. `scripts/train_and_diagnose.py:build_pipeline()` setzt deshalb
`"passthrough"` statt eines Scalers, sobald das Modell ein Baumverfahren ist:

```python
scaler = StandardScaler() if model_key in {"logistic", "linear_svm"} else "passthrough"

Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scl", scaler),                            # passthrough beim Random Forest
    ("mod", RandomForestClassifier(...)),
])
```

Imputer und Scaler werden ausschließlich auf dem Trainingsfenster gefittet — das
verhindert Preprocessing-Leakage.

### Ergebnisse

| Metrik | Majority Baseline | Logist. Regression | Random Forest |
|---|---|---|---|
| Accuracy | 59,1 % | 54,7 % | 51,9 % |
| Balanced Accuracy | 50,0 % | 50,4 % | 51,3 % |
| ROC-AUC | 0,500 | 0,517 | 0,519 |
| Walk-forward ROC-AUC | 0,500 | 0,491 | 0,514 |

### Interpretation

Der Random Forest schlägt die Mehrheits-Baseline **nicht bei roher Accuracy**,
liegt aber bei Balanced Accuracy und ROC-AUC knapp über Zufall. Das ist ein
wichtiges Negativergebnis: historische Preisdaten besitzen nur ein schwaches,
nicht stabiles Signal (Efficient Market Hypothesis, Fama 1970).

---

## C – Conclude / Fazit

### Kernergebnisse

- H1 nur schwach gestützt: RF-AUC liegt knapp über 0,5, der Vorsprung ist nicht handlungsrelevant
- H2 bestätigt: Streamlit-App macht Analyse verständlich zugänglich ✓
- H3 teilweise: Kombination aus ML + Scoring verbessert Nutzbarkeit ✓

### Grenzen

- Keine exogenen Faktoren (Sentiment, Makro)
- Statisches Modell (kein Online-Learning)
- Historische Daten ≠ Zukunftsgarantie

### Notebook

→ `05_conclude_evaluate.ipynb`

---

## K – Knowledge / Wissenstransfer

### Kommunikationsebenen

1. **Technisch**: Jupyter-Notebooks (reproduzierbar, kommentiert)
2. **Interaktiv**: Streamlit-App (`app.py` + `src/`)
3. **Dokumentarisch**: docs/, README.md
4. **Didaktisch**: Lernstudio mit direktem Feedback und arsnova.eu-Export

### App-Features für Wissenstransfer

- Methodik-Seite: QUA³CK erklärt
- Assistent (Gemini): Fachbegriffe erklärt
- Disclaimer: Grenzen transparent kommuniziert
- Export: PDF/CSV für Offline-Nutzung

### Notebooks

→ `06_knowledge_transfer_streamlit.ipynb`  
→ `07_newsapi_assistant_export.ipynb`

---

## Fazit

WealthScope AI durchläuft alle 7 QUA³CK-Phasen vollständig und dokumentiert  
jeden Schritt wissenschaftlich nachvollziehbar. Das Projekt zeigt exemplarisch,  
wie ein Data-Science-Workflow von der Fragestellung bis zur interaktiven  
Anwendung methodisch korrekt umgesetzt werden kann.
