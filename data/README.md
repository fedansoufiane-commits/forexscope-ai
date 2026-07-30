# Datenordner

## `raw/` — Rohdaten (gitignored, nicht im Repository)

- **`price-volume-data-for-all-us-stocks-etfs.zip`** (~492 MB): Original-Download von
  Kaggle, [„Huge Stock Market Dataset" von Boris Marjanovic](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs),
  CC0 Public Domain. Tägliche OHLCV-Daten für tausende US-Aktien und ETFs, 1962–2017.
- **`us_stocks_etfs/`** (~1.5 GB entpackt): Entpackter Inhalt — je eine `.txt`-Datei
  (CSV-Format) pro Ticker, aufgeteilt in `Data/Stocks/` und `Data/ETFs/`.

Beide sind zu groß für Git und werden lokal aus der Zip erzeugt — nicht committen.
Der für die App relevante Ausschnitt (26 Blue-Chip-Ticker, siehe `src/config.py:TICKERS`)
wird per Notebook (`02_understanding_the_data.ipynb`, `03_feature_engineering.ipynb`)
gefiltert und feature-engineered in `processed/` abgelegt.

## `processed/` — abgeleitete, versionierte Datensätze

- **`wealthscope_features.parquet` / `.csv`** (192.119 Zeilen × 27 Spalten): Das
  eigentliche Trainings- und App-Datenset. Enthält OHLCV, die 8 ML-Features
  (`daily_return`, `return_5d`, `return_20d`, `ma_20/50/200_distance`,
  `volatility_20d`, `drawdown`) sowie die Zielvariable `target_20d`. Wird von
  `scripts/train_and_diagnose.py` und `src/data.py` gelesen.
- **`wealthscope_market_dataset.csv`**: Schlankere Zwischenstufe (nur OHLCV + Ticker,
  ohne Features) aus der frühen EDA-Phase (`02_understanding_the_data.ipynb`).

Beide sind reproduzierbar aus `raw/` + den Notebooks — bei Bedarf einfach neu erzeugen.
