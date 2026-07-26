# Datenquellen für WealthScope AI

## 1. Kaggle — US Stocks & ETFs (Trainingsbasis)

**Quelle:** [„Huge Stock Market Dataset" von Boris Marjanovic](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs), CC0 Public Domain.

Tägliche OHLCV-Daten (Open/High/Low/Close/Volume), 1962–2017, für tausende US-Aktien
und ETFs. Für WealthScope AI auf 26 liquide Blue-Chip-Ticker eingegrenzt
(`src/config.py:TICKERS` — u. a. AAPL, MSFT, SPY, QQQ, JPM, XOM).

Nutzen für die App:

- **Trainingsdatensatz** für den RandomForest-Klassifikator (192.119 Zeilen, siehe
  `data/processed/wealthscope_features.parquet`)
- Feature Engineering: Renditen (1/5/20 Tage), Abstand zu gleitenden Durchschnitten
  (MA-20/50/200), annualisierte Volatilität, Drawdown
- Historische Charts, Korrelationsmatrix, EDA im **Datenlabor**

Grenzen:

- Endet 2017 → Survivorship-Bias (nur heute noch existierende Ticker), keine
  COVID-Krise, kein Zinszyklus 2022+ im Trainingssignal enthalten
- Reine Kursdaten, keine Fundamentaldaten (Bilanzen, KGV etc.)

## 2. yfinance (live, optional)

Für aktuelle Kurse außerhalb des Trainingszeitraums lädt `src/data.py:fetch_live_data()`
Live-Daten über die `yfinance`-Bibliothek (Yahoo-Finance-Wrapper) — dieselben
Feature-Formeln wie beim Training werden live nachgerechnet.

**Wichtig:** Das Modell wurde nie auf Daten nach 2017 trainiert. Wird „Live-Kurse"
in der Sidebar aktiviert, zeigt die App explizit einen **Distribution-Shift-Hinweis**
(siehe `src/pages/ml_insights.py`) — Vorhersagen auf Live-Daten sind mit erhöhter
Unsicherheit behaftet, das ist eine bewusste Design-Entscheidung, keine Einschränkung
die versteckt wird.

Grenzen:

- Gratis-Tier von Yahoo Finance: teils verzögerte/unvollständige Daten
- Für produktives Trading wäre eine Broker-API (z. B. Interactive Brokers, OANDA)
  nötig — nicht Ziel dieses Lernprojekts

## 3. NewsAPI (optional, benötigt API-Key)

`src/news.py` ruft `https://newsapi.org/v2/everything` mit dem Ticker-Symbol als
Suchbegriff ab und berechnet ein einfaches Lexikon-Sentiment (Positiv-/Negativ-
Wortlisten). Fließt mit 14 % Gewicht in den Confidence-Score ein.

Grenzen:

- Naive Keyword-Suche: ein Ticker wie „SPY" kann auch unrelated Treffer liefern
  (z. B. Artikel, die das Wort „spy" im Sinne von Spionage enthalten) — bekannte
  Schwäche einer bloßen String-Suche ohne Entity-Resolution
- Lexikon-Sentiment ist deutlich simpler als ein trainiertes NLP-Modell

## 4. Google Gemini (optional, benötigt API-Key)

`src/news.py:assistant_answer()` nutzt `google-genai` (Modell `gemini-2.0-flash`),
um Fragen zur aktuellen Analyse im Kontext der berechneten Scores zu beantworten
(**KI-Assistent**-Seite). Kein Bestandteil des ML-Modells selbst — rein für
Erklärbarkeit/Nutzerführung.

## Wichtig

WealthScope AI ist eine **Analyse- und Bildungs-App**, keine Live-Trading-Automation
und keine Finanzberatung.
