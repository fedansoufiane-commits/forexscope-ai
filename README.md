# ForexScope AI – Data-Driven Forex Insights

ForexScope AI ist eine ML-gestützte Streamlit-App zur Analyse von Forex-Marktdaten, technischen Indikatoren und risikobasierter Szenarioeinschätzung.

Die App lädt historische Kursdaten für Währungspaare, berechnet technische Indikatoren und unterstützt Nutzer dabei, Marktstruktur, Risiko, Positionsgröße und mögliche Szenarien besser zu verstehen.

## Projektidee

Viele Trading-Tools zeigen Charts und Indikatoren, erklären aber Risiko, Positionsgröße und Modellunsicherheit nicht ausreichend. ForexScope AI verbindet technische Analyse, Machine Learning und Capital/Risk Planning in einer App.

## Zentrale Fragestellung

Wie können historische Forex-Marktdaten genutzt werden, um eine ML-gestützte Streamlit-App zu entwickeln, die technische Marktanalyse, Szenarioeinschätzung und risikobasierte Positionsplanung verständlich kombiniert?

## QUA3CK-Bezug

Das Projekt orientiert sich am QUA3CK-Prozessmodell:

- Q: Question – Definition des Machine-Learning-Problems
- U: Understanding the data – Verständnis der Datenbasis
- A³: Algorithm selection, Adaption, Adjustment – Algorithmuswahl, Feature Engineering und Hyperparameter-Anpassung
- C: Conclude and compare – Ergebnisse bewerten und vergleichen
- K: Knowledge transfer – Umsetzung als App, Dokumentation und Wissenstransfer

## QUA3CK-Notebooks

Die fünf Phasen sind separat dokumentiert:

- notebooks/01_Q_Question.ipynb
- notebooks/02_U_Understanding_the_Data.ipynb
- notebooks/03_A3_Algorithm_Adaption_Adjustment.ipynb
- notebooks/04_C_Conclude_and_Compare.ipynb
- notebooks/05_K_Knowledge_Transfer.ipynb

## Hauptfunktionen

- Auswahl von Forex-Paaren, z. B. EUR/USD, GBP/USD, USD/JPY
- Analyse bis auf Intraday-Timeframes wie M1, M5, M15 und H1
- Candlestick-Charts
- Moving Averages
- RSI
- MACD
- ATR
- Support und Resistance
- Multi-Timeframe-Dashboard
- ML-Szenario für mögliche Kursrichtung
- Capital & Risk Planner
- Break-even-Trefferquote
- Automation Readiness als Ausblick für spätere Broker-Anbindung

## KPIs

ForexScope AI betrachtet mehrere KPI-Gruppen:

- Daten-KPIs: Datenpunkte, Timeframe-Abdeckung, Datenaktualität
- Analyse-KPIs: Trend-Score, ATR, Support/Resistance-Abstand
- ML-KPIs: Accuracy, Trainingsdaten, Szenario-Wahrscheinlichkeit
- Risiko-KPIs: Risiko pro Trade, CRV, Break-even-Trefferquote, Positionsgröße
- Transfer-KPIs: Verständlichkeit, Transparenz, Reproduzierbarkeit

Details stehen in:

docs/kpi_framework.md

## Datenquellen

Die App nutzt aktuell yfinance/Yahoo Finance für historische Forex-Marktdaten.

Zusätzlich werden als Big-Data-Grundlage und Erweiterung berücksichtigt:

- Kaggle Forex-Datensätze
- OpenML oder weitere ML-Datensätze
- Broker-APIs wie OANDA, Interactive Brokers, MetaTrader 5 oder cTrader als professioneller Ausblick

Details stehen in:

docs/data_sources.md

## Installation

Befehl:

pip install -r requirements.txt

## App starten

Befehl:

streamlit run app.py

## Projektstruktur

forexscope-ai/
- app.py
- README.md
- requirements.txt
- data/
- docs/data_sources.md
- docs/kpi_framework.md
- docs/qua3ck_process.md
- notebooks/01_Q_Question.ipynb
- notebooks/02_U_Understanding_the_Data.ipynb
- notebooks/03_A3_Algorithm_Adaption_Adjustment.ipynb
- notebooks/04_C_Conclude_and_Compare.ipynb
- notebooks/05_K_Knowledge_Transfer.ipynb

## Hinweis

Diese App dient nur Analyse- und Bildungszwecken. Sie ist keine Finanzberatung und keine Handelsaufforderung. Eine spätere Broker-Automation müsste zuerst über Paper-Trading, Risk Engine, Logging, Kill Switch und manuelle Freigabe abgesichert werden.
