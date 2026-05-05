# ForexScope AI

ForexScope AI ist eine ML-gestützte Streamlit-App zur Analyse von Forex-Marktdaten.

Die App lädt historische Kursdaten für Währungspaare, berechnet technische Indikatoren und erstellt eine risikobasierte Szenarioeinschätzung. Zusätzlich enthält sie einen Capital & Risk Planner zur Berechnung von Risiko, Positionsgröße, Chance/Risiko-Verhältnis und Break-even-Trefferquote.

## Projektidee

Viele Trading-Tools zeigen Charts und Indikatoren, aber erklären Risiko und Positionsgröße oft nicht ausreichend. ForexScope AI verbindet technische Analyse mit Risiko- und Kapitalplanung.

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
- Automation Readiness als Ausblick für spätere Broker-Anbindung

## Datenquellen

Die App nutzt aktuell yfinance/Yahoo Finance für historische Forex-Marktdaten.

Zusätzlich werden als Big-Data-Grundlage und Erweiterung berücksichtigt:

- Kaggle Forex-Datensätze
- OpenML oder weitere ML-Datensätze
- Broker-APIs wie OANDA, Interactive Brokers, MetaTrader 5 oder cTrader als professioneller Ausblick

Details stehen in docs/data_sources.md.

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
- notebooks/

## Hinweis

Diese App dient nur Analyse- und Bildungszwecken. Sie ist keine Finanzberatung und keine Handelsaufforderung. Eine spätere Broker-Automation müsste zuerst über Paper-Trading, Risk Engine, Logging, Kill Switch und manuelle Freigabe abgesichert werden.
