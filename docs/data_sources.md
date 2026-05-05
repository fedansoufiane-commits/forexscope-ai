# Datenquellen für ForexScope AI

## 1. Yahoo Finance / yfinance

ForexScope AI nutzt yfinance, um historische OHLC-Marktdaten für Währungspaare zu laden.

Beispiele:

- EUR/USD → EURUSD=X
- GBP/USD → GBPUSD=X
- USD/JPY → USDJPY=X

Genutzte Daten:

- Open
- High
- Low
- Close
- Zeitstempel

Nutzen für die App:

- technische Analyse
- Candlestick-Charts
- RSI
- MACD
- ATR
- Moving Averages
- Support und Resistance
- ML-Szenario

Grenzen:

- Daten können verzögert oder unvollständig sein
- für produktives Trading wäre eine Broker-API besser
- Forex-Volumen ist über Yahoo Finance nicht zuverlässig nutzbar

## 2. Kaggle Forex-Datensätze

Kaggle kann ergänzende historische Forex-Datensätze liefern. Diese können genutzt werden, um zusätzliche historische Kursdaten zu analysieren oder später Modellvergleiche zu bauen.

Mögliche Suchbegriffe:

- Forex Historical Data
- EURUSD Historical Data
- Foreign Exchange Rates
- Forex Dataset

Nutzen:

- zusätzliche Datenbasis
- Vergleich mit yfinance-Daten
- mögliche Big-Data-Erweiterung

## 3. OpenML / weitere Finanzdaten

OpenML kann als Plattform für zusätzliche ML-Datensätze genutzt werden. Für das Projekt können dort Finanz- oder Zeitreihendatensätze recherchiert werden.

Nutzen:

- Vergleichsdaten für ML-Methoden
- alternative Datensätze zur Modellbewertung

## 4. Professioneller Ausblick: Broker-API

Für eine spätere echte Automation wären Broker-APIs notwendig, z. B.:

- OANDA API
- Interactive Brokers API
- MetaTrader 5
- cTrader

Nutzen:

- echte Bid/Ask-Daten
- Spread
- Orderausführung
- Paper Trading
- Trade-Journal
- Automationslogik

Wichtig:

ForexScope AI ist aktuell eine Analyse- und Bildungs-App, keine Live-Trading-Automation.
