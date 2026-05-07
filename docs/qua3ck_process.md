# QUA3CK-Prozess für ForexScope AI

## Q-Phase: Question / Fragestellung

### Leitfrage

Wie können historische Forex-Marktdaten genutzt werden, um eine ML-gestützte Streamlit-App zu entwickeln, die technische Marktanalyse, Szenarioeinschätzung und risikobasierte Positionsplanung verständlich kombiniert?

### Ziel

Die Q-Phase klärt, welches Problem gelöst werden soll, für wen die App relevant ist und welche Datenquellen benötigt werden.

### Ergebnis der Q-Phase

ForexScope AI wird als datengetriebene Analyse-App konzipiert, nicht als automatischer Trading-Bot.

---

## U-Phase: Understanding / Daten- und Problemverständnis

### Problemverständnis

Forex-Trading ist stark risikobehaftet. Viele Nutzer betrachten Charts, Indikatoren und mögliche Setups getrennt voneinander. Die App soll technische Analyse und Risikomanagement zusammenführen.

### Datenverständnis

Genutzte bzw. geplante Datenquellen:

- yfinance / Yahoo Finance für historische Forex-OHLC-Daten
- Kaggle Forex-Datensätze als zusätzliche Big-Data-Grundlage
- OpenML oder weitere ML-Datensätze als methodische Ergänzung
- Broker-APIs als Ausblick für spätere Live- oder Paper-Trading-Daten

### Datenarten

- Open
- High
- Low
- Close
- Zeitstempel
- technische Indikatoren
- abgeleitete Risiko- und Positionsgrößen

---

## A-Phase: Analytics / Analyse und Modellierung

### Technische Analyse

Die App berechnet:

- Moving Averages
- RSI
- MACD
- ATR
- Support und Resistance
- Multi-Timeframe-Score

### Machine Learning

Ein Random-Forest-Modell wird verwendet, um ein einfaches bullish/bearish-Szenario für die nächsten Kerzen zu berechnen, sofern ausreichend Daten vorhanden sind.

### Risikoberechnung

Der Capital & Risk Planner berechnet:

- Risiko pro Trade
- Stop-Loss-Abstand in Pips
- Take-Profit-Abstand in Pips
- Chance/Risiko-Verhältnis
- Break-even-Trefferquote
- Positionsgröße

---

## 3-Phase: Drei zentrale Ergebnisse

### Ergebnis 1: Forex Pair Analyse

Die App zeigt aktuelle Kursdaten, Chart, Indikatoren, Support/Resistance und Trend-Score.

### Ergebnis 2: Multi-Timeframe Dashboard

Mehrere Timeframes werden zusammengefasst, um kurzfristige und langfristige Marktlage vergleichbar zu machen.

### Ergebnis 3: Capital & Risk Planner

Risiko, Positionsgröße, CRV und Break-even-Trefferquote werden transparent berechnet.

---

## C/K-Phase: Communication & Knowledge

### Kommunikation

Die App macht Ergebnisse durch Diagramme, KPIs, Tabellen und Warnhinweise verständlich.

### Knowledge

Die App dient als Lern- und Analysewerkzeug. Nutzer sollen nicht blind Signale übernehmen, sondern Daten, Modellgrenzen und Risiko verstehen.

### Grenzen

- keine Finanzberatung
- keine sichere Kursprognose
- keine Live-Trading-Automation
- yfinance-Daten können verzögert oder unvollständig sein
- Broker-Automation nur als Ausblick

---

## Projektfazit

ForexScope AI erfüllt den QUA3CK-Prozess, indem aus einer klaren Fragestellung über Datenverständnis und Analyse ein prototypisches Datenprodukt entsteht, das Ergebnisse kommuniziert und Wissen über datengetriebene Forex-Analyse vermittelt.
