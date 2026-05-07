# KPI Framework für ForexScope AI

## Ziel der KPIs

ForexScope AI soll nicht nur technische Indikatoren anzeigen, sondern messbare Kriterien liefern, mit denen die App, die Datenqualität, die Modellleistung und der Nutzen bewertet werden können.

## 1. Daten-KPIs

| KPI | Bedeutung | Ziel |
|---|---|---|
| Anzahl geladener Datenpunkte | Anzahl verfügbarer OHLC-Datenpunkte je Timeframe | ausreichend Daten je Analysefenster |
| Timeframe-Abdeckung | Anzahl unterstützter Timeframes | M1 bis W1 |
| Datenverfügbarkeit | Anteil erfolgreicher Datenabrufe je Forex-Paar | möglichst hoch |
| Datenaktualität | Zeitpunkt der letzten geladenen Kerze | möglichst aktuell |
| Missing-Value-Rate | Anteil fehlender Werte in den Marktdaten | möglichst gering |

## 2. Analyse-KPIs

| KPI | Bedeutung | Ziel |
|---|---|---|
| Trend-Score | Verdichtete Bewertung aus Moving Averages, MACD und RSI | nachvollziehbare Szenarioeinschätzung |
| ATR in Pips | Volatilität des Forex-Paares | Risiko besser einschätzen |
| Support-Abstand in Pips | Abstand zum nächsten Support | Downside-Risiko beurteilen |
| Resistance-Abstand in Pips | Abstand zum nächsten Widerstand | Upside-Potenzial beurteilen |
| Multi-Timeframe-Score | Gesamtbewertung über mehrere Timeframes | robustere Marktinterpretation |

## 3. ML-KPIs

| KPI | Bedeutung | Ziel |
|---|---|---|
| Accuracy | Anteil korrekt klassifizierter Testfälle | erste Modellbewertung |
| Anzahl Trainingsdatenpunkte | Menge der Trainingsdaten | ausreichend Datenbasis |
| Anzahl Testdatenpunkte | Menge der Testdaten | belastbare Prüfung |
| Szenario-Wahrscheinlichkeit | Wahrscheinlichkeit für bullish/bearish | keine sichere Prognose, sondern Szenario |
| Modellverfügbarkeit | Ob genug Daten für ML vorhanden sind | Transparenz bei Datenmangel |

## 4. Risiko-KPIs

| KPI | Bedeutung | Ziel |
|---|---|---|
| Risiko pro Trade | Anteil des Kontos, der riskiert wird | konservativ, z. B. 1 % |
| Maximaler Verlust | Verlustbetrag bei Stop Loss | klar begrenzt |
| Chance/Risiko-Verhältnis | Verhältnis von möglichem Gewinn zu Verlust | möglichst > 1,5 |
| Break-even-Trefferquote | benötigte Trefferquote für statistischen Break-even | automatisch berechnet |
| Positionsgröße | berechnete Units/Lots | risikobasiert statt willkürlich |

## 5. Business-/Nutzungs-KPIs

| KPI | Bedeutung | Ziel |
|---|---|---|
| Interpretierbarkeit | Ergebnisse sind für Nutzer verständlich erklärt | hoch |
| Entscheidungsunterstützung | Nutzer erkennt Risiko, Trend und Szenario | hoch |
| Automationsreife | Vorbereitung für Broker-Anbindung | nur nach Paper-Trading |
| Transparenz | Grenzen und Risiken werden klar genannt | hoch |
| Präsentationsfähigkeit | App und Notebook erklären das Projekt nachvollziehbar | hoch |

## Zusammenfassung

Die KPIs zeigen, dass ForexScope AI nicht als reiner Chart-Viewer verstanden wird, sondern als datengetriebenes Analysewerkzeug mit technischer Analyse, ML-Szenario und Risikomanagement.
