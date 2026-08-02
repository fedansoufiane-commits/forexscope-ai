# KPI-Framework – WealthScope AI

## Modell-KPIs

| KPI | Wert | Bewertung |
|---|---|---|
| **Accuracy (Out-of-Time-Test)** | 51,9 % | Unter Mehrheits-Baseline; allein ungeeignet |
| **Balanced Accuracy** | 51,3 % | Leicht über Zufall |
| **ROC-AUC** | 0,519 | Sehr schwache Trennfähigkeit |
| **F1-Score (weighted)** | 0,522 | Ergänzende klassenbezogene Sicht |
| **Walk-forward AUC (μ)** | 0,514 | Geringes zeitliches Signal |
| **Walk-forward AUC (σ)** | 0,013 | Regimeabhängige Schwankung |

## Datensatz-KPIs

| KPI | Wert |
|---|---|
| Gesamtzeilen | 192.119 |
| Zeilen im Modelldatensatz | 190.527 (nach Dropna auf Features + Target) |
| ML-Features | 8 |
| Fehlende Werte in den 8 ML-Features | max. 0,83 % (`ma_200_distance`, Warm-up-bedingt, MCAR) |
| Zielvariable Klasse 1 | 56,4 % im Gesamtdatensatz, 59,1 % im Out-of-Time-Testfenster |
| Ticker abgedeckt | 26 |

## App-Qualitäts-KPIs

| KPI | Status |
|---|---|
| Data-Leakage-freie Pipeline | ✅ |
| Purged Out-of-Time-Holdout | ✅ |
| 4-Fold Walk-forward-Validierung | ✅ |
| Historischer Fünf-Modell-Vergleich | ✅ |
| Disclaimer auf allen Seiten | ✅ |
| Wissenschaftliche Quellen | ✅ |
| QUA³CK vollständig dokumentiert | ✅ |

## Metriken-Erläuterung

**Accuracy:** Anteil korrekt klassifizierter Beobachtungen. Bei leichter Klassenimbalance
allein nicht ausreichend – daher zusätzlich F1 und AUC.

**ROC-AUC:** Area Under the Receiver Operating Characteristic Curve. Misst die
Trennfähigkeit unabhängig von der Entscheidungsschwelle. 0.5 = Zufall, 1.0 = perfekt.

**F1-Score:** Harmonisches Mittel aus Precision und Recall. Robust bei Imbalance.

**Walk-forward-Validierung:** Trainiert ausschließlich auf der Vergangenheit und
bewertet auf einem späteren Fenster. Eine 20-Tage-Sperrzone schützt zusätzlich
vor überlappenden Zielhorizonten.
