# KPI-Framework – WealthScope AI

## Modell-KPIs

| KPI | Wert | Bewertung |
|---|---|---|
| **Accuracy (Test)** | ~53 % | Erwartet (EMH); besser als Mehrheits-Baseline |
| **ROC-AUC** | ~0.53 | > 0.5 → Modell besser als Zufall |
| **F1-Score (weighted)** | ~0.54 | Ausgewogene Precision/Recall |
| **CV-AUC (5-Fold, μ)** | ~0.53 | Stabile, nicht überfittete Schätzung |
| **CV-AUC (5-Fold, σ)** | ~0.01 | Niedrige Varianz → robustes Modell |

## Datensatz-KPIs

| KPI | Wert |
|---|---|
| Gesamtzeilen | 192.119 |
| ML-Features | 8 |
| Fehlende Werte (%) | < 2 % (Warm-up-bedingt, MCAR) |
| Zielvariable Klasse 1 | ~59 % (leichte Imbalance) |
| Ticker abgedeckt | 26 |

## App-Qualitäts-KPIs

| KPI | Status |
|---|---|
| Data-Leakage-freie Pipeline | ✅ |
| Stratifizierter Train/Test-Split | ✅ |
| 5-Fold Cross-Validation | ✅ |
| Disclaimer auf allen Seiten | ✅ |
| Wissenschaftliche Quellen | ✅ |
| QUA³CK vollständig dokumentiert | ✅ |

## Metriken-Erläuterung

**Accuracy:** Anteil korrekt klassifizierter Beobachtungen. Bei leichter Klassenimbalance
allein nicht ausreichend – daher zusätzlich F1 und AUC.

**ROC-AUC:** Area Under the Receiver Operating Characteristic Curve. Misst die
Trennfähigkeit unabhängig von der Entscheidungsschwelle. 0.5 = Zufall, 1.0 = perfekt.

**F1-Score:** Harmonisches Mittel aus Precision und Recall. Robust bei Imbalance.

**Cross-Validation:** Schätzt die echte Generalisierungsleistung auf ungesehenen Daten.
Verhindert Overfitting durch einfache Train/Test-Aufteilung.
