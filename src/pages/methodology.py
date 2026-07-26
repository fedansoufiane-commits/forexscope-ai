from __future__ import annotations

import streamlit as st

from src.ui import card, disclaimer_footer, page_header


STEPS = [
    ("Q", "Question", "Wie können historische US-Aktienmarktdaten genutzt werden, um mit ML eine interaktive "
                       "Finanzanalyse-App zu entwickeln, die technische Analyse, ML-Signale und Risikoplanung "
                       "nachvollziehbar kombiniert? Zielgruppe: Data-Science-Studierende, keine Anlageberatung."),
    ("U", "Understanding the Data", "Kaggle US Stocks & ETFs (192.119 Zeilen, 26 Ticker, 1962–2017), CC0-Lizenz. "
                                     "EDA mit Histogrammen, Boxplots, Korrelationen. Fehlwerte = MCAR (MA-Warmup), "
                                     "median-imputiert. StandardScaler, nur auf Trainingsdaten gefittet."),
    ("A", "Analytics (Feature Engineering)", "8 Features: Renditen (1/5/20 Tage), Abstand zu MA-20/50/200, "
                                              "annualisierte Volatilität, Drawdown vom Allzeithoch. "
                                              "Zielvariable target_20d (binär)."),
    ("A", "Algorithm Selection", "DummyClassifier → logistische Regression → Entscheidungsbaum → Linear-SVM → "
                                  "RandomForest. Alle Modelle erhalten dieselben Features und Testdaten."),
    ("A", "Adaption (Optimierung)", "Pipeline mit medianer Imputation und Scaling; baumbasierte Modelle werden "
                                     "regularisiert. Die Evaluation nutzt expandierende Walk-forward-Fenster."),
    ("C", "Conclude & Compare", "Ein unangetasteter später 20%-Holdout und eine 20-Handelstage-Sperrzone "
                                 "verhindern Future Leakage. Accuracy, Balanced Accuracy, ROC-AUC, PR und Laufzeit "
                                 "machen Stärken und Grenzen jeder Modellgeneration sichtbar."),
    ("K", "Knowledge Transfer", "8 Jupyter-Notebooks (wissenschaftlich dokumentiert) → diese Streamlit-App "
                                 "(interaktive Exploration) → Export als PDF/CSV für offline Nutzung."),
]


def render() -> None:
    page_header("Methodik — QUA³CK-Prozess", "Stock et al. (2021), KIT ITIV — angewendet auf dieses Projekt")

    for letter, title, desc in STEPS:
        card(
            f'<div style="display:flex;gap:1rem;align-items:flex-start">'
            f'<div style="font-size:1.6rem;font-weight:900;color:var(--ws-primary);min-width:2.2rem">{letter}</div>'
            f'<div><b>{title}</b><br><span style="font-size:0.86rem;color:var(--ws-text-muted)">{desc}</span></div>'
            f'</div>'
        )

    st.markdown("#### Wissenschaftliche Quellen")
    st.markdown("""
- Stock et al. (2021): *QUA³CK — A Machine Learning Development Process.* KIT ITIV.
- Fama, E. (1970): *Efficient Capital Markets.* Journal of Finance, 25(2), 383–417.
- Li et al. (2024): *Comparison of Imputation Methods.* BMC Medical Research Methodology.
- Breiman (2001): *Random Forests.* Machine Learning, 45, 5–32.
- Lundberg & Lee (2017): *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Quibeldey-Cirkel (2026): Kursmaterialien zu Klassifikation, Modelltraining, SVM, Entscheidungsbäumen und Random Forests.
    """)

    disclaimer_footer()
