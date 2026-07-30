from __future__ import annotations

import numpy as np
import streamlit as st

from src.charts import chart_gauge, chart_radar
from src.config import MODEL_FEATURES
from src.context import get_context
from src.diagnostics import (
    chart_confusion_matrix,
    chart_correlation_heatmap,
    chart_feature_importance,
    chart_learning_curve,
    chart_model_comparison,
    chart_roc_pr,
    compute_correlation,
    learning_curve_diagnosis,
    load_diagnostics,
    load_learning_curve,
)
from src.icons import icon
from src.model import SCORE_WEIGHTS
from src.theme import THEMES
from src.ui import badge, bar, card, kpi_grid, page_header, section_title


def render() -> None:
    ctx = get_context()
    result, mode = ctx["result"], ctx["theme_mode"]
    diag = load_diagnostics()

    page_header("ML-Insights", "RandomForest-Diagnostik · Score-Zerlegung · Erklärbarkeit")

    n_rows = diag["n_rows"]
    tm_acc = diag["test_metrics"]["accuracy"]
    tm_auc = diag["test_metrics"]["roc_auc"]
    acc_badge = badge(f"Test-Accuracy {tm_acc*100:.1f}%", "info")
    auc_badge = badge(f"ROC-AUC {tm_auc:.3f}", "info")
    card(
        f'<div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:1rem;align-items:center">'
        f'<div><div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;color:var(--ws-text-muted)">'
        f'{icon("activity", 13)} Aktives Modell</div>'
        f'<b>RandomForestClassifier</b> · <span class="ws-mono">{n_rows:,}</span> Zeilen · '
        f'<span class="ws-mono">{len(MODEL_FEATURES)}</span> Features</div>'
        f'{acc_badge}{auc_badge}'
        f'</div>', variant="ws-accent",
    )

    if st.session_state.get("use_live_data"):
        st.warning(
            "⚠️ **Distribution Shift:** Das Modell wurde auf Kaggle-Daten von 1962–2017 trainiert. "
            "Live-Kurse liegen außerhalb dieser Trainingsverteilung (COVID-Crash 2020, KI-Rally 2023, "
            "aktuelle Zinszyklen wurden nie gesehen) — Vorhersagen darauf sind mit erhöhter Unsicherheit behaftet.",
            icon="⚠️",
        )

    tabs = st.tabs([
        "🕰️ Modell-Zeitreise", "🎯 Score-Zerlegung", "🔗 Korrelation",
        "🧩 Fehler & ROC", "📈 Lernkurve", "🌲 Importance", "🔍 SHAP", "📖 Methodik",
    ])

    # ---- Tab 1: historical model comparison -------------------------------------
    with tabs[0]:
        validation = diag["validation"]
        st.plotly_chart(chart_model_comparison(diag, mode), config={"displayModeBar": False})
        st.info(
            f"Fairer Vergleich: Training nur bis **{validation['train_end']}**, unangetasteter Test "
            f"von **{validation['test_start']} bis {validation['test_end']}**, dazwischen "
            f"**{validation['purge_trading_days']} Handelstage Sperrzone**. So kann das 20-Tage-Ziel "
            "keine Zukunftsinformation in das Training tragen."
        )
        rows = []
        for item in diag["model_comparison"].values():
            metric = item["test_metrics"]
            rows.append({
                "Epoche": item["year"],
                "Modell": item["label"],
                "Idee": item["family"],
                "ROC-AUC": metric["roc_auc"],
                "Balanced Accuracy": metric["balanced_accuracy"],
                "Walk-forward AUC": item["walk_forward_roc_auc_mean"],
                "Training (s)": metric["fit_seconds"],
            })
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "Die Zeitreise macht Fortschritt und Grenzen sichtbar: Ein komplexeres Modell ist nicht "
            "automatisch besser. Entscheidend sind Out-of-Time-Generalisation, Erklärbarkeit und Laufzeit."
        )

    # ---- Tab 2: Score breakdown -------------------------------------------------
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(chart_radar(result, mode), config={"displayModeBar": False})
        with c2:
            st.plotly_chart(chart_gauge(result, mode), config={"displayModeBar": False})

        t = THEMES.get(mode, THEMES["Hell"])
        rows = [
            ("Trend", result.trend_score, SCORE_WEIGHTS["trend"], t["primary"]),
            ("Volatilität", result.volatility_score, SCORE_WEIGHTS["volatility"], t["colorway"][3]),
            ("Drawdown", result.drawdown_score, SCORE_WEIGHTS["drawdown"], t["negative"]),
            ("News-Sentiment", round(50 + result.news_score * 10, 1), SCORE_WEIGHTS["news"], t["neutral"]),
            ("Positionsgröße", round(100 - max(0, result.asset_weight - 10) * 3, 1), SCORE_WEIGHTS["position"], t["colorway"][5]),
        ]
        for label, score, weight, color in rows:
            st.markdown(
                f'<div style="display:grid;grid-template-columns:140px 60px 1fr 60px;gap:0.8rem;align-items:center;margin-bottom:0.5rem">'
                f'<b>{label}</b><span class="ws-mono">{score:.0f}</span>{bar(score, color)}'
                f'<span class="ws-mono" style="color:var(--ws-text-muted)">{weight*100:.0f}%</span>'
                f'</div>', unsafe_allow_html=True,
            )
        st.caption(
            "Confidence = 0.36×Trend + 0.22×Volatilität + 0.18×Drawdown + 0.14×News + 0.10×Positionsgröße. "
            "Gewichte sind aus gruppierter RF-Feature-Importance abgeleitet (siehe Tab „Methodik“) — "
            "bewusst nicht gefittet, damit sie nachvollziehbar bleiben."
        )

    # ---- Tab 3: Correlation heatmap ---------------------------------------------
    with tabs[2]:
        section_title("Pearson-Korrelationsmatrix aller ML-Features", "grid")
        corr = compute_correlation(ctx["market"])
        st.plotly_chart(chart_correlation_heatmap(corr, mode), config={"displayModeBar": False})
        st.info(
            "Rot = positive Korrelation, Blau = negative Korrelation. Stark korrelierte Features (|r| > 0.8) "
            "sind potenziell redundant — RandomForest ist robust gegenüber Multikollinearität, lineare Modelle wären es nicht."
        )

    # ---- Tab 4: Confusion matrix + ROC/PR ---------------------------------------
    with tabs[3]:
        tm = diag["test_metrics"]
        kpi_grid([
            ("Accuracy vs. Baseline", f"{tm['accuracy']*100:.2f}%",
             f"Majority-Baseline {tm['majority_baseline']*100:.2f}% — {'schlägt' if tm['beats_baseline'] else 'unter'} Baseline",
             "target"),
            ("ROC-AUC", f"{tm['roc_auc']:.4f}", f"+{tm['roc_auc']-0.5:.4f} über Zufall (0.5)", "trend-up"),
            ("F1 (weighted)", f"{tm['f1_weighted']:.4f}", f"Precision {tm['precision']:.3f} · Recall {tm['recall']:.3f}", "check"),
        ])
        st.plotly_chart(chart_confusion_matrix(diag, mode), config={"displayModeBar": False})
        st.plotly_chart(chart_roc_pr(diag, mode), config={"displayModeBar": False})
        st.caption(
            "Klassenverteilung im Test-Set: Bullish "
            f"{tm['class_balance_test']['bullish_1']*100:.1f}% / Bearish {tm['class_balance_test']['bearish_0']*100:.1f}% "
            "→ bei Imbalance ist Accuracy allein irreführend, daher ROC-AUC als primäre Metrik (Kap. 3, Prof. Quibeldey-Cirkel)."
        )

    # ---- Tab 5: Learning curve ----------------------------------------------------
    with tabs[4]:
        lc = load_learning_curve()
        section_title("Lernkurve — Bias/Variance-Diagnose", "trend-up")
        st.plotly_chart(chart_learning_curve(lc, mode), config={"displayModeBar": False})
        card(learning_curve_diagnosis(lc), variant="ws-accent")
        st.caption(
            "Trainings- und Validierungs-Score (ROC-AUC) über wachsende Trainingsgröße in einer expandierenden "
            "Walk-forward-Validierung. Jede Validierung liegt zeitlich nach ihrem Training und besitzt eine "
            "20-Handelstage-Sperrzone. Diese Analyse zeigt, ob "
            "mehr Trainingsdaten den größten Hebel wären (Lücke schließt sich) oder ob das Modell strukturell "
            "an seine Grenze stößt (beide Kurven laufen flach — konsistent mit der Efficient Market Hypothesis, Fama 1970)."
        )

    # ---- Tab 6: Feature importance -----------------------------------------------
    with tabs[5]:
        section_title("RandomForest Feature Importance", "layers")
        st.plotly_chart(chart_feature_importance(diag, mode), config={"displayModeBar": False})
        st.caption("Gini-Importance des RandomForest — wie oft/wirksam ein Feature zum Aufteilen der Bäume genutzt wird.")

    # ---- Tab 7: SHAP ---------------------------------------------------------------
    with tabs[6]:
        _render_shap(ctx)

    # ---- Tab 8: Methodology --------------------------------------------------------
    with tabs[7]:
        cv = diag["cross_validation"]
        st.markdown(f"""
**Algorithmus:** RandomForestClassifier (sklearn {diag['sklearn_version']})
**Zielvariable:** `target_20d` — Kurs in 20 Handelstagen höher? (binär)
**Features ({len(diag['features'])}):** {', '.join(diag['features'])}
**Train/Test:** {diag['n_train']:,} / {diag['n_test']:,} Zeilen, purged Out-of-Time-Holdout
**Sperrzone:** {diag['validation']['purge_trading_days']} Handelstage passend zum Zielhorizont
**Hyperparameter:** n_estimators={diag['hyperparams']['n_estimators']}, max_depth={diag['hyperparams']['max_depth']}, class_weight={diag['hyperparams']['class_weight']}
**Walk-forward CV:** Accuracy {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f} · ROC-AUC {cv['auc_mean']:.4f} ± {cv['auc_std']:.4f}

**Wissenschaftliche Einordnung:** Die Out-of-Time-ROC-AUC liegt nur knapp über 0,5
und die rohe Accuracy unter der Mehrheits-Baseline. Dieses ehrliche Negativergebnis
ist mit der Efficient Market Hypothesis (Fama 1970) vereinbar. Das Modell ist ein
Lerndemonstrator für den vollständigen ML-Workflow, **kein** Handelssignal-Generator.

> *„Aktuelle Kursdaten sind bereits im Marktpreis eingepreist."* — Fama (1970), Efficient Capital Markets
        """)
        with st.expander("Score-Formel (Python)"):
            st.code(
                "confidence = round(\n"
                "    0.36 * trend_score\n"
                "  + 0.22 * volatility_score\n"
                "  + 0.18 * drawdown_score\n"
                "  + 0.14 * (50 + news_score * 10)\n"
                "  + 0.10 * weight_risk,\n"
                "  1,\n)",
                language="python",
            )


def _render_shap(ctx) -> None:
    section_title("SHAP — warum sagt das Modell das, was es sagt?", "flask")
    try:
        import shap

        from src.model import load_model

        model = load_model()
        df = ctx["market"][MODEL_FEATURES + ["target_20d"]].dropna()
        sample = df[MODEL_FEATURES].sample(min(300, len(df)), random_state=42)

        imp, scl, rf = model.named_steps["imp"], model.named_steps["scl"], model.named_steps["mod"]
        transformed = scl.transform(imp.transform(sample))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(transformed)
        sv1 = shap_values[:, :, 1] if np.array(shap_values).ndim == 3 else shap_values
        mean_abs = np.abs(sv1).mean(axis=0)

        import plotly.graph_objects as go
        from src.config import FEATURE_LABELS
        from src.theme import apply_chart_theme

        t = THEMES.get(ctx["theme_mode"], THEMES["Hell"])
        order = np.argsort(mean_abs)
        fig = go.Figure(go.Bar(
            x=mean_abs[order], y=[FEATURE_LABELS.get(MODEL_FEATURES[i], MODEL_FEATURES[i]) for i in order],
            orientation="h", marker_color=t["primary"],
        ))
        fig.update_layout(title="Globale Feature-Wichtigkeit (Ø |SHAP|)", height=340)
        st.plotly_chart(apply_chart_theme(fig, ctx["theme_mode"], 340),
                         config={"displayModeBar": False})
        st.caption(
            "SHAP (Lundberg & Lee, 2017) misst den tatsächlichen Beitrag jedes Features pro Datenpunkt — "
            "robuster als impurity-basierte RF-Importance, da unverzerrt gegenüber hochkardinalen Features."
        )
    except ImportError:
        st.info("`shap` ist nicht installiert — `pip install shap`.")
    except Exception as exc:
        st.warning(f"SHAP-Analyse aktuell nicht verfügbar: {exc}")
