"""ML diagnostics: correlation heatmap, confusion matrix, ROC/PR, learning
curve, cross-validation and SHAP — the analytical core of the ML-Insights page.

Everything that requires refitting the model (confusion matrix, ROC/PR, CV,
learning curve) is read from `models/diagnostics.json` /
`models/learning_curve.json`, produced once by
`scripts/train_and_diagnose.py`. Only the feature correlation matrix — cheap
on ~190k x 8 floats — is computed live and cached.
"""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.config import DIAGNOSTICS_PATH, FEATURE_LABELS, LEARNING_CURVE_PATH, MODEL_FEATURES
from src.theme import THEMES, apply_chart_theme


def _t(mode: str) -> dict:
    return THEMES.get(mode, THEMES["Hell"])


@st.cache_data(show_spinner=False)
def load_diagnostics() -> Dict[str, Any]:
    return json.loads(DIAGNOSTICS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_learning_curve() -> Dict[str, Any]:
    return json.loads(LEARNING_CURVE_PATH.read_text())


# ---------------------------------------------------------------- Correlation
@st.cache_data(show_spinner="Berechne Korrelationsmatrix ...")
def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in MODEL_FEATURES + ["future_return_20d", "target_20d"] if c in df.columns]
    return df[cols].dropna().corr(method="pearson")


def chart_correlation_heatmap(corr: pd.DataFrame, mode: str) -> go.Figure:
    labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}", textfont={"size": 11},
        colorbar=dict(title="r"),
    ))
    fig.update_layout(title="Pearson-Korrelationsmatrix — ML-Features", height=460)
    return apply_chart_theme(fig, mode, height=460)


# ------------------------------------------------------------- Confusion mtx
def chart_confusion_matrix(diag: Dict[str, Any], mode: str) -> go.Figure:
    cm = diag["confusion_matrix"]
    counts = np.array(cm["counts"])
    norm = np.array(cm["normalized"])
    fig = go.Figure(go.Heatmap(
        z=norm, colorscale="Blues", zmin=0, zmax=1,
        x=["Vorhersage: Bearish (0)", "Vorhersage: Bullish (1)"],
        y=["Real: Bearish (0)", "Real: Bullish (1)"],
        text=[[f"{counts[i][j]:,}<br>({norm[i][j]*100:.1f}%)" for j in range(2)] for i in range(2)],
        texttemplate="%{text}", textfont={"size": 14},
        showscale=True, colorbar=dict(title="Anteil"),
    ))
    fig.update_layout(title="Konfusionsmatrix (Test-Set, normalisiert je Zeile)", height=380)
    return apply_chart_theme(fig, mode, height=380)


# ----------------------------------------------------------------- ROC / PR
def chart_roc_pr(diag: Dict[str, Any], mode: str) -> go.Figure:
    roc, pr = diag["roc_curve"], diag["pr_curve"]
    auc = diag["test_metrics"]["roc_auc"]
    baseline = diag["test_metrics"]["class_balance_test"]["bullish_1"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"ROC-Kurve (AUC={auc:.3f})", "Precision-Recall-Kurve"])
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="Random Forest",
                              line=dict(width=2.5), fill="tozeroy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Zufall",
                              line=dict(dash="dash", color="#94a3b8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR-Kurve",
                              line=dict(width=2.5)), row=1, col=2)
    fig.add_hline(y=baseline, line_dash="dash", line_color="#94a3b8",
                  annotation_text=f"Baseline {baseline:.2f}", row=1, col=2)
    fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
    fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    fig.update_layout(height=420, margin=dict(t=80))
    fig = apply_chart_theme(fig, mode, height=420)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5))
    return fig


# ------------------------------------------------------------- Learning curve
def chart_learning_curve(lc: Dict[str, Any], mode: str) -> go.Figure:
    t = _t(mode)
    sizes = lc["train_sizes_abs"]
    tr_mean, tr_std = np.array(lc["train_scores_mean"]), np.array(lc["train_scores_std"])
    va_mean, va_std = np.array(lc["val_scores_mean"]), np.array(lc["val_scores_std"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=tr_mean, mode="lines+markers", name="Trainings-Score",
                              line=dict(color=t["primary"], width=2.5)))
    fig.add_trace(go.Scatter(x=sizes + sizes[::-1], y=list(tr_mean + tr_std) + list((tr_mean - tr_std)[::-1]),
                              fill="toself", fillcolor=t["primary-soft"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sizes, y=va_mean, mode="lines+markers", name="Validierungs-Score (CV)",
                              line=dict(color=t["positive"], width=2.5)))
    fig.add_trace(go.Scatter(x=sizes + sizes[::-1], y=list(va_mean + va_std) + list((va_mean - va_std)[::-1]),
                              fill="toself", fillcolor=t["positive-soft"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        title=f"Lernkurve — {lc['scoring']} vs. Trainingsgröße ({lc['cv_n_splits']}-Fold CV)",
        xaxis_title="Anzahl Trainingsbeispiele", yaxis_title=lc["scoring"].upper(),
        height=420,
    )
    return apply_chart_theme(fig, mode, height=420)


def learning_curve_diagnosis(lc: Dict[str, Any]) -> str:
    tr_final = lc["train_scores_mean"][-1]
    va_final = lc["val_scores_mean"][-1]
    gap = tr_final - va_final
    va_trend = lc["val_scores_mean"][-1] - lc["val_scores_mean"][0]
    if gap > 0.08:
        return (f"<b>Hohe Varianz (Overfitting-Tendenz):</b> Trainings-Score ({tr_final:.3f}) liegt deutlich über "
                f"dem Validierungs-Score ({va_final:.3f}), Gap = {gap:.3f}. Mehr Daten oder stärkere "
                f"Regularisierung (kleinere <code>max_depth</code>) könnten helfen.")
    if va_final < 0.55:
        return (f"<b>Hoher Bias (Underfitting-Tendenz):</b> Beide Kurven konvergieren auf niedrigem Niveau "
                f"(Val = {va_final:.3f}). Mehr Trainingsdaten allein würde kaum helfen — das Modell ist "
                f"vermutlich zu einfach für das Signal, das in den Daten steckt (vgl. EMH, Fama 1970).")
    return (f"<b>Gute Balance:</b> Train ({tr_final:.3f}) und Validierung ({va_final:.3f}) liegen nah beieinander "
            f"(Gap = {gap:.3f}). Validierungs-Score verändert sich seit dem ersten Punkt um "
            f"{va_trend:+.3f} — zusätzliche Trainingsdaten bringen aktuell {'kaum' if abs(va_trend) < 0.01 else 'noch'} weiteren Nutzen.")


# --------------------------------------------------------- Feature importance
def chart_feature_importance(diag: Dict[str, Any], mode: str) -> go.Figure:
    t = _t(mode)
    fi = diag["feature_importance"]
    items = sorted(fi.items(), key=lambda kv: kv[1])
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    values = [v for _, v in items]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                            marker_color=t["primary"],
                            text=[f"{v:.3f}" for v in values], textposition="outside"))
    fig.update_layout(title="RandomForest Feature Importance", height=340,
                       xaxis_title="Importance (Gini)")
    return apply_chart_theme(fig, mode, height=340)
