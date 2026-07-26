"""
WealthScope AI — Model Training & Diagnostics Pipeline
========================================================
Single source of truth for the RandomForest classifier used across the app.

Trains on the Kaggle-derived feature parquet only (no live network calls —
fully reproducible offline) and writes THREE artifacts:

  models/wealthscope_model.joblib   trained sklearn Pipeline (impute→scale→RF)
  models/diagnostics.json           test-set metrics, confusion matrix, ROC/PR,
                                     5-fold CV scores, RF feature importance
  models/learning_curve.json        sklearn.model_selection.learning_curve()
                                     output (train/val score vs. training size)

Rationale for precomputing instead of recomputing on every Streamlit rerun:
learning_curve() alone fits the 200-tree RandomForest ~40 times (8 train
sizes x 5 CV folds) on up to ~144k rows — far too slow to run inside a page
render. Run this script once after any change to features/hyperparameters/
data, then the app just reads the JSON.

Usage: python3 scripts/train_and_diagnose.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "wealthscope_features.parquet"
MODEL_PATH = BASE_DIR / "models" / "wealthscope_model.joblib"
DIAGNOSTICS_PATH = BASE_DIR / "models" / "diagnostics.json"
LEARNING_CURVE_PATH = BASE_DIR / "models" / "learning_curve.json"
REPORT_PATH = BASE_DIR / "models" / "wealthscope_model_report.txt"

FEATURES = [
    "daily_return", "return_5d", "return_20d",
    "ma_20_distance", "ma_50_distance", "ma_200_distance",
    "volatility_20d", "drawdown",
]
TARGET = "target_20d"
RANDOM_STATE = 42

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mod", RandomForestClassifier(**RF_PARAMS)),
    ])


def main() -> None:
    t0 = time.time()
    print(f"[1/6] Loading {DATA_PATH.relative_to(BASE_DIR)} ...")
    df = pd.read_parquet(DATA_PATH)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    print(f"      {len(df):,} rows after dropna · class balance: "
          f"{(y == 1).mean():.3f} positive")

    print("[2/6] Splitting 75/25 (stratified, random_state=42) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    print("[3/6] Fitting RandomForest pipeline on training set ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    majority_baseline = float(max(y_test.mean(), 1 - y_test.mean()))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    auc = float(roc_auc_score(y_test, y_proba))

    cm_counts = confusion_matrix(y_test, y_pred).tolist()
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true").tolist()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_proba)

    print("[4/6] 5-fold stratified cross-validation on full dataset ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_acc = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_auc = cross_val_score(build_pipeline(), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    importances = pipe.named_steps["mod"].feature_importances_
    feature_importance = {f: float(v) for f, v in zip(FEATURES, importances)}

    print("[5/6] Computing learning curve (this fits ~40 RF models, be patient) ...")
    lc_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_sizes_abs, train_scores, val_scores = learning_curve(
        build_pipeline(), X, y,
        cv=lc_cv,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="roc_auc",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    print("[6/6] Writing artifacts ...")
    joblib.dump(pipe, MODEL_PATH)

    diagnostics = {
        "trained_at_note": "generated by scripts/train_and_diagnose.py",
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "hyperparams": {**RF_PARAMS, "test_size": 0.25},
        "test_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_weighted": f1,
            "roc_auc": auc,
            "majority_baseline": majority_baseline,
            "beats_baseline": acc > majority_baseline,
            "class_balance_test": {"bearish_0": float(1 - y_test.mean()), "bullish_1": float(y_test.mean())},
        },
        "confusion_matrix": {
            "labels": ["0_bearish", "1_bullish"],
            "counts": cm_counts,
            "normalized": cm_norm,
        },
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": pr_prec.tolist(), "recall": pr_rec.tolist()},
        "cross_validation": {
            "n_splits": 5,
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
            "auc_mean": float(cv_auc.mean()),
            "auc_std": float(cv_auc.std()),
        },
        "feature_importance": feature_importance,
    }
    DIAGNOSTICS_PATH.write_text(json.dumps(diagnostics, indent=2))

    learning_curve_data = {
        "scoring": "roc_auc",
        "cv_n_splits": 5,
        "train_sizes_abs": train_sizes_abs.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }
    LEARNING_CURVE_PATH.write_text(json.dumps(learning_curve_data, indent=2))

    report_lines = [
        "WealthScope AI — Model Report",
        "=" * 32,
        f"Algorithm: RandomForestClassifier (sklearn {sklearn.__version__})",
        f"Target: {TARGET} (1 = price higher in 20 trading days)",
        "",
        f"Dataset rows: {len(df):,}  |  Train: {len(X_train):,}  |  Test: {len(X_test):,}",
        f"Features ({len(FEATURES)}): {', '.join(FEATURES)}",
        "",
        f"Accuracy:          {acc:.4f}",
        f"Majority baseline:  {majority_baseline:.4f}  ({'beats' if acc > majority_baseline else 'below'} baseline)",
        f"ROC-AUC:           {auc:.4f}",
        f"Precision:         {prec:.4f}",
        f"Recall:            {rec:.4f}",
        f"F1 (weighted):     {f1:.4f}",
        "",
        f"5-fold CV accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}",
        f"5-fold CV ROC-AUC:  {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}",
        "",
        "Feature importance (RandomForest):",
        *[f"  {f:<18} {v:.4f}" for f, v in sorted(feature_importance.items(), key=lambda kv: -kv[1])],
        "",
        "Interpretation: Accuracy modestly above random is expected and consistent",
        "with the Efficient Market Hypothesis (Fama, 1970) — historical price data",
        "alone has limited predictive power over 20-day horizons.",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s")
    print(f"  Accuracy={acc:.4f}  ROC-AUC={auc:.4f}  CV-AUC={cv_auc.mean():.4f}+/-{cv_auc.std():.4f}")
    print(f"  -> {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"  -> {DIAGNOSTICS_PATH.relative_to(BASE_DIR)}")
    print(f"  -> {LEARNING_CURVE_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    sys.exit(main())
