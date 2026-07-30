"""Reproducible out-of-time model benchmark for WealthScope AI 1.0.

Financial rows are ordered observations, not exchangeable samples. The
pipeline therefore uses:

* a final 20% out-of-time holdout,
* a 20-trading-day purge before every validation window (matching target_20d),
* four expanding walk-forward folds,
* five algorithms that make the historical development of classification
  visible: Dummy, Logistic Regression, Decision Tree, Linear SVM and RF.

Run with the project environment:
    python scripts/train_and_diagnose.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "wealthscope_features.parquet"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "wealthscope_model.joblib"
DIAGNOSTICS_PATH = MODEL_DIR / "diagnostics.json"
LEARNING_CURVE_PATH = MODEL_DIR / "learning_curve.json"
REPORT_PATH = MODEL_DIR / "wealthscope_model_report.txt"

FEATURES = [
    "daily_return", "return_5d", "return_20d",
    "ma_20_distance", "ma_50_distance", "ma_200_distance",
    "volatility_20d", "drawdown",
]
TARGET = "target_20d"
RANDOM_STATE = 42
PURGE_PERIODS = 20
TEST_FRACTION = 0.20

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "dummy": {
        "label": "Dummy (Mehrheitsklasse)", "year": "Baseline",
        "family": "Kontrollmodell",
        "estimator": DummyClassifier(strategy="prior"),
    },
    "logistic": {
        "label": "Logistische Regression", "year": "1958",
        "family": "Statistisch-linear",
        "estimator": LogisticRegression(
            solver="liblinear", max_iter=1_000,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    },
    "decision_tree": {
        "label": "Entscheidungsbaum", "year": "1984",
        "family": "Regelbasiert / White Box",
        "estimator": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    },
    "linear_svm": {
        "label": "Linear SVM", "year": "1995",
        "family": "Maximum Margin",
        "estimator": LinearSVC(
            class_weight="balanced", dual="auto",
            max_iter=5_000, random_state=RANDOM_STATE
        ),
    },
    "random_forest": {
        "label": "Random Forest", "year": "2001",
        "family": "Ensemble Learning",
        "estimator": RandomForestClassifier(**RF_PARAMS),
    },
}


def build_pipeline(model_key: str = "random_forest") -> Pipeline:
    """Create an isolated sklearn pipeline for one benchmark candidate."""
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {model_key}")
    scaler = (
        StandardScaler()
        if model_key in {"logistic", "linear_svm"}
        else "passthrough"
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", scaler),
        ("mod", clone(MODEL_SPECS[model_key]["estimator"])),
    ])


def _scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return np.asarray(decision, dtype=float)


def _metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def _window_masks(
    dates: pd.Series,
    unique_dates: pd.DatetimeIndex,
    validation_start_idx: int,
    validation_end_idx: int,
) -> Tuple[pd.Series, pd.Series, pd.Timestamp, pd.Timestamp]:
    train_end_idx = validation_start_idx - PURGE_PERIODS - 1
    if train_end_idx <= 0:
        raise ValueError("Validation starts too early for the purge window.")
    train_end = unique_dates[train_end_idx]
    val_start = unique_dates[validation_start_idx]
    val_end = unique_dates[min(validation_end_idx, len(unique_dates) - 1)]
    return dates <= train_end, dates.between(val_start, val_end), train_end, val_start


def _fit_and_measure(
    model_key: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> Tuple[Pipeline, Dict[str, float], np.ndarray, np.ndarray]:
    model = build_pipeline(model_key)
    started = time.perf_counter()
    with warnings.catch_warnings():
        # Apple's Accelerate-backed NumPy can emit spurious intermediate
        # matmul RuntimeWarnings in sklearn although final coefficients and
        # scores are finite. The explicit check below remains authoritative.
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning,
            module=r"sklearn\.utils\.extmath",
        )
        model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    y_pred = model.predict(X_eval)
    y_score = _scores(model, X_eval)
    if not np.isfinite(y_score).all():
        raise ValueError(f"{model_key} produced non-finite evaluation scores.")
    predict_ms_per_1k = (
        (time.perf_counter() - predict_started) * 1_000
        / max(len(X_eval), 1) * 1_000
    )
    metrics = _metrics(y_eval, y_pred, y_score)
    metrics.update({
        "fit_seconds": float(fit_seconds),
        "predict_ms_per_1k": float(predict_ms_per_1k),
    })
    return model, metrics, y_pred, y_score


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std())


def _downsample_curve(x: np.ndarray, y: np.ndarray, limit: int = 800) -> Tuple[list, list]:
    """Keep JSON and browser payloads small without changing displayed shape."""
    if len(x) <= limit:
        return x.tolist(), y.tolist()
    indices = np.unique(np.linspace(0, len(x) - 1, limit, dtype=int))
    return x[indices].tolist(), y[indices].tolist()


def main() -> None:
    started = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    print(f"[1/6] Lade {DATA_PATH.relative_to(BASE_DIR)} ...")
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.dropna(subset=FEATURES + [TARGET, "date"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    X, y, dates = df[FEATURES], df[TARGET].astype(int), df["date"]
    unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    print(f"      {len(df):,} Zeilen · {len(unique_dates):,} Handelstage · "
          f"positive Klasse {(y == 1).mean():.3f}")

    print("[2/6] Erzeuge purged Out-of-Time-Holdout ...")
    test_start_idx = int(len(unique_dates) * (1 - TEST_FRACTION))
    train_mask, test_mask, train_end, test_start = _window_masks(
        dates, unique_dates, test_start_idx, len(unique_dates) - 1
    )
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]
    print(f"      Train bis {train_end.date()} ({len(X_train):,}) · "
          f"Test ab {test_start.date()} ({len(X_test):,}) · "
          f"Purge {PURGE_PERIODS} Handelstage")

    print("[3/6] Vergleiche fünf Modellgenerationen auf identischem Holdout ...")
    comparison: Dict[str, Dict[str, Any]] = {}
    fitted_models: Dict[str, Pipeline] = {}
    holdout_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for key, spec in MODEL_SPECS.items():
        model, metrics, pred, score = _fit_and_measure(
            key, X_train, y_train, X_test, y_test
        )
        fitted_models[key] = model
        holdout_predictions[key] = (pred, score)
        comparison[key] = {
            "label": spec["label"], "year": spec["year"],
            "family": spec["family"], "test_metrics": metrics,
            "walk_forward": {},
        }
        print(f"      {spec['label']:<26} AUC={metrics['roc_auc']:.4f} "
              f"BalAcc={metrics['balanced_accuracy']:.4f}")

    print("[4/6] Expanding walk-forward evaluation (4 Folds) ...")
    fold_starts = [0.40, 0.50, 0.60, 0.70]
    fold_width = 0.08
    learning_points = []
    for fold_no, fraction in enumerate(fold_starts, start=1):
        val_start_idx = int(len(unique_dates) * fraction)
        val_end_idx = int(len(unique_dates) * (fraction + fold_width))
        fold_train, fold_val, fold_train_end, fold_val_start = _window_masks(
            dates, unique_dates, val_start_idx, val_end_idx
        )
        fold_key = f"fold_{fold_no}"
        for model_key in MODEL_SPECS:
            model, metrics, _, val_score = _fit_and_measure(
                model_key, X.loc[fold_train], y.loc[fold_train],
                X.loc[fold_val], y.loc[fold_val],
            )
            comparison[model_key]["walk_forward"][fold_key] = {
                **metrics,
                "train_end": str(fold_train_end.date()),
                "validation_start": str(fold_val_start.date()),
                "n_train": int(fold_train.sum()),
                "n_validation": int(fold_val.sum()),
            }
            if model_key == "random_forest":
                train_score = _scores(model, X.loc[fold_train])
                learning_points.append({
                    "n_train": int(fold_train.sum()),
                    "train_auc": float(roc_auc_score(y.loc[fold_train], train_score)),
                    "validation_auc": float(roc_auc_score(y.loc[fold_val], val_score)),
                })

    for model_data in comparison.values():
        folds = list(model_data["walk_forward"].values())
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "average_precision"):
            mean, std = _mean_std(f[metric] for f in folds)
            model_data[f"walk_forward_{metric}_mean"] = mean
            model_data[f"walk_forward_{metric}_std"] = std

    print("[5/6] Erzeuge RF-Diagnostik und Lernkurve ...")
    pipe = fitted_models["random_forest"]
    y_pred, y_score = holdout_predictions["random_forest"]
    rf_metrics = comparison["random_forest"]["test_metrics"]
    majority_baseline = float(max(y_test.mean(), 1 - y_test.mean()))
    cm_counts = confusion_matrix(y_test, y_pred).tolist()
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true").tolist()
    fpr, tpr, _ = roc_curve(y_test, y_score)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_score)
    roc_fpr, roc_tpr = _downsample_curve(fpr, tpr)
    pr_recall, pr_precision = _downsample_curve(pr_rec, pr_prec)
    importances = pipe.named_steps["mod"].feature_importances_
    feature_importance = {f: float(v) for f, v in zip(FEATURES, importances)}

    rf_folds = list(comparison["random_forest"]["walk_forward"].values())
    cv_acc_mean, cv_acc_std = _mean_std(f["accuracy"] for f in rf_folds)
    cv_auc_mean, cv_auc_std = _mean_std(f["roc_auc"] for f in rf_folds)

    diagnostics = {
        "schema_version": 2,
        "app_version": "1.0.2",
        "trained_at_note": "generated by scripts/train_and_diagnose.py",
        "sklearn_version": sklearn.__version__,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "preprocessing": {
            "imputation": "Median imputer fitted within each training window.",
            "scaling": {
                "logistic": "StandardScaler fitted on training data only.",
                "linear_svm": "StandardScaler fitted on training data only.",
                "dummy": "No scaling required.",
                "decision_tree": "No scaling required (tree thresholds are scale-invariant).",
                "random_forest": "No scaling required (tree thresholds are scale-invariant).",
            },
        },
        "validation": {
            "strategy": "purged out-of-time holdout + expanding walk-forward",
            "purge_trading_days": PURGE_PERIODS,
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(unique_dates[-1].date()),
            "rationale": "Prevents future observations and overlapping 20-day labels leaking into training.",
        },
        "hyperparams": {**RF_PARAMS, "test_fraction": TEST_FRACTION},
        "test_metrics": {
            **rf_metrics,
            "majority_baseline": majority_baseline,
            "beats_baseline": rf_metrics["accuracy"] > majority_baseline,
            "class_balance_test": {
                "bearish_0": float(1 - y_test.mean()),
                "bullish_1": float(y_test.mean()),
            },
        },
        "confusion_matrix": {
            "labels": ["0_bearish", "1_bullish"],
            "counts": cm_counts,
            "normalized": cm_norm,
        },
        "roc_curve": {"fpr": roc_fpr, "tpr": roc_tpr},
        "pr_curve": {"precision": pr_precision, "recall": pr_recall},
        "cross_validation": {
            "strategy": "4-fold expanding walk-forward with 20-day purge",
            "n_splits": 4,
            "accuracy_mean": cv_acc_mean,
            "accuracy_std": cv_acc_std,
            "auc_mean": cv_auc_mean,
            "auc_std": cv_auc_std,
        },
        "feature_importance": feature_importance,
        "model_comparison": comparison,
    }

    learning_curve_data = {
        "scoring": "roc_auc",
        "cv_n_splits": len(learning_points),
        "strategy": "expanding walk-forward",
        "train_sizes_abs": [p["n_train"] for p in learning_points],
        "train_scores_mean": [p["train_auc"] for p in learning_points],
        "train_scores_std": [0.0 for _ in learning_points],
        "val_scores_mean": [p["validation_auc"] for p in learning_points],
        "val_scores_std": [0.0 for _ in learning_points],
    }

    print("[6/6] Schreibe deploybare Artefakte ...")
    joblib.dump(pipe, MODEL_PATH)
    DIAGNOSTICS_PATH.write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LEARNING_CURVE_PATH.write_text(
        json.dumps(learning_curve_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_key = max(
        comparison,
        key=lambda key: comparison[key]["walk_forward_roc_auc_mean"],
    )
    report = [
        "WealthScope AI 1.0 - Model Report",
        "=" * 38,
        f"Validation: {diagnostics['validation']['strategy']}",
        f"Train end: {train_end.date()} | Test: {test_start.date()} to {unique_dates[-1].date()}",
        f"Purge: {PURGE_PERIODS} trading days (target horizon)",
        f"Rows: {len(df):,} | Train: {len(X_train):,} | Test: {len(X_test):,}",
        "",
        "Out-of-time model comparison:",
        *[
            f"  {data['label']:<27} "
            f"AUC={data['test_metrics']['roc_auc']:.4f} "
            f"BalAcc={data['test_metrics']['balanced_accuracy']:.4f} "
            f"WF-AUC={data['walk_forward_roc_auc_mean']:.4f}"
            for data in comparison.values()
        ],
        "",
        f"Best walk-forward AUC: {comparison[best_key]['label']}",
        "Production demonstrator: Random Forest (interpretability + nonlinear ensemble).",
        "Not financial advice. Historical data cannot guarantee future performance.",
    ]
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")

    elapsed = time.time() - started
    print(f"\nFertig in {elapsed:.1f}s · RF Holdout-AUC={rf_metrics['roc_auc']:.4f} "
          f"· RF Walk-forward-AUC={cv_auc_mean:.4f} +/- {cv_auc_std:.4f}")
    return None


if __name__ == "__main__":
    sys.exit(main())
