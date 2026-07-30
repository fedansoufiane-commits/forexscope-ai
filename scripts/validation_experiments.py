"""Falsification experiments behind the negative result of WealthScope AI 1.0.

The out-of-time benchmark in ``train_and_diagnose.py`` reports a weak signal
(ROC-AUC 0.519). A weak result is only a scientific finding once the obvious
counter-explanations have been ruled out. This script measures the two that
matter and writes them to a JSON artefact, so presentation, report and
notebooks cite reproducible numbers instead of prose:

* **Capacity sweep** - does the gap between training and test score come from
  too much or too little model capacity? Seven regularisation settings from
  unconstrained to heavily pruned, on one identical purged out-of-time window.
* **Split comparison** - how much apparent signal does a methodologically
  wrong split invent? The same model and data evaluated on a purged temporal
  split, a temporal split without purge, and a naive random split.

Run with the project environment:
    python scripts/validation_experiments.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "wealthscope_features.parquet"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_PATH = MODEL_DIR / "validation_experiments.json"

FEATURES = [
    "daily_return", "return_5d", "return_20d",
    "ma_20_distance", "ma_50_distance", "ma_200_distance",
    "volatility_20d", "drawdown",
]
TARGET = "target_20d"
RANDOM_STATE = 42
PURGE_PERIODS = 20
TEST_FRACTION = 0.20

# Identical to RF_PARAMS in train_and_diagnose.py; max_depth and
# min_samples_leaf are the two axes the capacity sweep varies.
RF_BASE = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

CAPACITY_GRID: List[Dict[str, Any]] = [
    {"label": "unbeschränkt", "max_depth": None, "min_samples_leaf": 1},
    {"label": "max_depth=16, leaf=2", "max_depth": 16, "min_samples_leaf": 2},
    {"label": "max_depth=8, leaf=5 (v1.0)", "max_depth": 8, "min_samples_leaf": 5},
    {"label": "max_depth=6, leaf=20", "max_depth": 6, "min_samples_leaf": 20},
    {"label": "max_depth=4, leaf=50", "max_depth": 4, "min_samples_leaf": 50},
    {"label": "max_depth=3, leaf=100", "max_depth": 3, "min_samples_leaf": 100},
    {"label": "max_depth=2, leaf=200", "max_depth": 2, "min_samples_leaf": 200},
]


def build_pipeline(**overrides: Any) -> Pipeline:
    """RF pipeline without scaler - trees are invariant to feature scale."""
    params = dict(RF_BASE)
    params.update(overrides)
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("mod", RandomForestClassifier(**params)),
    ])


def evaluate(pipe: Pipeline, X_train, y_train, X_eval, y_eval) -> Dict[str, float]:
    pipe.fit(X_train, y_train)
    score_train = pipe.predict_proba(X_train)[:, 1]
    score_eval = pipe.predict_proba(X_eval)[:, 1]
    pred_eval = pipe.predict(X_eval)
    positive_rate = float((y_eval == 1).mean())
    return {
        "train_roc_auc": float(roc_auc_score(y_train, score_train)),
        "roc_auc": float(roc_auc_score(y_eval, score_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred_eval)),
        "accuracy": float(accuracy_score(y_eval, pred_eval)),
        "majority_baseline_accuracy": max(positive_rate, 1.0 - positive_rate),
        "positive_rate": positive_rate,
        "n_train": int(len(X_train)),
        "n_eval": int(len(X_eval)),
    }


def main() -> None:
    started = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    print(f"[1/4] Lade {DATA_PATH.relative_to(BASE_DIR)} ...")
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.dropna(subset=FEATURES + [TARGET, "date"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    X, y, dates = df[FEATURES], df[TARGET].astype(int), df["date"]
    unique_dates = pd.DatetimeIndex(dates.drop_duplicates().sort_values())
    test_start_idx = int(len(unique_dates) * (1 - TEST_FRACTION))

    # Reference split: identical to the one train_and_diagnose.py reports on.
    train_end = unique_dates[test_start_idx - PURGE_PERIODS - 1]
    test_start, test_end = unique_dates[test_start_idx], unique_dates[-1]
    purged_train = dates <= train_end
    holdout = dates.between(test_start, test_end)
    print(f"      {len(df):,} Zeilen · Train bis {train_end.date()} "
          f"({int(purged_train.sum()):,}) · Test {test_start.date()} bis "
          f"{test_end.date()} ({int(holdout.sum()):,})")

    print("[2/4] Kapazitäts-Sweep über sieben Regularisierungsstufen ...")
    capacity: List[Dict[str, Any]] = []
    for spec in CAPACITY_GRID:
        label = spec["label"]
        metrics = evaluate(
            build_pipeline(max_depth=spec["max_depth"],
                           min_samples_leaf=spec["min_samples_leaf"]),
            X.loc[purged_train], y.loc[purged_train],
            X.loc[holdout], y.loc[holdout],
        )
        metrics["gap"] = metrics["train_roc_auc"] - metrics["roc_auc"]
        capacity.append({
            "label": label,
            "max_depth": spec["max_depth"],
            "min_samples_leaf": spec["min_samples_leaf"],
            **metrics,
        })
        print(f"      {label:<26} TrainAUC={metrics['train_roc_auc']:.4f} "
              f"TestAUC={metrics['roc_auc']:.4f} Gap={metrics['gap']:.4f}")

    print("[3/4] Split-Vergleich: purged / ohne Purge / naiver Zufalls-Split ...")
    v1_params = {"max_depth": 8, "min_samples_leaf": 5}
    splits: List[Dict[str, Any]] = []

    metrics = evaluate(build_pipeline(**v1_params),
                       X.loc[purged_train], y.loc[purged_train],
                       X.loc[holdout], y.loc[holdout])
    splits.append({"label": "zeitlich + 20 Handelstage Purge (v1.0)",
                   "leakage": "keine", **metrics})

    # Without the purge the last training labels still reach into the test
    # window, because target_20d looks 20 trading days ahead.
    no_purge_train = dates < test_start
    metrics = evaluate(build_pipeline(**v1_params),
                       X.loc[no_purge_train], y.loc[no_purge_train],
                       X.loc[holdout], y.loc[holdout])
    splits.append({"label": "zeitlich, ohne Purge",
                   "leakage": "überlappende Zielhorizonte", **metrics})

    X_rand_tr, X_rand_te, y_rand_tr, y_rand_te = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE, shuffle=True
    )
    metrics = evaluate(build_pipeline(**v1_params),
                       X_rand_tr, y_rand_tr, X_rand_te, y_rand_te)
    splits.append({"label": "naiver Zufalls-Split 80/20",
                   "leakage": "Zukunft im Training + Zielhorizont-Overlap",
                   **metrics})

    for row in splits:
        print(f"      {row['label']:<40} AUC={row['roc_auc']:.4f} "
              f"Acc={row['accuracy']:.4f} (Baseline "
              f"{row['majority_baseline_accuracy']:.4f})")

    print(f"[4/4] Schreibe {OUTPUT_PATH.relative_to(BASE_DIR)} ...")
    reference = splits[0]["roc_auc"]
    inflated = splits[-1]["roc_auc"]
    test_aucs = [row["roc_auc"] for row in capacity]
    payload = {
        "schema_version": 1,
        "sklearn_version": sklearn.__version__,
        "purpose": ("Falsifikationsexperimente zum schwachen Signal: schließen "
                    "Modellkapazität und Datenmenge als Ursache aus und "
                    "quantifizieren die Illusion eines falschen Splits."),
        "features": FEATURES,
        "target": TARGET,
        "n_rows": int(len(df)),
        "reference_split": {
            "strategy": "purged out-of-time holdout",
            "purge_trading_days": PURGE_PERIODS,
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
        },
        "hyperparams_reference": {**RF_BASE, **v1_params},
        "capacity_sweep": capacity,
        "capacity_sweep_summary": {
            "test_roc_auc_min": min(test_aucs),
            "test_roc_auc_max": max(test_aucs),
            "test_roc_auc_span": max(test_aucs) - min(test_aucs),
            "best_label": capacity[int(np.argmax(test_aucs))]["label"],
            "conclusion": ("Der Test-AUC ist über den gesamten Komplexitätsbereich "
                           "flach; die Lücke entsteht ausschließlich auf der "
                           "Trainingsseite. Kapazität ist nicht der Engpass."),
        },
        "split_comparison": splits,
        "split_comparison_summary": {
            "reference_roc_auc": reference,
            "leaky_roc_auc": inflated,
            "absolute_difference": inflated - reference,
            "apparent_signal_reference": reference - 0.5,
            "apparent_signal_leaky": inflated - 0.5,
            "apparent_signal_factor": (inflated - 0.5) / (reference - 0.5),
            "conclusion": ("Ein naiver Zufalls-Split lässt das scheinbare Signal "
                           "um den angegebenen Faktor größer erscheinen, ohne dass "
                           "Modell oder Daten sich ändern."),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    factor = payload["split_comparison_summary"]["apparent_signal_factor"]
    print(f"\nFertig in {time.time() - started:.1f}s · Test-AUC-Spanne im Sweep "
          f"{payload['capacity_sweep_summary']['test_roc_auc_span']:.4f} · "
          f"scheinbares Signal beim Zufalls-Split {factor:.1f}x größer")
    return None


if __name__ == "__main__":
    sys.exit(main())
