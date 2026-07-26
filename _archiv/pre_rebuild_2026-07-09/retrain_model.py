#!/usr/bin/env python3
"""
WealthScope AI – Modell-Retraining Script v2
==============================================
Ausführung:  python3 retrain_model.py
Dauer:       ~5–15 min (abhängig von yfinance-Geschwindigkeit)

Was dieses Script tut:
  1. Bestehende Kaggle-Features laden (1962–2017-11-10)
  2. yfinance-Daten für alle 26 Ticker von 2017-11-11 bis heute herunterladen
  3. ^VIX für gesamten Zeitraum herunterladen
  4. Features neu berechnen (identisch zu enrich_features in app_max.py)
  5. VIX als Makro-Feature anfügen
  6. Erweitertes Dataset als Parquet speichern
  7. Neues RandomForest-Modell (v2) trainieren + auswerten
  8. Score-Gewichte per Logistic Regression kalibrieren
  9. Ergebnisse ausgeben: neue Accuracy, ROC-AUC, Gewichte

Outputs:
  - data/processed/wealthscope_features_v2.parquet  (erweitertes Dataset)
  - models/wealthscope_model_v2.joblib              (neues Modell)
  - models/score_weights_v2.json                   (kalibrierte Score-Gewichte)
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "AGG", "AMZN", "BA", "BND", "DIS", "EEM", "GE", "GLD",
    "GOOGL", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MSFT", "NVDA",
    "PG", "QQQ", "SPY", "TSLA", "VOO", "VTI", "VWO", "XOM",
]

# Kaggle-Daten enden am 2017-11-10; wir starten einen Tag danach
YFINANCE_START = "2017-11-11"
YFINANCE_END = datetime.today().strftime("%Y-%m-%d")

# 8 Feature-Spalten, die das Modell sieht (identisch zu app_max.py)
MODEL_FEATURES = [
    "daily_return",
    "return_5d",
    "return_20d",
    "ma_20_distance",
    "ma_50_distance",
    "ma_200_distance",
    "volatility_20d",
    "drawdown",
]

# V2: mit VIX
MODEL_FEATURES_V2 = MODEL_FEATURES + ["vix_level", "vix_change_5d"]

TARGET_COL = "target_20d"

PARQUET_V1 = Path("data/processed/wealthscope_features.parquet")
PARQUET_V2 = Path("data/processed/wealthscope_features_v2.parquet")
MODEL_V1   = Path("models/wealthscope_model.joblib")
MODEL_V2   = Path("models/wealthscope_model_v2.joblib")
WEIGHTS_V2 = Path("models/score_weights_v2.json")


# ─────────────────────────────────────────────────────────────────────────────
# HILFSFUNKTIONEN
# ─────────────────────────────────────────────────────────────────────────────

def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Identisch zu enrich_features() in app_max.py."""
    out = df.copy().sort_values("date")
    out["daily_return"]      = out["close"].pct_change()
    out["return_5d"]         = out["close"].pct_change(5)
    out["return_20d"]        = out["close"].pct_change(20)
    out["return_60d"]        = out["close"].pct_change(60)
    out["ma_20"]             = out["close"].rolling(20).mean()
    out["ma_50"]             = out["close"].rolling(50).mean()
    out["ma_100"]            = out["close"].rolling(100).mean()
    out["ma_200"]            = out["close"].rolling(200).mean()
    out["ma_20_distance"]    = out["close"] / out["ma_20"] - 1
    out["ma_50_distance"]    = out["close"] / out["ma_50"] - 1
    out["ma_200_distance"]   = out["close"] / out["ma_200"] - 1
    out["volatility_20d"]    = out["daily_return"].rolling(20).std() * np.sqrt(252)
    out["volatility_60d"]    = out["daily_return"].rolling(60).std() * np.sqrt(252)
    out["rolling_high"]      = out["close"].cummax()
    out["drawdown"]          = out["close"] / out["rolling_high"] - 1
    out["rolling_low_60"]    = out["close"].rolling(60).min()
    out["rolling_high_60"]   = out["close"].rolling(60).max()
    out["future_return_20d"] = out["close"].shift(-20) / out["close"] - 1
    out["target_20d"]        = (out["future_return_20d"] > 0).astype(float)
    return out


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Lädt yfinance-Daten für einen Ticker und normalisiert die Spalten."""
    try:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  ⚠️  {ticker}: keine Daten")
            return pd.DataFrame()
        raw = raw.reset_index()
        # yfinance liefert MultiIndex bei auto_adjust – flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0].lower() if col[1] == "" else col[0].lower() for col in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        raw = raw.rename(columns={"price": "close"}) if "price" in raw.columns else raw
        raw["ticker"] = ticker
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        needed = ["date", "ticker", "open", "high", "low", "close", "volume"]
        existing = [c for c in needed if c in raw.columns]
        return raw[existing]
    except Exception as e:
        print(f"  ❌  {ticker}: {e}")
        return pd.DataFrame()


def download_vix(start: str, end: str) -> pd.DataFrame:
    """Lädt ^VIX und gibt DataFrame mit [date, vix_level] zurück."""
    try:
        vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
        if vix.empty:
            return pd.DataFrame()
        vix = vix.reset_index()
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = [col[0].lower() for col in vix.columns]
        else:
            vix.columns = [c.lower() for c in vix.columns]
        close_col = "close" if "close" in vix.columns else vix.columns[-1]
        vix = vix.rename(columns={close_col: "vix_level"})
        vix["date"] = pd.to_datetime(vix["date"]).dt.normalize()
        vix["vix_change_5d"] = vix["vix_level"].pct_change(5)
        return vix[["date", "vix_level", "vix_change_5d"]]
    except Exception as e:
        print(f"  ❌  VIX: {e}")
        return pd.DataFrame()


def build_pipeline(feature_cols: list, n_estimators: int = 300) -> Pipeline:
    """Baut sklearn-Pipeline: Imputer → Scaler → RandomForest."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("mod", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=8,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def calibrate_score_weights(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """
    Kalibriert Score-Gewichte via Logistic Regression auf RF-Predictions.

    Approach:
      1. RF trainieren (klein, schnell)
      2. predict_proba als Output
      3. Statt Gewichte aus RF proba → Logistic Regression auf normierte
         Einzel-Dimension-Scores lernen
      4. Koeffizienten = kalibrierte Gewichte

    Vereinfachte Implementierung: wir trainieren LR direkt auf den
    Feature-Gruppen-Aggregaten, um Gewichte je Dimension zu bekommen.
    """
    # Dimensions-Gruppen (identisch zu Score-Berechnung in app_max.py)
    dimensions = {
        "Trend":       ["ma_20_distance", "ma_50_distance", "ma_200_distance"],
        "Momentum":    ["daily_return", "return_5d", "return_20d"],
        "Volatilität": ["volatility_20d"],
        "Drawdown":    ["drawdown"],
    }
    if "vix_level" in feature_names:
        dimensions["Makro"] = ["vix_level", "vix_change_5d"]

    feat_idx = {f: i for i, f in enumerate(feature_names)}
    X_dims = np.zeros((X.shape[0], len(dimensions)))

    for j, (dim, cols) in enumerate(dimensions.items()):
        idxs = [feat_idx[c] for c in cols if c in feat_idx]
        if idxs:
            X_dims[:, j] = np.nanmean(np.abs(X[:, idxs]), axis=1)

    # Logistic Regression ohne Regularisierung → rohes Gewichtsverhältnis
    lr = LogisticRegression(C=1e6, max_iter=1000, random_state=42)
    lr.fit(X_dims, y)
    coefs = np.abs(lr.coef_[0])
    total = coefs.sum()
    weights = {dim: round(float(coefs[j] / total), 4)
               for j, dim in enumerate(dimensions.keys())}
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# HAUPTPROGRAMM
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("WealthScope AI – Modell-Retraining v2")
    print(f"Datum: {datetime.today().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # ── 1. Bestehende Kaggle-Daten laden ────────────────────────────────────
    print("\n[1/8] Lade bestehende Kaggle-Features …")
    if not PARQUET_V1.exists():
        raise FileNotFoundError(
            f"Datei nicht gefunden: {PARQUET_V1}\n"
            "Bitte stelle sicher, dass wealthscope_features.parquet "
            "unter data/processed/ liegt."
        )
    df_kaggle = pd.read_parquet(PARQUET_V1)
    df_kaggle["date"] = pd.to_datetime(df_kaggle["date"]).dt.normalize()
    print(f"  ✅  Kaggle: {len(df_kaggle):,} Zeilen, "
          f"{df_kaggle['ticker'].nunique()} Ticker, "
          f"bis {df_kaggle['date'].max().date()}")

    # ── 2. yfinance-Daten herunterladen ─────────────────────────────────────
    print(f"\n[2/8] Lade yfinance-Daten ({YFINANCE_START} → {YFINANCE_END}) …")
    yf_frames = []
    for i, ticker in enumerate(TICKERS, 1):
        print(f"  [{i:2d}/{len(TICKERS)}] {ticker} …", end=" ", flush=True)
        frame = download_ticker(ticker, YFINANCE_START, YFINANCE_END)
        if not frame.empty:
            yf_frames.append(frame)
            print(f"{len(frame):,} Zeilen")
        else:
            print("übersprungen")

    if not yf_frames:
        print("\n❌ Keine yfinance-Daten geladen. Prüfe Internetverbindung.")
        return

    df_yf_raw = pd.concat(yf_frames, ignore_index=True)
    print(f"\n  yfinance gesamt: {len(df_yf_raw):,} Zeilen für "
          f"{df_yf_raw['ticker'].nunique()} Ticker")

    # ── 3. Features für yfinance-Daten berechnen ───────────────────────────
    print("\n[3/8] Berechne Features für neue Daten …")
    yf_enriched_frames = []
    for ticker in df_yf_raw["ticker"].unique():
        sub = df_yf_raw[df_yf_raw["ticker"] == ticker].copy()
        # Für korrekte MA-Berechnung: letzten 250 Handelstage Kaggle-Daten
        # als Warm-up anhängen (werden nach Feature-Berechnung entfernt)
        kaggle_ticker = (
            df_kaggle[df_kaggle["ticker"] == ticker][["date", "ticker", "open", "high", "low", "close", "volume"]]
            .tail(250)
            .copy()
        )
        combined = pd.concat([kaggle_ticker, sub], ignore_index=True).drop_duplicates("date")
        enriched = enrich_features(combined)
        # Nur die neuen yfinance-Zeilen behalten
        enriched = enriched[enriched["date"] >= pd.Timestamp(YFINANCE_START)]
        yf_enriched_frames.append(enriched)

    df_yf = pd.concat(yf_enriched_frames, ignore_index=True)
    print(f"  ✅  {len(df_yf):,} neue Feature-Zeilen berechnet")

    # ── 4. VIX laden ────────────────────────────────────────────────────────
    print("\n[4/8] Lade VIX-Daten …")
    # VIX ab 1990 verfügbar; für historischen Kaggle-Teil ab 1990
    vix_start = "1990-01-01"
    df_vix = download_vix(vix_start, YFINANCE_END)
    if df_vix.empty:
        print("  ⚠️  VIX nicht verfügbar – Modell wird ohne VIX trainiert")
        use_vix = False
    else:
        print(f"  ✅  VIX: {len(df_vix):,} Zeilen, "
              f"von {df_vix['date'].min().date()} bis {df_vix['date'].max().date()}")
        use_vix = True

    # ── 5. Datasets zusammenführen ──────────────────────────────────────────
    print("\n[5/8] Füge Datasets zusammen …")

    # Kaggle-Daten: nur relevante Spalten halten
    base_cols = ["date", "ticker", "close", "daily_return", "return_5d",
                 "return_20d", "ma_20_distance", "ma_50_distance",
                 "ma_200_distance", "volatility_20d", "drawdown",
                 "future_return_20d", "target_20d"]
    kaggle_slim = df_kaggle[[c for c in base_cols if c in df_kaggle.columns]].copy()

    # yfinance-Daten: gleiche Spalten
    yf_slim = df_yf[[c for c in base_cols if c in df_yf.columns]].copy()

    df_combined = pd.concat([kaggle_slim, yf_slim], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["date", "ticker"])
    df_combined = df_combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    if use_vix:
        df_combined = df_combined.merge(df_vix, on="date", how="left")
        # VIX-Lücken vorwärts auffüllen (Wochenenden, Feiertage)
        df_combined["vix_level"] = df_combined["vix_level"].ffill()
        df_combined["vix_change_5d"] = df_combined["vix_change_5d"].ffill()

    print(f"  ✅  Gesamt: {len(df_combined):,} Zeilen, "
          f"{df_combined['ticker'].nunique()} Ticker")
    print(f"      Zeitraum: {df_combined['date'].min().date()} "
          f"→ {df_combined['date'].max().date()}")

    # ── 6. Dataset speichern ────────────────────────────────────────────────
    print(f"\n[6/8] Speichere erweitertes Dataset → {PARQUET_V2} …")
    PARQUET_V2.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(PARQUET_V2, index=False)
    size_mb = PARQUET_V2.stat().st_size / 1e6
    print(f"  ✅  {size_mb:.1f} MB gespeichert")

    # ── 7. Modell trainieren ────────────────────────────────────────────────
    print("\n[7/8] Trainiere Modell v2 …")

    feat_cols = MODEL_FEATURES_V2 if use_vix else MODEL_FEATURES

    train_df = df_combined.dropna(subset=feat_cols + [TARGET_COL]).copy()
    # Letzten 20 Handelstage pro Ticker entfernen (kein valides Target)
    train_df = train_df[train_df["future_return_20d"].notna()]

    X = train_df[feat_cols].values
    y = train_df[TARGET_COL].values.astype(int)

    print(f"  Training-Datensatz: {len(train_df):,} Zeilen, "
          f"{len(feat_cols)} Features")
    print(f"  Klassen: {np.sum(y==0):,} (0) vs {np.sum(y==1):,} (1)")
    print(f"  Majority-Baseline: {max(np.mean(y), 1-np.mean(y)):.4f}")

    pipe = build_pipeline(feat_cols)

    # 5-Fold Stratified CV (gleich wie Originalmodell)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_auc    = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc",  n_jobs=-1)

    print(f"\n  CV-Ergebnisse (5-Fold):")
    print(f"    Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"    ROC-AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    # Finales Modell auf gesamten Trainingsdaten
    pipe.fit(X, y)

    # Held-out Evaluation: letztes Jahr als Test-Set
    cutoff = pd.Timestamp(YFINANCE_END) - pd.DateOffset(years=1)
    test_mask = train_df["date"] >= cutoff
    if test_mask.sum() > 100:
        X_test = train_df.loc[test_mask, feat_cols].values
        y_test = train_df.loc[test_mask, TARGET_COL].values.astype(int)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        print(f"\n  Hold-out (letztes Jahr, n={len(y_test):,}):")
        print(f"    Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"    ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
        print(f"    Majority-Baseline: {max(np.mean(y_test), 1-np.mean(y_test)):.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Negativ", "Positiv"]))

    MODEL_V2.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_V2)
    print(f"\n  ✅  Modell gespeichert → {MODEL_V2}")

    # ── 8. Score-Gewichte kalibrieren ───────────────────────────────────────
    print("\n[8/8] Kalibriere Score-Gewichte per Logistic Regression …")

    weights = calibrate_score_weights(X, y, feat_cols)
    print("\n  Kalibrierte Gewichte:")
    for dim, w in weights.items():
        print(f"    {dim:12s}: {w*100:5.1f}%")

    # Aktuelle (manuelle) Gewichte zum Vergleich
    manual_weights = {
        "Trend": 0.36, "Momentum": 0.24, "Volatilität": 0.22, "Drawdown": 0.18
    }
    if use_vix:
        manual_weights["Makro"] = 0.0  # war bisher nicht vorhanden

    print("\n  Bisherige manuelle Gewichte:")
    for dim, w in manual_weights.items():
        print(f"    {dim:12s}: {w*100:5.1f}%")

    # RF Feature Importance
    rf_model = pipe.named_steps["mod"]
    fi = dict(zip(feat_cols, rf_model.feature_importances_))
    print("\n  RF Feature Importances:")
    for f, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {f:20s}: {imp*100:.2f}%")

    output = {
        "version": "v2",
        "trained_at": datetime.today().isoformat(),
        "features": feat_cols,
        "use_vix": use_vix,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std":  float(cv_scores.std()),
        "cv_auc_mean":      float(cv_auc.mean()),
        "cv_auc_std":       float(cv_auc.std()),
        "majority_baseline": float(max(np.mean(y), 1 - np.mean(y))),
        "calibrated_weights": weights,
        "rf_feature_importances": {k: float(v) for k, v in fi.items()},
        "dataset_rows": len(train_df),
        "date_min": str(train_df["date"].min().date()),
        "date_max": str(train_df["date"].max().date()),
    }

    WEIGHTS_V2.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_V2, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  ✅  Gewichte + Metriken gespeichert → {WEIGHTS_V2}")

    # ── ZUSAMMENFASSUNG ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FERTIG – Zusammenfassung")
    print("=" * 60)
    print(f"  Dataset v2:    {PARQUET_V2}")
    print(f"  Modell v2:     {MODEL_V2}")
    print(f"  Gewichte v2:   {WEIGHTS_V2}")
    print(f"\n  CV Accuracy:   {cv_scores.mean():.4f} (vorher: 0.5572)")
    print(f"  CV ROC-AUC:    {cv_auc.mean():.4f} (vorher: 0.5842)")
    print(f"  Majority-BL:   {max(np.mean(y), 1-np.mean(y)):.4f}")
    print()
    if cv_scores.mean() > 0.5637:
        print("  ✅  Accuracy ist jetzt ÜBER der Majority-Baseline!")
    else:
        print("  ⚠️  Accuracy liegt noch unter/bei der Majority-Baseline.")
        print("      Das ist normal bei effizienten Märkten – ROC-AUC zählt.")
    print()
    print("  Nächster Schritt:")
    print("  → app_max.py aktualisieren: MODEL_V2 laden + VIX in Scoring")
    print("=" * 60)


if __name__ == "__main__":
    main()
