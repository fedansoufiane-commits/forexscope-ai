"""Central configuration: paths, constants, feature/ticker lists, page registry."""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

MARKET_PARQUET = DATA_DIR / "wealthscope_features.parquet"
MODEL_PATH = MODEL_DIR / "wealthscope_model.joblib"
DIAGNOSTICS_PATH = MODEL_DIR / "diagnostics.json"
LEARNING_CURVE_PATH = MODEL_DIR / "learning_curve.json"

APP_NAME = "WealthScope AI"
APP_VERSION = "1.0.2"
APP_TAGLINE = "ML-gestützte Finanzanalyse · QUA³CK-Prozess · IU Internationale Hochschule"

# Same 8 engineered features the model was trained on (scripts/train_and_diagnose.py).
MODEL_FEATURES = [
    "daily_return", "return_5d", "return_20d",
    "ma_20_distance", "ma_50_distance", "ma_200_distance",
    "volatility_20d", "drawdown",
]
TARGET_COL = "target_20d"

FEATURE_LABELS = {
    "daily_return": "Tägl. Rendite",
    "return_5d": "5-Tage-Rendite",
    "return_20d": "20-Tage-Rendite",
    "ma_20_distance": "Abstand MA-20",
    "ma_50_distance": "Abstand MA-50",
    "ma_200_distance": "Abstand MA-200",
    "volatility_20d": "Volatilität (20d)",
    "drawdown": "Drawdown",
    "future_return_20d": "Zukünftige 20d-Rendite",
    "target_20d": "Ziel (Kurs steigt in 20d)",
}

TICKERS = [
    "AAPL", "AGG", "AMZN", "BA", "BND", "DIS", "EEM", "GE", "GLD", "GOOGL",
    "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MSFT", "NVDA", "PG", "QQQ",
    "SPY", "TSLA", "VOO", "VTI", "VWO", "XOM",
]

RISK_FREE_RATE = 0.02

DISCLAIMER = (
    "Dieses Projekt dient ausschließlich Lernzwecken (IU-Modul Data Analytics und Big Data, DSDABD072501). "
    "Keine Finanzberatung. Alle Analysen sind Demonstrationen ohne Gewähr."
)
