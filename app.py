import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="ForexScope AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1550px;
    }

    .main-title {
        font-size: 2.95rem;
        font-weight: 950;
        letter-spacing: -0.055em;
        margin-bottom: 0rem;
        line-height: 1.05;
    }

    .subtitle {
        color: #8f8f8f;
        font-size: 1.08rem;
        margin-bottom: 1.4rem;
    }

    .hero-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 26px;
        padding: 1.6rem 1.8rem;
        background: linear-gradient(135deg, rgba(110,110,110,0.22), rgba(90,90,90,0.04));
        margin-bottom: 1.2rem;
    }

    .info-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        background: rgba(128,128,128,0.06);
        margin-bottom: 1rem;
    }

    .mini-card {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 16px;
        padding: 0.95rem 1.1rem;
        background: rgba(128,128,128,0.045);
        height: 100%;
    }

    .news-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 18px;
        padding: 1.05rem 1.2rem;
        background: rgba(128,128,128,0.045);
        margin-bottom: 0.95rem;
    }

    .good-card {
        border-left: 8px solid #2ca02c;
    }

    .warn-card {
        border-left: 8px solid #ff7f0e;
    }

    .bad-card {
        border-left: 8px solid #d62728;
    }

    .neutral-card {
        border-left: 8px solid #1f77b4;
    }

    .big-number {
        font-size: 2rem;
        font-weight: 850;
        letter-spacing: -0.03em;
    }

    .muted {
        color: #8b8b8b;
    }

    a {
        text-decoration: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# FOREX CONFIG
# =========================================================

FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
    "USD/CAD": "USDCAD=X",
    "EUR/GBP": "EURGBP=X",
    "EUR/JPY": "EURJPY=X",
    "GBP/JPY": "GBPJPY=X",
    "AUD/JPY": "AUDJPY=X",
    "CHF/JPY": "CHFJPY=X",
    "EUR/CHF": "EURCHF=X",
    "EUR/CAD": "EURCAD=X",
    "GBP/CHF": "GBPCHF=X",
}

PAIR_INFO = {
    "EUR/USD": "Euro gegen US-Dollar. Stark beeinflusst durch EZB, Federal Reserve, Inflation und US-/Eurozonen-Konjunktur.",
    "GBP/USD": "Britisches Pfund gegen US-Dollar. Relevant sind Bank of England, Federal Reserve, UK-Daten und US-Dollar-Stärke.",
    "USD/JPY": "US-Dollar gegen japanischen Yen. Stark beeinflusst durch Zinsdifferenzen, Federal Reserve und Bank of Japan.",
    "USD/CHF": "US-Dollar gegen Schweizer Franken. CHF gilt häufig als sicherer Hafen.",
    "AUD/USD": "Australischer Dollar gegen US-Dollar. Sensibel gegenüber Rohstoffen, China-Risiko und US-Dollar-Stärke.",
    "NZD/USD": "Neuseeland-Dollar gegen US-Dollar. Häufig abhängig von Risikoappetit und Rohstoff-/Asien-Kontext.",
    "USD/CAD": "US-Dollar gegen kanadischen Dollar. CAD reagiert häufig auf Ölpreis, US-Daten und Bank of Canada.",
    "EUR/GBP": "Euro gegen britisches Pfund. Wichtig sind EZB, Bank of England, UK- und Eurozonen-Daten.",
    "EUR/JPY": "Euro gegen Yen. Cross-Pair mit Zins-, Risiko- und Carry-Komponente.",
    "GBP/JPY": "Pfund gegen Yen. Sehr volatiles Cross-Pair mit Risiko- und Zinsdifferenz-Komponente.",
}

PAIR_NEWS_CONTEXT = {
    "EUR/USD": ["EUR USD", "European Central Bank", "Federal Reserve", "Eurozone inflation", "US dollar"],
    "GBP/USD": ["GBP USD", "Bank of England", "Federal Reserve", "UK inflation", "US dollar"],
    "USD/JPY": ["USD JPY", "Bank of Japan", "Federal Reserve", "Japanese yen", "US Treasury yields"],
    "USD/CHF": ["USD CHF", "Swiss franc", "Swiss National Bank", "Federal Reserve", "safe haven"],
    "AUD/USD": ["AUD USD", "Reserve Bank of Australia", "China economy", "commodities", "US dollar"],
    "NZD/USD": ["NZD USD", "Reserve Bank of New Zealand", "risk sentiment", "US dollar"],
    "USD/CAD": ["USD CAD", "Bank of Canada", "oil prices", "Federal Reserve", "Canadian dollar"],
    "EUR/GBP": ["EUR GBP", "European Central Bank", "Bank of England", "UK economy", "Eurozone"],
    "EUR/JPY": ["EUR JPY", "European Central Bank", "Bank of Japan", "yen", "risk sentiment"],
    "GBP/JPY": ["GBP JPY", "Bank of England", "Bank of Japan", "yen", "risk sentiment"],
}

LEARN_LINKS = {
    "Federal Reserve": "https://www.federalreserve.gov/",
    "European Central Bank": "https://www.ecb.europa.eu/",
    "Bank of England": "https://www.bankofengland.co.uk/",
    "Bank of Japan": "https://www.boj.or.jp/en/",
    "Swiss National Bank": "https://www.snb.ch/en/",
    "Bank of Canada": "https://www.bankofcanada.ca/",
    "Reserve Bank of Australia": "https://www.rba.gov.au/",
    "Reserve Bank of New Zealand": "https://www.rbnz.govt.nz/",
    "US inflation": "https://www.bls.gov/cpi/",
    "Eurozone inflation": "https://ec.europa.eu/eurostat/",
    "UK inflation": "https://www.ons.gov.uk/economy/inflationandpriceindices",
    "US Treasury yields": "https://home.treasury.gov/",
    "oil prices": "https://www.eia.gov/",
    "risk sentiment": "https://www.investopedia.com/terms/r/risksentiment.asp",
    "Forex basics": "https://www.investopedia.com/terms/f/foreign-exchange.asp",
    "Risk management": "https://www.investopedia.com/terms/r/riskmanagement.asp",
}


@dataclass
class TimeframeConfig:
    label: str
    period: str
    interval: str
    fast_ma: int
    slow_ma: int
    rsi_period: int
    atr_period: int
    min_rows_for_ml: int
    horizon: int


TIMEFRAME_CONFIGS = {
    "M1": TimeframeConfig("M1", "5d", "1m", 9, 21, 7, 7, 300, 10),
    "M5": TimeframeConfig("M5", "1mo", "5m", 9, 21, 14, 14, 300, 8),
    "M15": TimeframeConfig("M15", "60d", "15m", 20, 50, 14, 14, 250, 6),
    "M30": TimeframeConfig("M30", "60d", "30m", 20, 50, 14, 14, 220, 5),
    "H1": TimeframeConfig("H1", "60d", "1h", 20, 50, 14, 14, 180, 5),
    "H4": TimeframeConfig("H4", "6mo", "1h", 50, 100, 14, 14, 250, 8),
    "D1": TimeframeConfig("D1", "2y", "1d", 20, 50, 14, 14, 180, 5),
    "W1": TimeframeConfig("W1", "10y", "1wk", 20, 50, 14, 14, 180, 4),
}


# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_pair(pair: str) -> str:
    pair = pair.strip().upper().replace(" ", "")
    if "/" in pair:
        base, quote = pair.split("/", 1)
        return f"{base}{quote}=X"
    if pair.endswith("=X"):
        return pair
    if len(pair) == 6:
        return f"{pair}=X"
    return pair


def format_pair_label(pair: str) -> str:
    clean = pair.upper().replace("=X", "").replace("/", "")
    if len(clean) == 6:
        return f"{clean[:3]}/{clean[3:]}"
    return pair.upper()


def pip_factor(pair_label: str) -> int:
    return 100 if "JPY" in pair_label.upper() else 10000


def pips_between(pair_label: str, price_a: float, price_b: float) -> float:
    return abs(price_a - price_b) * pip_factor(pair_label)


def explain_pair(pair_label: str) -> str:
    if "/" not in pair_label:
        return "Dieses Währungspaar zeigt das Verhältnis zweier Währungen."
    base, quote = pair_label.split("/")
    return f"{pair_label} bedeutet: Wie viele {quote} man für 1 {base} bekommt."


def risk_level(risk_percent: float, rr, pip_risk: float) -> dict:
    score = 0
    reasons = []

    if risk_percent <= 1:
        score += 2
        reasons.append("Das Risiko pro Trade ist konservativ.")
    elif risk_percent <= 2:
        score += 1
        reasons.append("Das Risiko pro Trade ist moderat.")
    else:
        score -= 2
        reasons.append("Das Risiko pro Trade ist für Anfänger hoch.")

    if rr is not None:
        if rr >= 2:
            score += 2
            reasons.append("Das Chance/Risiko-Verhältnis ist rechnerisch stark.")
        elif rr >= 1.5:
            score += 1
            reasons.append("Das Chance/Risiko-Verhältnis ist brauchbar.")
        elif rr >= 1:
            score -= 1
            reasons.append("Das Chance/Risiko-Verhältnis ist nur mittelmäßig.")
        else:
            score -= 2
            reasons.append("Das Chance/Risiko-Verhältnis ist schwach.")

    if pip_risk < 3:
        score -= 2
        reasons.append("Der Stop ist sehr eng. Spread und Schwankungen können problematisch sein.")
    elif pip_risk < 8:
        score -= 1
        reasons.append("Der Stop ist relativ eng.")
    else:
        score += 1
        reasons.append("Der Stop-Abstand ist nicht extrem eng.")

    if score >= 3:
        return {"label": "Kontrolliert", "card": "good-card", "color": "Grün", "reasons": reasons}
    if score <= -2:
        return {"label": "Gefährlich", "card": "bad-card", "color": "Rot", "reasons": reasons}
    return {"label": "Vorsichtig prüfen", "card": "warn-card", "color": "Gelb", "reasons": reasons}


def beginner_interpretation(ts: dict, latest_atr_pips: float, news_count = None) -> list[str]:
    output = []

    if ts["label"] == "Bullish":
        output.append("Die technische Lage wirkt eher positiv. Das heißt aber nicht, dass der Kurs sicher steigt.")
    elif ts["label"] == "Bearish":
        output.append("Die technische Lage wirkt eher negativ. Das heißt aber nicht, dass der Kurs sicher fällt.")
    else:
        output.append("Die technische Lage ist gemischt. Für Anfänger ist das ein Zeichen, besonders vorsichtig zu sein.")

    if latest_atr_pips >= 25:
        output.append("Die aktuelle Schwankungsbreite wirkt hoch. Ein zu enger Stop kann schnell ausgelöst werden.")
    elif latest_atr_pips >= 10:
        output.append("Die Schwankungsbreite ist moderat. Risiko und Stop-Abstand sollten bewusst gewählt werden.")
    else:
        output.append("Die Schwankungsbreite wirkt eher niedrig. Trotzdem bleibt Forex riskant.")

    if news_count is not None:
        if news_count > 0:
            output.append("Es gibt aktuelle Nachrichten zum Währungspaar. Diese sollten vor einer Entscheidung gelesen werden.")
        else:
            output.append("Es wurden keine starken aktuellen News gefunden. Das heißt nicht, dass kein Risiko besteht.")

    output.append("Für neues Kapital oder geerbtes Geld ist Paper-Trading sinnvoller als direkt echtes Geld einzusetzen.")
    return output


# =========================================================
# INDICATORS / DATA
# =========================================================

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(close: pd.Series):
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def calculate_atr(data: pd.DataFrame, period: int) -> pd.Series:
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


@st.cache_data(show_spinner="Forex-Marktdaten werden geladen...")
def load_fx_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else "Datetime"
    data = data.rename(columns={date_col: "Date"})

    required = ["Date", "Open", "High", "Low", "Close"]
    existing = [col for col in required if col in data.columns]
    data = data[existing]

    if "Volume" not in data.columns:
        data["Volume"] = 0

    return data.dropna()


def prepare_features(data: pd.DataFrame, cfg: TimeframeConfig) -> pd.DataFrame:
    df = data.copy()

    df["return_1"] = df["Close"].pct_change()
    df["return_3"] = df["Close"].pct_change(3)
    df["return_5"] = df["Close"].pct_change(5)
    df["return_10"] = df["Close"].pct_change(10)

    df["ma_fast"] = df["Close"].rolling(cfg.fast_ma).mean()
    df["ma_slow"] = df["Close"].rolling(cfg.slow_ma).mean()
    df["ema_fast"] = df["Close"].ewm(span=cfg.fast_ma, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=cfg.slow_ma, adjust=False).mean()

    if len(df) >= 220:
        df["ma_200"] = df["Close"].rolling(200).mean()
    else:
        df["ma_200"] = np.nan

    df["ma_distance_fast"] = (df["Close"] - df["ma_fast"]) / df["ma_fast"]
    df["ma_distance_slow"] = (df["Close"] - df["ma_slow"]) / df["ma_slow"]

    df["rsi"] = calculate_rsi(df["Close"], cfg.rsi_period)

    macd, signal, hist = calculate_macd(df["Close"])
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    df["atr"] = calculate_atr(df, cfg.atr_period)
    df["volatility_10"] = df["return_1"].rolling(10).std()
    df["volatility_20"] = df["return_1"].rolling(20).std()

    df["future_return"] = df["Close"].shift(-cfg.horizon) / df["Close"] - 1
    df["target"] = (df["future_return"] > 0).astype(int)

    df = df.replace([np.inf, -np.inf], np.nan)

    needed = [
        "return_1",
        "return_3",
        "return_5",
        "ma_fast",
        "ma_slow",
        "ma_distance_fast",
        "ma_distance_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "volatility_10",
        "volatility_20",
    ]

    return df.dropna(subset=needed)


def create_candlestick(data: pd.DataFrame, pair_label: str, cfg: TimeframeConfig):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=data["Date"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Kurs",
        )
    )

    for col, label in [
        ("ma_fast", f"Durchschnitt {cfg.fast_ma}"),
        ("ma_slow", f"Durchschnitt {cfg.slow_ma}"),
        ("ma_200", "Durchschnitt 200"),
    ]:
        if col in data.columns and data[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(x=data["Date"], y=data[col], mode="lines", name=label))

    fig.update_layout(
        title=f"{pair_label} | {cfg.label} Kursverlauf",
        xaxis_title="Zeit",
        yaxis_title="Wechselkurs",
        height=620,
        xaxis_rangeslider_visible=False,
    )
    return fig


def detect_support_resistance(feature_data: pd.DataFrame, lookback: int = 120) -> dict:
    if feature_data.empty:
        return {}

    recent = feature_data.tail(min(lookback, len(feature_data))).copy()
    last_close = float(recent["Close"].iloc[-1])

    windows = [20, 50, min(120, len(feature_data))]
    highs = [float(feature_data["High"].tail(w).max()) for w in windows if len(feature_data) >= max(5, w)]
    lows = [float(feature_data["Low"].tail(w).min()) for w in windows if len(feature_data) >= max(5, w)]

    if not highs or not lows:
        highs = [float(feature_data["High"].max())]
        lows = [float(feature_data["Low"].min())]

    resistance_above = [x for x in highs if x >= last_close]
    support_below = [x for x in lows if x <= last_close]

    nearest_resistance = min(resistance_above, key=lambda x: abs(x - last_close)) if resistance_above else max(highs)
    nearest_support = min(support_below, key=lambda x: abs(x - last_close)) if support_below else min(lows)

    return {
        "last_close": last_close,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "range_high": max(highs),
        "range_low": min(lows),
    }


def trend_score(latest: pd.Series) -> dict:
    score = 0
    reasons = []

    close = latest.get("Close", np.nan)
    fast = latest.get("ma_fast", np.nan)
    slow = latest.get("ma_slow", np.nan)
    ma200 = latest.get("ma_200", np.nan)
    rsi = latest.get("rsi", np.nan)
    macd_hist = latest.get("macd_hist", np.nan)

    if pd.notna(close) and pd.notna(fast):
        if close > fast:
            score += 1
            reasons.append("Der Kurs liegt über dem kurzfristigen Durchschnitt.")
        else:
            score -= 1
            reasons.append("Der Kurs liegt unter dem kurzfristigen Durchschnitt.")

    if pd.notna(close) and pd.notna(slow):
        if close > slow:
            score += 1
            reasons.append("Der Kurs liegt über dem mittelfristigen Durchschnitt.")
        else:
            score -= 1
            reasons.append("Der Kurs liegt unter dem mittelfristigen Durchschnitt.")

    if pd.notna(ma200):
        if close > ma200:
            score += 1
            reasons.append("Der Kurs liegt über dem langfristigen Durchschnitt.")
        else:
            score -= 1
            reasons.append("Der Kurs liegt unter dem langfristigen Durchschnitt.")

    if pd.notna(macd_hist):
        if macd_hist > 0:
            score += 1
            reasons.append("Das Momentum wirkt eher positiv.")
        else:
            score -= 1
            reasons.append("Das Momentum wirkt eher negativ.")

    if pd.notna(rsi):
        if rsi > 70:
            score -= 1
            reasons.append("Der Markt wirkt kurzfristig überhitzt.")
        elif rsi < 30:
            score += 1
            reasons.append("Der Markt wirkt kurzfristig stark gefallen.")
        else:
            reasons.append("Der RSI liegt im neutralen Bereich.")

    if score >= 3:
        label = "Eher positiv"
        card = "good-card"
    elif score <= -3:
        label = "Eher negativ"
        card = "bad-card"
    else:
        label = "Gemischt / unklar"
        card = "neutral-card"

    return {"score": score, "label": label, "reasons": reasons, "card": card}


def train_direction_model(feature_df: pd.DataFrame, cfg: TimeframeConfig):
    features = [
        "return_1",
        "return_3",
        "return_5",
        "return_10",
        "ma_distance_fast",
        "ma_distance_slow",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "volatility_10",
        "volatility_20",
    ]

    model_df = feature_df.dropna(subset=features + ["target"]).copy()

    if len(model_df) < cfg.min_rows_for_ml:
        return None, None, None

    X = model_df[features]
    y = model_df["target"]

    if y.nunique() < 2:
        return None, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        shuffle=False,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=9,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "report": classification_report(y_test, preds, output_dict=True, zero_division=0),
        "test_size": len(X_test),
        "train_size": len(X_train),
        "rows_available": len(model_df),
    }

    latest_input = model_df[features].tail(1)
    return model, metrics, latest_input


def get_period_performance(data: pd.DataFrame) -> float:
    if data.empty or len(data) < 2:
        return np.nan
    return float(data["Close"].iloc[-1] / data["Close"].iloc[0] - 1)


def risk_reward(entry, stop, target):
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk == 0:
        return None
    return reward / risk


def break_even_winrate(rr: float) -> float:
    if rr is None or rr <= 0:
        return np.nan
    return 1 / (1 + rr)


def calc_position_units(account_size: float, risk_percent: float, entry: float, stop: float, pair_label: str) -> dict:
    risk_amount = account_size * (risk_percent / 100)
    pip_risk = pips_between(pair_label, entry, stop)

    if pip_risk <= 0:
        return {
            "risk_amount": risk_amount,
            "pip_risk": pip_risk,
            "units": np.nan,
            "micro_lots": np.nan,
            "mini_lots": np.nan,
            "standard_lots": np.nan,
        }

    pip_value_per_standard_lot = 10.0
    standard_lots = risk_amount / (pip_risk * pip_value_per_standard_lot)
    units = standard_lots * 100000

    return {
        "risk_amount": risk_amount,
        "pip_risk": pip_risk,
        "units": units,
        "micro_lots": units / 1000,
        "mini_lots": units / 10000,
        "standard_lots": standard_lots,
    }


def simulate_equity(account_size: float, risk_amount: float, rr: float, winrate: float, trades: int) -> pd.DataFrame:
    equity = account_size
    rows = []
    expected_r = (winrate * rr) - (1 - winrate)

    for i in range(1, trades + 1):
        expected_profit = risk_amount * expected_r
        equity += expected_profit
        rows.append(
            {
                "Versuch": i,
                "Erwarteter Kontostand": equity,
                "Erwarteter Gewinn kumuliert": equity - account_size,
            }
        )

    return pd.DataFrame(rows)


def automation_precheck(account_size, risk_percent, rr, stop, entry, target, pair_label):
    warnings = []
    passed = []

    if risk_percent <= 1:
        passed.append("Das Risiko pro Versuch ist konservativ.")
    elif risk_percent <= 2:
        warnings.append("Das Risiko pro Versuch ist moderat. Für Anfänger besser niedriger halten.")
    else:
        warnings.append("Das Risiko pro Versuch ist hoch. Für neues Kapital nicht empfehlenswert.")

    if rr is not None and rr >= 1.5:
        passed.append("Das Chance/Risiko-Verhältnis ist mindestens 1,5.")
    else:
        warnings.append("Das Chance/Risiko-Verhältnis ist schwach oder nicht berechenbar.")

    if stop == entry:
        warnings.append("Stop Loss darf nicht identisch mit Entry sein.")
    else:
        passed.append("Stop Loss ist definiert.")

    if target == entry:
        warnings.append("Take Profit darf nicht identisch mit Entry sein.")
    else:
        passed.append("Take Profit ist definiert.")

    pip_risk = pips_between(pair_label, entry, stop)
    if pip_risk < 3:
        warnings.append("Der Stop-Abstand ist sehr klein. Spread/Slippage können stark stören.")
    else:
        passed.append("Der Stop-Abstand ist nicht extrem klein.")

    return passed, warnings


# =========================================================
# NEWS INTELLIGENCE
# =========================================================

def get_news_api_key():
    try:
        return st.secrets["NEWS_API_KEY"]
    except Exception:
        return None


def build_news_query(pair_label: str) -> str:
    keywords = PAIR_NEWS_CONTEXT.get(pair_label, [pair_label.replace("/", " ")])
    return " OR ".join([f'"{kw}"' for kw in keywords[:4]])


@st.cache_data(ttl=900, show_spinner="Aktuelle Nachrichten werden geladen...")
def fetch_market_news(query: str, api_key: str, language: str = "en", page_size: int = 8):
    if not api_key:
        return {"status": "missing_key", "articles": [], "message": "NEWS_API_KEY fehlt."}

    url = "https://newsapi.org/v2/everything"
    today = dt.date.today()
    from_date = today - dt.timedelta(days=7)

    params = {
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "from": from_date.isoformat(),
        "apiKey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=12)
        if response.status_code != 200:
            return {
                "status": "error",
                "articles": [],
                "message": f"NewsAPI Fehler {response.status_code}: {response.text[:250]}",
            }

        data = response.json()
        return {
            "status": "ok",
            "articles": data.get("articles", []),
            "message": f"{data.get('totalResults', 0)} Treffer gefunden.",
        }

    except Exception as exc:
        return {"status": "error", "articles": [], "message": str(exc)}


def news_context_summary(pair_label: str):
    base, quote = pair_label.split("/") if "/" in pair_label else (pair_label[:3], pair_label[3:])

    drivers = {
        "USD": ["Federal Reserve", "US inflation", "US jobs data", "US Treasury yields", "risk sentiment"],
        "EUR": ["European Central Bank", "Eurozone inflation", "PMI data", "energy prices", "Eurozone growth"],
        "GBP": ["Bank of England", "UK inflation", "UK GDP", "wage growth", "UK risk"],
        "JPY": ["Bank of Japan", "Japanese yields", "intervention risk", "safe haven flows", "carry trades"],
        "CHF": ["Swiss National Bank", "safe haven demand", "Swiss inflation", "risk sentiment"],
        "AUD": ["Reserve Bank of Australia", "China economy", "commodities", "risk sentiment"],
        "NZD": ["Reserve Bank of New Zealand", "risk sentiment", "commodities", "China demand"],
        "CAD": ["Bank of Canada", "oil prices", "US economy", "Canadian inflation"],
    }

    return {
        "base": base,
        "quote": quote,
        "base_drivers": drivers.get(base, []),
        "quote_drivers": drivers.get(quote, []),
    }


def driver_link(driver: str) -> str:
    return LEARN_LINKS.get(driver, "https://www.investopedia.com/")


def render_driver_links(drivers: list[str]):
    for item in drivers:
        url = driver_link(item)
        st.markdown(f"- [{item}]({url})")


def render_news_articles(articles):
    if not articles:
        st.info("Keine aktuellen News gefunden oder API-Key fehlt.")
        return

    for article in articles:
        title = article.get("title") or "Ohne Titel"
        source = (article.get("source") or {}).get("name", "Unbekannte Quelle")
        published_at = article.get("publishedAt", "")
        description = article.get("description") or "Keine Kurzbeschreibung verfügbar."
        url = article.get("url", "")

        st.markdown(
            f"""
            <div class="news-card">
            <h4>{title}</h4>
            <p><b>Quelle:</b> {source} &nbsp; | &nbsp; <b>Zeit:</b> {published_at}</p>
            <p>{description}</p>
            <p><a href="{url}" target="_blank">Mehr dazu lesen</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )




# =========================================================
# FRESHNESS + CONFIDENCE ENGINE
# =========================================================

def safe_to_datetime(value):
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def market_data_freshness(data, interval_label):
    """
    Bewertet, wie aktuell die letzte Marktdaten-Kerze ist.
    Diese Logik ist bewusst einfach und erklärbar.
    """

    if data is None or data.empty or "Date" not in data.columns:
        return {
            "status": "Keine Daten",
            "card": "bad-card",
            "minutes_old": None,
            "last_timestamp": None,
            "score": 0,
            "explanation": "Es konnten keine Marktdaten geladen werden."
        }

    last_ts = safe_to_datetime(data["Date"].iloc[-1])

    if last_ts is None:
        return {
            "status": "Unbekannt",
            "card": "warn-card",
            "minutes_old": None,
            "last_timestamp": None,
            "score": 30,
            "explanation": "Der Zeitpunkt der letzten Kerze konnte nicht sauber gelesen werden."
        }

    now = dt.datetime.now(last_ts.tzinfo) if getattr(last_ts, "tzinfo", None) else dt.datetime.now()
    diff_minutes = max((now - last_ts).total_seconds() / 60, 0)

    interval = str(interval_label).upper()

    # Erwartete Frische je Timeframe
    if interval in ["M1", "M5", "M15", "M30", "H1"]:
        fresh_limit = 90
        stale_limit = 360
    elif interval in ["H4", "D1"]:
        fresh_limit = 24 * 60
        stale_limit = 3 * 24 * 60
    else:
        fresh_limit = 7 * 24 * 60
        stale_limit = 21 * 24 * 60

    # Forex ist am Wochenende oft geschlossen.
    weekday = now.weekday()
    is_weekend = weekday >= 5

    if diff_minutes <= fresh_limit:
        status = "Aktuell"
        card = "good-card"
        score = 100
        explanation = "Die letzte Marktdaten-Kerze wirkt für diesen Timeframe frisch."
    elif diff_minutes <= stale_limit:
        status = "Leicht verzögert"
        card = "warn-card"
        score = 65
        explanation = "Die Daten sind nutzbar, aber nicht ganz frisch. Für kurzfristige Entscheidungen vorsichtig interpretieren."
    else:
        status = "Veraltet"
        card = "bad-card"
        score = 25
        explanation = "Die Daten sind für aktuelle Szenarien zu alt. Analyse neu laden oder andere Datenquelle prüfen."

    if is_weekend:
        explanation += " Hinweis: Forex-Märkte sind am Wochenende typischerweise geschlossen, daher können Daten älter wirken."

    return {
        "status": status,
        "card": card,
        "minutes_old": diff_minutes,
        "last_timestamp": last_ts,
        "score": score,
        "explanation": explanation
    }


def parse_news_time(article):
    raw = article.get("publishedAt")
    if not raw:
        return None
    try:
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


def news_freshness(articles):
    """
    Bewertet Aktualität der News.
    """

    if not articles:
        return {
            "status": "Keine News",
            "card": "warn-card",
            "hours_old": None,
            "latest_timestamp": None,
            "score": 35,
            "explanation": "Es wurden keine aktuellen News geladen. Der Outlook stützt sich stärker auf historische Daten."
        }

    timestamps = [parse_news_time(a) for a in articles]
    timestamps = [t for t in timestamps if t is not None]

    if not timestamps:
        return {
            "status": "Unbekannt",
            "card": "warn-card",
            "hours_old": None,
            "latest_timestamp": None,
            "score": 40,
            "explanation": "Die News-Zeitpunkte konnten nicht sauber gelesen werden."
        }

    latest = max(timestamps)
    now = dt.datetime.now(latest.tzinfo) if getattr(latest, "tzinfo", None) else dt.datetime.now()
    hours_old = max((now - latest).total_seconds() / 3600, 0)

    if hours_old <= 6:
        return {
            "status": "Sehr aktuell",
            "card": "good-card",
            "hours_old": hours_old,
            "latest_timestamp": latest,
            "score": 100,
            "explanation": "Die News-Lage ist sehr aktuell."
        }

    if hours_old <= 24:
        return {
            "status": "Aktuell",
            "card": "good-card",
            "hours_old": hours_old,
            "latest_timestamp": latest,
            "score": 80,
            "explanation": "Die News-Lage ist aktuell genug für Kontextanalyse."
        }

    if hours_old <= 72:
        return {
            "status": "Mittel",
            "card": "warn-card",
            "hours_old": hours_old,
            "latest_timestamp": latest,
            "score": 55,
            "explanation": "Die News sind nicht mehr ganz frisch. Für kurzfristige Forex-Szenarien vorsichtig interpretieren."
        }

    return {
        "status": "Alt",
        "card": "bad-card",
        "hours_old": hours_old,
        "latest_timestamp": latest,
        "score": 25,
        "explanation": "Die News sind für einen aktuellen Outlook zu alt."
    }


def compute_outlook_confidence(technical_score, news_score, atr_pips, market_fresh, news_fresh, ml_accuracy=None, articles_count=0):
    """
    Confidence Score != Trefferwahrscheinlichkeit.
    Er bewertet, wie belastbar die aktuelle Einschätzung wirkt.
    """

    confidence = 0
    components = []

    # 1. Technische Eindeutigkeit
    tech_strength = min(abs(technical_score) / 6, 1)
    tech_points = tech_strength * 25
    confidence += tech_points
    components.append({
        "Faktor": "Technische Eindeutigkeit",
        "Punkte": round(tech_points, 1),
        "Erklärung": "Je klarer Chart/Indikatoren in eine Richtung zeigen, desto höher dieser Anteil."
    })

    # 2. News-Eindeutigkeit
    news_strength = min(abs(news_score) / 6, 1)
    news_points = news_strength * 20
    confidence += news_points
    components.append({
        "Faktor": "News-Eindeutigkeit",
        "Punkte": round(news_points, 1),
        "Erklärung": "Je eindeutiger die News-Lage zum Währungspaar wirkt, desto höher dieser Anteil."
    })

    # 3. Marktdaten-Aktualität
    market_points = (market_fresh.get("score", 0) / 100) * 20
    confidence += market_points
    components.append({
        "Faktor": "Marktdaten-Aktualität",
        "Punkte": round(market_points, 1),
        "Erklärung": f"Status: {market_fresh.get('status', 'unbekannt')}."
    })

    # 4. News-Aktualität
    news_fresh_points = (news_fresh.get("score", 0) / 100) * 15
    confidence += news_fresh_points
    components.append({
        "Faktor": "News-Aktualität",
        "Punkte": round(news_fresh_points, 1),
        "Erklärung": f"Status: {news_fresh.get('status', 'unbekannt')}."
    })

    # 5. ML Accuracy als historische Zusatzinfo
    if ml_accuracy is not None:
        # Alles unter 50 % bringt fast nichts; 60 % ist schon brauchbar für ein einfaches Modell.
        normalized_ml = max(min((ml_accuracy - 0.50) / 0.20, 1), 0)
        ml_points = normalized_ml * 10
    else:
        ml_points = 0

    confidence += ml_points
    components.append({
        "Faktor": "Historische ML-Accuracy",
        "Punkte": round(ml_points, 1),
        "Erklärung": "Historische Testleistung des Modells. Keine Garantie für die Zukunft."
    })

    # 6. Volatilitätsabschlag
    if atr_pips >= 35:
        vol_penalty = -15
        vol_text = "Hohe Volatilität senkt die Belastbarkeit deutlich."
    elif atr_pips >= 20:
        vol_penalty = -8
        vol_text = "Erhöhte Volatilität senkt die Belastbarkeit."
    else:
        vol_penalty = 0
        vol_text = "Volatilität erzeugt keinen starken Abschlag."

    confidence += vol_penalty
    components.append({
        "Faktor": "Volatilitätsabschlag",
        "Punkte": round(vol_penalty, 1),
        "Erklärung": vol_text
    })

    # 7. News-Menge
    if articles_count >= 5:
        article_points = 5
    elif articles_count >= 2:
        article_points = 2.5
    else:
        article_points = 0

    confidence += article_points
    components.append({
        "Faktor": "News-Abdeckung",
        "Punkte": round(article_points, 1),
        "Erklärung": f"{articles_count} Artikel wurden berücksichtigt."
    })

    confidence = max(0, min(100, confidence))

    if confidence >= 75:
        label = "Hoch"
        card = "good-card"
    elif confidence >= 50:
        label = "Mittel"
        card = "neutral-card"
    elif confidence >= 30:
        label = "Niedrig bis mittel"
        card = "warn-card"
    else:
        label = "Niedrig"
        card = "bad-card"

    return {
        "score": round(confidence, 0),
        "label": label,
        "card": card,
        "components": components,
        "explanation": "Der Confidence Score bewertet die Qualität der aktuellen Einschätzung. Er ist keine garantierte Trefferwahrscheinlichkeit."
    }


def format_age_minutes(minutes):
    if minutes is None:
        return "unbekannt"
    if minutes < 60:
        return f"{minutes:.0f} Minuten"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} Stunden"
    days = hours / 24
    return f"{days:.1f} Tage"


def format_age_hours(hours):
    if hours is None:
        return "unbekannt"
    if hours < 24:
        return f"{hours:.1f} Stunden"
    days = hours / 24
    return f"{days:.1f} Tage"


def render_freshness_cards(market_fresh, news_fresh):
    c1, c2, c3 = st.columns(3)

    with c1:
        last_ts = market_fresh.get("last_timestamp")
        last_txt = last_ts.strftime("%d.%m.%Y %H:%M") if last_ts else "unbekannt"
        st.markdown(
            f"""
            <div class="info-card {market_fresh.get('card', 'neutral-card')}">
            <h3>Marktdaten</h3>
            <div class="big-number">{market_fresh.get('status', 'Unbekannt')}</div>
            <p>Letzte Kerze: <b>{last_txt}</b><br>
            Alter: <b>{format_age_minutes(market_fresh.get('minutes_old'))}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        latest_news = news_fresh.get("latest_timestamp")
        news_txt = latest_news.strftime("%d.%m.%Y %H:%M") if latest_news else "unbekannt"
        st.markdown(
            f"""
            <div class="info-card {news_fresh.get('card', 'neutral-card')}">
            <h3>News</h3>
            <div class="big-number">{news_fresh.get('status', 'Unbekannt')}</div>
            <p>Neueste News: <b>{news_txt}</b><br>
            Alter: <b>{format_age_hours(news_fresh.get('hours_old'))}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        now_txt = dt.datetime.now().strftime("%d.%m.%Y %H:%M")
        st.markdown(
            f"""
            <div class="info-card neutral-card">
            <h3>Outlook berechnet</h3>
            <div class="big-number">Jetzt</div>
            <p>Zeitpunkt: <b>{now_txt}</b><br>
            Empfehlung: bei News oder starken Bewegungen neu laden.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# MARKET OUTLOOK ENGINE
# =========================================================

def score_news_for_pair(pair_label, articles):
    """
    Einfache regelbasierte News-Auswertung.
    Ziel: Kontextbewertung, keine sichere Marktprognose.
    """

    if not articles:
        return {
            "score": 0,
            "label": "Keine starke News-Lage erkannt",
            "drivers": [],
            "confidence": "Niedrig",
            "explanation": "Es wurden keine ausreichenden Nachrichten gefunden. Die App stützt sich daher stärker auf historische Daten."
        }

    pair = pair_label.upper()
    text_blob = " ".join([
        str(a.get("title", "")) + " " + str(a.get("description", ""))
        for a in articles
    ]).lower()

    score = 0
    drivers = []

    usd_positive = [
        "fed hawkish", "federal reserve hawkish", "rate hike", "higher rates",
        "us inflation rises", "inflation hotter", "dollar strengthens",
        "strong dollar", "treasury yields rise", "yields rise", "strong jobs"
    ]

    usd_negative = [
        "fed dovish", "rate cut", "lower rates", "dollar weakens",
        "weak dollar", "us inflation cools", "soft inflation",
        "treasury yields fall", "yields fall", "weak jobs"
    ]

    eur_positive = [
        "ecb hawkish", "european central bank hawkish", "euro strengthens",
        "eurozone inflation rises", "euro gains"
    ]

    eur_negative = [
        "ecb dovish", "euro weakens", "eurozone recession",
        "eurozone inflation cools", "euro falls"
    ]

    gbp_positive = [
        "bank of england hawkish", "boe hawkish", "pound strengthens",
        "sterling gains", "uk inflation rises"
    ]

    gbp_negative = [
        "bank of england dovish", "boe dovish", "pound weakens",
        "sterling falls", "uk recession"
    ]

    jpy_positive = [
        "bank of japan hawkish", "boj hawkish", "yen strengthens",
        "yen gains", "japan intervention"
    ]

    jpy_negative = [
        "bank of japan dovish", "boj dovish", "yen weakens",
        "yen falls", "carry trade"
    ]

    def contains_any(words):
        return [w for w in words if w in text_blob]

    usd_pos_hits = contains_any(usd_positive)
    usd_neg_hits = contains_any(usd_negative)
    eur_pos_hits = contains_any(eur_positive)
    eur_neg_hits = contains_any(eur_negative)
    gbp_pos_hits = contains_any(gbp_positive)
    gbp_neg_hits = contains_any(gbp_negative)
    jpy_pos_hits = contains_any(jpy_positive)
    jpy_neg_hits = contains_any(jpy_negative)

    base = pair.split("/")[0] if "/" in pair else pair[:3]
    quote = pair.split("/")[1] if "/" in pair else pair[3:6]

    def apply_currency_impact(currency, positive_hits, negative_hits):
        nonlocal score, drivers

        # Wenn Basiswährung stärker wird: Paar eher hoch
        # Wenn Kurswährung stärker wird: Paar eher runter
        if currency == base:
            if positive_hits:
                score += len(positive_hits)
                drivers += [f"{currency} stärkende News: {x}" for x in positive_hits[:3]]
            if negative_hits:
                score -= len(negative_hits)
                drivers += [f"{currency} schwächende News: {x}" for x in negative_hits[:3]]

        if currency == quote:
            if positive_hits:
                score -= len(positive_hits)
                drivers += [f"{currency} stärkende News drücken tendenziell {pair_label}: {x}" for x in positive_hits[:3]]
            if negative_hits:
                score += len(negative_hits)
                drivers += [f"{currency} schwächende News stützen tendenziell {pair_label}: {x}" for x in negative_hits[:3]]

    apply_currency_impact("USD", usd_pos_hits, usd_neg_hits)
    apply_currency_impact("EUR", eur_pos_hits, eur_neg_hits)
    apply_currency_impact("GBP", gbp_pos_hits, gbp_neg_hits)
    apply_currency_impact("JPY", jpy_pos_hits, jpy_neg_hits)

    if score >= 3:
        label = "News-Kontext eher positiv"
    elif score <= -3:
        label = "News-Kontext eher negativ"
    elif score > 0:
        label = "News-Kontext leicht positiv"
    elif score < 0:
        label = "News-Kontext leicht negativ"
    else:
        label = "News-Kontext neutral oder uneindeutig"

    if abs(score) >= 4:
        confidence = "Mittel"
    elif abs(score) >= 2:
        confidence = "Niedrig bis mittel"
    else:
        confidence = "Niedrig"

    return {
        "score": score,
        "label": label,
        "drivers": drivers[:6],
        "confidence": confidence,
        "explanation": "Die News werden regelbasiert ausgewertet. Das ist ein Kontextsignal, keine sichere Prognose."
    }


def build_market_outlook(technical_score, news_score, atr_pips):
    """
    Kombiniert technische Lage, News-Kontext und Volatilität.
    Ergebnis ist ein Szenario-Ausblick, keine Handelsentscheidung.
    """

    combined = technical_score + news_score

    volatility_penalty = 0
    volatility_note = "Normale bis moderate Schwankung."

    if atr_pips >= 35:
        volatility_penalty = -2
        volatility_note = "Die Schwankung ist hoch. Das erhöht das Risiko deutlich."
    elif atr_pips >= 20:
        volatility_penalty = -1
        volatility_note = "Die Schwankung ist erhöht. Stop Loss und Positionsgröße müssen vorsichtig gewählt werden."

    adjusted = combined + volatility_penalty

    if adjusted >= 5:
        outlook = "Vorsichtig positiv"
        card = "good-card"
        action = "Nur beobachten oder im Paper-Trading simulieren. Kein Blind-Trade."
    elif adjusted <= -5:
        outlook = "Vorsichtig negativ"
        card = "bad-card"
        action = "Kein Blind-Trade. Erst News und Risiko prüfen. Für Anfänger eher beobachten."
    elif adjusted >= 2:
        outlook = "Leicht positiv, aber unsicher"
        card = "neutral-card"
        action = "Interessantes Szenario, aber nur mit Risikoplan und Simulation."
    elif adjusted <= -2:
        outlook = "Leicht negativ, aber unsicher"
        card = "warn-card"
        action = "Vorsicht. Setup wirkt nicht stabil genug für Anfänger."
    else:
        outlook = "Unklar / kein Vorteil erkennbar"
        card = "warn-card"
        action = "Für Anfänger: kein Setup erzwingen. Besser beobachten oder lernen."

    if abs(adjusted) >= 6:
        confidence = "Mittel"
    elif abs(adjusted) >= 3:
        confidence = "Niedrig bis mittel"
    else:
        confidence = "Niedrig"

    return {
        "combined_raw": combined,
        "adjusted_score": adjusted,
        "outlook": outlook,
        "card": card,
        "confidence": confidence,
        "volatility_note": volatility_note,
        "action": action
    }



# =========================================================
# PREMIUM UX + DECISION PASSPORT ENGINE
# =========================================================

def capital_protection_score(risk_percent, rr, confidence_score, market_status, news_status, atr_pips):
    """
    Bewertet, ob ein Szenario für eine nicht-technische Person mit neuem Kapital
    eher lerngeeignet, beobachtbar oder zu riskant ist.
    Keine Trading-Empfehlung.
    """

    score = 100
    reasons = []

    if risk_percent <= 0.5:
        reasons.append("Sehr konservatives Risiko pro Versuch.")
    elif risk_percent <= 1:
        score -= 5
        reasons.append("Konservatives Risiko pro Versuch.")
    elif risk_percent <= 2:
        score -= 20
        reasons.append("Risiko pro Versuch ist moderat und für Anfänger bereits sensibel.")
    else:
        score -= 40
        reasons.append("Risiko pro Versuch ist hoch für neues Kapital.")

    if rr is None:
        score -= 25
        reasons.append("Chance/Risiko-Verhältnis ist nicht berechenbar.")
    elif rr >= 2:
        reasons.append("Chance/Risiko-Verhältnis ist stark.")
    elif rr >= 1.5:
        score -= 5
        reasons.append("Chance/Risiko-Verhältnis ist akzeptabel.")
    elif rr >= 1:
        score -= 20
        reasons.append("Chance/Risiko-Verhältnis ist nur mittelmäßig.")
    else:
        score -= 35
        reasons.append("Chance/Risiko-Verhältnis ist schwach.")

    if confidence_score >= 70:
        reasons.append("Confidence Score ist solide.")
    elif confidence_score >= 50:
        score -= 10
        reasons.append("Confidence Score ist mittel.")
    elif confidence_score >= 30:
        score -= 25
        reasons.append("Confidence Score ist niedrig bis mittel.")
    else:
        score -= 40
        reasons.append("Confidence Score ist niedrig.")

    if market_status == "Veraltet":
        score -= 25
        reasons.append("Marktdaten sind veraltet.")
    elif market_status == "Leicht verzögert":
        score -= 10
        reasons.append("Marktdaten sind leicht verzögert.")

    if news_status in ["Alt", "Keine News"]:
        score -= 15
        reasons.append("News-Kontext ist schwach oder veraltet.")
    elif news_status == "Mittel":
        score -= 8
        reasons.append("News-Kontext ist nur mittelaktuell.")

    if atr_pips >= 35:
        score -= 25
        reasons.append("Volatilität ist hoch.")
    elif atr_pips >= 20:
        score -= 12
        reasons.append("Volatilität ist erhöht.")

    score = max(0, min(100, round(score)))

    if score >= 80:
        label = "Lern- und simulationsgeeignet"
        card = "good-card"
        action = "Geeignet zum Verstehen und Paper-Trading. Kein automatischer Echtgeld-Trade."
    elif score >= 60:
        label = "Beobachten und simulieren"
        card = "neutral-card"
        action = "Nicht erzwingen. Erst weiter beobachten, News lesen und im Demo-Modus testen."
    elif score >= 40:
        label = "Nur Lernen, kein Trade-Plan"
        card = "warn-card"
        action = "Für Anfänger zu unsicher. Fokus auf Lernen, nicht auf Umsetzung."
    else:
        label = "Nicht geeignet"
        card = "bad-card"
        action = "Szenario ist zu unsicher oder riskant. Kein Echtgeld-Einsatz."

    return {
        "score": score,
        "label": label,
        "card": card,
        "action": action,
        "reasons": reasons
    }


def build_decision_passport(pair_label, selected_tf, outlook, confidence, market_fresh, news_fresh, ts, news_eval, atr_pips, rr=None, risk_percent=None):
    """
    Kompakter Entscheidungs-Pass für nicht-technische Nutzer.
    """

    lines = []
    lines.append(f"# ForexScope AI Decision Passport")
    lines.append("")
    lines.append(f"**Währungspaar:** {pair_label}")
    lines.append(f"**Timeframe:** {selected_tf}")
    lines.append(f"**Zeitpunkt:** {dt.datetime.now().strftime('%d.%m.%Y %H:%M')}")
    lines.append("")
    lines.append("## 1. Marktausblick")
    lines.append(f"- Szenario: {outlook.get('outlook', 'unbekannt')}")
    lines.append(f"- Outlook Score: {outlook.get('adjusted_score', 'unbekannt')}")
    lines.append(f"- Confidence Score: {confidence.get('score', 'unbekannt')} / 100")
    lines.append(f"- Confidence Einordnung: {confidence.get('label', 'unbekannt')}")
    lines.append("")
    lines.append("## 2. Aktualität")
    lines.append(f"- Marktdatenstatus: {market_fresh.get('status', 'unbekannt')}")
    lines.append(f"- Newsstatus: {news_fresh.get('status', 'unbekannt')}")
    lines.append("")
    lines.append("## 3. Historische / technische Lage")
    lines.append(f"- Technisches Bild: {ts.get('label', 'unbekannt')}")
    lines.append(f"- Technischer Score: {ts.get('score', 'unbekannt')}")
    lines.append(f"- ATR / Schwankung: {atr_pips:.1f} Pips")
    lines.append("")
    lines.append("## 4. News-Kontext")
    lines.append(f"- News Score: {news_eval.get('score', 'unbekannt')}")
    lines.append(f"- News Einordnung: {news_eval.get('label', 'unbekannt')}")
    if news_eval.get("drivers"):
        lines.append("- Erkannte News-Treiber:")
        for d in news_eval.get("drivers", [])[:6]:
            lines.append(f"  - {d}")
    else:
        lines.append("- Keine eindeutigen News-Treiber erkannt.")
    lines.append("")
    lines.append("## 5. Risiko")
    if rr is not None:
        lines.append(f"- Chance/Risiko-Verhältnis: {rr:.2f}")
    if risk_percent is not None:
        lines.append(f"- Risiko pro Versuch: {risk_percent:.2f}%")
    lines.append("")
    lines.append("## 6. Hinweis")
    lines.append("Diese Auswertung ist keine Finanzberatung, keine Handelsaufforderung und keine Garantie. Sie dient Analyse, Lernen und Simulation.")
    return "\n".join(lines)


def render_executive_outlook_summary(outlook, confidence, protection=None):
    """
    Obere Premium-Zusammenfassung.
    """

    protection_html = ""
    if protection:
        protection_html = f"""
        <p><b>Kapital-Schutz-Status:</b> {protection['label']} ({protection['score']} / 100)</p>
        """

    st.markdown(
        f"""
        <div class="hero-card">
        <h2>Executive Summary</h2>
        <p><b>Market Outlook:</b> {outlook.get('outlook', 'unbekannt')}</p>
        <p><b>Confidence:</b> {confidence.get('score', 'unbekannt')} / 100 – {confidence.get('label', 'unbekannt')}</p>
        {protection_html}
        <p><b>Nächster sicherer Schritt:</b> {outlook.get('action', 'Erst prüfen und simulieren.')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_beginner_next_steps(outlook, confidence, protection=None):
    st.subheader("Nächster sicherer Schritt")

    if protection and protection["score"] < 40:
        st.error("Nicht handeln. Erst verstehen, News lesen und Risiko reduzieren.")
        st.write("- Wechsel in den Kapital-Kompass.")
        st.write("- Reduziere Risiko pro Versuch.")
        st.write("- Nutze Paper-Trading statt Echtgeld.")
        return

    if confidence["score"] < 30:
        st.warning("Die Einschätzung ist zu unsicher.")
        st.write("- Keine Entscheidung aus diesem Outlook ableiten.")
        st.write("- Aktualität prüfen.")
        st.write("- News lesen.")
        st.write("- Später erneut analysieren.")
    elif confidence["score"] < 60:
        st.warning("Die Einschätzung ist nur begrenzt belastbar.")
        st.write("- Nur beobachten oder im Paper-Trading simulieren.")
        st.write("- Keine hohe Positionsgröße.")
        st.write("- Erst weitere Bestätigung abwarten.")
    else:
        st.info("Die Einschätzung ist strukturierter, aber trotzdem keine Garantie.")
        st.write("- Nur mit kleinem Risiko simulieren.")
        st.write("- Stop Loss und Take Profit klar definieren.")
        st.write("- Ergebnis dokumentieren.")


def render_mode_explanation(user_mode):
    if user_mode == "Einfach":
        st.caption("Einfache Ansicht: Fokus auf Klartext, Ampel, Kapital-Schutz und nächste sichere Schritte.")
    else:
        st.caption("Fortgeschrittene Ansicht: Zusätzlich technische Score-Zerlegung, ML-Werte, News-Regeln und Detailtabellen.")




# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("ForexScope AI")
st.sidebar.caption("Kapital schützen. Märkte verstehen.")

user_mode = st.sidebar.radio("Ansicht", ["Einfach", "Fortgeschritten"], horizontal=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Start",
        "Kapital-Kompass",
        "Forex einfach erklärt",
        "Marktanalyse",
        "News & Marktkontext",
        "Daten + News Gesamtbild",
        "Market Outlook",
        "Risiko-Simulator",
        "Expert Dashboard",
        "Automation Readiness",
        "Methodik & Grenzen",
    ],
)

st.sidebar.divider()

pair_mode = st.sidebar.radio("Währungspaar wählen", ["Liste", "Manuell"], horizontal=True)

if pair_mode == "Liste":
    pair_label = st.sidebar.selectbox("Forex-Paar", list(FOREX_PAIRS.keys()), index=1)
    yf_ticker = FOREX_PAIRS[pair_label]
else:
    manual_pair = st.sidebar.text_input("Forex-Paar", value="GBP/USD", help="Beispiele: EUR/USD, GBP/USD, USD/JPY oder EURUSD=X")
    yf_ticker = normalize_pair(manual_pair)
    pair_label = format_pair_label(manual_pair)

selected_tf = st.sidebar.selectbox("Zeitebene", list(TIMEFRAME_CONFIGS.keys()), index=4)
cfg = TIMEFRAME_CONFIGS[selected_tf]

if user_mode == "Fortgeschritten":
    st.sidebar.write(f"Daten: `{cfg.period}` / `{cfg.interval}`")
    st.sidebar.write(f"ML-Horizont: `{cfg.horizon}` Kerzen")

st.sidebar.warning("Keine Finanzberatung. Nur Analyse, Lernen und Simulation.")


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="main-title">ForexScope AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Historische Daten, aktuelle News und Risikoaufklärung für Menschen, die neues Kapital verantwortungsvoll verstehen möchten.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATA
# =========================================================

market_data = load_fx_data(yf_ticker, cfg.period, cfg.interval)

if market_data.empty and page not in ["Start", "Forex einfach erklärt", "Methodik & Grenzen", "News & Marktkontext"]:
    st.error(f"Keine Forex-Daten gefunden für {yf_ticker}. Prüfe Paar, Zeitraum oder Intervall.")
    st.stop()

if not market_data.empty:
    feature_data = prepare_features(market_data, cfg)
else:
    feature_data = pd.DataFrame()


# =========================================================
# START
# =========================================================

if page == "Start":
    st.markdown(
        """
        <div class="hero-card">
        <h2>Kapital schützen. Märkte verstehen. Entscheidungen simulieren.</h2>
        <p>
        ForexScope AI hilft Menschen mit neuem Kapital – zum Beispiel durch Erbe, Abfindung oder Rücklagen –
        Forex-Märkte verständlich zu analysieren. Die App kombiniert historische Marktdaten, aktuelle Nachrichten
        und Risikokennzahlen, bevor echtes Geld eingesetzt wird.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Kapital schützen</h3>
            <p>Berechne zuerst, wie viel Verlust überhaupt tragbar wäre.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Märkte verstehen</h3>
            <p>Sieh historische Kursdaten, technische Lage und aktuelle News zusammen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Nur simulieren</h3>
            <p>Die App ist bewusst keine Live-Trading-Automation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Empfohlener Ablauf")

    st.write("1. Starte mit dem **Kapital-Kompass**.")
    st.write("2. Lies **Forex einfach erklärt**, wenn Begriffe unklar sind.")
    st.write("3. Prüfe die **Marktanalyse** und den **News-Kontext**.")
    st.write("4. Nutze den **Risiko-Simulator**, bevor du überhaupt an echtes Trading denkst.")

    st.warning("Besonders bei geerbtem oder neuem Kapital gilt: Kapitalerhalt ist wichtiger als schnelle Gewinne.")


# =========================================================
# CAPITAL COMPASS
# =========================================================

elif page == "Kapital-Kompass":
    st.header("Kapital-Kompass")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Wie viel Risiko wäre überhaupt vertretbar?</h2>
        <p>
        Bevor ein Markt analysiert wird, sollte klar sein, wie viel Verlust psychologisch und finanziell tragbar wäre.
        Dieser Bereich übersetzt Kapital in konkrete Risikogrenzen.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        capital_origin = st.selectbox(
            "Woher kommt das Kapital?",
            ["Erbe", "Abfindung", "Ersparnisse", "Bonus", "Sonstiges"],
        )
        account_size = st.number_input("Kapitalbetrag", min_value=100.0, value=10000.0, step=500.0)
        experience = st.selectbox("Erfahrung mit Forex", ["Keine Erfahrung", "Ein wenig", "Fortgeschritten"])
        goal = st.selectbox(
            "Ziel",
            ["Nur verstehen", "Paper-Trading simulieren", "Risiko einschätzen", "Echtes Trading vorbereiten"],
        )
        risk_percent = st.slider("Maximaler Verlust pro Versuch (%)", 0.1, 5.0, 1.0, 0.1)

    with col2:
        max_loss = account_size * (risk_percent / 100)

        if risk_percent <= 1 and goal != "Echtes Trading vorbereiten":
            compass_card = "good-card"
            compass_label = "Schutzorientiert"
            compass_text = "Dein Risikoansatz ist eher konservativ. Für Anfänger ist das sinnvoll."
        elif risk_percent <= 2:
            compass_card = "warn-card"
            compass_label = "Vorsichtig prüfen"
            compass_text = "Das Risiko ist nicht extrem, aber für neues Kapital sollte man sehr vorsichtig bleiben."
        else:
            compass_card = "bad-card"
            compass_label = "Zu riskant"
            compass_text = "Für Anfänger oder neues Kapital ist dieses Risiko pro Versuch hoch."

        st.markdown(
            f"""
            <div class="info-card {compass_card}">
            <h3>{compass_label}</h3>
            <p>{compass_text}</p>
            <div class="big-number">{max_loss:,.2f}</div>
            <p>Maximaler Verlust pro Versuch bei {risk_percent:.1f}% Risiko.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("**Was bedeutet das?**")
        st.write(f"- Bei {account_size:,.2f} Kapital wären {max_loss:,.2f} der Betrag, der pro Versuch maximal verloren gehen dürfte.")
        st.write("- Diese Grenze sollte vor jeder Analyse feststehen.")
        st.write("- Wenn dich dieser Verlust emotional stark belasten würde, ist das Risiko zu hoch.")

    st.divider()

    st.subheader("Persönliche Empfehlung")

    if experience == "Keine Erfahrung":
        st.info("Für dich wäre Paper-Trading der sinnvollste nächste Schritt. Kein echtes Geld, bevor du über längere Zeit simuliert hast.")
    elif experience == "Ein wenig":
        st.info("Nutze die App zuerst zur Analyse und Simulation. Echte Trades sollten erst nach dokumentierten Tests folgen.")
    else:
        st.info("Auch mit Erfahrung bleibt Risikobegrenzung zentral. Die App ersetzt keine Strategieprüfung.")

    st.warning("Diese App soll Kapital schützen helfen, nicht zu schnellen Spekulationen verleiten.")


# =========================================================
# FOREX BASICS
# =========================================================

elif page == "Forex einfach erklärt":
    st.header("Forex einfach erklärt")

    st.markdown(
        f"""
        <div class="info-card">
        <h3>{pair_label}</h3>
        <p>{explain_pair(pair_label)}</p>
        <p>{PAIR_INFO.get(pair_label, "Dieses Währungspaar wird von wirtschaftlichen und geldpolitischen Faktoren beeinflusst.")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Die wichtigsten Begriffe")

    basics = pd.DataFrame(
        [
            {"Begriff": "Forex", "Einfache Erklärung": "Handel mit Währungen, z. B. Euro gegen US-Dollar."},
            {"Begriff": "Währungspaar", "Einfache Erklärung": "Zwei Währungen im Vergleich, z. B. EUR/USD."},
            {"Begriff": "Pip", "Einfache Erklärung": "Kleine Messeinheit für Kursbewegungen."},
            {"Begriff": "Stop Loss", "Einfache Erklärung": "Eine Grenze, bei der ein Verlust begrenzt werden soll."},
            {"Begriff": "Take Profit", "Einfache Erklärung": "Ein Ziel, bei dem Gewinn mitgenommen werden soll."},
            {"Begriff": "CRV", "Einfache Erklärung": "Vergleich zwischen möglichem Gewinn und möglichem Verlust."},
            {"Begriff": "ATR", "Einfache Erklärung": "Zeigt, wie stark der Markt durchschnittlich schwankt."},
            {"Begriff": "RSI", "Einfache Erklärung": "Zeigt, ob ein Markt kurzfristig überhitzt oder stark gefallen wirkt."},
            {"Begriff": "MACD", "Einfache Erklärung": "Hilft, Momentum und Trendrichtung einzuschätzen."},
        ]
    )

    st.dataframe(basics, use_container_width=True)

    st.subheader("Mehr lesen")

    st.markdown(f"- [Forex-Grundlagen]({LEARN_LINKS['Forex basics']})")
    st.markdown(f"- [Risikomanagement]({LEARN_LINKS['Risk management']})")
    st.markdown("- [Investopedia: Pip](https://www.investopedia.com/terms/p/pip.asp)")
    st.markdown("- [Investopedia: Stop-Loss Order](https://www.investopedia.com/terms/s/stop-lossorder.asp)")

    st.warning("Wenn diese Begriffe noch nicht sitzen, sollte kein echtes Geld riskiert werden.")


# =========================================================
# MARKET ANALYSIS BEGINNER
# =========================================================

elif page == "Marktanalyse":
    st.header("Marktanalyse")

    if feature_data.empty:
        st.error("Nicht genug Daten für eine sinnvolle Analyse. Wähle eine andere Zeitebene.")
        st.stop()

    latest_close = float(market_data["Close"].iloc[-1])
    first_close = float(market_data["Close"].iloc[0])
    performance = latest_close / first_close - 1

    sr = detect_support_resistance(feature_data)
    latest = feature_data.iloc[-1]
    ts = trend_score(latest)
    atr_pips = pips_between(pair_label, 0, latest["atr"])

    st.markdown(
        f"""
        <div class="hero-card">
        <h2>{pair_label}: verständliche Marktübersicht</h2>
        <p>{explain_pair(pair_label)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    market_fresh = market_data_freshness(market_data, selected_tf)
    last_ts = market_fresh.get("last_timestamp")
    last_txt = last_ts.strftime("%d.%m.%Y %H:%M") if last_ts else "unbekannt"

    st.markdown(
        f"""
        <div class="info-card {market_fresh.get('card', 'neutral-card')}">
        <h3>Daten-Aktualität: {market_fresh.get('status', 'Unbekannt')}</h3>
        <p>Letzte Marktdaten-Kerze: <b>{last_txt}</b> | Alter: <b>{format_age_minutes(market_fresh.get('minutes_old'))}</b></p>
        <p>{market_fresh.get('explanation', '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Aktueller Kurs", f"{latest_close:.5f}")
    k2.metric("Bewegung im Zeitraum", f"{performance:.2%}")
    k3.metric("Schwankung", f"{atr_pips:.1f} Pips")
    k4.metric("Einschätzung", ts["label"])

    st.markdown(
        f"""
        <div class="info-card {ts['card']}">
        <h3>Was sagt die technische Lage?</h3>
        <p><b>{ts['label']}</b> – Score: {ts['score']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("In einfachen Worten")

    for line in beginner_interpretation(ts, atr_pips):
        st.write(f"- {line}")

    st.subheader("Kursverlauf")

    fig = create_candlestick(feature_data, pair_label, cfg)
    fig.add_hline(y=sr["nearest_support"], line_dash="dash", annotation_text="Möglicher Unterstützungsbereich")
    fig.add_hline(y=sr["nearest_resistance"], line_dash="dash", annotation_text="Möglicher Widerstandsbereich")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Warum kommt die App zu dieser Einschätzung?"):
        for reason in ts["reasons"]:
            st.write(f"- {reason}")

        st.write("Support bedeutet: ein Bereich, in dem der Kurs in der Vergangenheit eher Halt gefunden hat.")
        st.write("Resistance bedeutet: ein Bereich, in dem der Kurs in der Vergangenheit eher gebremst wurde.")


# =========================================================
# NEWS CONTEXT
# =========================================================

elif page == "News & Marktkontext":
    st.header("News & Marktkontext")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Was bewegt den Markt gerade?</h2>
        <p>
        Historische Daten zeigen, was passiert ist. Nachrichten helfen zu verstehen, warum sich Erwartungen ändern können.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    context = news_context_summary(pair_label)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Wichtige Themen für {context['base']}")
        render_driver_links(context["base_drivers"])

    with c2:
        st.subheader(f"Wichtige Themen für {context['quote']}")
        render_driver_links(context["quote_drivers"])

    api_key = get_news_api_key()

    if not api_key:
        st.warning(
            "NEWS_API_KEY fehlt. Lege lokal `.streamlit/secrets.toml` mit NEWS_API_KEY an. "
            "Ohne Key kann die App keine Live-News abrufen."
        )

    default_query = build_news_query(pair_label)
    query = st.text_input("News-Suchbegriff", value=default_query)
    language = st.selectbox("Sprache", ["en", "de"], index=0)
    page_size = st.slider("Anzahl News", 3, 20, 8)

    if st.button("Aktuelle News laden", type="primary"):
        result = fetch_market_news(query, api_key, language=language, page_size=page_size)
        st.session_state["news_result"] = result
        st.session_state["news_query"] = query

    result = st.session_state.get("news_result")

    if result:
        if result["status"] == "ok":
            st.success(result["message"])
            render_news_articles(result["articles"])
        else:
            st.error(result["message"])
    else:
        st.info("Klicke auf „Aktuelle News laden“, um aktuelle Nachrichten mit Links abzurufen.")

    st.warning("News erklären Kontext, aber sie sind kein automatisches Handelssignal.")


# =========================================================
# DATA + NEWS FUSION
# =========================================================

elif page == "Daten + News Gesamtbild":
    st.header("Daten + News Gesamtbild")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Historische Marktdaten + aktuelle Nachrichten</h2>
        <p>
        Das ist der Kern von ForexScope AI: Die App verbindet messbare Kursdaten mit aktuellem Kontext.
        Dadurch entsteht ein verständlicheres Gesamtbild.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if feature_data.empty:
        st.error("Nicht genug historische Daten für die technische Analyse.")
        st.stop()

    latest = feature_data.iloc[-1]
    sr = detect_support_resistance(feature_data)
    ts = trend_score(latest)
    atr_pips = pips_between(pair_label, 0, latest["atr"])

    api_key = get_news_api_key()
    query = build_news_query(pair_label)
    news_result = fetch_market_news(query, api_key, language="en", page_size=5) if api_key else {"status": "missing_key", "articles": []}
    news_count = len(news_result.get("articles", [])) if news_result.get("status") == "ok" else 0

    col_hist, col_news = st.columns([1, 1])

    with col_hist:
        st.subheader("Historische Daten")
        st.markdown(
            f"""
            <div class="info-card {ts['card']}">
            <h3>Technisches Bild: {ts['label']}</h3>
            <p>
            Support: <b>{sr['nearest_support']:.5f}</b><br>
            Resistance: <b>{sr['nearest_resistance']:.5f}</b><br>
            Schwankung: <b>{atr_pips:.1f} Pips</b>
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for reason in ts["reasons"]:
            st.write(f"- {reason}")

    with col_news:
        st.subheader("Aktuelle News")

        if not api_key:
            st.warning("NEWS_API_KEY fehlt. News-Kontext kann ergänzt werden.")
        elif news_result["status"] == "ok":
            st.metric("Geladene News", news_count)
            for article in news_result.get("articles", [])[:3]:
                title = article.get("title", "Ohne Titel")
                source = (article.get("source") or {}).get("name", "Unbekannte Quelle")
                url = article.get("url", "")
                st.markdown(f"**[{title}]({url})**")
                st.caption(source)
                st.write(article.get("description") or "Keine Beschreibung verfügbar.")
                st.write("---")
        else:
            st.warning(news_result.get("message", "News konnten nicht geladen werden."))

    st.divider()

    st.subheader("Was bedeutet das für mich?")

    for line in beginner_interpretation(ts, atr_pips, news_count):
        st.write(f"- {line}")

    st.warning("Diese Gesamtansicht ist eine Lern- und Analysehilfe, keine Empfehlung zum Kauf oder Verkauf.")



# =========================================================
# MARKET OUTLOOK
# =========================================================

elif page == "Market Outlook":
    st.header("Market Outlook")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Szenario-Prognose aus historischen Daten, aktuellen News und Risiko-Kontext</h2>
        <p>
        Dieser Bereich ist das Herzstück von ForexScope AI. Er verbindet Chartdaten, technische Indikatoren,
        News-Kontext, Volatilität, historische Modellgüte und Daten-Aktualität zu einem verständlichen Marktausblick.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_mode_explanation(user_mode)

    if feature_data.empty:
        st.error("Nicht genug historische Daten für einen Market Outlook.")
        st.stop()

    latest = feature_data.iloc[-1]
    sr = detect_support_resistance(feature_data)
    ts = trend_score(latest)
    atr_pips = pips_between(pair_label, 0, latest["atr"])

    api_key = get_news_api_key()
    query = build_news_query(pair_label)

    if api_key:
        news_result = fetch_market_news(query, api_key, language="en", page_size=8)
        articles = news_result.get("articles", []) if news_result.get("status") == "ok" else []
    else:
        news_result = {"status": "missing_key", "articles": [], "message": "NEWS_API_KEY fehlt."}
        articles = []

    news_eval = score_news_for_pair(pair_label, articles)
    outlook = build_market_outlook(ts["score"], news_eval["score"], atr_pips)

    market_fresh = market_data_freshness(market_data, selected_tf)
    news_fresh = news_freshness(articles)

    ml_accuracy = None
    ml_model, ml_metrics, latest_input = train_direction_model(feature_data, cfg)
    if ml_metrics is not None:
        ml_accuracy = ml_metrics.get("accuracy")

    confidence = compute_outlook_confidence(
        technical_score=ts["score"],
        news_score=news_eval["score"],
        atr_pips=atr_pips,
        market_fresh=market_fresh,
        news_fresh=news_fresh,
        ml_accuracy=ml_accuracy,
        articles_count=len(articles),
    )

    # Mini-Risikoprofil für Executive Summary
    default_risk_percent = 1.0
    default_rr = 1.5
    protection = capital_protection_score(
        risk_percent=default_risk_percent,
        rr=default_rr,
        confidence_score=confidence["score"],
        market_status=market_fresh.get("status"),
        news_status=news_fresh.get("status"),
        atr_pips=atr_pips,
    )

    render_executive_outlook_summary(outlook, confidence, protection)

    st.subheader("1. Aktualität der Einschätzung")
    render_freshness_cards(market_fresh, news_fresh)

    if user_mode == "Fortgeschritten":
        st.caption(
            "Aktualitätslogik: Die App bewertet, wie alt die letzte Marktdaten-Kerze und die neuesten News sind. "
            "Je kürzer der Timeframe, desto wichtiger ist Datenfrische."
        )

    st.divider()

    st.subheader("2. Drei Bausteine des Outlooks")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="info-card {ts['card']}">
            <h3>Historische Daten</h3>
            <div class="big-number">{ts['label']}</div>
            <p>Technischer Score: <b>{ts['score']}</b></p>
            <p class="muted">Chart, Trend, Momentum und Schwankung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        news_card = "good-card" if news_eval["score"] > 0 else "bad-card" if news_eval["score"] < 0 else "neutral-card"
        st.markdown(
            f"""
            <div class="info-card {news_card}">
            <h3>News-Kontext</h3>
            <div class="big-number">{news_eval['score']}</div>
            <p>{news_eval['label']}</p>
            <p class="muted">Zentralbanken, Inflation, Dollar-Stärke, Risiko-Stimmung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        vol_card = "bad-card" if atr_pips >= 35 else "warn-card" if atr_pips >= 20 else "neutral-card"
        st.markdown(
            f"""
            <div class="info-card {vol_card}">
            <h3>Volatilität</h3>
            <div class="big-number">{atr_pips:.1f} Pips</div>
            <p>{outlook['volatility_note']}</p>
            <p class="muted">Hohe Schwankung senkt die Belastbarkeit.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("3. Ergebnis: Market Outlook")

    c_outlook, c_conf, c_protect = st.columns([1.15, 0.9, 0.9])

    with c_outlook:
        st.markdown(
            f"""
            <div class="info-card {outlook['card']}">
            <h2>{outlook['outlook']}</h2>
            <p><b>Outlook Score:</b> {outlook['adjusted_score']}</p>
            <p><b>Handlungslogik:</b> {outlook['action']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_conf:
        st.markdown(
            f"""
            <div class="info-card {confidence['card']}">
            <h2>Confidence</h2>
            <div class="big-number">{confidence['score']:.0f} / 100</div>
            <p><b>{confidence['label']}</b></p>
            <p class="muted">Belastbarkeit der aktuellen Einschätzung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_protect:
        st.markdown(
            f"""
            <div class="info-card {protection['card']}">
            <h2>Kapital-Schutz</h2>
            <div class="big-number">{protection['score']} / 100</div>
            <p><b>{protection['label']}</b></p>
            <p class="muted">Für Anfänger mit neuem Kapital.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(
        "Confidence ist keine Trefferwahrscheinlichkeit. Sie bewertet nur die aktuelle Belastbarkeit der Einschätzung. "
        "Historische ML-Accuracy ist ebenfalls keine Garantie für die Zukunft."
    )

    if ml_accuracy is not None:
        st.metric("Historische ML-Accuracy", f"{ml_accuracy:.1%}")
    elif user_mode == "Fortgeschritten":
        st.metric("Historische ML-Accuracy", "nicht verfügbar")

    if user_mode == "Fortgeschritten":
        st.subheader("Confidence-Zerlegung")

        conf_df = pd.DataFrame(confidence["components"])
        st.dataframe(conf_df, use_container_width=True)

        conf_fig = px.bar(
            conf_df,
            x="Faktor",
            y="Punkte",
            text="Punkte",
            title="Zusammensetzung des Confidence Scores",
        )
        conf_fig.update_traces(textposition="outside")
        st.plotly_chart(conf_fig, use_container_width=True)

    st.divider()

    st.subheader("4. Erklärung in Klartext")

    reason_col1, reason_col2 = st.columns(2)

    with reason_col1:
        st.write("**Was sagen die historischen Daten?**")
        for reason in ts["reasons"]:
            st.write(f"- {reason}")

        st.write(f"- Unterstützungsbereich ungefähr bei **{sr['nearest_support']:.5f}**.")
        st.write(f"- Widerstandsbereich ungefähr bei **{sr['nearest_resistance']:.5f}**.")
        st.write(f"- Aktuelle Schwankung ungefähr **{atr_pips:.1f} Pips**.")

        if user_mode == "Fortgeschritten":
            st.write("**Technische Rohwerte**")
            st.write(f"- RSI: **{latest.get('rsi', np.nan):.2f}**")
            st.write(f"- MACD: **{latest.get('macd', np.nan):.6f}**")
            st.write(f"- MACD Histogramm: **{latest.get('macd_hist', np.nan):.6f}**")
            st.write(f"- ATR: **{latest.get('atr', np.nan):.6f}**")

    with reason_col2:
        st.write("**Was sagen die aktuellen News?**")

        if not api_key:
            st.warning("Kein NEWS_API_KEY gefunden. News können aktuell nicht live bewertet werden.")
        elif news_eval["drivers"]:
            for driver in news_eval["drivers"]:
                st.write(f"- {driver}")
        else:
            st.write("- Die News-Lage ist aktuell neutral oder nicht eindeutig genug.")

        if user_mode == "Fortgeschritten":
            st.write("**News-Auswertung technisch**")
            st.write(f"- News Score: **{news_eval['score']}**")
            st.write(f"- News Confidence: **{news_eval['confidence']}**")
            st.write(f"- Anzahl Artikel: **{len(articles)}**")
            st.write(f"- Suchquery: `{query}`")

    st.divider()

    st.subheader("5. Was bedeutet das für mich?")

    for line in beginner_interpretation(ts, atr_pips, len(articles)):
        st.write(f"- {line}")

    render_beginner_next_steps(outlook, confidence, protection)

    st.divider()

    st.subheader("6. Aktuelle Quellen zum Nachlesen")

    if articles:
        visible_articles = articles[:4] if user_mode == "Einfach" else articles[:8]
        for article in visible_articles:
            title = article.get("title", "Ohne Titel")
            source = (article.get("source") or {}).get("name", "Unbekannte Quelle")
            url = article.get("url", "")
            published_at = article.get("publishedAt", "")
            description = article.get("description") or "Keine Kurzbeschreibung verfügbar."

            st.markdown(
                f"""
                <div class="news-card">
                <h4>{title}</h4>
                <p><b>Quelle:</b> {source} &nbsp; | &nbsp; <b>Zeit:</b> {published_at}</p>
                <p>{description}</p>
                <p><a href="{url}" target="_blank">Mehr dazu lesen</a></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("Keine News-Artikel verfügbar. Prüfe NEWS_API_KEY oder lade News im Bereich „News & Marktkontext“.")

    st.divider()

    st.subheader("7. Decision Passport exportieren")

    report = build_decision_passport(
        pair_label=pair_label,
        selected_tf=selected_tf,
        outlook=outlook,
        confidence=confidence,
        market_fresh=market_fresh,
        news_fresh=news_fresh,
        ts=ts,
        news_eval=news_eval,
        atr_pips=atr_pips,
        rr=None,
        risk_percent=None,
    )

    st.download_button(
        label="Decision Passport als Markdown herunterladen",
        data=report,
        file_name=f"forexscope_decision_passport_{pair_label.replace('/', '')}_{selected_tf}.md",
        mime="text/markdown",
    )

    st.error(
        "Keine Finanzberatung. Keine Handelsaufforderung. Keine Garantie. Bei neuem Kapital oder Erbe sollte zuerst gelernt und simuliert werden."
    )





# =========================================================
# RISK SIMULATOR
# =========================================================

elif page == "Risiko-Simulator":
    st.header("Risiko-Simulator")

    if market_data.empty:
        st.error("Keine Marktdaten vorhanden.")
        st.stop()

    latest_close = float(market_data["Close"].iloc[-1])

    if not feature_data.empty:
        sr = detect_support_resistance(feature_data)
        default_stop_long = sr["nearest_support"]
        default_target_long = sr["nearest_resistance"]
    else:
        default_stop_long = latest_close * 0.997
        default_target_long = latest_close * 1.006

    st.markdown(
        """
        <div class="hero-card">
        <h2>Wie viel könnte ich verlieren?</h2>
        <p>
        Dieser Bereich stellt Risiko vor Gewinn. Gerade bei neuem Kapital ist das entscheidend.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        account_size = st.number_input("Kapitalbetrag", min_value=50.0, value=10000.0, step=500.0)
        risk_percent = st.slider("Maximaler Verlust pro Versuch (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        trade_direction = st.radio("Richtung", ["Long", "Short"], horizontal=True)
        entry = st.number_input("Einstiegskurs", value=round(latest_close, 5), format="%.5f")

        if trade_direction == "Long":
            stop = st.number_input("Stop Loss", value=round(min(default_stop_long, entry * 0.997), 5), format="%.5f")
            target = st.number_input("Take Profit", value=round(max(default_target_long, entry * 1.006), 5), format="%.5f")
        else:
            stop = st.number_input("Stop Loss", value=round(max(default_target_long, entry * 1.003), 5), format="%.5f")
            target = st.number_input("Take Profit", value=round(min(default_stop_long, entry * 0.994), 5), format="%.5f")

    with col2:
        rr = risk_reward(entry, stop, target)
        pos = calc_position_units(account_size, risk_percent, entry, stop, pair_label)
        pip_risk = pos["pip_risk"]
        pip_reward = pips_between(pair_label, entry, target)
        risk_amount = pos["risk_amount"]

        if rr is None:
            st.error("CRV kann nicht berechnet werden, weil Einstieg und Stop identisch sind.")
            st.stop()

        potential_profit = risk_amount * rr
        be_wr = break_even_winrate(rr)
        level = risk_level(risk_percent, rr, pip_risk)

        st.markdown(
            f"""
            <div class="info-card {level['card']}">
            <h3>Risiko-Ampel: {level['color']} – {level['label']}</h3>
            <p>Diese Bewertung prüft Risiko, Stop-Abstand und Chance/Risiko-Verhältnis.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        m1.metric("Max. Verlust", f"{risk_amount:,.2f}")
        m2.metric("Möglicher Gewinn", f"{potential_profit:,.2f}")

        m3, m4 = st.columns(2)
        m3.metric("Risiko in Pips", f"{pip_risk:.1f}")
        m4.metric("Chance in Pips", f"{pip_reward:.1f}")

        m5, m6 = st.columns(2)
        m5.metric("CRV", f"{rr:.2f}")
        m6.metric("Benötigte Trefferquote", f"{be_wr:.1%}")

        st.write("**Einfache Interpretation:**")
        for reason in level["reasons"]:
            st.write(f"- {reason}")

    st.divider()

    with st.expander("Optional: Was-wäre-wenn-Simulation"):
        st.write("Die Trefferquote ist unbekannt. Diese Simulation ist nur ein Lernwerkzeug.")

        planned_trades = st.slider("Anzahl Versuche", min_value=1, max_value=100, value=20)
        assumed_winrate = st.slider("Angenommene Trefferquote (%)", min_value=10, max_value=90, value=45) / 100

        equity_df = simulate_equity(
            account_size=account_size,
            risk_amount=risk_amount,
            rr=rr,
            winrate=assumed_winrate,
            trades=planned_trades,
        )

        fig = px.line(
            equity_df,
            x="Versuch",
            y="Erwarteter Kontostand",
            title="Was-wäre-wenn-Kontoverlauf",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        final_equity = float(equity_df["Erwarteter Kontostand"].iloc[-1])
        st.metric("Erwarteter Kontostand nach Simulation", f"{final_equity:,.2f}", delta=f"{final_equity - account_size:,.2f}")

        if assumed_winrate < be_wr:
            st.warning("Die angenommene Trefferquote liegt unter der Break-even-Trefferquote.")
        else:
            st.success("Die angenommene Trefferquote liegt über der Break-even-Trefferquote.")

    st.warning("Wenn du mit echtem Kapital emotional nicht ruhig bleiben könntest, ist das Risiko zu hoch.")


# =========================================================
# EXPERT DASHBOARD
# =========================================================

elif page == "Expert Dashboard":
    st.header("Expert Dashboard")

    if feature_data.empty:
        st.error("Nicht genug Daten für Expert Dashboard.")
        st.stop()

    latest_close = float(market_data["Close"].iloc[-1])
    first_close = float(market_data["Close"].iloc[0])
    performance = latest_close / first_close - 1
    latest = feature_data.iloc[-1]
    sr = detect_support_resistance(feature_data)
    ts = trend_score(latest)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Aktueller Kurs", f"{latest_close:.5f}")
    k2.metric("Performance", f"{performance:.2%}")
    k3.metric("Datenpunkte", f"{len(market_data):,}")
    k4.metric("ATR", f"{latest['atr']:.5f}")
    k5.metric("ATR Pips", f"{pips_between(pair_label, 0, latest['atr']):.1f}")

    st.markdown(
        f"""
        <div class="info-card {ts['card']}">
        <h3>Technisches Szenario: {ts['label']} | Score: {ts['score']}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = create_candlestick(feature_data, pair_label, cfg)
    fig.add_hline(y=sr["nearest_support"], line_dash="dash", annotation_text="Support")
    fig.add_hline(y=sr["nearest_resistance"], line_dash="dash", annotation_text="Resistance")
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        rsi_fig = px.line(feature_data, x="Date", y="rsi", title=f"RSI {cfg.rsi_period}")
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        st.plotly_chart(rsi_fig, use_container_width=True)

    with right:
        macd_fig = px.line(feature_data, x="Date", y=["macd", "macd_signal"], title="MACD")
        st.plotly_chart(macd_fig, use_container_width=True)

    st.subheader("Multi-Timeframe")

    rows = []
    for tf_label, tf_cfg in TIMEFRAME_CONFIGS.items():
        data = load_fx_data(yf_ticker, tf_cfg.period, tf_cfg.interval)
        if data.empty:
            continue
        prepared = prepare_features(data, tf_cfg)
        perf = get_period_performance(data)

        if prepared.empty:
            rows.append(
                {
                    "Timeframe": tf_label,
                    "Daten": len(data),
                    "Performance": perf,
                    "Trend": "zu wenig Daten",
                    "Score": np.nan,
                    "Letzter Kurs": float(data["Close"].iloc[-1]),
                    "RSI": np.nan,
                    "ATR Pips": np.nan,
                }
            )
            continue

        latest_tf = prepared.iloc[-1]
        ts_tf = trend_score(latest_tf)
        rows.append(
            {
                "Timeframe": tf_label,
                "Daten": len(data),
                "Performance": perf,
                "Trend": ts_tf["label"],
                "Score": ts_tf["score"],
                "Letzter Kurs": float(prepared["Close"].iloc[-1]),
                "RSI": float(latest_tf["rsi"]),
                "ATR Pips": pips_between(pair_label, 0, float(latest_tf["atr"])),
            }
        )

    summary = pd.DataFrame(rows)
    st.dataframe(summary, use_container_width=True)

    fig_score = px.bar(summary, x="Timeframe", y="Score", color="Trend", text="Score", title="Trend-Score je Timeframe")
    fig_score.update_traces(textposition="outside")
    st.plotly_chart(fig_score, use_container_width=True)

    st.subheader("ML Szenario")

    ml_model, metrics, latest_input = train_direction_model(feature_data, cfg)

    if ml_model is None:
        st.warning(
            f"Für {selected_tf} sind nicht genug saubere Daten für ein sinnvolles ML-Modell vorhanden. "
            f"Rohdaten: {len(market_data):,}, Feature-Daten: {len(feature_data):,}."
        )
    else:
        proba = ml_model.predict_proba(latest_input)[0]
        classes = list(ml_model.classes_)

        probability_df = pd.DataFrame(
            {
                "Szenario": ["Eher niedriger" if c == 0 else "Eher höher" for c in classes],
                "Wahrscheinlichkeit": proba,
            }
        )

        fig_ml = px.bar(
            probability_df,
            x="Szenario",
            y="Wahrscheinlichkeit",
            text=probability_df["Wahrscheinlichkeit"].apply(lambda x: f"{x:.1%}"),
            title="ML-Szenario-Wahrscheinlichkeiten",
        )
        fig_ml.update_yaxes(tickformat=".0%")
        fig_ml.update_traces(textposition="outside")
        st.plotly_chart(fig_ml, use_container_width=True)

        st.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
        st.dataframe(pd.DataFrame(metrics["report"]).transpose(), use_container_width=True)

    st.warning("Expert Dashboard ist für Analyse gedacht, nicht für automatische Handelsentscheidungen.")


# =========================================================
# AUTOMATION READINESS
# =========================================================

elif page == "Automation Readiness":
    st.header("Automation Readiness")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Warum diese App noch kein Trading-Bot ist</h2>
        <p>
        Eine spätere Broker-Automation wäre nur mit Paper-Trading, Risk Engine, Logging, Kill Switch und manueller Freigabe vertretbar.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    architecture = pd.DataFrame(
        [
            {"Baustein": "Market Data", "Zweck": "Echte Bid/Ask-Preise, Spreads, aktuelle Handelbarkeit", "Status": "Später Broker-API nötig"},
            {"Baustein": "News Context", "Zweck": "Aktuelle Makro- und Zentralbanknachrichten", "Status": "NewsAPI-Prototyp"},
            {"Baustein": "Signal Engine", "Zweck": "Regeln/ML-Szenario, aber kein blindes Signal", "Status": "Teilweise vorhanden"},
            {"Baustein": "Risk Engine", "Zweck": "Max. Risiko, Positionsgröße, Tagesverlust, Exposure", "Status": "Im Planner vorbereitet"},
            {"Baustein": "Order Manager", "Zweck": "Market/Limit/Stop Orders, SL/TP, Orderstatus", "Status": "Noch nicht angebunden"},
            {"Baustein": "Paper Trading", "Zweck": "Test ohne echtes Geld", "Status": "Pflicht vor Live"},
            {"Baustein": "Audit Log", "Zweck": "Jede Entscheidung und Order nachvollziehbar speichern", "Status": "Noch offen"},
            {"Baustein": "Kill Switch", "Zweck": "Automatik sofort stoppen", "Status": "Pflicht vor Live"},
        ]
    )

    st.dataframe(architecture, use_container_width=True)

    st.warning("Kein Live-Trading ohne Paper-Trading, Limits, Kill Switch, Logging und manuelle Freigabe.")


# =========================================================
# METHODIK & GRENZEN
# =========================================================

elif page == "Methodik & Grenzen":
    st.header("Methodik & Grenzen")

    st.subheader("Warum diese App anders positioniert ist")

    st.write(
        "ForexScope AI richtet sich an Menschen, die Märkte zuerst verstehen wollen. "
        "Die App soll nicht zu schnellen Trades motivieren, sondern Risiko sichtbar machen."
    )

    st.subheader("Warum historische Daten + News?")

    st.write(
        "Historische Daten liefern messbare technische Signale. Aktuelle News liefern Kontext zu möglichen makroökonomischen Markttreibern. "
        "Die Kombination beider Perspektiven ist stärker als ein reiner Chart."
    )

    st.subheader("Was die App berechnet")

    st.markdown(
        """
        - OHLC-Daten je Timeframe
        - Moving Averages
        - RSI
        - MACD
        - ATR
        - Support/Resistance
        - Multi-Timeframe-Score
        - ML-Szenario, wenn genug Daten vorhanden sind
        - News-Kontext zu Zentralbanken, Inflation, Währungen und Makrotreibern
        - Kapital-, Risiko-, Lot- und CRV-Rechner
        """
    )

    st.subheader("Grenzen")

    st.markdown(
        """
        - Keine Finanzberatung
        - Keine Handelsaufforderung
        - Keine sichere Prognose
        - yfinance-Daten können verzögert, begrenzt oder unvollständig sein
        - NewsAPI liefert Kontext, aber keine sichere Marktrichtung
        - Lot-Berechnung ist vereinfacht
        - Für Broker-Automation braucht man echte Brokerdaten, Bid/Ask, Spread, Slippage, Order-Handling und Sicherheitslimits
        """
    )

    st.subheader("Weiterführende Links")

    st.markdown(f"- [Forex-Grundlagen]({LEARN_LINKS['Forex basics']})")
    st.markdown(f"- [Risikomanagement]({LEARN_LINKS['Risk management']})")
    st.markdown(f"- [Federal Reserve]({LEARN_LINKS['Federal Reserve']})")
    st.markdown(f"- [European Central Bank]({LEARN_LINKS['European Central Bank']})")
    st.markdown(f"- [Bank of England]({LEARN_LINKS['Bank of England']})")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("ForexScope AI | Historische Daten + aktuelle News + Risikoaufklärung | Keine Finanzberatung")
