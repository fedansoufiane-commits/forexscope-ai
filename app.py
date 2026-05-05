import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from dataclasses import dataclass
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
        font-size: 2.9rem;
        font-weight: 950;
        letter-spacing: -0.05em;
        margin-bottom: 0rem;
    }
    .subtitle {
        color: #8f8f8f;
        font-size: 1.08rem;
        margin-bottom: 1.4rem;
    }
    .hero-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 24px;
        padding: 1.5rem 1.7rem;
        background: linear-gradient(135deg, rgba(110,110,110,0.22), rgba(90,90,90,0.04));
        margin-bottom: 1.2rem;
    }
    .info-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        background: rgba(128,128,128,0.06);
        margin-bottom: 1rem;
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
    "EUR/USD": "Euro gegen US-Dollar. Sehr liquides Hauptwährungspaar.",
    "GBP/USD": "Britisches Pfund gegen US-Dollar. Oft volatiler als EUR/USD.",
    "USD/JPY": "US-Dollar gegen japanischen Yen. Stark durch Zinsdifferenzen und Risikoappetit beeinflusst.",
    "USD/CHF": "US-Dollar gegen Schweizer Franken. CHF gilt häufig als sicherer Hafen.",
    "AUD/USD": "Australischer Dollar gegen US-Dollar. Sensibel gegenüber Rohstoffen und China-Risiko.",
    "NZD/USD": "Neuseeland-Dollar gegen US-Dollar. Kleinere, oft volatile FX-Struktur.",
    "USD/CAD": "US-Dollar gegen kanadischen Dollar. CAD reagiert oft auf Ölpreis und US-Konjunktur.",
    "EUR/GBP": "Euro gegen britisches Pfund. Relevant für Eurozone/UK-Risiken.",
    "EUR/JPY": "Euro gegen japanischen Yen. Cross-Pair mit Risiko- und Zinskomponente.",
    "GBP/JPY": "Pfund gegen Yen. Sehr volatiles Cross-Pair.",
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
# HELPERS
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
            name="FX-Kurs",
        )
    )

    for col, label in [
        ("ma_fast", f"MA {cfg.fast_ma}"),
        ("ma_slow", f"MA {cfg.slow_ma}"),
        ("ma_200", "MA 200"),
    ]:
        if col in data.columns and data[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(x=data["Date"], y=data[col], mode="lines", name=label))

    fig.update_layout(
        title=f"{pair_label} | {cfg.label} Chart",
        xaxis_title="Zeit",
        yaxis_title="Wechselkurs",
        height=650,
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
            reasons.append("Kurs liegt über der schnellen MA.")
        else:
            score -= 1
            reasons.append("Kurs liegt unter der schnellen MA.")

    if pd.notna(close) and pd.notna(slow):
        if close > slow:
            score += 1
            reasons.append("Kurs liegt über der langsamen MA.")
        else:
            score -= 1
            reasons.append("Kurs liegt unter der langsamen MA.")

    if pd.notna(ma200):
        if close > ma200:
            score += 1
            reasons.append("Kurs liegt über MA 200.")
        else:
            score -= 1
            reasons.append("Kurs liegt unter MA 200.")

    if pd.notna(macd_hist):
        if macd_hist > 0:
            score += 1
            reasons.append("MACD-Histogramm positiv.")
        else:
            score -= 1
            reasons.append("MACD-Histogramm negativ.")

    if pd.notna(rsi):
        if rsi > 70:
            score -= 1
            reasons.append("RSI über 70: kurzfristig überkauft.")
        elif rsi < 30:
            score += 1
            reasons.append("RSI unter 30: kurzfristig überverkauft.")
        else:
            reasons.append("RSI neutral.")

    if score >= 3:
        label = "Bullish"
        card = "good-card"
    elif score <= -3:
        label = "Bearish"
        card = "bad-card"
    else:
        label = "Neutral / gemischt"
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
                "Trade": i,
                "Erwarteter Kontostand": equity,
                "Erwarteter Gewinn kumuliert": equity - account_size,
            }
        )

    return pd.DataFrame(rows)


def automation_precheck(account_size, risk_percent, rr, stop, entry, target, pair_label):
    warnings = []
    passed = []

    if risk_percent <= 1:
        passed.append("Risiko pro Trade ist konservativ.")
    elif risk_percent <= 2:
        warnings.append("Risiko pro Trade ist moderat. Für Automatisierung besser konservativer prüfen.")
    else:
        warnings.append("Risiko pro Trade ist hoch. Für Automation nicht empfehlenswert.")

    if rr is not None and rr >= 1.5:
        passed.append("CRV ist mindestens 1.5.")
    else:
        warnings.append("CRV ist schwach oder nicht berechenbar.")

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
        warnings.append("Stop-Abstand ist sehr klein. Spread/Slippage können den Trade stark verzerren.")
    else:
        passed.append("Stop-Abstand ist nicht extrem klein.")

    return passed, warnings


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("ForexScope AI")
st.sidebar.caption("Intraday Forex Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Start",
        "Live Forex Analyse",
        "Multi-Timeframe Dashboard",
        "ML Szenario",
        "Capital & Risk Planner",
        "Automation Readiness",
        "Methodik & Grenzen",
    ],
)

st.sidebar.divider()

pair_mode = st.sidebar.radio("Paar wählen", ["Liste", "Manuell"], horizontal=True)

if pair_mode == "Liste":
    pair_label = st.sidebar.selectbox("Forex-Paar", list(FOREX_PAIRS.keys()), index=1)
    yf_ticker = FOREX_PAIRS[pair_label]
else:
    manual_pair = st.sidebar.text_input("Forex-Paar", value="GBP/USD", help="Beispiele: EUR/USD, GBP/USD, USD/JPY oder EURUSD=X")
    yf_ticker = normalize_pair(manual_pair)
    pair_label = format_pair_label(manual_pair)

selected_tf = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_CONFIGS.keys()), index=4)
cfg = TIMEFRAME_CONFIGS[selected_tf]

st.sidebar.write(f"Daten: `{cfg.period}` / `{cfg.interval}`")
st.sidebar.write(f"ML-Horizont: `{cfg.horizon}` Kerzen")

st.sidebar.warning("Keine Finanzberatung. Forex ist riskant. Nur Analyse- und Bildungszweck.")


# =========================================================
# HEADER
# =========================================================

st.markdown('<div class="main-title">ForexScope AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Forex-Paar wählen, bis M1 analysieren, Szenario ableiten und Risiko professionell planen</div>',
    unsafe_allow_html=True,
)


# =========================================================
# DATA
# =========================================================

market_data = load_fx_data(yf_ticker, cfg.period, cfg.interval)

if market_data.empty and page not in ["Start", "Methodik & Grenzen"]:
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
    st.header("Produktlogik")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Forex-Paar rein. Intraday-Analyse raus. Risiko zuerst.</h2>
        <p>
        ForexScope AI analysiert Währungspaare bis auf M1-Basis, nutzt je Timeframe passende Indikatoren
        und ergänzt die technische Lage um Positionsgrößen-, Kapital- und Automations-Checks.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("1. Timeframe")
        st.write("M1, M5, M15, M30, H1, H4, D1 oder W1.")

    with c2:
        st.subheader("2. Marktstruktur")
        st.write("MA, RSI, MACD, ATR, Support, Resistance und Trend-Score.")

    with c3:
        st.subheader("3. Risk Engine")
        st.write("Budget, Risiko, Lots, CRV, Break-even und Automation-Precheck.")

    st.info("Empfohlener Flow: Live Forex Analyse → Capital & Risk Planner → Automation Readiness.")


# =========================================================
# LIVE FOREX ANALYSE
# =========================================================

elif page == "Live Forex Analyse":
    st.header("Live Forex Analyse")

    st.markdown(
        f"""
        <div class="info-card">
        <b>{pair_label}</b> | yfinance-Ticker: <b>{yf_ticker}</b><br>
        Timeframe: <b>{selected_tf}</b> | Datenfenster: <b>{cfg.period}</b> | Intervall: <b>{cfg.interval}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if feature_data.empty:
        st.error("Nicht genug Daten für Indikatoren. Wähle einen anderen Timeframe oder versuche es später erneut.")
        st.stop()

    latest_close = float(market_data["Close"].iloc[-1])
    first_close = float(market_data["Close"].iloc[0])
    performance = latest_close / first_close - 1

    sr = detect_support_resistance(feature_data)
    latest = feature_data.iloc[-1]
    ts = trend_score(latest)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Aktueller Kurs", f"{latest_close:.5f}")
    k2.metric("Performance Fenster", f"{performance:.2%}")
    k3.metric("Datenpunkte", f"{len(market_data):,}")
    k4.metric("ATR", f"{latest['atr']:.5f}")
    k5.metric("ATR in Pips", f"{pips_between(pair_label, 0, latest['atr']):.1f}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Range High", f"{sr['range_high']:.5f}")
    m2.metric("Range Low", f"{sr['range_low']:.5f}")
    m3.metric("Support", f"{sr['nearest_support']:.5f}")
    m4.metric("Resistance", f"{sr['nearest_resistance']:.5f}")

    st.markdown(
        f"""
        <div class="info-card {ts['card']}">
        <h3>Szenario: {ts['label']} | Score: {ts['score']}</h3>
        <p>Diese Einschätzung kombiniert schnelle/langsame MA, MACD, RSI und optional MA 200.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for reason in ts["reasons"]:
        st.write(f"- {reason}")

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


# =========================================================
# MULTI TIMEFRAME
# =========================================================

elif page == "Multi-Timeframe Dashboard":
    st.header("Multi-Timeframe Dashboard")

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
                    "Trend": "zu wenig Indikator-Daten",
                    "Score": np.nan,
                    "Letzter Kurs": float(data["Close"].iloc[-1]),
                    "Support": np.nan,
                    "Resistance": np.nan,
                    "RSI": np.nan,
                    "ATR Pips": np.nan,
                }
            )
            continue

        latest = prepared.iloc[-1]
        ts = trend_score(latest)
        sr = detect_support_resistance(prepared)

        rows.append(
            {
                "Timeframe": tf_label,
                "Daten": len(data),
                "Performance": perf,
                "Trend": ts["label"],
                "Score": ts["score"],
                "Letzter Kurs": sr["last_close"],
                "Support": sr["nearest_support"],
                "Resistance": sr["nearest_resistance"],
                "RSI": float(latest["rsi"]),
                "ATR Pips": pips_between(pair_label, 0, float(latest["atr"])),
            }
        )

    summary = pd.DataFrame(rows)

    if summary.empty:
        st.error("Keine Multi-Timeframe-Daten verfügbar.")
        st.stop()

    st.dataframe(summary, use_container_width=True)

    fig = px.bar(
        summary,
        x="Timeframe",
        y="Score",
        color="Trend",
        text="Score",
        title="Trend-Score je Timeframe",
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    perf_fig = px.bar(
        summary,
        x="Timeframe",
        y="Performance",
        color="Trend",
        text=summary["Performance"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else ""),
        title="Performance je Timeframe-Fenster",
    )
    perf_fig.update_yaxes(tickformat=".0%")
    perf_fig.update_traces(textposition="outside")
    st.plotly_chart(perf_fig, use_container_width=True)

    score_sum = summary["Score"].dropna().sum()

    if score_sum >= 8:
        overall = "Übergeordnet eher bullish"
        card = "good-card"
    elif score_sum <= -8:
        overall = "Übergeordnet eher bearish"
        card = "bad-card"
    else:
        overall = "Übergeordnet gemischt / neutral"
        card = "neutral-card"

    st.markdown(
        f"""
        <div class="info-card {card}">
        <h3>{overall}</h3>
        <p>Diese Einschätzung verdichtet mehrere Timeframes. Sie ersetzt keine Trading-Entscheidung.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# ML SZENARIO
# =========================================================

elif page == "ML Szenario":
    st.header("ML Szenario")

    if feature_data.empty:
        st.error("Nicht genug Daten für Feature Engineering.")
        st.stop()

    ml_model, metrics, latest_input = train_direction_model(feature_data, cfg)

    if ml_model is None:
        st.warning(
            f"Für {selected_tf} sind aktuell nicht genug saubere Daten für ein sinnvolles ML-Modell vorhanden. "
            f"Geladene Rohdaten: {len(market_data):,}. Indikator-Daten: {len(feature_data):,}. "
            f"Benötigt für diesen Timeframe: ca. {cfg.min_rows_for_ml}."
        )
        st.info("Trotzdem kannst du die Live Forex Analyse und den Capital & Risk Planner nutzen.")
        st.stop()

    proba = ml_model.predict_proba(latest_input)[0]
    classes = list(ml_model.classes_)

    probability_df = pd.DataFrame(
        {
            "Szenario": ["Bearish / niedriger" if c == 0 else "Bullish / höher" for c in classes],
            "Wahrscheinlichkeit": proba,
        }
    )

    predicted_class = classes[int(np.argmax(proba))]
    predicted_label = "Bullish / höher" if predicted_class == 1 else "Bearish / niedriger"
    card = "good-card" if predicted_class == 1 else "bad-card"

    c1, c2 = st.columns([0.9, 1.1])

    with c1:
        st.markdown(
            f"""
            <div class="info-card {card}">
            <h2>{predicted_label}</h2>
            <p>Modellbasierte Einschätzung für die nächsten {cfg.horizon} Kerzen.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
        st.metric("Verfügbare ML-Zeilen", f"{metrics['rows_available']:,}")
        st.metric("Trainingsdatenpunkte", f"{metrics['train_size']:,}")
        st.metric("Testdatenpunkte", f"{metrics['test_size']:,}")

    with c2:
        fig = px.bar(
            probability_df,
            x="Szenario",
            y="Wahrscheinlichkeit",
            text=probability_df["Wahrscheinlichkeit"].apply(lambda x: f"{x:.1%}"),
            title="ML-Szenario-Wahrscheinlichkeiten",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Klassifikationsbericht")
    st.dataframe(pd.DataFrame(metrics["report"]).transpose(), use_container_width=True)

    st.warning("Keine Finanzberatung. Das ML-Modell ist eine einfache Szenarioeinschätzung auf historischen Daten.")


# =========================================================
# CAPITAL & RISK PLANNER
# =========================================================

elif page == "Capital & Risk Planner":
    st.header("Capital & Risk Planner")

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
        <div class="info-card">
        <b>Professionelle Logik:</b> Die App fragt keine unbekannte Trefferquote als Pflichtwert ab.
        Sie berechnet zuerst CRV, Break-even-Trefferquote, Risiko, Positionsgröße und Automationsfähigkeit.
        Trefferquote ist nur optional für Was-wäre-wenn-Szenarien.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        account_size = st.number_input("Kontogröße / Budget", min_value=50.0, value=1000.0, step=50.0)
        risk_percent = st.slider("Risiko pro Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

        trade_direction = st.radio("Trade-Richtung", ["Long", "Short"], horizontal=True)

        entry = st.number_input("Entry", value=round(latest_close, 5), format="%.5f")

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
            st.error("CRV kann nicht berechnet werden, weil Entry und Stop identisch sind.")
            st.stop()

        potential_profit = risk_amount * rr
        be_wr = break_even_winrate(rr)

        m1, m2 = st.columns(2)
        m1.metric("Max. Verlust bei Stop", f"{risk_amount:.2f}")
        m2.metric("Pot. Gewinn bei Target", f"{potential_profit:.2f}")

        m3, m4 = st.columns(2)
        m3.metric("Risiko in Pips", f"{pip_risk:.1f}")
        m4.metric("Chance in Pips", f"{pip_reward:.1f}")

        m5, m6 = st.columns(2)
        m5.metric("CRV", f"{rr:.2f}")
        m6.metric("Benötigte Trefferquote", f"{be_wr:.1%}")

        st.subheader("Positionsgröße grob")
        st.write(f"Units: **{pos['units']:,.0f}**")
        st.write(f"Micro Lots: **{pos['micro_lots']:.2f}**")
        st.write(f"Mini Lots: **{pos['mini_lots']:.2f}**")
        st.write(f"Standard Lots: **{pos['standard_lots']:.3f}**")

        st.caption(
            "Lot-Berechnung ist vereinfacht. Für Broker-Automation müssen Kontowährung, Quote-Währung, Bid/Ask, Spread, Slippage und Kontraktgröße exakt berücksichtigt werden."
        )

    st.divider()

    st.subheader("Trade-Plan Bewertung")

    passed, warnings = automation_precheck(account_size, risk_percent, rr, stop, entry, target, pair_label)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card good-card">
            <h3>Bestandene Checks</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if passed:
            for item in passed:
                st.write(f"✅ {item}")
        else:
            st.write("Keine bestandenen Checks.")

    with c2:
        st.markdown(
            """
            <div class="info-card warn-card">
            <h3>Warnungen</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if warnings:
            for item in warnings:
                st.write(f"⚠️ {item}")
        else:
            st.write("Keine Warnungen.")

    st.divider()

    with st.expander("Optional: Was-wäre-wenn mit angenommener Trefferquote"):
        st.write(
            "Diese Trefferquote ist keine bekannte Wahrheit. Sie dient nur als Szenario, bis du echte Backtest- oder Journal-Daten hast."
        )

        planned_trades = st.slider("Szenario: Anzahl Trades", min_value=1, max_value=100, value=20)
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
            x="Trade",
            y="Erwarteter Kontostand",
            title="Was-wäre-wenn-Kontoverlauf",
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)

        final_equity = float(equity_df["Erwarteter Kontostand"].iloc[-1])
        st.metric("Erwarteter Kontostand nach Szenario", f"{final_equity:.2f}", delta=f"{final_equity - account_size:.2f}")

        if assumed_winrate < be_wr:
            st.warning("Die angenommene Trefferquote liegt unter der Break-even-Trefferquote. Das Szenario ist statistisch negativ.")
        else:
            st.success("Die angenommene Trefferquote liegt über der Break-even-Trefferquote. Das Szenario ist rechnerisch positiv.")

    st.warning(
        "Spätere Broker-Automation darf nicht blind handeln. Dafür brauchst du Risk Engine, Order-Freigabe, Spread-/Slippage-Prüfung, Max-Loss-Regeln und zuerst Paper-Trading."
    )


# =========================================================
# AUTOMATION READINESS
# =========================================================

elif page == "Automation Readiness":
    st.header("Automation Readiness")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Vorbereitung für spätere Broker-Automation</h2>
        <p>
        Diese Seite ist keine Orderausführung. Sie beschreibt, welche Bausteine vor einer echten Broker-Anbindung notwendig sind.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Mindestarchitektur")

    architecture = pd.DataFrame(
        [
            {
                "Baustein": "Market Data",
                "Zweck": "Echte Bid/Ask-Preise, Spreads, aktuelle Handelbarkeit",
                "Status": "Später Broker-API nötig",
            },
            {
                "Baustein": "Signal Engine",
                "Zweck": "Regeln/ML-Szenario, aber kein blindes Signal",
                "Status": "Teilweise vorhanden",
            },
            {
                "Baustein": "Risk Engine",
                "Zweck": "Max. Risiko, Positionsgröße, Tagesverlust, Exposure",
                "Status": "Im Planner vorbereitet",
            },
            {
                "Baustein": "Order Manager",
                "Zweck": "Market/Limit/Stop Orders, SL/TP, Orderstatus",
                "Status": "Noch nicht angebunden",
            },
            {
                "Baustein": "Paper Trading",
                "Zweck": "Test ohne echtes Geld",
                "Status": "Pflicht vor Live",
            },
            {
                "Baustein": "Audit Log",
                "Zweck": "Jede Entscheidung und Order nachvollziehbar speichern",
                "Status": "Noch offen",
            },
            {
                "Baustein": "Kill Switch",
                "Zweck": "Automatik sofort stoppen",
                "Status": "Pflicht vor Live",
            },
        ]
    )

    st.dataframe(architecture, use_container_width=True)

    st.subheader("Broker-API Kandidat")

    st.info(
        "Für Forex wäre OANDA ein realistischer Demo-/Paper-Trading-Kandidat. Die v20 REST API bietet programmatischen Zugriff auf eine Trading Engine. Vor Live-Trading müssen Konto, Regulatorik, API-Keys, Rate Limits und Order-Risiken sauber geprüft werden."
    )

    st.subheader("Nächster professioneller Schritt")

    st.markdown(
        """
        1. Paper-Trading-Modus bauen  
        2. Jede geplante Order zuerst nur simulieren  
        3. Trade-Journal speichern  
        4. Trefferquote und Erwartungswert aus echten Journal-Daten berechnen  
        5. Erst danach Broker-Anbindung vorbereiten  
        """
    )

    st.warning("Kein Live-Trading ohne Paper-Trading, Limits, Kill Switch, Logging und manuelle Freigabe.")


# =========================================================
# METHODIK
# =========================================================

elif page == "Methodik & Grenzen":
    st.header("Methodik & Grenzen")

    st.subheader("Warum M1/M5 anders behandelt werden")
    st.write(
        "M1- und M5-Daten liefern viele kurze Kerzen, aber nur für kurze historische Fenster. "
        "Deshalb nutzt die App dort kurze Indikatoren wie MA 9/21, RSI 7/14 und ATR 7/14 statt SMA 200."
    )

    st.subheader("Berechnete Größen")
    st.markdown(
        """
        - OHLC-Daten je Timeframe
        - schnelle/langsame Moving Averages je Timeframe
        - RSI
        - MACD
        - ATR
        - Support/Resistance aus historischen Highs/Lows
        - Multi-Timeframe-Score
        - ML-Szenario, wenn genug Daten vorhanden sind
        - Budget-, Risiko-, Lot- und CRV-Rechner
        """
    )

    st.subheader("Grenzen")
    st.markdown(
        """
        - Keine Finanzberatung
        - Yahoo/yfinance-Daten können verzögert, begrenzt oder unvollständig sein
        - Forex-Volumen ist über Yahoo häufig nicht aussagekräftig
        - Lot-Berechnung ist vereinfacht
        - Für Broker-Automation braucht man echte Brokerdaten, Bid/Ask, Spread, Slippage, Order-Handling und Sicherheitslimits
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.caption("ForexScope AI | Intraday Forex-Marktdatenanalyse | Keine Finanzberatung")
