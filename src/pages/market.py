from __future__ import annotations

import streamlit as st

from src.charts import chart_candlestick, chart_drawdown, chart_price
from src.context import get_context
from src.ui import disclaimer_footer, kpi_grid, page_header, section_title


def render() -> None:
    ctx = get_context()
    df, ticker, mode = ctx["period_df"], ctx["ticker"], ctx["theme_mode"]

    page_header("Marktanalyse", f"{ticker} — technische Indikatoren über den gewählten Zeitraum")

    if df.empty or len(df) < 5:
        st.warning("Nicht genug Daten für diesen Ticker/Zeitraum.")
        return

    last, first = df.iloc[-1], df.iloc[0]
    period_return = (last["close"] / first["close"] - 1) * 100 if first["close"] else 0.0
    kpi_grid([
        ("Letzter Kurs", f"{last['close']:.2f}", ticker, "trend-up"),
        ("Periodenrendite", f"{period_return:+.1f}%", f"seit {first['date']:%Y-%m-%d}", "activity"),
        ("Volatilität (20d)", f"{last.get('volatility_20d', 0)*100:.1f}%", "annualisiert", "pulse"),
        ("Max Drawdown", f"{df['drawdown'].min()*100:.1f}%", "im Zeitraum", "trend-down"),
    ])

    tab1, tab2, tab3 = st.tabs(["📈 Linienchart", "🕯️ Candlestick", "📉 Drawdown"])
    with tab1:
        st.plotly_chart(chart_price(df, ticker, mode), config={"displayModeBar": False})
    with tab2:
        st.plotly_chart(chart_candlestick(df, ticker, mode), config={"displayModeBar": False})
    with tab3:
        st.plotly_chart(chart_drawdown(df, ticker, mode), config={"displayModeBar": False})

    section_title("Datenbasis", "layers")
    with st.expander("Rohdaten anzeigen"):
        st.dataframe(df.tail(200), width="stretch")
        st.download_button("Als CSV herunterladen", df.to_csv(index=False), f"{ticker}_daten.csv", icon=":material/download:")

    disclaimer_footer()
