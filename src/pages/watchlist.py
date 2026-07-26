from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import chart_risk_return_scatter
from src.context import get_context
from src.model import compute_scores
from src.ui import disclaimer_footer, page_header, section_title


@st.cache_data(show_spinner="Berechne Watchlist-Ranking ...")
def _build_ranking(market_df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        sub = market_df[market_df["ticker"] == t].sort_values("date")
        if sub.empty:
            continue
        result = compute_scores(sub, t)
        last = sub.iloc[-1]
        rows.append({
            "ticker": t,
            "confidence": result.confidence,
            "outlook": result.outlook,
            "return_20d": last.get("return_20d", 0) * 100,
            "volatility_20d": last.get("volatility_20d", 0) * 100,
            "rf_proba": result.rf_proba * 100,
        })
    return pd.DataFrame(rows).sort_values("confidence", ascending=False)


def render() -> None:
    ctx = get_context()
    page_header("Watchlist-Vergleich", "Alle Ticker im Ranking nach Confidence-Score")

    ranking = _build_ranking(ctx["market"], ctx["tickers"])
    if ranking.empty:
        st.info("Keine Daten verfügbar.")
        return

    section_title("Ranking", "layers")
    st.dataframe(
        ranking.style.format({"confidence": "{:.0f}", "return_20d": "{:+.2f}%",
                               "volatility_20d": "{:.2f}%", "rf_proba": "{:.1f}%"}),
        width="stretch", hide_index=True,
    )

    section_title("Risiko/Rendite-Karte", "target")
    st.plotly_chart(chart_risk_return_scatter(ranking, ctx["theme_mode"]),
                     config={"displayModeBar": False})

    disclaimer_footer()
