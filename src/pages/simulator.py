from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import chart_portfolio_allocation
from src.context import get_context
from src.ui import disclaimer_footer, kpi_grid, page_header, section_title


def render() -> None:
    ctx = get_context()
    mode = ctx["theme_mode"]

    page_header("Portfolio-Simulator", "Kapitalplanung, Allokation und Chance-Risiko-Verhältnis")

    capital = st.number_input("Gesamtkapital (€)", min_value=100.0, value=5000.0, step=100.0)

    section_title("Allokations-Szenario", "briefcase")
    default_tickers = ctx["tickers"][:5] if len(ctx["tickers"]) >= 5 else ctx["tickers"]
    n = st.slider("Anzahl Positionen", 2, min(8, len(ctx["tickers"])), min(4, len(ctx["tickers"])))
    chosen = st.multiselect("Ticker", ctx["tickers"], default=default_tickers[:n])[:n] or default_tickers[:n]

    if chosen:
        equal = round(100 / len(chosen), 1)
        rows = []
        total = 0.0
        for i, t in enumerate(chosen):
            key = f"alloc_{t}"
            default_val = equal if i < len(chosen) - 1 else max(0.0, 100 - equal * (len(chosen) - 1))
            val = st.session_state.get(key, default_val)
            rows.append({"ticker": t, "allocation": val})
            total += val
        df = pd.DataFrame(rows)
        edited = st.data_editor(df, width="stretch", hide_index=True, key="alloc_editor",
                                 column_config={"allocation": st.column_config.NumberColumn("Allokation (%)", min_value=0.0, max_value=100.0)})
        total_alloc = edited["allocation"].sum()
        if abs(total_alloc - 100) > 0.5:
            st.warning(f"Allokation summiert sich auf {total_alloc:.1f}% — sollte 100% sein.")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(chart_portfolio_allocation(edited["ticker"].tolist(), edited["allocation"].tolist(), mode),
                             config={"displayModeBar": False})
        with c2:
            edited["kapital_eur"] = capital * edited["allocation"] / 100
            st.dataframe(edited, width="stretch", hide_index=True)
            kpi_grid([
                ("Positionen", str(len(edited)), "aktiv", "layers"),
                ("Größte Position", f"{edited['allocation'].max():.1f}%", edited.loc[edited['allocation'].idxmax(), 'ticker'], "target"),
                ("Konzentration (HHI)", f"{((edited['allocation']/100)**2).sum():.3f}", "0=diversifiziert, 1=konzentriert", "grid"),
            ])

    disclaimer_footer()
