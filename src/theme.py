"""Design system: CSS variables per light/dark mode + matching Plotly theme.

Palette grounded in the subject — a quant-research ledger, not a generic SaaS
dashboard: deep pine/petrol as the brand accent (kept distinct from the
red/green/amber semantic colors used for signals), warm-neutral paper in
light mode, ink-black in dark mode. Fraunces for display type, a quiet system
sans for body/UI, and a monospace face for every number (prices, scores,
tables) — functionally justified (columns of digits line up) and distinctive.

Two-layer approach: `styles/base.css` holds static structural rules written
against `var(--ws-*)` custom properties; this module only injects the
variable *values* per mode plus the Plotly layout equivalents, so charts and
markdown cards always stay in sync with the active theme.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import plotly.graph_objects as go
import streamlit as st

from src.config import BASE_DIR

CSS_PATH = BASE_DIR / "styles" / "base.css"

FONT_DISPLAY = "'Fraunces', 'Iowan Old Style', Georgia, serif"
FONT_BODY = "-apple-system, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES: Dict[str, Dict[str, str]] = {
    "Hell": {
        "bg": "#f4f5f0",
        "bg-elevated": "#ffffff",
        "surface": "#ffffff",
        "surface-muted": "#eceee0",
        "border": "rgba(23,26,23,0.12)",
        "text": "#171a17",
        "text-muted": "#5b6259",
        "primary": "#2f5d53",
        "primary-soft": "rgba(47,93,83,0.10)",
        "positive": "#1f7a4d",
        "positive-soft": "rgba(31,122,77,0.10)",
        "negative": "#a83c30",
        "negative-soft": "rgba(168,60,48,0.10)",
        "neutral": "#96681f",
        "neutral-soft": "rgba(150,104,31,0.10)",
        "shadow": "0 12px 30px rgba(23,26,23,0.08)",
        "plotly-template": "plotly_white",
        "grid": "rgba(23,26,23,0.08)",
        "colorway": ["#2f5d53", "#96681f", "#a83c30", "#3d5a80", "#7a8c6b", "#6b4c6b", "#5b6259"],
    },
    "Dunkel": {
        "bg": "#10130f",
        "bg-elevated": "#171a15",
        "surface": "#171a15",
        "surface-muted": "#1d2118",
        "border": "rgba(255,255,255,0.10)",
        "text": "#eef1ea",
        "text-muted": "#9aa398",
        "primary": "#6fae9d",
        "primary-soft": "rgba(111,174,157,0.16)",
        "positive": "#4fbf82",
        "positive-soft": "rgba(79,191,130,0.14)",
        "negative": "#e0776a",
        "negative-soft": "rgba(224,119,106,0.14)",
        "neutral": "#d1a44e",
        "neutral-soft": "rgba(209,164,78,0.14)",
        "shadow": "0 12px 30px rgba(0,0,0,0.5)",
        "plotly-template": "plotly_dark",
        "grid": "rgba(255,255,255,0.09)",
        "colorway": ["#6fae9d", "#d1a44e", "#e0776a", "#7f9fc4", "#9db38c", "#a883a8", "#9aa398"],
    },
}


@st.cache_data(show_spinner=False)
def _read_css() -> str:
    return CSS_PATH.read_text()


def inject_base_css() -> None:
    st.markdown(f"<style>{_read_css()}</style>", unsafe_allow_html=True)


def inject_theme_vars(mode: str) -> None:
    t = THEMES.get(mode, THEMES["Hell"])
    vars_css = "\n".join(
        f"  --ws-{k}: {v};" for k, v in t.items() if k not in ("plotly-template", "grid", "colorway")
    )
    fonts_css = (
        f"  --ws-font-display: {FONT_DISPLAY};\n"
        f"  --ws-font-body: {FONT_BODY};\n"
        f"  --ws-font-mono: {FONT_MONO};"
    )
    st.markdown(
        f"<style id='ws-theme-vars'>:root {{\n{vars_css}\n{fonts_css}\n}}</style>",
        unsafe_allow_html=True,
    )


def plotly_theme(mode: str) -> Dict[str, Any]:
    t = THEMES.get(mode, THEMES["Hell"])
    return {
        "template": t["plotly-template"],
        "font_color": t["text"],
        "grid_color": t["grid"],
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "colorway": t["colorway"],
    }


def apply_chart_theme(fig: go.Figure, mode: str, height: int | None = None) -> go.Figure:
    pt = plotly_theme(mode)
    fig.update_layout(
        template=pt["template"],
        paper_bgcolor=pt["paper_bgcolor"],
        plot_bgcolor=pt["plot_bgcolor"],
        font_color=pt["font_color"],
        font_family=FONT_BODY,
        colorway=pt["colorway"],
        margin=dict(l=10, r=20, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(gridcolor=pt["grid_color"], zerolinecolor=pt["grid_color"], tickfont_family=FONT_MONO)
    fig.update_yaxes(gridcolor=pt["grid_color"], zerolinecolor=pt["grid_color"], tickfont_family=FONT_MONO)
    return fig
