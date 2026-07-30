"""Small reusable markdown/HTML UI building blocks shared across pages."""
from __future__ import annotations

from typing import Iterable, Tuple

import streamlit as st

from src.config import DISCLAIMER
from src.icons import icon


def card(body_html: str, variant: str = "") -> None:
    cls = f"ws-card {variant}".strip()
    st.markdown(f'<div class="{cls}">{body_html}</div>', unsafe_allow_html=True)


def kpi_grid(items: Iterable[Tuple[str, str, str] | Tuple[str, str, str, str]]) -> None:
    """items: iterable of (label, value, sub) or (label, value, sub, icon_name)."""
    cells = ""
    for item in items:
        label, value, sub = item[0], item[1], item[2]
        icon_name = item[3] if len(item) > 3 else ""
        icon_html = icon(icon_name, size=13) if icon_name else ""
        cells += (
            f'<div class="ws-kpi"><div class="ws-kpi-label">{icon_html}{label}</div>'
            f'<div class="ws-kpi-value ws-mono">{value}</div>'
            f'<div class="ws-kpi-sub">{sub}</div></div>'
        )
    st.markdown(f'<div class="ws-kpi-grid">{cells}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "info", icon_name: str = "") -> str:
    icon_html = icon(icon_name, size=12) if icon_name else ""
    return f'<span class="ws-badge {kind}">{icon_html}{text}</span>'


def section_title(text: str, icon_name: str = "") -> None:
    icon_html = icon(icon_name, size=13) if icon_name else ""
    st.markdown(f'<div class="ws-section-title">{icon_html}{text}</div>', unsafe_allow_html=True)


def bar(pct: float, color: str) -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div class="ws-bar-track"><div class="ws-bar-fill" '
        f'style="width:{pct:.0f}%;background:{color}"></div></div>'
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def disclaimer_footer() -> None:
    st.markdown(
        f'<div class="ws-disclaimer">{icon("warning", 13)} {DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )
