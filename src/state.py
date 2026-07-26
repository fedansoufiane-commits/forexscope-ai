"""Session-state defaults."""
from __future__ import annotations

import streamlit as st

DEFAULTS = {
    "theme_mode": "Hell",
    "app_mode": "Geführte Ansicht",
    "ticker": "SPY",
    "period": "1J",
    "use_live_data": False,
    "asset_weight": 15,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value)
