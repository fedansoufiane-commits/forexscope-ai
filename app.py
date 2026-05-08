import datetime as dt
from pathlib import Path
from urllib.parse import quote, unquote

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from styles.theme import load_base_css, render_runtime_theme, apply_chart_theme
from components.charts import render_interactive_chart, add_hover_data

try:
    import joblib
except Exception:
    joblib = None


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="WealthScope AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# DESIGN
# =========================================================

st.markdown(
    """
    <style>
    /* =========================
       GLOBAL PREMIUM THEME
       ========================= */

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(95, 116, 255, 0.22), transparent 26%),
            radial-gradient(circle at 85% 0%, rgba(255, 95, 135, 0.13), transparent 25%),
            linear-gradient(135deg, #080b14 0%, #101522 45%, #171a28 100%) !important;
        color: #f6f8ff;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4 {
        color: #f8f9ff !important;
        letter-spacing: -0.035em;
    }

    p, li, label, span {
        color: #dce2f2;
    }

    .main-title {
        font-size: clamp(3rem, 7vw, 6rem);
        font-weight: 950;
        letter-spacing: -0.08em;
        line-height: 0.9;
        margin-bottom: 0.35rem;
        background: linear-gradient(90deg, #ffffff, #cbd7ff 45%, #ffb6c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 18px 35px rgba(0,0,0,0.30));
    }

    .subtitle {
        color: #aab2c7;
        font-size: 1.08rem;
        margin-bottom: 1.7rem;
        max-width: 980px;
    }

    /* =========================
       SIDEBAR
       ========================= */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(12, 16, 30, 0.98), rgba(15, 19, 33, 0.96)) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
        width: 250px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f4f6fb !important;
    }

    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
    }

    /* =========================
       CARDS
       ========================= */

    .hero-card,
    .info-card,
    .mini-card,
    .news-card,
    .premium-strip,
    .advisor-box {
        border: 1px solid rgba(255,255,255,0.12);
        background:
            linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
        box-shadow:
            0 20px 55px rgba(0,0,0,0.26),
            inset 0 1px 0 rgba(255,255,255,0.07);
        backdrop-filter: blur(18px);
    }

    .hero-card {
        border-radius: 30px;
        padding: 1.7rem 2rem;
        margin-bottom: 1.35rem;
        background:
            linear-gradient(135deg, rgba(87, 116, 255, 0.19), rgba(255,255,255,0.055)),
            rgba(255,255,255,0.045);
    }

    .info-card {
        border-radius: 23px;
        padding: 1.15rem 1.35rem;
        margin-bottom: 1rem;
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
    }

    .info-card:hover,
    .mini-card:hover,
    .news-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255,255,255,0.22);
        box-shadow: 0 24px 62px rgba(0,0,0,0.34);
    }

    .mini-card {
        border-radius: 22px;
        padding: 1.15rem 1.3rem;
        height: 100%;
    }

    .news-card {
        border-radius: 22px;
        padding: 1.12rem 1.3rem;
        margin-bottom: 1rem;
    }

    .premium-strip {
        border-radius: 28px;
        padding: 1.1rem 1.35rem;
        background:
            linear-gradient(135deg, rgba(86, 110, 255, 0.22), rgba(255,255,255,0.045)),
            rgba(255,255,255,0.035);
        margin-bottom: 1.1rem;
    }

    .advisor-box {
        border-radius: 26px;
        padding: 1.35rem 1.55rem;
        background:
            linear-gradient(135deg, rgba(40, 209, 124, 0.12), rgba(111,140,255,0.09)),
            rgba(255,255,255,0.04);
        margin-bottom: 1.15rem;
    }

    .good-card { border-left: 7px solid #28d17c; }
    .warn-card { border-left: 7px solid #ffb020; }
    .bad-card { border-left: 7px solid #ff4d5e; }
    .neutral-card { border-left: 7px solid #6f8cff; }

    .big-number {
        font-size: 2.15rem;
        font-weight: 900;
        letter-spacing: -0.045em;
        color: #ffffff;
        margin-top: .15rem;
        margin-bottom: .2rem;
    }

    .section-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: #9fa8c1;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .decision-step {
        border-left: 5px solid #6f8cff;
        padding: 0.82rem 1rem;
        background: rgba(255,255,255,0.055);
        border-radius: 16px;
        margin-bottom: 0.65rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }

    /* =========================
       REAL CLICKABLE INFO BUTTON
       ========================= */

    .info-wrap {
        display: inline-block;
        position: relative;
        vertical-align: middle;
        margin-left: 0.25rem;
    }

    .info-wrap details {
        display: inline-block;
        position: relative;
    }

    .info-wrap summary {
        list-style: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 19px;
        height: 19px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.36);
        background: rgba(255,255,255,0.10);
        color: #e8edff;
        font-size: 0.74rem;
        font-weight: 900;
        cursor: pointer;
        user-select: none;
        box-shadow: 0 8px 18px rgba(0,0,0,0.22);
    }

    .info-wrap summary::-webkit-details-marker {
        display: none;
    }

    .info-pop {
        position: absolute;
        z-index: 9999;
        top: 28px;
        left: -10px;
        width: 310px;
        padding: 0.85rem 0.95rem;
        border-radius: 16px;
        background: rgba(16, 21, 34, 0.98);
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: 0 18px 50px rgba(0,0,0,0.45);
        color: #f5f7ff;
        font-size: 0.92rem;
        line-height: 1.35;
    }

    .info-pop b {
        color: #ffffff;
    }

    .glossary-chip {
        display: inline-block;
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 999px;
        padding: 0.32rem 0.75rem;
        margin: 0.18rem;
        background: rgba(255,255,255,0.07);
        font-size: 0.9rem;
        color: #e7ebf8;
    }

    /* =========================
       BADGES
       ========================= */

    .status-badge {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.78rem;
        font-weight: 850;
        font-size: 0.88rem;
        margin-right: 0.38rem;
        margin-bottom: 0.35rem;
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
    }

    .badge-green { background: rgba(40, 209, 124, 0.15); color: #6ff0ac; }
    .badge-yellow { background: rgba(255, 176, 32, 0.15); color: #ffd073; }
    .badge-red { background: rgba(255, 77, 94, 0.15); color: #ff9aa4; }
    .badge-blue { background: rgba(111, 140, 255, 0.16); color: #b9c7ff; }

    /* =========================
       STREAMLIT WIDGETS
       ========================= */

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 16px !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        background: linear-gradient(135deg, #6f8cff, #8d6dff) !important;
        color: white !important;
        font-weight: 800 !important;
        padding: 0.65rem 1rem !important;
        box-shadow: 0 14px 30px rgba(80,100,255,0.23) !important;
        transition: transform .15s ease, box-shadow .15s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 38px rgba(80,100,255,0.32) !important;
    }

    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.055);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 14px 32px rgba(0,0,0,0.18);
    }

    [data-testid="stMetricLabel"] {
        color: #aeb6cc !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 900;
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        background: rgba(255,255,255,0.045);
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stMultiSelect div[data-baseweb="select"] > div,
    .stTextArea textarea {
        border-radius: 15px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: rgba(255,255,255,0.075) !important;
        color: #ffffff !important;
    }

    [data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 16px 34px rgba(0,0,0,0.20);
    }

    a {
        color: #aebdff !important;
        text-decoration: none;
        font-weight: 700;
    }

    a:hover {
        color: #ffffff !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.12);
    }

    hr {
        border-color: rgba(255,255,255,0.10);
        margin-top: 1.6rem;
        margin-bottom: 1.6rem;
    }

    /* =========================
       LIGHT MODE OVERRIDES
       ========================= */

    .stApp:has(.theme-light) {
        background:
            radial-gradient(circle at 15% 10%, rgba(95, 116, 255, 0.13), transparent 26%),
            radial-gradient(circle at 85% 0%, rgba(255, 95, 135, 0.08), transparent 25%),
            linear-gradient(135deg, #f6f8fc 0%, #eef2f8 45%, #ffffff 100%) !important;
        color: #1f2430 !important;
    }

    .stApp:has(.theme-light) h1,
    .stApp:has(.theme-light) h2,
    .stApp:has(.theme-light) h3,
    .stApp:has(.theme-light) h4,
    .stApp:has(.theme-light) p,
    .stApp:has(.theme-light) li,
    .stApp:has(.theme-light) label,
    .stApp:has(.theme-light) span {
        color: #1f2430 !important;
    }

    .stApp:has(.theme-light) .subtitle,
    .stApp:has(.theme-light) .muted,
    .stApp:has(.theme-light) .section-label {
        color: #5d6578 !important;
    }

    .stApp:has(.theme-light) .main-title {
        background: linear-gradient(90deg, #111827, #3046a3 55%, #a7345d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stApp:has(.theme-light) .hero-card,
    .stApp:has(.theme-light) .info-card,
    .stApp:has(.theme-light) .mini-card,
    .stApp:has(.theme-light) .news-card,
    .stApp:has(.theme-light) .premium-strip,
    .stApp:has(.theme-light) .advisor-box,
    .stApp:has(.theme-light) .status-row {
        background: rgba(255,255,255,0.78) !important;
        border: 1px solid rgba(15, 23, 42, 0.10) !important;
        box-shadow: 0 18px 45px rgba(15,23,42,0.10) !important;
    }

    .stApp:has(.theme-light) .big-number {
        color: #111827 !important;
    }

    .stApp:has(.theme-light) section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #eef2f8) !important;
        border-right: 1px solid rgba(15,23,42,0.10) !important;
    }

    .stApp:has(.theme-light) section[data-testid="stSidebar"] * {
        color: #1f2430 !important;
    }

    .stApp:has(.theme-light) section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    .stApp:has(.theme-light) section[data-testid="stSidebar"] input,
    .stApp:has(.theme-light) .stTextInput input,
    .stApp:has(.theme-light) .stNumberInput input,
    .stApp:has(.theme-light) .stSelectbox div[data-baseweb="select"] > div,
    .stApp:has(.theme-light) .stMultiSelect div[data-baseweb="select"] > div,
    .stApp:has(.theme-light) .stTextArea textarea {
        background: rgba(255,255,255,0.95) !important;
        border: 1px solid rgba(15,23,42,0.12) !important;
        color: #111827 !important;
    }

    .stApp:has(.theme-light) .info-pop {
        background: rgba(255,255,255,0.98);
        color: #111827;
        border: 1px solid rgba(15,23,42,0.14);
    }

    .stApp:has(.theme-light) .info-pop b {
        color: #111827;
    }

    /* =========================
       BOTTOM STATUS BAR
       ========================= */

    .bottom-status-bar {
        position: sticky;
        bottom: 0;
        z-index: 999;
        margin-top: 2rem;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 22px;
        padding: 0.8rem 1rem;
        background: rgba(12,16,30,0.92);
        backdrop-filter: blur(18px);
        box-shadow: 0 -12px 35px rgba(0,0,0,0.22);
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem 1.2rem;
        align-items: center;
        font-size: 0.9rem;
    }

    .bottom-status-item {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        white-space: nowrap;
    }

    .bottom-status-link {
        font-weight: 800;
        opacity: 0.95;
    }

    .stApp:has(.theme-light) .bottom-status-bar {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(15,23,42,0.10);
        box-shadow: 0 -12px 35px rgba(15,23,42,0.10);
    }
    .status-dot {
        display: inline-block;
        width: 13px;
        height: 13px;
        border-radius: 50%;
        margin-right: 0.45rem;
        box-shadow: 0 0 18px rgba(255,255,255,0.25);
    }

    .dot-green {
        background: #28d17c;
        box-shadow: 0 0 18px rgba(40,209,124,0.65);
    }

    .dot-yellow {
        background: #ffb020;
        box-shadow: 0 0 18px rgba(255,176,32,0.65);
    }

    .dot-red {
        background: #ff4d5e;
        box-shadow: 0 0 18px rgba(255,77,94,0.65);
    }

    .status-row {
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        background: rgba(255,255,255,0.045);
        margin-bottom: 0.65rem;
    }

    
    /* =====================================================
       FINAL OVERRIDE: ARSNOVA-LIKE FIXED BOTTOM BAR
       ===================================================== */

    .bottom-status-bar {
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        min-height: 72px !important;
        z-index: 2147483647 !important;
        border-radius: 0 !important;
        border-top: 1px solid rgba(255,255,255,0.18) !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: none !important;
        padding: 0.85rem 1.25rem !important;
        background:
            linear-gradient(180deg, rgba(95, 15, 170, 0.96), rgba(98, 0, 185, 0.98)) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        box-shadow:
            0 -18px 45px rgba(0,0,0,0.40),
            inset 0 1px 0 rgba(255,255,255,0.14) !important;
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0.7rem 2rem !important;
        font-size: 0.98rem !important;
    }

    .bottom-status-item {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        white-space: nowrap !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        text-decoration: none !important;
        opacity: 0.96 !important;
    }

    .bottom-status-item:hover {
        opacity: 1 !important;
        transform: translateY(-1px);
    }

    .bottom-status-icon {
        display: inline-flex !important;
        width: 25px !important;
        height: 25px !important;
        border-radius: 999px !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(255,255,255,0.16) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        font-size: 0.9rem !important;
    }

    .bottom-spacer {
        height: 110px !important;
    }

    .block-container {
        padding-bottom: 7.5rem !important;
    }

    /* alten Streamlit-Footer verstecken */
    footer {
        visibility: hidden !important;
    }

    /* Light Mode für feste Bottom-Bar */
    .stApp:has(.theme-light) .bottom-status-bar {
        background:
            linear-gradient(180deg, rgba(110, 32, 190, 0.95), rgba(117, 0, 205, 0.97)) !important;
        border-top: 1px solid rgba(15,23,42,0.12) !important;
    }


    /* =====================================================
       FINAL CLEAN SIDEBAR + WEALTHSCOPE BOTTOM BAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        max-width: 250px !important;
        min-width: 250px !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start !important;
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        box-shadow: none !important;
        color: #eef3ff !important;
        font-weight: 750 !important;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(111,140,255,0.12) !important;
        border-color: rgba(111,140,255,0.22) !important;
    }

    .bottom-status-bar {
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        min-height: 70px !important;
        z-index: 2147483647 !important;
        border-radius: 0 !important;
        border-top: 1px solid rgba(255,255,255,0.12) !important;
        border-left: none !important;
        border-right: none !important;
        border-bottom: none !important;
        padding: 0.8rem 1.4rem !important;
        background:
            linear-gradient(180deg, rgba(12, 18, 34, 0.98), rgba(6, 10, 20, 0.99)) !important;
        backdrop-filter: blur(26px) !important;
        -webkit-backdrop-filter: blur(26px) !important;
        box-shadow:
            0 -18px 45px rgba(0,0,0,0.46),
            inset 0 1px 0 rgba(255,255,255,0.08) !important;
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0.65rem 1.45rem !important;
        font-size: 0.94rem !important;
    }

    .bottom-status-item {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        white-space: nowrap !important;
        color: #eef3ff !important;
        font-weight: 780 !important;
        text-decoration: none !important;
        opacity: 0.93 !important;
    }

    .bottom-status-item:hover {
        opacity: 1 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    .bottom-status-icon {
        display: inline-flex !important;
        width: 25px !important;
        height: 25px !important;
        border-radius: 999px !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(111, 140, 255, 0.16) !important;
        border: 1px solid rgba(111, 140, 255, 0.24) !important;
        font-size: 0.9rem !important;
    }

    .bottom-status-meta {
        opacity: 0.78 !important;
        font-weight: 650 !important;
    }

    .bottom-spacer {
        height: 115px !important;
    }

    .block-container {
        padding-bottom: 8rem !important;
    }

    footer {
        visibility: hidden !important;
    }

    .stApp:has(.theme-light) .bottom-status-bar {
        background:
            linear-gradient(180deg, rgba(248,250,255,0.98), rgba(232,237,247,0.98)) !important;
        border-top: 1px solid rgba(15,23,42,0.12) !important;
        box-shadow: 0 -18px 45px rgba(15,23,42,0.12) !important;
    }

    .stApp:has(.theme-light) .bottom-status-item {
        color: #172033 !important;
    }

    .stApp:has(.theme-light) .bottom-status-icon {
        background: rgba(47, 78, 180, 0.10) !important;
        border: 1px solid rgba(47, 78, 180, 0.16) !important;
    }


    /* =====================================================
       FINAL COLLAPSED SIDEBAR / HEADER CLEANUP
       ===================================================== */

    header[data-testid="stHeader"] {
        background: rgba(5, 9, 18, 0.88) !important;
        height: 3.2rem !important;
    }

    div[data-testid="collapsedControl"] {
        left: 0.7rem !important;
        top: 0.7rem !important;
        z-index: 2147483647 !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        width: 0 !important;
        max-width: 0 !important;
        overflow: hidden !important;
        border-right: none !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] + div {
        margin-left: 0 !important;
    }

    .main .block-container {
        padding-left: 3.2rem !important;
        padding-right: 3.2rem !important;
    }

    @media (max-width: 900px) {
        .main .block-container {
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
        }
    }


    /* =====================================================
       FINAL CLEAN WEALTHSCOPE UI
       ===================================================== */

    .bottom-status-bar {
        position: fixed !important;
        left: 0 !important;
        right: 0 !important;
        bottom: 0 !important;
        width: 100vw !important;
        min-height: 70px !important;
        z-index: 2147483647 !important;
        border-radius: 0 !important;
        border-top: 1px solid rgba(111, 140, 255, 0.18) !important;
        padding: 0.8rem 1.4rem !important;
        background:
            linear-gradient(180deg, rgba(10, 16, 30, 0.98), rgba(5, 9, 18, 0.99)) !important;
        backdrop-filter: blur(26px) !important;
        -webkit-backdrop-filter: blur(26px) !important;
        box-shadow: 0 -18px 45px rgba(0,0,0,0.46) !important;
        display: flex !important;
        flex-wrap: wrap !important;
        justify-content: center !important;
        align-items: center !important;
        gap: 0.65rem 1.45rem !important;
        font-size: 0.94rem !important;
    }

    .bottom-status-item {
        display: inline-flex !important;
        align-items: center !important;
        gap: 0.45rem !important;
        white-space: nowrap !important;
        color: #eef3ff !important;
        font-weight: 780 !important;
        text-decoration: none !important;
        opacity: 0.94 !important;
    }

    .bottom-status-item:hover {
        opacity: 1 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
    }

    .bottom-status-icon {
        display: inline-flex !important;
        width: 25px !important;
        height: 25px !important;
        border-radius: 999px !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(111, 140, 255, 0.16) !important;
        border: 1px solid rgba(111, 140, 255, 0.24) !important;
        font-size: 0.9rem !important;
    }

    .bottom-status-meta {
        opacity: 0.78 !important;
        font-weight: 650 !important;
    }

    .bottom-spacer {
        height: 115px !important;
    }

    .block-container {
        padding-bottom: 8rem !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0 !important;
        width: 0 !important;
        max-width: 0 !important;
        overflow: hidden !important;
        border-right: none !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(5, 9, 18, 0.88) !important;
        height: 3.2rem !important;
    }

    footer {
        visibility: hidden !important;
    }

    .stApp:has(.theme-light) .bottom-status-bar {
        background:
            linear-gradient(180deg, rgba(248,250,255,0.98), rgba(232,237,247,0.98)) !important;
        border-top: 1px solid rgba(15,23,42,0.12) !important;
        box-shadow: 0 -18px 45px rgba(15,23,42,0.12) !important;
    }

    .stApp:has(.theme-light) .bottom-status-item {
        color: #172033 !important;
    }

    .stApp:has(.theme-light) .bottom-status-icon {
        background: rgba(47, 78, 180, 0.10) !important;
        border: 1px solid rgba(47, 78, 180, 0.16) !important;
    }


    /* =====================================================
       FINAL MODE SWITCH FIX
       ===================================================== */

    .stApp:has(.theme-light) {
        background:
            radial-gradient(circle at 20% 5%, rgba(64, 104, 255, 0.12), transparent 28%),
            radial-gradient(circle at 88% 8%, rgba(255, 108, 150, 0.10), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #eef3fb 48%, #ffffff 100%) !important;
        color: #111827 !important;
    }

    .stApp:has(.theme-light) [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #edf2fa) !important;
        border-right: 1px solid rgba(15,23,42,0.12) !important;
    }

    .stApp:has(.theme-light) [data-testid="stSidebar"] * {
        color: #111827 !important;
    }

    .stApp:has(.theme-light) .hero-card,
    .stApp:has(.theme-light) .mini-card,
    .stApp:has(.theme-light) .info-card,
    .stApp:has(.theme-light) .export-card,
    .stApp:has(.theme-light) .decision-step {
        background: rgba(255,255,255,0.86) !important;
        border: 1px solid rgba(15,23,42,0.12) !important;
        box-shadow: 0 18px 45px rgba(15,23,42,0.10) !important;
    }

    .stApp:has(.theme-light) .main-title {
        background: linear-gradient(90deg, #111827, #2f4eb4 55%, #7c3aed) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .stApp:has(.theme-light) h1,
    .stApp:has(.theme-light) h2,
    .stApp:has(.theme-light) h3,
    .stApp:has(.theme-light) p,
    .stApp:has(.theme-light) li,
    .stApp:has(.theme-light) label,
    .stApp:has(.theme-light) span {
        color: #111827 !important;
    }

    .stApp:has(.mode-expert) .hero-card {
        border-left: 5px solid rgba(111, 140, 255, 0.75) !important;
    }

    .stApp:has(.mode-guided) .hero-card {
        border-left: 5px solid rgba(34, 197, 94, 0.65) !important;
    }

    .stApp:has(.theme-light) .bottom-status-bar {
        background:
            linear-gradient(180deg, rgba(248,250,255,0.98), rgba(232,237,247,0.98)) !important;
        border-top: 1px solid rgba(15,23,42,0.12) !important;
    }

    .stApp:has(.theme-light) .bottom-status-item {
        color: #172033 !important;
    }

</style>
    """,
    unsafe_allow_html=True,
)



# =========================================================
# ASSET CONFIG
# =========================================================

ASSETS = {
    "ETF – S&P 500 (SPY)": {"ticker": "SPY", "type": "ETF", "category": "US-Aktienmarkt breit"},
    "ETF – S&P 500 (VOO)": {"ticker": "VOO", "type": "ETF", "category": "US-Aktienmarkt breit"},
    "ETF – Nasdaq 100 (QQQ)": {"ticker": "QQQ", "type": "ETF", "category": "Technologie / Wachstum"},
    "ETF – US Total Market (VTI)": {"ticker": "VTI", "type": "ETF", "category": "US-Gesamtmarkt"},
    "ETF – Emerging Markets (EEM)": {"ticker": "EEM", "type": "ETF", "category": "Schwellenländer"},
    "ETF – Emerging Markets (VWO)": {"ticker": "VWO", "type": "ETF", "category": "Schwellenländer"},
    "ETF – Bonds (AGG)": {"ticker": "AGG", "type": "ETF", "category": "Anleihen"},
    "ETF – Bonds (BND)": {"ticker": "BND", "type": "ETF", "category": "Anleihen"},
    "ETF – Gold (GLD)": {"ticker": "GLD", "type": "ETF", "category": "Gold / Krisenschutz"},
    "Aktie – Apple (AAPL)": {"ticker": "AAPL", "type": "Stock", "category": "Einzelaktie Technologie"},
    "Aktie – Microsoft (MSFT)": {"ticker": "MSFT", "type": "Stock", "category": "Einzelaktie Technologie"},
    "Aktie – Nvidia (NVDA)": {"ticker": "NVDA", "type": "Stock", "category": "Einzelaktie KI / Halbleiter"},
    "Aktie – Amazon (AMZN)": {"ticker": "AMZN", "type": "Stock", "category": "Einzelaktie Konsum / Cloud"},
    "Aktie – Alphabet (GOOGL)": {"ticker": "GOOGL", "type": "Stock", "category": "Einzelaktie Technologie"},
    "Aktie – Tesla (TSLA)": {"ticker": "TSLA", "type": "Stock", "category": "Einzelaktie Wachstum"},
    "Aktie – JPMorgan (JPM)": {"ticker": "JPM", "type": "Stock", "category": "Einzelaktie Banken"},
    "Aktie – Johnson & Johnson (JNJ)": {"ticker": "JNJ", "type": "Stock", "category": "Einzelaktie Gesundheit"},
    "Aktie – Exxon Mobil (XOM)": {"ticker": "XOM", "type": "Stock", "category": "Einzelaktie Energie"},
}

LOCAL_FEATURES_PATH = Path("data/processed/wealthscope_features.csv")
LOCAL_MODEL_PATH = Path("models/wealthscope_model.joblib")
MODEL_REPORT_PATH = Path("models/wealthscope_model_report.txt")



# =========================================================
# GLOSSARY / INFO HELPERS
# =========================================================

GLOSSARY = {
    "Wealth Outlook": "Gesamtbewertung aus historischen Daten, News, Risiko, Volatilität und ML-Szenario. Keine sichere Prognose.",
    "Confidence": "Bewertet, wie belastbar die aktuelle Einschätzung wirkt. Das ist keine Trefferwahrscheinlichkeit.",
    "Kapital-Schutz": "Score, der einschätzt, wie vorsichtig ein Szenario aus Sicht des Kapitalerhalts ist.",
    "Drawdown": "Rückgang vom bisherigen Hoch. Beispiel: -20 % bedeutet, dass der Kurs 20 % unter seinem Hoch liegt.",
    "Volatilität": "Schwankungsintensität. Hohe Volatilität bedeutet stärkere Kursbewegungen und höheres Risiko.",
    "RSI": "Relative-Stärke-Index. Ein Momentum-Indikator, der überkaufte oder überverkaufte Phasen anzeigen kann.",
    "MACD": "Trend- und Momentum-Indikator. Er zeigt, ob sich kurzfristige und längerfristige Trends annähern oder entfernen.",
    "Moving Average": "Gleitender Durchschnitt. Er glättet Kurse und hilft, Trends zu erkennen.",
    "MA 20": "Durchschnittskurs der letzten 20 Handelstage. Kurzfristige Trendlinie.",
    "MA 50": "Durchschnittskurs der letzten 50 Handelstage. Mittelfristige Trendlinie.",
    "MA 200": "Durchschnittskurs der letzten 200 Handelstage. Langfristige Trendlinie.",
    "Support": "Unterstützungsbereich. Dort hat der Kurs in der Vergangenheit häufiger Halt gefunden.",
    "Resistance": "Widerstandsbereich. Dort ist der Kurs in der Vergangenheit häufiger abgeprallt.",
    "News Score": "Regelbasierte Bewertung, ob aktuelle Nachrichten eher positiv, negativ oder neutral wirken.",
    "Impact": "Einschätzung, wie stark eine Nachricht den Markt beeinflussen könnte.",
    "Relevanz": "Einschätzung, wie stark eine Nachricht zum ausgewählten Asset passt.",
    "Sentiment": "Stimmung einer Nachricht: positiv, neutral oder negativ.",
    "Target 20d": "ML-Zielvariable: Ob der Kurs in 20 Handelstagen höher liegt oder nicht.",
    "ML-Accuracy": "Historische Testgenauigkeit des Modells. Keine Garantie für zukünftige Treffer.",
    "Decision Passport": "Herunterladbarer Report mit Asset, Kapital-Kontext, Scores, News und Warnhinweisen.",
    "ETF": "Börsengehandelter Fonds. Meist breiter gestreut als eine Einzelaktie.",
    "Einzelaktie": "Anteil an einem einzelnen Unternehmen. Höheres Konzentrationsrisiko als ein breit gestreuter ETF.",
}


def info_icon(term):
    explanation = GLOSSARY.get(term, "Keine Erklärung hinterlegt.")
    safe_term = str(term).replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    safe_explanation = str(explanation).replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
    <span class="info-wrap">
        <details>
            <summary>i</summary>
            <span class="info-pop"><b>{safe_term}</b><br>{safe_explanation}</span>
        </details>
    </span>
    """


def render_glossary_box(mode="Geführte Ansicht"):
    if mode == "Geführte Ansicht":
        with st.expander("Begriffe einfach erklärt"):
            important_terms = [
                "Wealth Outlook", "Confidence", "Kapital-Schutz", "Drawdown",
                "Volatilität", "ETF", "Einzelaktie", "Decision Passport"
            ]
            for term in important_terms:
                st.markdown(f"**{term}:** {GLOSSARY[term]}")
    else:
        with st.expander("Glossar anzeigen"):
            for term, explanation in GLOSSARY.items():
                st.markdown(f"**{term}:** {explanation}")


def render_metric_card(title, value, body, card="neutral-card", info_term=None):
    icon = info_icon(info_term or title)
    st.markdown(
        f"""
        <div class="info-card {card}">
        <h3>{title} {icon}</h3>
        <div class="big-number">{value}</div>
        <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



# =========================================================
# GERMAN NUMBER FORMATTING
# =========================================================

def de_number(value, decimals=2):
    try:
        value = float(value)
        formatted = f"{value:,.{decimals}f}"
        return formatted.replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(value)


def de_currency(value):
    return f"{de_number(value, 2)} €"


def de_percent(value, decimals=2):
    try:
        return f"{de_number(float(value), decimals)} %"
    except Exception:
        return str(value)



# =========================================================
# LIST / SELECTION HELPERS
# =========================================================

def unique_keep_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def unique_asset_labels():
    return unique_keep_order(list(ASSETS.keys()))


def deduplicate_selection(selection):
    return unique_keep_order(selection or [])



# =========================================================
# DYNAMIC UI MODE CSS
# =========================================================

def apply_dynamic_ui_mode(theme_mode, app_mode):
    is_light = theme_mode == "Light Mode"
    is_expert = app_mode == "Expertenansicht"

    if is_light:
        app_bg = """
            radial-gradient(circle at 20% 6%, rgba(59, 130, 246, 0.12), transparent 28%),
            radial-gradient(circle at 88% 4%, rgba(168, 85, 247, 0.10), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #eef3fb 48%, #ffffff 100%)
        """
        text_color = "#111827"
        muted_color = "#475569"
        card_bg = "rgba(255,255,255,0.88)"
        card_border = "rgba(15,23,42,0.12)"
        sidebar_bg = "linear-gradient(180deg, #ffffff 0%, #edf2fa 100%)"
        bottom_bg = "linear-gradient(180deg, rgba(248,250,255,0.98), rgba(232,237,247,0.98))"
        bottom_text = "#172033"
        header_bg = "rgba(248,250,255,0.92)"
        input_bg = "rgba(255,255,255,0.92)"
        input_text = "#111827"
    else:
        app_bg = """
            radial-gradient(circle at 12% 4%, rgba(73, 100, 255, 0.16), transparent 24%),
            radial-gradient(circle at 85% 8%, rgba(176, 72, 122, 0.13), transparent 28%),
            linear-gradient(135deg, #07101f 0%, #0b1020 48%, #12111d 100%)
        """
        text_color = "#eef3ff"
        muted_color = "#aeb7c8"
        card_bg = "rgba(255,255,255,0.065)"
        card_border = "rgba(255,255,255,0.13)"
        sidebar_bg = "linear-gradient(180deg, #080d1b 0%, #0b1020 100%)"
        bottom_bg = "linear-gradient(180deg, rgba(10,16,30,0.98), rgba(5,9,18,0.99))"
        bottom_text = "#eef3ff"
        header_bg = "rgba(5,9,18,0.88)"
        input_bg = "rgba(255,255,255,0.08)"
        input_text = "#eef3ff"

    mode_border = "rgba(111, 140, 255, 0.78)" if is_expert else "rgba(34, 197, 94, 0.70)"
    mode_badge_bg = "rgba(111, 140, 255, 0.15)" if is_expert else "rgba(34, 197, 94, 0.13)"
    mode_badge_border = "rgba(111, 140, 255, 0.28)" if is_expert else "rgba(34, 197, 94, 0.25)"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {app_bg} !important;
            color: {text_color} !important;
        }}

        header[data-testid="stHeader"] {{
            background: {header_bg} !important;
        }}

        section[data-testid="stSidebar"] {{
            background: {sidebar_bg} !important;
            border-right: 1px solid {card_border} !important;
        }}

        section[data-testid="stSidebar"] * {{
            color: {text_color} !important;
        }}

        .main-title {{
            background: linear-gradient(90deg, {text_color}, #8da2ff 60%, #c084fc) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }}

        h1, h2, h3, h4, h5, h6, p, li, label, span, div {{
            color: {text_color} !important;
        }}

        .subtitle, .muted, .stCaptionContainer, small {{
            color: {muted_color} !important;
        }}

        .hero-card,
        .mini-card,
        .info-card,
        .export-card,
        .decision-step,
        .status-row,
        .neutral-card,
        .good-card,
        .warning-card {{
            background: {card_bg} !important;
            border: 1px solid {card_border} !important;
            box-shadow: 0 18px 45px rgba(0,0,0,0.16) !important;
        }}

        .hero-card {{
            border-left: 6px solid {mode_border} !important;
        }}

        .mode-status-banner {{
            background: {mode_badge_bg} !important;
            border: 1px solid {mode_badge_border} !important;
            border-radius: 18px !important;
            padding: 0.9rem 1.1rem !important;
            margin: 1rem 0 1.2rem 0 !important;
            font-weight: 800 !important;
        }}

        .bottom-status-bar {{
            background: {bottom_bg} !important;
            border-top: 1px solid {card_border} !important;
        }}

        .bottom-status-item {{
            color: {bottom_text} !important;
        }}

        .bottom-status-item * {{
            color: {bottom_text} !important;
        }}

        input, textarea, select,
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {{
            background: {input_bg} !important;
            color: {input_text} !important;
            border-color: {card_border} !important;
        }}

        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input {{
            color: {input_text} !important;
        }}

        .stButton > button {{
            border-color: {card_border} !important;
        }}

        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {{
            color: {text_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



# =========================================================
# FINAL UI MODE RENDERING
# =========================================================

def apply_final_ui_mode(theme_mode, app_mode):
    is_light = theme_mode == "Light Mode"
    is_expert = app_mode == "Expertenansicht"

    if is_light:
        app_bg = "linear-gradient(135deg, #f8fafc 0%, #eef3fb 45%, #ffffff 100%)"
        sidebar_bg = "linear-gradient(180deg, #ffffff 0%, #eef3fb 100%)"
        card_bg = "rgba(255,255,255,0.90)"
        card_border = "rgba(15,23,42,0.13)"
        text = "#111827"
        muted = "#475569"
        input_bg = "rgba(255,255,255,0.95)"
        bottom_bg = "linear-gradient(180deg, rgba(248,250,255,0.98), rgba(232,237,247,0.98))"
        bottom_text = "#172033"
        top_bg = "rgba(248,250,255,0.92)"
    else:
        app_bg = "linear-gradient(135deg, #07101f 0%, #0b1020 48%, #12111d 100%)"
        sidebar_bg = "linear-gradient(180deg, #080d1b 0%, #0b1020 100%)"
        card_bg = "rgba(255,255,255,0.065)"
        card_border = "rgba(255,255,255,0.14)"
        text = "#eef3ff"
        muted = "#aeb7c8"
        input_bg = "rgba(255,255,255,0.08)"
        bottom_bg = "linear-gradient(180deg, rgba(10,16,30,0.98), rgba(5,9,18,0.99))"
        bottom_text = "#eef3ff"
        top_bg = "rgba(5,9,18,0.90)"

    accent = "#6f8cff" if is_expert else "#22c55e"
    accent_soft = "rgba(111,140,255,0.18)" if is_expert else "rgba(34,197,94,0.16)"
    accent_border = "rgba(111,140,255,0.42)" if is_expert else "rgba(34,197,94,0.36)"

    st.markdown(
        f"""
        <style id="wealthscope-final-mode-style">
            .stApp {{
                background: {app_bg} !important;
                color: {text} !important;
            }}

            header[data-testid="stHeader"] {{
                background: {top_bg} !important;
            }}

            section[data-testid="stSidebar"] {{
                background: {sidebar_bg} !important;
                border-right: 1px solid {card_border} !important;
            }}

            section[data-testid="stSidebar"] * {{
                color: {text} !important;
            }}

            .main-title {{
                background: linear-gradient(90deg, {text}, #8da2ff 62%, #c084fc) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
            }}

            h1, h2, h3, h4, h5, h6,
            p, li, label, span, div {{
                color: {text} !important;
            }}

            small,
            .subtitle,
            .stCaptionContainer,
            [data-testid="stCaptionContainer"],
            .muted {{
                color: {muted} !important;
            }}

            .hero-card,
            .mini-card,
            .info-card,
            .export-card,
            .decision-step,
            .status-row,
            .neutral-card,
            .good-card,
            .warning-card {{
                background: {card_bg} !important;
                border: 1px solid {card_border} !important;
                box-shadow: 0 18px 45px rgba(0,0,0,0.18) !important;
            }}

            .hero-card {{
                border-left: 6px solid {accent} !important;
            }}

            .mode-status-banner,
            .mode-content-panel {{
                background: {accent_soft} !important;
                border: 1px solid {accent_border} !important;
                border-left: 6px solid {accent} !important;
                border-radius: 22px !important;
                padding: 1.05rem 1.25rem !important;
                margin: 1rem 0 1.35rem 0 !important;
                box-shadow: 0 18px 45px rgba(0,0,0,0.16) !important;
            }}

            .mode-content-grid {{
                display: grid !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: 1rem !important;
                margin-top: 1rem !important;
            }}

            .mode-content-card {{
                background: {card_bg} !important;
                border: 1px solid {card_border} !important;
                border-radius: 20px !important;
                padding: 1rem 1.1rem !important;
            }}

            input, textarea, select,
            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div {{
                background: {input_bg} !important;
                color: {text} !important;
                border-color: {card_border} !important;
            }}

            div[data-baseweb="select"] span,
            div[data-baseweb="input"] input {{
                color: {text} !important;
            }}

            .bottom-status-bar {{
                background: {bottom_bg} !important;
                border-top: 1px solid {card_border} !important;
            }}

            .bottom-status-item,
            .bottom-status-item * {{
                color: {bottom_text} !important;
            }}

            [data-testid="stMetricValue"],
            [data-testid="stMetricLabel"] {{
                color: {text} !important;
            }}

            .stButton > button {{
                border-color: {card_border} !important;
            }}

            @media (max-width: 1000px) {{
                .mode-content-grid {{
                    grid-template-columns: 1fr !important;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mode_status_panel(theme_mode, app_mode):
    if theme_mode == "Light Mode":
        theme_text = "🌕 Sie befinden sich jetzt im Light Mode."
    else:
        theme_text = "🌑 Sie befinden sich jetzt im Dark Mode."

    if app_mode == "Geführte Ansicht":
        mode_text = "🧭 Geführte Ansicht aktiv: Fokus auf Klartext, Risiko und verständliche nächste Schritte."
        c1_title = "Klartext-Einschätzung"
        c1_text = "Die App erklärt, was die Datenlage für dein Kapital bedeutet."
        c2_title = "Kapital-Schutz"
        c2_text = "Risiko, Verlusttoleranz und Anlagehorizont werden verständlich eingeordnet."
        c3_title = "Nächster Schritt"
        c3_text = "Die App gibt eine vorsichtige Orientierung, ohne eine Anlageberatung zu ersetzen."
    else:
        mode_text = "📊 Expertenansicht aktiv: Fokus auf Kennzahlen, Modellgrenzen, Datenstatus und Score-Zerlegung."
        c1_title = "Technische Kennzahlen"
        c1_text = "Volatilität, Drawdown, gleitende Durchschnitte und Trendindikatoren werden stärker sichtbar."
        c2_title = "Modelltransparenz"
        c2_text = "Das ML-Modell wird als Prototyp mit ca. 53,25 % Accuracy kritisch eingeordnet."
        c3_title = "Daten- und Newsstatus"
        c3_text = "Aktualität, News-Score und Datenbasis werden für die Bewertung explizit offengelegt."

    st.markdown(
        f"""
        <div class="mode-status-banner">
            <b>{theme_text}</b><br>
            {mode_text}
        </div>

        <div class="mode-content-panel">
            <h3>Aktive Darstellung</h3>
            <p>Die App passt ihre Erklärung und Gewichtung jetzt sichtbar an deine Auswahl an.</p>
            <div class="mode-content-grid">
                <div class="mode-content-card">
                    <h4>{c1_title}</h4>
                    <p>{c1_text}</p>
                </div>
                <div class="mode-content-card">
                    <h4>{c2_title}</h4>
                    <p>{c2_text}</p>
                </div>
                <div class="mode-content-card">
                    <h4>{c3_title}</h4>
                    <p>{c3_text}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



# =========================================================
# FINAL THEME + VIEW RENDERER
# =========================================================


def get_news_api_key():
    try:
        return st.secrets["NEWS_API_KEY"]
    except Exception:
        return None


def safe_to_datetime(value):
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def format_age_minutes(minutes):
    if minutes is None:
        return "unbekannt"
    if minutes < 60:
        return f"{minutes:.0f} Minuten"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} Stunden"
    return f"{hours / 24:.1f} Tage"


def format_age_hours(hours):
    if hours is None:
        return "unbekannt"
    if hours < 24:
        return f"{hours:.1f} Stunden"
    return f"{hours / 24:.1f} Tage"


def card_for_score(score):
    if score >= 75:
        return "good-card"
    if score >= 50:
        return "neutral-card"
    if score >= 30:
        return "warn-card"
    return "bad-card"


def label_for_confidence(score):
    if score >= 75:
        return "Hoch"
    if score >= 50:
        return "Mittel"
    if score >= 30:
        return "Niedrig bis mittel"
    return "Niedrig"


def render_mode_hint(mode):
    if mode == "Geführte Ansicht":
        st.caption("Geführte Ansicht: Klartext, Kapital-Schutz, nächste Schritte und wenige technische Begriffe.")
    else:
        st.caption("Expertenansicht: Technische Kennzahlen, Modellwerte, Score-Zerlegung und Datenstatus im Detail.")


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data(show_spinner="Aktuelle Marktdaten werden geladen...")
def load_market_data(ticker, period, interval):
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
    data = data.rename(columns={date_col: "date"})

    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    data = data.rename(columns=rename)

    needed = ["date", "open", "high", "low", "close", "volume"]
    existing = [c for c in needed if c in data.columns]
    data = data[existing].dropna()

    return data


@st.cache_data(show_spinner="Lokale Feature-Daten werden geladen...")
def load_local_features():
    if not LOCAL_FEATURES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOCAL_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"])


@st.cache_resource(show_spinner="ML-Modell wird geladen...")
def load_model_bundle():
    if joblib is None:
        return None
    if not LOCAL_MODEL_PATH.exists():
        return None
    try:
        return joblib.load(LOCAL_MODEL_PATH)
    except Exception:
        return None


def market_data_freshness(data, interval):
    if data is None or data.empty or "date" not in data.columns:
        return {
            "status": "Keine Daten",
            "card": "bad-card",
            "minutes_old": None,
            "last_timestamp": None,
            "score": 0,
            "explanation": "Es konnten keine Marktdaten geladen werden.",
        }

    last_ts = safe_to_datetime(data["date"].iloc[-1])
    if last_ts is None:
        return {
            "status": "Unbekannt",
            "card": "warn-card",
            "minutes_old": None,
            "last_timestamp": None,
            "score": 35,
            "explanation": "Zeitpunkt der letzten Kerze konnte nicht gelesen werden.",
        }

    now = dt.datetime.now(last_ts.tzinfo) if getattr(last_ts, "tzinfo", None) else dt.datetime.now()
    diff_minutes = max((now - last_ts).total_seconds() / 60, 0)

    if interval in ["1m", "5m", "15m", "30m", "1h"]:
        fresh_limit = 180
        stale_limit = 24 * 60
    elif interval in ["1d"]:
        fresh_limit = 3 * 24 * 60
        stale_limit = 10 * 24 * 60
    else:
        fresh_limit = 14 * 24 * 60
        stale_limit = 45 * 24 * 60

    if diff_minutes <= fresh_limit:
        status = "Aktuell"
        card = "good-card"
        score = 100
        explanation = "Die Marktdaten wirken für den gewählten Zeitraum frisch."
    elif diff_minutes <= stale_limit:
        status = "Leicht verzögert"
        card = "warn-card"
        score = 65
        explanation = "Die Daten sind nutzbar, aber nicht ganz frisch."
    else:
        status = "Veraltet"
        card = "bad-card"
        score = 25
        explanation = "Die Daten sind für einen aktuellen Outlook zu alt."

    return {
        "status": status,
        "card": card,
        "minutes_old": diff_minutes,
        "last_timestamp": last_ts,
        "score": score,
        "explanation": explanation,
    }


# =========================================================
# FEATURES / INDICATORS
# =========================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(data, period=14):
    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift()).abs()
    low_close = (data["low"] - data["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_macd(close):
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def prepare_live_features(data):
    df = data.copy().sort_values("date")
    df["daily_return"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["return_20d"] = df["close"].pct_change(20)

    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()
    df["ma_200"] = df["close"].rolling(200).mean()

    df["ma_20_distance"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["ma_50_distance"] = (df["close"] - df["ma_50"]) / df["ma_50"]
    df["ma_200_distance"] = (df["close"] - df["ma_200"]) / df["ma_200"]

    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    df["rolling_high"] = df["close"].cummax()
    df["drawdown"] = (df["close"] / df["rolling_high"]) - 1

    df["rsi"] = calculate_rsi(df["close"])
    df["atr"] = calculate_atr(df)

    macd, signal, hist = calculate_macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    return df.replace([np.inf, -np.inf], np.nan)


def trend_score(latest):
    score = 0
    reasons = []

    close = latest.get("close", np.nan)
    ma20 = latest.get("ma_20", np.nan)
    ma50 = latest.get("ma_50", np.nan)
    ma200 = latest.get("ma_200", np.nan)
    rsi = latest.get("rsi", np.nan)
    macd_hist = latest.get("macd_hist", np.nan)
    drawdown = latest.get("drawdown", np.nan)

    if pd.notna(close) and pd.notna(ma20):
        if close > ma20:
            score += 1
            reasons.append("Der Kurs liegt über dem kurzfristigen Durchschnitt.")
        else:
            score -= 1
            reasons.append("Der Kurs liegt unter dem kurzfristigen Durchschnitt.")

    if pd.notna(close) and pd.notna(ma50):
        if close > ma50:
            score += 1
            reasons.append("Der Kurs liegt über dem mittelfristigen Durchschnitt.")
        else:
            score -= 1
            reasons.append("Der Kurs liegt unter dem mittelfristigen Durchschnitt.")

    if pd.notna(close) and pd.notna(ma200):
        if close > ma200:
            score += 2
            reasons.append("Der Kurs liegt über dem langfristigen Durchschnitt.")
        else:
            score -= 2
            reasons.append("Der Kurs liegt unter dem langfristigen Durchschnitt.")

    if pd.notna(macd_hist):
        if macd_hist > 0:
            score += 1
            reasons.append("Das Momentum wirkt positiv.")
        else:
            score -= 1
            reasons.append("Das Momentum wirkt negativ.")

    if pd.notna(rsi):
        if rsi > 75:
            score -= 1
            reasons.append("Der Markt wirkt kurzfristig überhitzt.")
        elif rsi < 30:
            score += 1
            reasons.append("Der Markt wirkt kurzfristig stark gefallen.")
        else:
            reasons.append("Der RSI liegt im neutralen Bereich.")

    if pd.notna(drawdown):
        if drawdown < -0.20:
            score -= 1
            reasons.append("Der aktuelle Rückgang vom Hoch ist deutlich.")
        elif drawdown > -0.05:
            score += 1
            reasons.append("Der Rückgang vom Hoch ist aktuell eher gering.")

    if score >= 4:
        return {"label": "Eher positiv", "score": score, "card": "good-card", "reasons": reasons}
    if score <= -4:
        return {"label": "Eher negativ", "score": score, "card": "bad-card", "reasons": reasons}
    return {"label": "Gemischt / unklar", "score": score, "card": "warn-card", "reasons": reasons}


def detect_levels(df):
    if df.empty:
        return {}
    recent = df.tail(min(len(df), 252))
    return {
        "last_close": float(recent["close"].iloc[-1]),
        "one_year_high": float(recent["high"].max()),
        "one_year_low": float(recent["low"].min()),
        "recent_support": float(recent["low"].tail(min(60, len(recent))).min()),
        "recent_resistance": float(recent["high"].tail(min(60, len(recent))).max()),
    }


def max_drawdown(close):
    rolling_high = close.cummax()
    drawdown = close / rolling_high - 1
    return float(drawdown.min())


# =========================================================
# NEWS
# =========================================================

def build_news_query(ticker, asset_label, category):
    if "ETF" in asset_label:
        if ticker in ["SPY", "VOO", "VTI"]:
            return '"S&P 500" OR "US stock market" OR "Federal Reserve" OR inflation'
        if ticker == "QQQ":
            return '"Nasdaq 100" OR technology stocks OR Nvidia OR "Federal Reserve"'
        if ticker in ["EEM", "VWO"]:
            return '"emerging markets" OR China economy OR "US dollar" OR "global markets"'
        if ticker in ["AGG", "BND"]:
            return 'bonds OR "Treasury yields" OR "Federal Reserve" OR interest rates'
        if ticker == "GLD":
            return 'gold OR inflation OR "US dollar" OR "central banks"'
    return f'"{ticker}" stock OR "{category}" OR market OR earnings'


@st.cache_data(ttl=900, show_spinner="Aktuelle Nachrichten werden geladen...")
def fetch_news(query, api_key, language="en", page_size=8):
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
                "message": f"NewsAPI Fehler {response.status_code}: {response.text[:200]}",
            }
        data = response.json()
        return {"status": "ok", "articles": data.get("articles", []), "message": f"{data.get('totalResults', 0)} Treffer gefunden."}
    except Exception as exc:
        return {"status": "error", "articles": [], "message": str(exc)}


def parse_news_time(article):
    raw = article.get("publishedAt")
    if not raw:
        return None
    try:
        return pd.to_datetime(raw).to_pydatetime()
    except Exception:
        return None


def news_freshness(articles):
    if not articles:
        return {
            "status": "Keine News",
            "card": "warn-card",
            "hours_old": None,
            "latest_timestamp": None,
            "score": 35,
            "explanation": "Es wurden keine aktuellen News geladen.",
        }

    times = [parse_news_time(a) for a in articles]
    times = [t for t in times if t is not None]

    if not times:
        return {
            "status": "Unbekannt",
            "card": "warn-card",
            "hours_old": None,
            "latest_timestamp": None,
            "score": 40,
            "explanation": "News-Zeitpunkte konnten nicht gelesen werden.",
        }

    latest = max(times)
    now = dt.datetime.now(latest.tzinfo) if getattr(latest, "tzinfo", None) else dt.datetime.now()
    hours_old = max((now - latest).total_seconds() / 3600, 0)

    if hours_old <= 6:
        return {"status": "Sehr aktuell", "card": "good-card", "hours_old": hours_old, "latest_timestamp": latest, "score": 100, "explanation": "News-Lage ist sehr aktuell."}
    if hours_old <= 24:
        return {"status": "Aktuell", "card": "good-card", "hours_old": hours_old, "latest_timestamp": latest, "score": 80, "explanation": "News-Lage ist aktuell."}
    if hours_old <= 72:
        return {"status": "Mittel", "card": "warn-card", "hours_old": hours_old, "latest_timestamp": latest, "score": 55, "explanation": "News sind nicht mehr ganz frisch."}
    return {"status": "Alt", "card": "bad-card", "hours_old": hours_old, "latest_timestamp": latest, "score": 25, "explanation": "News sind alt."}


def score_news(articles):
    if not articles:
        return {"score": 0, "label": "Keine starke News-Lage", "drivers": [], "confidence": "Niedrig"}

    blob = " ".join([str(a.get("title", "")) + " " + str(a.get("description", "")) for a in articles]).lower()

    positive_terms = [
        "rally", "gains", "beats expectations", "strong earnings", "cuts rates",
        "rate cuts", "inflation cools", "soft landing", "growth", "bullish",
    ]
    negative_terms = [
        "falls", "drops", "selloff", "recession", "inflation rises", "higher rates",
        "misses expectations", "warning", "lawsuit", "bearish", "risk",
    ]

    pos_hits = [x for x in positive_terms if x in blob]
    neg_hits = [x for x in negative_terms if x in blob]

    score = len(pos_hits) - len(neg_hits)
    drivers = [f"positiv: {x}" for x in pos_hits[:4]] + [f"negativ: {x}" for x in neg_hits[:4]]

    if score >= 3:
        label = "News-Kontext eher positiv"
    elif score <= -3:
        label = "News-Kontext eher negativ"
    elif score > 0:
        label = "News-Kontext leicht positiv"
    elif score < 0:
        label = "News-Kontext leicht negativ"
    else:
        label = "News-Kontext neutral"

    confidence = "Mittel" if abs(score) >= 3 else "Niedrig"
    return {"score": score, "label": label, "drivers": drivers, "confidence": confidence}


def render_news_cards(articles, max_articles=6):
    if not articles:
        st.info("Keine News verfügbar oder kein NEWS_API_KEY gesetzt.")
        return

    for article in articles[:max_articles]:
        title = article.get("title") or "Ohne Titel"
        source = (article.get("source") or {}).get("name", "Unbekannte Quelle")
        published = article.get("publishedAt", "")
        desc = article.get("description") or "Keine Kurzbeschreibung verfügbar."
        url = article.get("url") or ""

        st.markdown(
            f"""
            <div class="news-card">
            <h4>{title}</h4>
            <p><b>Quelle:</b> {source} &nbsp; | &nbsp; <b>Zeit:</b> {published}</p>
            <p>{desc}</p>
            <p><a href="{url}" target="_blank">Mehr dazu lesen</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )



# =========================================================
# ADVANCED NEWS INTELLIGENCE
# =========================================================

NEWS_CATEGORIES = {
    "Zinsen / Notenbanken": [
        "federal reserve", "fed", "ecb", "central bank", "interest rate",
        "rate cut", "rate hike", "monetary policy", "powell", "lagarde"
    ],
    "Inflation": [
        "inflation", "cpi", "ppi", "prices", "deflation", "consumer prices"
    ],
    "Unternehmenszahlen": [
        "earnings", "revenue", "profit", "guidance", "beats expectations",
        "misses expectations", "quarterly results"
    ],
    "Rezession / Makro": [
        "recession", "gdp", "slowdown", "unemployment", "jobs report",
        "labor market", "consumer confidence"
    ],
    "Geopolitik / Risiko": [
        "war", "sanctions", "geopolitical", "conflict", "oil shock",
        "tariffs", "trade war"
    ],
    "Technologie / KI": [
        "artificial intelligence", "ai", "semiconductor", "chips",
        "cloud", "technology", "nvidia"
    ],
    "Anleihen / Renditen": [
        "treasury yields", "bond yields", "yields rise", "yields fall",
        "bond market", "treasuries"
    ],
    "Währung / Dollar": [
        "dollar", "us dollar", "currency", "forex", "exchange rate"
    ],
}

POSITIVE_NEWS_TERMS = [
    "rally", "gains", "surges", "jumps", "beats expectations", "strong earnings",
    "record high", "optimism", "soft landing", "inflation cools", "rate cuts",
    "growth", "upgrade", "bullish", "outperform", "resilient"
]

NEGATIVE_NEWS_TERMS = [
    "falls", "drops", "selloff", "plunges", "misses expectations", "warning",
    "recession", "inflation rises", "higher rates", "lawsuit", "downgrade",
    "bearish", "risk", "uncertainty", "weak demand", "slowdown", "cuts guidance"
]

HIGH_IMPACT_TERMS = [
    "federal reserve", "fed", "ecb", "inflation", "cpi", "interest rate",
    "earnings", "recession", "war", "sanctions", "treasury yields",
    "jobs report", "gdp", "guidance"
]


def classify_news_article(article, ticker, asset_type, category):
    title = str(article.get("title") or "")
    description = str(article.get("description") or "")
    source = (article.get("source") or {}).get("name", "Unbekannte Quelle")
    url = article.get("url") or ""
    published = article.get("publishedAt") or ""

    blob = f"{title} {description}".lower()

    pos_hits = [term for term in POSITIVE_NEWS_TERMS if term in blob]
    neg_hits = [term for term in NEGATIVE_NEWS_TERMS if term in blob]
    high_hits = [term for term in HIGH_IMPACT_TERMS if term in blob]

    category_hits = {}
    for cat, terms in NEWS_CATEGORIES.items():
        hits = [term for term in terms if term in blob]
        if hits:
            category_hits[cat] = hits

    raw_score = len(pos_hits) - len(neg_hits)

    if raw_score >= 2:
        sentiment = "Positiv"
        sentiment_score = 1
    elif raw_score <= -2:
        sentiment = "Negativ"
        sentiment_score = -1
    elif raw_score == 1:
        sentiment = "Leicht positiv"
        sentiment_score = 0.5
    elif raw_score == -1:
        sentiment = "Leicht negativ"
        sentiment_score = -0.5
    else:
        sentiment = "Neutral"
        sentiment_score = 0

    relevance_score = 0

    if ticker.lower() in blob:
        relevance_score += 3

    if asset_type == "ETF":
        if any(x in blob for x in ["market", "s&p 500", "nasdaq", "stocks", "economy", "fed", "inflation"]):
            relevance_score += 2
    else:
        if any(x in blob for x in ["earnings", "revenue", "stock", "guidance", category.lower()]):
            relevance_score += 2

    relevance_score += min(len(category_hits), 3)
    relevance_score += min(len(high_hits), 2)

    if relevance_score >= 5:
        relevance = "Hoch"
    elif relevance_score >= 3:
        relevance = "Mittel"
    else:
        relevance = "Niedrig"

    impact_score = len(high_hits) + len(category_hits)

    if impact_score >= 4:
        impact = "Hoch"
    elif impact_score >= 2:
        impact = "Mittel"
    else:
        impact = "Niedrig"

    if category_hits:
        main_category = sorted(category_hits.items(), key=lambda x: len(x[1]), reverse=True)[0][0]
    else:
        main_category = "Allgemeiner Markt"

    if sentiment in ["Negativ", "Leicht negativ"] and impact in ["Mittel", "Hoch"]:
        beginner_text = "Diese Nachricht kann das Risiko erhöhen. Für Anfänger bedeutet das: nicht überstürzt handeln."
    elif sentiment in ["Positiv", "Leicht positiv"] and impact in ["Mittel", "Hoch"]:
        beginner_text = "Diese Nachricht kann das Umfeld verbessern, ersetzt aber keine eigene Risikoprüfung."
    elif relevance == "Niedrig":
        beginner_text = "Diese Nachricht ist wahrscheinlich nur begrenzt relevant für die aktuelle Entscheidung."
    else:
        beginner_text = "Die Nachricht liefert Kontext, ist aber alleine nicht entscheidend."

    return {
        "title": title,
        "description": description,
        "source": source,
        "url": url,
        "published": published,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "impact": impact,
        "relevance": relevance,
        "relevance_score": relevance_score,
        "main_category": main_category,
        "category_hits": category_hits,
        "positive_hits": pos_hits,
        "negative_hits": neg_hits,
        "high_impact_hits": high_hits,
        "beginner_text": beginner_text,
    }


def analyze_news_intelligence(articles, ticker, asset_type, category):
    if not articles:
        return {
            "articles": [],
            "summary_score": 0,
            "summary_label": "Keine News verfügbar",
            "risk_label": "Unbekannt",
            "main_driver": "Keine",
            "impact_label": "Unbekannt",
            "beginner_summary": "Es wurden keine aktuellen News ausgewertet.",
            "category_summary": {},
        }

    analyzed = [
        classify_news_article(article, ticker, asset_type, category)
        for article in articles
    ]

    weighted_scores = []
    category_counter = {}

    for item in analyzed:
        relevance_weight = {"Hoch": 1.5, "Mittel": 1.0, "Niedrig": 0.4}.get(item["relevance"], 0.5)
        impact_weight = {"Hoch": 1.5, "Mittel": 1.0, "Niedrig": 0.6}.get(item["impact"], 0.6)
        weighted_scores.append(item["sentiment_score"] * relevance_weight * impact_weight)

        category_counter[item["main_category"]] = category_counter.get(item["main_category"], 0) + 1

    summary_score = sum(weighted_scores)

    if summary_score >= 3:
        summary_label = "News-Lage positiv"
        risk_label = "Risikoumfeld eher freundlicher"
    elif summary_score <= -3:
        summary_label = "News-Lage negativ"
        risk_label = "Risikoumfeld angespannt"
    elif summary_score > 0.75:
        summary_label = "News-Lage leicht positiv"
        risk_label = "Risikoumfeld leicht verbessert"
    elif summary_score < -0.75:
        summary_label = "News-Lage leicht negativ"
        risk_label = "Risikoumfeld leicht angespannt"
    else:
        summary_label = "News-Lage gemischt"
        risk_label = "Keine klare Richtung"

    main_driver = max(category_counter, key=category_counter.get) if category_counter else "Allgemeiner Markt"

    high_impact_count = sum(1 for item in analyzed if item["impact"] == "Hoch")
    if high_impact_count >= 3:
        impact_label = "Hoch"
    elif high_impact_count >= 1:
        impact_label = "Mittel"
    else:
        impact_label = "Niedrig"

    if "negativ" in summary_label.lower():
        beginner_summary = "Die Nachrichtenlage spricht aktuell eher für Vorsicht. Für nicht-technische Nutzer bedeutet das: keine schnelle Entscheidung erzwingen."
    elif "positiv" in summary_label.lower():
        beginner_summary = "Die Nachrichtenlage wirkt unterstützend, sollte aber nur zusammen mit Daten, Risiko und Zeithorizont bewertet werden."
    else:
        beginner_summary = "Die Nachrichtenlage ist nicht eindeutig. Das spricht eher für Beobachten und Simulieren statt sofortiger Entscheidung."

    return {
        "articles": analyzed,
        "summary_score": round(summary_score, 2),
        "summary_label": summary_label,
        "risk_label": risk_label,
        "main_driver": main_driver,
        "impact_label": impact_label,
        "beginner_summary": beginner_summary,
        "category_summary": category_counter,
    }


def render_news_intelligence_summary(news_intel, mode):
    score = news_intel["summary_score"]
    if score >= 2:
        card = "good-card"
    elif score <= -2:
        card = "bad-card"
    elif abs(score) >= 0.75:
        card = "warn-card"
    else:
        card = "neutral-card"

    st.markdown(
        f"""
        <div class="info-card {card}">
        <h3>News Intelligence</h3>
        <div class="big-number">{news_intel['summary_label']}</div>
        <p><b>News Score:</b> {news_intel['summary_score']} | <b>Wichtigster Treiber:</b> {news_intel['main_driver']} | <b>Impact:</b> {news_intel['impact_label']}</p>
        <p>{news_intel['beginner_summary']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if mode == "Expertenansicht":
        if news_intel["category_summary"]:
            cat_df = pd.DataFrame(
                [{"Kategorie": k, "Anzahl": v} for k, v in news_intel["category_summary"].items()]
            ).sort_values("Anzahl", ascending=False)

            fig = px.bar(cat_df, x="Kategorie", y="Anzahl", text="Anzahl", title="News-Kategorien")
            fig.update_traces(textposition="outside")
            fig = apply_chart_theme(fig, theme_mode)
            render_interactive_chart(fig, data=None, key="interactive_chart_1")


def render_analyzed_news_cards(news_intel, mode, max_articles=8):
    analyzed = news_intel.get("articles", [])

    if not analyzed:
        st.info("Keine News verfügbar oder kein NEWS_API_KEY gesetzt.")
        return

    visible = analyzed[:max_articles]

    for item in visible:
        if item["sentiment"] in ["Positiv", "Leicht positiv"]:
            card = "good-card"
        elif item["sentiment"] in ["Negativ", "Leicht negativ"]:
            card = "bad-card"
        elif item["impact"] == "Hoch":
            card = "warn-card"
        else:
            card = "neutral-card"

        st.markdown(
            f"""
            <div class="news-card {card}">
            <h4>{item['title']}</h4>
            <p><b>Quelle:</b> {item['source']} &nbsp; | &nbsp; <b>Zeit:</b> {item['published']}</p>
            <p><b>Bewertung:</b> {item['sentiment']} | <b>Impact:</b> {item['impact']} | <b>Relevanz:</b> {item['relevance']} | <b>Kategorie:</b> {item['main_category']}</p>
            <p>{item['description']}</p>
            <p><b>Einfach erklärt:</b> {item['beginner_text']}</p>
            <p><a href="{item['url']}" target="_blank">Quelle öffnen</a></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if mode == "Expertenansicht":
            with st.expander("Technische News-Treffer anzeigen"):
                st.write("Positive Treffer:", item["positive_hits"])
                st.write("Negative Treffer:", item["negative_hits"])
                st.write("High-Impact Treffer:", item["high_impact_hits"])
                st.write("Kategorie-Treffer:", item["category_hits"])


# =========================================================
# MODEL
# =========================================================

def predict_with_local_model(ticker, live_feature_df):
    bundle = load_model_bundle()
    if bundle is None or live_feature_df.empty:
        return None

    model = bundle.get("model")
    feature_cols = bundle.get("feature_cols")
    accuracy = bundle.get("accuracy")

    latest = live_feature_df.dropna(subset=feature_cols).tail(1)
    if latest.empty:
        return None

    try:
        proba = model.predict_proba(latest[feature_cols])[0]
        classes = list(model.classes_)
        p_up = float(proba[classes.index(1)]) if 1 in classes else None
        prediction = int(model.predict(latest[feature_cols])[0])
        return {
            "probability_up": p_up,
            "prediction": prediction,
            "accuracy": accuracy,
            "feature_cols": feature_cols,
        }
    except Exception:
        return None


# =========================================================
# SCORING
# =========================================================

def build_wealth_outlook(technical_score, news_score, volatility_20d, drawdown, model_prob_up, market_fresh, news_fresh):
    combined = technical_score + news_score

    if model_prob_up is not None:
        if model_prob_up >= 0.60:
            combined += 2
        elif model_prob_up <= 0.40:
            combined -= 2

    vol_penalty = 0
    vol_note = "Volatilität wirkt moderat."
    if pd.notna(volatility_20d):
        annualized = volatility_20d * np.sqrt(252)
        if annualized >= 0.35:
            vol_penalty = -2
            vol_note = "Volatilität ist hoch. Risiko steigt deutlich."
        elif annualized >= 0.22:
            vol_penalty = -1
            vol_note = "Volatilität ist erhöht."

    dd_penalty = 0
    dd_note = "Drawdown wirkt nicht extrem."
    if pd.notna(drawdown):
        if drawdown <= -0.25:
            dd_penalty = -2
            dd_note = "Der aktuelle Rückgang vom Hoch ist deutlich."
        elif drawdown <= -0.12:
            dd_penalty = -1
            dd_note = "Der Rückgang vom Hoch ist spürbar."

    adjusted = combined + vol_penalty + dd_penalty

    if adjusted >= 5:
        outlook = "Konstruktiv, aber kontrolliert prüfen"
        card = "good-card"
        action = "Nicht blind investieren. Schrittweise analysieren oder Sparplan/Simulation prüfen."
    elif adjusted <= -5:
        outlook = "Defensiv bleiben"
        card = "bad-card"
        action = "Kein übereilter Einstieg. Kapital schützen, News lesen, später erneut prüfen."
    elif adjusted >= 2:
        outlook = "Leicht konstruktiv, aber unsicher"
        card = "neutral-card"
        action = "Nur mit kleiner Gewichtung oder Simulation prüfen."
    elif adjusted <= -2:
        outlook = "Leicht negativ, Vorsicht"
        card = "warn-card"
        action = "Beobachten, nicht erzwingen. Risiko und Zeithorizont prüfen."
    else:
        outlook = "Unklar / kein klarer Vorteil"
        card = "warn-card"
        action = "Keine Entscheidung erzwingen. Erst verstehen, vergleichen und simulieren."

    return {
        "outlook": outlook,
        "card": card,
        "adjusted_score": adjusted,
        "action": action,
        "vol_note": vol_note,
        "drawdown_note": dd_note,
    }


def compute_confidence(technical_score, news_score, market_fresh, news_fresh, model_accuracy, model_prob_up, volatility_20d):
    score = 0
    components = []

    tech_points = min(abs(technical_score) / 7, 1) * 25
    score += tech_points
    components.append({"Faktor": "Technische Eindeutigkeit", "Punkte": round(tech_points, 1)})

    news_points = min(abs(news_score) / 5, 1) * 18
    score += news_points
    components.append({"Faktor": "News-Eindeutigkeit", "Punkte": round(news_points, 1)})

    market_points = (market_fresh.get("score", 0) / 100) * 20
    score += market_points
    components.append({"Faktor": "Marktdaten-Aktualität", "Punkte": round(market_points, 1)})

    news_fresh_points = (news_fresh.get("score", 0) / 100) * 15
    score += news_fresh_points
    components.append({"Faktor": "News-Aktualität", "Punkte": round(news_fresh_points, 1)})

    if model_accuracy is not None:
        model_points = max(min((model_accuracy - 0.50) / 0.20, 1), 0) * 12
    else:
        model_points = 0
    score += model_points
    components.append({"Faktor": "Historische ML-Accuracy", "Punkte": round(model_points, 1)})

    if model_prob_up is not None:
        model_signal_points = min(abs(model_prob_up - 0.5) / 0.25, 1) * 5
    else:
        model_signal_points = 0
    score += model_signal_points
    components.append({"Faktor": "ML-Signalstärke", "Punkte": round(model_signal_points, 1)})

    vol_penalty = 0
    if pd.notna(volatility_20d):
        annualized = volatility_20d * np.sqrt(252)
        if annualized >= 0.35:
            vol_penalty = -15
        elif annualized >= 0.22:
            vol_penalty = -8

    score += vol_penalty
    components.append({"Faktor": "Volatilitätsabschlag", "Punkte": round(vol_penalty, 1)})

    score = max(0, min(100, round(score)))
    return {
        "score": score,
        "label": label_for_confidence(score),
        "card": card_for_score(score),
        "components": components,
    }


def capital_protection_score(capital, planned_allocation_pct, confidence_score, drawdown, volatility_20d, asset_type):
    score = 100
    reasons = []

    if planned_allocation_pct <= 5:
        reasons.append("Geplante Gewichtung ist sehr klein.")
    elif planned_allocation_pct <= 15:
        score -= 8
        reasons.append("Geplante Gewichtung ist moderat.")
    elif planned_allocation_pct <= 30:
        score -= 22
        reasons.append("Geplante Gewichtung ist hoch.")
    else:
        score -= 40
        reasons.append("Geplante Gewichtung ist sehr hoch.")

    if asset_type == "Stock":
        score -= 15
        reasons.append("Einzelaktien haben Konzentrationsrisiko.")
    else:
        reasons.append("ETF-Struktur reduziert Einzeltitelrisiko.")

    if confidence_score < 30:
        score -= 25
        reasons.append("Confidence ist niedrig.")
    elif confidence_score < 50:
        score -= 15
        reasons.append("Confidence ist niedrig bis mittel.")
    elif confidence_score < 75:
        score -= 5
        reasons.append("Confidence ist mittel.")

    if pd.notna(drawdown):
        if drawdown <= -0.25:
            score -= 20
            reasons.append("Aktueller Drawdown ist deutlich.")
        elif drawdown <= -0.12:
            score -= 10
            reasons.append("Aktueller Drawdown ist spürbar.")

    if pd.notna(volatility_20d):
        ann_vol = volatility_20d * np.sqrt(252)
        if ann_vol >= 0.35:
            score -= 20
            reasons.append("Volatilität ist hoch.")
        elif ann_vol >= 0.22:
            score -= 10
            reasons.append("Volatilität ist erhöht.")

    score = max(0, min(100, round(score)))

    if score >= 80:
        label = "Kapital-schonend"
        card = "good-card"
        action = "Eher geeignet für Lernen, Beobachten und kleine Simulationen."
    elif score >= 60:
        label = "Kontrolliert prüfen"
        card = "neutral-card"
        action = "Nur mit klarer Gewichtung und langfristigem Plan prüfen."
    elif score >= 40:
        label = "Vorsichtig"
        card = "warn-card"
        action = "Für unerfahrene Nutzer nur beobachten oder simulieren."
    else:
        label = "Zu riskant"
        card = "bad-card"
        action = "Nicht als erste Handlung geeignet. Kapital schützen."

    return {"score": score, "label": label, "card": card, "action": action, "reasons": reasons}



# =========================================================
# PREMIUM PRODUCT FEATURES
# =========================================================

def traffic_light_status(confidence_score, protection_score, outlook_score):
    if protection_score < 40 or confidence_score < 30:
        return {
            "label": "Nicht handeln",
            "badge": "badge-red",
            "emoji": "🔴",
            "explanation": "Das Szenario ist aktuell zu unsicher oder zu riskant."
        }

    if protection_score < 65 or confidence_score < 55 or outlook_score < 0:
        return {
            "label": "Nur simulieren",
            "badge": "badge-yellow",
            "emoji": "🟡",
            "explanation": "Das Szenario ist interessant, aber nicht belastbar genug für eine echte Entscheidung."
        }

    return {
        "label": "Kontrolliert prüfen",
        "badge": "badge-green",
        "emoji": "🟢",
        "explanation": "Das Szenario wirkt strukturierter, bleibt aber trotzdem unsicher."
    }


def render_premium_status_strip(ticker, asset_type, traffic, confidence, protection, news_intel, market_fresh, news_fresh):
    st.markdown(
        f"""
        <div class="premium-strip">
            <div class="section-label">Premium Status</div>
            <span class="status-badge {traffic['badge']}">{traffic['emoji']} {traffic['label']}</span>
            <span class="status-badge badge-blue">Asset: {ticker}</span>
            <span class="status-badge badge-blue">Typ: {asset_type}</span>
            <span class="status-badge badge-blue">Confidence: {confidence['score']} / 100</span>
            <span class="status-badge badge-blue">Kapital-Schutz: {protection['score']} / 100</span>
            <span class="status-badge badge-blue">News: {news_intel['summary_label']}</span>
            <p style="margin-top:0.75rem; margin-bottom:0;">{traffic['explanation']} Marktdaten: <b>{market_fresh['status']}</b>. News: <b>{news_fresh['status']}</b>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_progress(title, score, help_text):
    safe_score = int(max(0, min(100, float(score)))) if score is not None else 0
    icon = info_icon(title) if title in GLOSSARY else ""
    st.markdown(f"**{title}** {icon}", unsafe_allow_html=True)
    st.progress(safe_score)
    st.caption(help_text)


def advisor_explanation(outlook, confidence, protection, news_intel, asset_type, allocation_pct):
    if protection["score"] < 40:
        start = "Wenn Kapitalerhalt im Vordergrund steht, wäre ich hier aktuell defensiv."
    elif confidence["score"] < 50:
        start = "Ich würde dieses Szenario aktuell nicht als belastbare Entscheidung betrachten."
    elif outlook["adjusted_score"] > 2:
        start = "Das Szenario wirkt grundsätzlich interessant, aber nur kontrolliert."
    else:
        start = "Ich würde hier keine Entscheidung erzwingen."

    concentration = "Bei einer Einzelaktie ist das Konzentrationsrisiko höher." if asset_type == "Stock" else "Ein ETF reduziert Einzeltitelrisiken, kann aber trotzdem schwanken."

    allocation = (
        "Die geplante Gewichtung ist hoch und sollte reduziert oder auf mehrere Bausteine verteilt werden."
        if allocation_pct > 25 else
        "Die geplante Gewichtung wirkt eher kontrolliert, sofern sie zum Gesamtplan passt."
    )

    return (
        f"{start} Der Wealth Outlook lautet: {outlook['outlook']}. "
        f"Die Confidence liegt bei {confidence['score']} von 100, der Kapital-Schutz bei {protection['score']} von 100. "
        f"Die News-Lage wird als '{news_intel['summary_label']}' bewertet, wichtigster Treiber ist '{news_intel['main_driver']}'. "
        f"{concentration} {allocation} Sinnvoll wäre: erst dokumentieren, simulieren und nur bei klarem Zeithorizont weiter prüfen."
    )


def render_advisor_box(text_value):
    st.markdown(
        f"""
        <div class="advisor-box">
        <h3>Advisor Explanation {info_icon("Decision Passport")}</h3>
        <p>{text_value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_risk_profile(horizon_years, loss_tolerance, experience, liquidity_need, goal):
    score = 0

    if horizon_years >= 10:
        score += 25
    elif horizon_years >= 5:
        score += 18
    elif horizon_years >= 3:
        score += 10
    else:
        score += 3

    score += min(loss_tolerance, 40)

    if experience == "Anfänger":
        score -= 10
    elif experience == "Fortgeschritten":
        score += 5
    else:
        score += 10

    if liquidity_need == "Ja, wahrscheinlich":
        score -= 20
    elif liquidity_need == "Vielleicht":
        score -= 10

    if goal == "Kapital erhalten":
        score -= 10
    elif goal == "Ausgewogen wachsen":
        score += 5
    else:
        score += 12

    score = max(0, min(100, score))

    if score < 35:
        label = "Defensiv"
        explanation = "Kapitalerhalt, Liquidität und breite Streuung sollten im Vordergrund stehen."
        card = "bad-card"
    elif score < 60:
        label = "Ausgewogen"
        explanation = "Eine Mischung aus defensiven und wachstumsorientierten Bausteinen wirkt plausibel."
        card = "warn-card"
    elif score < 80:
        label = "Wachstumsorientiert"
        explanation = "Mehr Schwankung kann tragbar sein, wenn Zeithorizont und Liquidität passen."
        card = "neutral-card"
    else:
        label = "Offensiv"
        explanation = "Hohe Schwankungen können akzeptiert werden, aber Klumpenrisiken bleiben kritisch."
        card = "good-card"

    return {"score": score, "label": label, "explanation": explanation, "card": card}


def simulate_what_if(capital, allocation_pct, drop_pct, monthly_saving=0, years=0, annual_return=0.05):
    allocation_amount = capital * allocation_pct / 100
    loss_on_position = allocation_amount * drop_pct / 100
    portfolio_after_drop = capital - loss_on_position

    future_value = None
    if years > 0:
        months = years * 12
        monthly_return = (1 + annual_return) ** (1 / 12) - 1
        lump = capital * ((1 + annual_return) ** years)
        savings = 0
        for m in range(int(months)):
            savings = (savings + monthly_saving) * (1 + monthly_return)
        future_value = lump + savings

    return {
        "allocation_amount": allocation_amount,
        "loss_on_position": loss_on_position,
        "portfolio_after_drop": portfolio_after_drop,
        "future_value": future_value,
    }


def portfolio_diversification_score(weights):
    # weights: dict category -> percentage
    if not weights:
        return {"score": 0, "label": "Unbekannt", "card": "warn-card", "notes": ["Keine Gewichtungen vorhanden."]}

    total = sum(weights.values())
    if total <= 0:
        return {"score": 0, "label": "Unbekannt", "card": "warn-card", "notes": ["Gesamtgewichtung ist 0."]}

    normalized = {k: v / total for k, v in weights.items()}
    max_weight = max(normalized.values())
    categories = len([v for v in normalized.values() if v > 0.01])

    score = 100
    notes = []

    if max_weight > 0.70:
        score -= 45
        notes.append("Sehr starke Konzentration in einem Bereich.")
    elif max_weight > 0.50:
        score -= 30
        notes.append("Hohe Konzentration in einem Bereich.")
    elif max_weight > 0.35:
        score -= 15
        notes.append("Spürbare Schwerpunktbildung.")
    else:
        notes.append("Keine extreme Einzelkonzentration sichtbar.")

    if categories >= 4:
        score += 5
        notes.append("Mehrere Anlageklassen/Sektoren vorhanden.")
    elif categories <= 2:
        score -= 20
        notes.append("Nur wenige Bausteine. Diversifikation ausbaufähig.")

    score = max(0, min(100, round(score)))

    if score >= 80:
        label = "Gut diversifiziert"
        card = "good-card"
    elif score >= 60:
        label = "Ordentlich diversifiziert"
        card = "neutral-card"
    elif score >= 40:
        label = "Einseitig"
        card = "warn-card"
    else:
        label = "Stark konzentriert"
        card = "bad-card"

    return {"score": score, "label": label, "card": card, "notes": notes}


def render_news_risk_radar(news_intel):
    categories = news_intel.get("category_summary", {})
    if not categories:
        st.info("Keine ausreichenden News-Kategorien für ein Risiko-Radar verfügbar.")
        return

    all_categories = list(NEWS_CATEGORIES.keys())
    values = [categories.get(cat, 0) for cat in all_categories]

    radar_df = pd.DataFrame({"Kategorie": all_categories, "Relevanz": values})

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=all_categories,
            fill="toself",
            name="News-Risiko-Radar",
            hovertemplate="<b>%{theta}</b><br>Treffer: %{r}<extra></extra>",
        )
    )

    fig.update_layout(
        title="News-Risiko-Radar",
        polar=dict(radialaxis=dict(visible=True, rangemode="tozero")),
        height=520,
    )

    fig = apply_chart_theme(fig, theme_mode)

    render_interactive_chart(fig, data=None, key="interactive_chart_2")

    with st.expander("Radar-Daten anzeigen"):
        st.dataframe(radar_df, width='stretch')


def current_update_timer(market_fresh, news_fresh):
    market_age = format_age_minutes(market_fresh.get("minutes_old"))
    news_age = format_age_hours(news_fresh.get("hours_old"))

    if market_fresh.get("status") == "Veraltet" or news_fresh.get("status") in ["Alt", "Keine News"]:
        recommendation = "Jetzt aktualisieren oder Datenquelle prüfen."
        card = "bad-card"
    elif market_fresh.get("status") == "Leicht verzögert" or news_fresh.get("status") == "Mittel":
        recommendation = "Vor einer Entscheidung erneut prüfen."
        card = "warn-card"
    else:
        recommendation = "Aktualität wirkt ausreichend."
        card = "good-card"

    return {
        "market_age": market_age,
        "news_age": news_age,
        "recommendation": recommendation,
        "card": card,
        "generated_at": dt.datetime.now().strftime("%d.%m.%Y %H:%M"),
    }


def render_update_timer(timer):
    icon = info_icon("Confidence") if "Confidence" in GLOSSARY else ""
    st.markdown(
        f"""
        <div class="info-card {timer['card']}">
        <h3>Aktualitäts-Timer {icon}</h3>
        <p><b>Outlook berechnet:</b> {timer['generated_at']}</p>
        <p><b>Marktdaten-Alter:</b> {timer['market_age']} | <b>News-Alter:</b> {timer['news_age']}</p>
        <p><b>Empfehlung:</b> {timer['recommendation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def save_journal_entry(entry):
    journal_path = Path("data/journal/decision_journal.csv")
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([entry])
    if journal_path.exists():
        old = pd.read_csv(journal_path)
        out = pd.concat([old, df_new], ignore_index=True)
    else:
        out = df_new

    out.to_csv(journal_path, index=False)
    return journal_path


def load_journal():
    journal_path = Path("data/journal/decision_journal.csv")
    if not journal_path.exists():
        return pd.DataFrame()
    return pd.read_csv(journal_path)


# =========================================================
# CHARTS / REPORT
# =========================================================

def create_price_chart(df, ticker):
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
        )
    )

    for col, label in [("ma_20", "MA 20"), ("ma_50", "MA 50"), ("ma_200", "MA 200")]:
        if col in df.columns and df[col].notna().sum() > 0:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=label, mode="lines"))

    fig.update_layout(
        title=f"{ticker} Kursverlauf",
        xaxis_title="Datum",
        yaxis_title="Preis",
        height=580,
        xaxis_rangeslider_visible=False,
    )
    return fig


def build_decision_passport(ticker, asset_label, capital, allocation_pct, latest_price, outlook, confidence, protection, market_fresh, news_fresh, model_result):
    amount = capital * allocation_pct / 100

    model_line = "Nicht verfügbar"
    if model_result is not None and model_result.get("probability_up") is not None:
        model_line = f"{model_result['probability_up']:.1%} Wahrscheinlichkeit für höheren Kurs in 20 Handelstagen laut Modell"

    return f"""# WealthScope AI Decision Passport

**Asset:** {asset_label}  
**Ticker:** {ticker}  
**Zeitpunkt:** {dt.datetime.now().strftime('%d.%m.%Y %H:%M')}  

## 1. Kapital-Kontext

- Kapital: {capital:,.2f}
- Geplante Gewichtung: {allocation_pct:.1f} %
- Geplanter Betrag: {amount:,.2f}
- Letzter Preis: {latest_price:.2f}

## 2. Wealth Outlook

- Ausblick: {outlook['outlook']}
- Outlook Score: {outlook['adjusted_score']}
- Nächster Schritt: {outlook['action']}

## 3. Confidence

- Confidence Score: {confidence['score']} / 100
- Einordnung: {confidence['label']}
- Hinweis: Confidence ist keine Trefferwahrscheinlichkeit.

## 4. Kapital-Schutz

- Kapital-Schutz-Score: {protection['score']} / 100
- Einordnung: {protection['label']}
- Handlung: {protection['action']}

## 5. Aktualität

- Marktdaten: {market_fresh['status']}
- News: {news_fresh['status']}

## 6. ML-Modell

- Modellsignal: {model_line}

## 7. Hinweis

Diese Auswertung ist keine Finanzberatung, keine Handelsaufforderung und keine Garantie. Sie dient Analyse, Lernen und Simulation.
"""



# =========================================================
# COOKIE / PRIVACY NOTICE
# =========================================================

def render_cookie_notice():
    if "cookie_consent" not in st.session_state:
        st.session_state.cookie_consent = False

    if not st.session_state.cookie_consent:
        st.markdown(
            """
            <div class="info-card neutral-card">
            <h3>Cookie- & Datenschutz-Hinweis</h3>
            <p>
            Diese lokale Demo-App speichert keine Tracking-Cookies und verkauft keine Daten.
            Eingaben werden nur während der lokalen Nutzung verarbeitet. Journal-Einträge werden lokal im Projektordner gespeichert.
            Externe Dienste können genutzt werden, wenn Live-Marktdaten oder News geladen werden.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Verstanden"):
                st.session_state.cookie_consent = True
                st.rerun()
        with c2:
            st.caption("Für eine echte Veröffentlichung müsste ein vollständiges Impressum, Datenschutztext und Cookie-Management ergänzt werden.")


def privacy_status_table():
    rows = [
        {"Bereich": "Tracking-Cookies", "Status": "Nicht aktiv", "Hinweis": "Keine Werbe- oder Tracking-Cookies in dieser lokalen Demo."},
        {"Bereich": "Session State", "Status": "Aktiv", "Hinweis": "Wird für UI-Zustände wie Cookie-Hinweis genutzt."},
        {"Bereich": "Entscheidungsjournal", "Status": "Lokal", "Hinweis": "Speichert CSV lokal unter data/journal/."},
        {"Bereich": "Marktdaten", "Status": "Extern", "Hinweis": "yfinance lädt Daten aus externer Quelle."},
        {"Bereich": "News", "Status": "Optional extern", "Hinweis": "NewsAPI nur bei gesetztem NEWS_API_KEY."},
        {"Bereich": "GitHub", "Status": "Geschützt", "Hinweis": "Rohdaten, Modelle und Journal werden per .gitignore ausgeschlossen."},
    ]
    return pd.DataFrame(rows)



# =========================================================
# ROUTING HELPERS
# =========================================================



def get_query_page():
    try:
        value = st.query_params.get("page", None)
        if isinstance(value, list):
            return value[0] if value else None
        return value
    except Exception:
        return None


def route_to_page(target_page):
    st.session_state["current_page"] = target_page
    try:
        st.query_params["page"] = target_page
    except Exception:
        pass
    st.rerun()




# =========================================================
# ROUTING + SIDEBAR
# =========================================================

def normalize_page_name(value):
    if value is None:
        return None

    try:
        value = unquote(str(value)).strip()
    except Exception:
        value = str(value).strip()

    aliases = {
        "So funktioniert%27s": "So funktioniert's",
        "So funktioniert’s": "So funktioniert's",
        "Projekt": "Über das Projekt",
        "Status": "Betriebsstatus",
        "News": "News-Archiv",
        "Outlook": "Wealth Outlook",
        "Portfolio": "Portfolio-Simulator",
    }

    return aliases.get(value, value) if value else None


def get_query_page():
    try:
        value = st.query_params.get("page", None)
        if isinstance(value, list):
            value = value[0] if value else None
        return normalize_page_name(value)
    except Exception:
        return None



def route_to_page(target_page):
    target_page = normalize_page_name(target_page)
    if not target_page:
        return

    st.session_state["current_page"] = target_page

    try:
        st.query_params["page"] = target_page
    except Exception:
        pass

    st.rerun()


main_pages = [
    "Start",
    "Wealth Outlook",
    "Kapital-Kompass",
    "Portfolio-Simulator",
    "Watchlist-Vergleich",
]

bottom_pages = [
    "News & Aktualität",
    "News-Archiv",
    "So funktioniert's",
    "Betriebsstatus",
    "Datenschutz",
    "Impressum",
    "Über das Projekt",
    "Methodik & Grenzen",
]

internal_pages = [
    "Entscheidungsjournal",
    "Professor-Export",
    "Stabilität & QA",
    "Expertenanalyse",
    "Modell & Datenbasis",
]

all_pages = main_pages + bottom_pages + internal_pages

query_page = get_query_page()

# Query-Parameter hat immer Vorrang.
if query_page in all_pages:
    page = query_page
    st.session_state["current_page"] = page
else:
    page = st.session_state.get("current_page", "Start")
    if page not in all_pages:
        page = "Start"
    st.session_state["current_page"] = page


# =========================================================
# CLEAN SIDEBAR
# =========================================================

st.sidebar.caption("Kapital verstehen. Risiken prüfen. Entscheidungen simulieren.")

# =========================================================
# UI MODE STATE
# =========================================================

if "ui_theme_mode" not in st.session_state:
    st.session_state["ui_theme_mode"] = "Dark Mode"

if "ui_app_mode" not in st.session_state:
    st.session_state["ui_app_mode"] = "Geführte Ansicht"

# =========================================================
# DARSTELLUNG + ANSICHT
# =========================================================


# URL-State für Theme/View übernehmen
try:
    qp = st.query_params
    url_theme = qp.get("theme")
    url_view = qp.get("view")

    if url_theme in ["Dark Mode", "Light Mode"]:
        st.session_state["theme_mode"] = url_theme

    if url_view in ["Geführte Ansicht", "Expertenansicht"]:
        st.session_state["app_mode"] = url_view
except Exception:
    pass


# =========================================================
# CENTRAL UI STATE HELPERS
# =========================================================

VALID_THEMES = ["Dark Mode", "Light Mode"]
VALID_VIEWS = ["Geführte Ansicht", "Expertenansicht"]


def init_ui_state():
    """Initialisiert Theme, Ansicht und Seite zentral."""
    qp = st.query_params

    url_theme = qp.get("theme", None)
    url_view = qp.get("view", None)
    url_page = qp.get("page", None)

    if isinstance(url_theme, list):
        url_theme = url_theme[0] if url_theme else None

    if isinstance(url_view, list):
        url_view = url_view[0] if url_view else None

    if isinstance(url_page, list):
        url_page = url_page[0] if url_page else None

    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = url_theme if url_theme in VALID_THEMES else "Dark Mode"

    if "app_mode" not in st.session_state:
        st.session_state["app_mode"] = url_view if url_view in VALID_VIEWS else "Geführte Ansicht"

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = url_page if url_page else "Start"

    # URL darf bestehenden Session-State nicht ungewollt überschreiben,
    # außer der Wert ist explizit gültig vorhanden.
    if url_theme in VALID_THEMES:
        st.session_state["theme_mode"] = url_theme

    if url_view in VALID_VIEWS:
        st.session_state["app_mode"] = url_view

    if url_page:
        st.session_state["current_page"] = url_page


def sync_query_params():
    """Schreibt aktuellen UI-State in die URL."""
    try:
        st.query_params["page"] = st.session_state.get("current_page", "Start")
        st.query_params["theme"] = st.session_state.get("theme_mode", "Dark Mode")
        st.query_params["view"] = st.session_state.get("app_mode", "Geführte Ansicht")
    except Exception:
        pass


def route_to_page(page_name):
    """Wechselt Seite und behält Theme/View bei."""
    st.session_state["current_page"] = page_name
    sync_query_params()
    st.rerun()



# =========================================================
# START
# =========================================================




# Gemeinsamer Asset-Kontext für alle Seiten
try:
    selected_asset = st.session_state.get("selected_asset", "ETF – S&P 500 (SPY)")
    asset_label = st.session_state.get("asset_label", selected_asset)
    ticker = st.session_state.get("ticker", "SPY")
    period = st.session_state.get("period", "5y")
    interval = st.session_state.get("interval", "1d")
except Exception:
    selected_asset = "ETF – S&P 500 (SPY)"
    asset_label = "ETF – S&P 500 (SPY)"
    ticker = "SPY"
    period = "5y"
    interval = "1d"

try:
    asset_label = selected_asset
    ticker = selected_asset.split("(")[-1].replace(")", "").strip() if "(" in selected_asset else "SPY"
except Exception:
    asset_label = "ETF – S&P 500 (SPY)"
    ticker = "SPY"

try:
    category = ASSET_META.get(ticker, {}).get("category", "Allgemeiner Markt")
except Exception:
    category = "Allgemeiner Markt"



# =========================================================
# GLOBALER APP-KONTEXT FÜR ALLE SEITEN
# =========================================================

# Gemeinsame Asset-Basis für alle Seiten
try:
    selected_asset = st.session_state.get("selected_asset", "ETF – S&P 500 (SPY)")
except Exception:
    selected_asset = "ETF – S&P 500 (SPY)"

try:
    asset_label = selected_asset
    ticker = selected_asset.split("(")[-1].replace(")", "").strip() if "(" in selected_asset else "SPY"
except Exception:
    asset_label = "ETF – S&P 500 (SPY)"
    ticker = "SPY"

try:
    asset_meta = ASSET_META.get(ticker, {})
except Exception:
    asset_meta = {}

try:
    category = asset_meta.get("category", "Allgemeiner Markt")
except Exception:
    category = "Allgemeiner Markt"

try:
    asset_type = asset_meta.get("type", "ETF")
except Exception:
    asset_type = "ETF"

# Gemeinsame Marktdaten für Service-Seiten wie Betriebsstatus, News-Archiv, Export
try:
    period
except NameError:
    period = "5y"

try:
    interval
except NameError:
    interval = "1d"

try:
    market_data = load_market_data(ticker, period, interval)
except Exception:
    market_data = pd.DataFrame()

try:
    if market_data is not None and not market_data.empty:
        feature_data = prepare_live_features(market_data)
    else:
        feature_data = pd.DataFrame()
except Exception:
    feature_data = pd.DataFrame()






def route_link(page_name):
    """Erzeugt interne Links mit aktuellem Theme und aktueller Ansicht."""
    import urllib.parse

    params = {
        "page": page_name,
        "theme": st.session_state.get("theme_mode", "Dark Mode"),
        "view": st.session_state.get("app_mode", "Geführte Ansicht"),
    }

    return "?" + urllib.parse.urlencode(params)


def render_sidebar():
    """Harmonische Sidebar: klickbares Branding, Darstellung, Ansicht und Hauptnavigation."""
    home_link = route_link("Start")

    with st.sidebar:
        st.markdown(
            f"""
            <a class="sidebar-home-link" href="{home_link}" target="_self">
                <div class="sidebar-brand-card">
                    <div class="sidebar-brand-title">WealthScope AI</div>
                    <div class="sidebar-brand-subtitle">
                        Kapital verstehen.<br>
                        Risiken prüfen.<br>
                        Entscheidungen simulieren.
                    </div>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">Darstellung</div>', unsafe_allow_html=True)

        theme_mode = st.radio(
            "Darstellung",
            VALID_THEMES,
            index=VALID_THEMES.index(st.session_state.get("theme_mode", "Dark Mode")),
            key="sidebar_theme_mode",
            label_visibility="collapsed",
        )

        st.session_state["theme_mode"] = theme_mode

        st.markdown('<div class="sidebar-section-label">Ansicht</div>', unsafe_allow_html=True)

        app_mode = st.radio(
            "Ansicht",
            VALID_VIEWS,
            index=VALID_VIEWS.index(st.session_state.get("app_mode", "Geführte Ansicht")),
            key="sidebar_app_mode",
            label_visibility="collapsed",
        )

        st.session_state["app_mode"] = app_mode

        st.markdown(
            f"""
            <div class="sidebar-state-box">
                <span>Darstellung aktiv:</span>
                <b>{theme_mode}</b>
                <span>Ansicht aktiv:</span>
                <b>{app_mode}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section-label">Hauptnavigation</div>', unsafe_allow_html=True)

        main_pages = [
            "Start",
            "Wealth Outlook",
            "Kapital-Kompass",
            "Portfolio-Simulator",
            "Watchlist-Vergleich",
        ]

        current_page = st.session_state.get("current_page", "Start")

        for nav_page in main_pages:
            active = "• " if current_page == nav_page else ""
            if st.button(
                f"{active}{nav_page}",
                key=f"sidebar_nav_{nav_page}",
                use_container_width=True,
            ):
                st.session_state["current_page"] = nav_page
                sync_query_params()
                st.rerun()

        st.markdown(
            """
            <div class="sidebar-help-box">
                Weitere Seiten wie Impressum, Datenschutz, Export und Betriebsstatus findest du unten in der festen Statusleiste.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state["theme_mode"], st.session_state["app_mode"], st.session_state.get("current_page", "Start")


def render_analysis_settings():
    """Zentrale Analyse-Einstellungen im Hauptbereich."""
    st.markdown("")

    with st.expander("Analyse-Einstellungen", expanded=True):
        c_asset, c_custom, c_period, c_interval = st.columns([2.2, 1.2, 0.8, 0.8])

        asset_options = [
            "ETF – S&P 500 (SPY)",
            "ETF – Nasdaq 100 (QQQ)",
            "ETF – Bonds (BND)",
            "ETF – Gold (GLD)",
            "ETF – Emerging Markets (EEM)",
            "ETF – US Total Market (VTI)",
            "Aktie – Apple (AAPL)",
            "Aktie – Microsoft (MSFT)",
            "Aktie – Nvidia (NVDA)",
            "Aktie – Tesla (TSLA)",
            "Aktie – Amazon (AMZN)",
        ]

        if "selected_asset" not in st.session_state:
            st.session_state["selected_asset"] = "ETF – S&P 500 (SPY)"

        if st.session_state["selected_asset"] not in asset_options:
            st.session_state["selected_asset"] = "ETF – S&P 500 (SPY)"

        with c_asset:
            selected_asset = st.selectbox(
                "Asset auswählen",
                asset_options,
                index=asset_options.index(st.session_state["selected_asset"]),
                key="selected_asset_selector",
            )

        with c_custom:
            custom_ticker = st.text_input(
                "Optional eigener Ticker",
                value=st.session_state.get("custom_ticker", ""),
                key="custom_ticker_selector",
                placeholder="z. B. AMD",
            )

        with c_period:
            period = st.selectbox(
                "Zeitraum",
                ["6mo", "1y", "2y", "5y", "10y", "max"],
                index=["6mo", "1y", "2y", "5y", "10y", "max"].index(st.session_state.get("period", "5y")),
                key="period_selector",
            )

        with c_interval:
            interval = st.selectbox(
                "Intervall",
                ["1d", "1wk", "1mo"],
                index=["1d", "1wk", "1mo"].index(st.session_state.get("interval", "1d")),
                key="interval_selector",
            )

        st.session_state["selected_asset"] = selected_asset
        st.session_state["custom_ticker"] = custom_ticker
        st.session_state["period"] = period
        st.session_state["interval"] = interval

        if custom_ticker.strip():
            ticker = custom_ticker.strip().upper()
            asset_label = f"Eigener Ticker ({ticker})"
        else:
            asset_label = selected_asset
            ticker = selected_asset.split("(")[-1].replace(")", "").strip() if "(" in selected_asset else "SPY"

        st.session_state["ticker"] = ticker
        st.session_state["asset_label"] = asset_label

        st.caption("Keine Finanzberatung. Analyse, Lernen und Simulation. Die Einstellungen gelten für die aktuelle Auswertung.")

        return selected_asset, asset_label, ticker, period, interval


# Zentralen UI-State initialisieren, bevor Seiten gerendert werden
init_ui_state()

theme_mode = st.session_state.get("theme_mode", "Dark Mode")
app_mode = st.session_state.get("app_mode", "Geführte Ansicht")
page = st.session_state.get("current_page", page if "page" in globals() else "Start")

# Sidebar immer rendern und UI-State daraus übernehmen
theme_mode, app_mode, page = render_sidebar()
sync_query_params()


# Zentrale Analyse-Einstellungen oben im Hauptbereich anzeigen
selected_asset, asset_label, ticker, period, interval = render_analysis_settings()


if page == "Start":

    # FINAL VIEW DIFFERENTIATION ON START
    if app_mode == "Expertenansicht":
        st.markdown(
            """
            <div class="mode-proof-card">
                <h3>Expertenmodus-Inhalt</h3>
                <p>
                Diese Ansicht legt den Fokus auf Modellgüte, technische Indikatoren, News-Sentiment,
                Datenfrische und Score-Zerlegung. Sie ist für Nutzer gedacht, die die Herleitung prüfen möchten.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="mode-proof-card">
                <h3>Geführter Modus-Inhalt</h3>
                <p>
                Diese Ansicht erklärt die Ergebnisse bewusst einfach: Was bedeutet die Einschätzung,
                wie hoch ist das Risiko und welcher nächste Schritt ist sinnvoll?
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="hero-card">
        <h2>Kapital schützen. Märkte verstehen. Entscheidungen simulieren.</h2>
        <p>
        WealthScope AI richtet sich an Menschen, die Kapital verantwortungsvoll einordnen möchten:
        zum Beispiel nach einem Erbe, einer Abfindung oder dem Aufbau größerer Rücklagen.
        Die App kombiniert historische Marktbewegungen, aktuelle Nachrichten, Risikokennzahlen
        und ein ML-Szenario zu einer verständlichen Entscheidungshilfe.
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
            <h3>Geführte Ansicht</h3>
            <p>Für nicht-technische Nutzer: klare Antwort, warum, was bedeutet das fürs Kapital, nächster Schritt.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Expertenansicht</h3>
            <p>Für Nutzer mit Vorwissen: Chart, Indikatoren, ML-Werte, News-Score und Confidence Breakdown.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Aktualität</h3>
            <p>Die App zeigt, wie frisch Marktdaten und Nachrichten sind, bevor ein Outlook bewertet wird.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # START MODE DIFFERENTIATION
    if app_mode == "Geführte Ansicht":
        st.markdown(
            """
            <div class="hero-card">
            <h2>Geführte Startlogik</h2>
            <p>
            Diese Ansicht richtet sich an Nutzer, die eine verständliche Antwort möchten:
            Was bedeutet das Asset für mein Kapital, wie riskant ist es und was sollte ich als Nächstes prüfen?
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hero-card">
            <h2>Experten-Startlogik</h2>
            <p>
            Diese Ansicht richtet sich an Nutzer mit Vorwissen:
            Relevant sind Modellgüte, technische Indikatoren, Datenfrische, News-Sentiment und Score-Zerlegung.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Empfohlener Ablauf")
    st.write("1. Starte mit **Kapital-Kompass**.")
    st.write("2. Prüfe den **Wealth Outlook**.")
    st.write("3. Lies die wichtigsten **News & Aktualität**.")
    st.write("4. Nutze den **Decision Passport** als Report.")
    st.write("5. In der Expertenansicht kannst du die technische Herleitung prüfen.")

    st.divider()

    st.subheader("Was macht WealthScope stärker als eine normale Watchlist?")
    k1, k2, k3 = st.columns(3)

    with k1:
        st.markdown(
            """
            <div class="mini-card">
            <h3>News werden bewertet</h3>
            <p>Nicht nur Links anzeigen: Jede Nachricht bekommt Sentiment, Relevanz, Impact und Klartext-Erklärung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Kapital statt Kurs</h3>
            <p>Die App fragt nicht nur nach Ticker, sondern nach Kapital, Gewichtung und Risikotragfähigkeit.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            """
            <div class="mini-card">
            <h3>Zwei echte Ansichten</h3>
            <p>Geführte Ansicht für Kapitalinhaber. Expertenansicht für Daten, Modellwerte und Score-Zerlegung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# WEALTH OUTLOOK
# =========================================================

if page == "Wealth Outlook":
    st.header("Wealth Outlook")
    render_mode_hint(app_mode)

    if feature_data.empty or len(feature_data.dropna(subset=["ma_200"])) < 1:
        st.error("Nicht genug Marktdaten für einen vollständigen Wealth Outlook. Wähle einen längeren Zeitraum.")
        st.stop()

    capital = st.number_input("Kapitalbetrag", min_value=1000.0, value=100000.0, step=5000.0)
    allocation_pct = st.slider("Geplante Gewichtung für dieses Asset (%)", 1.0, 100.0, 10.0, 1.0)

    latest = feature_data.dropna(subset=["ma_20", "ma_50", "ma_200", "volatility_20d", "drawdown"]).tail(1).iloc[0]
    latest_price = float(latest["close"])
    amount = capital * allocation_pct / 100

    levels = detect_levels(feature_data)
    ts = trend_score(latest)

    api_key = get_news_api_key()
    news_query = build_news_query(ticker, asset_label, category)
    news_result = fetch_news(news_query, api_key, language="en", page_size=8) if api_key else {"status": "missing_key", "articles": [], "message": "NEWS_API_KEY fehlt."}
    articles = news_result.get("articles", []) if news_result.get("status") == "ok" else []

    nf = news_freshness(articles)
    mf = market_data_freshness(market_data, interval)
    ne = score_news(articles)
    news_intel = analyze_news_intelligence(articles, ticker, asset_type, category)

    # Advanced News Intelligence beeinflusst den News Score.
    # Klassischer Keyword-Score bleibt als Basis erhalten.
    ne["score"] = ne["score"] + int(round(news_intel["summary_score"]))

    model_result = predict_with_local_model(ticker, feature_data)

    model_prob_up = None
    model_accuracy = None
    if model_result is not None:
        model_prob_up = model_result.get("probability_up")
        model_accuracy = model_result.get("accuracy")

    outlook = build_wealth_outlook(
        technical_score=ts["score"],
        news_score=ne["score"],
        volatility_20d=latest["volatility_20d"],
        drawdown=latest["drawdown"],
        model_prob_up=model_prob_up,
        market_fresh=mf,
        news_fresh=nf,
    )

    confidence = compute_confidence(
        technical_score=ts["score"],
        news_score=ne["score"],
        market_fresh=mf,
        news_fresh=nf,
        model_accuracy=model_accuracy,
        model_prob_up=model_prob_up,
        volatility_20d=latest["volatility_20d"],
    )

    protection = capital_protection_score(
        capital=capital,
        planned_allocation_pct=allocation_pct,
        confidence_score=confidence["score"],
        drawdown=latest["drawdown"],
        volatility_20d=latest["volatility_20d"],
        asset_type=asset_type,
    )

    traffic = traffic_light_status(confidence["score"], protection["score"], outlook["adjusted_score"])
    timer = current_update_timer(mf, nf)

    render_premium_status_strip(ticker, asset_type, traffic, confidence, protection, news_intel, mf, nf)
    render_update_timer(timer)

    advisor_text = advisor_explanation(outlook, confidence, protection, news_intel, asset_type, allocation_pct)
    render_advisor_box(advisor_text)

    st.markdown(
        f"""
        <div class="hero-card">
        <h2>Executive Answer</h2>
        <p><b>Asset:</b> {asset_label}</p>
        <p><b>Status:</b> {traffic['emoji']} {traffic['label']}</p>
        <p><b>Aktuelle Einschätzung:</b> {outlook['outlook']}</p>
        <p><b>Nächster sinnvoller Schritt:</b> {outlook['action']}</p>
        <p><b>Geplanter Betrag:</b> {de_currency(amount)} bei {de_number(allocation_pct, 1)} % Gewichtung</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f"""
            <div class="info-card {outlook['card']}">
            <h3>Wealth Outlook</h3>
            <div class="big-number">{outlook['adjusted_score']}</div>
            <p>{outlook['outlook']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="info-card {confidence['card']}">
            <h3>Confidence</h3>
            <div class="big-number">{confidence['score']} / 100</div>
            <p>{confidence['label']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="info-card {protection['card']}">
            <h3>Kapital-Schutz</h3>
            <div class="big-number">{protection['score']} / 100</div>
            <p>{protection['label']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info("Confidence ist keine Trefferwahrscheinlichkeit. Sie bewertet nur die aktuelle Belastbarkeit der Einschätzung.")

    st.subheader("Score-Fortschritt")
    p1, p2, p3 = st.columns(3)
    with p1:
        render_score_progress("Confidence", confidence["score"], "Wie belastbar die aktuelle Einschätzung wirkt.")
    with p2:
        render_score_progress("Kapital-Schutz", protection["score"], "Wie vorsichtig das Szenario aus Sicht des Kapitalerhalts ist.")
    with p3:
        normalized_outlook = max(0, min(100, (outlook["adjusted_score"] + 10) * 5))
        render_score_progress("Wealth Outlook", normalized_outlook, "Skalierte Darstellung des Outlook Scores.")

    st.subheader("News Intelligence Summary")
    render_news_intelligence_summary(news_intel, app_mode)

    if app_mode == "Expertenansicht":
        render_news_risk_radar(news_intel)

    st.divider()

    st.subheader("Warum?")

    col_a, col_b = st.columns(2)

    with col_a:
        st.write("**Historische Daten / Chart**")
        for reason in ts["reasons"]:
            st.write(f"- {reason}")
        st.write(f"- Aktueller Preis: **{latest_price:.2f}**")
        st.write(f"- 1-Jahres-Hoch: **{levels.get('one_year_high', np.nan):.2f}**")
        st.write(f"- 1-Jahres-Tief: **{levels.get('one_year_low', np.nan):.2f}**")
        st.write(f"- Drawdown vom Hoch: **{latest['drawdown']:.2%}**")
        st.write(f"- {outlook['vol_note']}")
        st.write(f"- {outlook['drawdown_note']}")

    with col_b:
        st.write("**News / Aktualität / Modell**")
        st.write(f"- News-Kontext: **{ne['label']}**")
        if ne["drivers"]:
            for driver in ne["drivers"][:5]:
                st.write(f"- {driver}")
        st.write(f"- Marktdatenstatus: **{mf['status']}**")
        st.write(f"- Newsstatus: **{nf['status']}**")
        if model_result is not None and model_prob_up is not None:
            st.write(f"- ML-Szenario: **{model_prob_up:.1%}** Wahrscheinlichkeit für höheren Kurs in 20 Handelstagen")
            st.write(f"- Historische Modell-Accuracy: **{model_accuracy:.2%}**")
        else:
            st.write("- ML-Szenario: nicht verfügbar")

    st.divider()

    if app_mode == "Geführte Ansicht":
        st.subheader("Was bedeutet das für dein Kapital?")
        st.write(f"- Bei {capital:,.2f} Kapital entspricht {allocation_pct:.1f}% einem Betrag von **{amount:,.2f}**.")
        st.write(f"- Dieses Asset ist als **{asset_type}** eingeordnet.")
        st.write(f"- Kapital-Schutz-Einschätzung: **{protection['label']}**.")
        st.write(f"- Handlung: **{protection['action']}**.")
        st.write("- Für größere Beträge ist ein schrittweises Vorgehen sinnvoller als eine einmalige Entscheidung.")
        st.write("- Bei Unsicherheit: erst beobachten, simulieren oder mit professioneller Beratung abgleichen.")

        st.subheader("Nächste Schritte")
        if protection["score"] < 40 or confidence["score"] < 30:
            st.error("Nicht direkt investieren. Erst Risiko reduzieren, News lesen und später erneut prüfen.")
        elif protection["score"] < 70:
            st.warning("Nur kontrolliert prüfen. Kleine Gewichtung, langer Zeithorizont und keine übereilte Entscheidung.")
        else:
            st.info("Das Szenario wirkt strukturierter, aber bleibt unsicher. Schrittweise prüfen und dokumentieren.")

        st.markdown(
            """
            <div class="decision-step"><b>1. Entscheidung nicht erzwingen:</b> Erst verstehen, dann handeln.</div>
            <div class="decision-step"><b>2. Gewichtung prüfen:</b> Keine einzelne Position darf das Kapital dominieren.</div>
            <div class="decision-step"><b>3. News lesen:</b> Vor allem hohe Impact-News beachten.</div>
            <div class="decision-step"><b>4. Entscheidung dokumentieren:</b> Decision Passport speichern.</div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.subheader("Experten-Details")

        details = pd.DataFrame(
            [
                {"Kennzahl": "Technischer Score", "Wert": ts["score"]},
                {"Kennzahl": "News Score", "Wert": ne["score"]},
                {"Kennzahl": "Adjusted Outlook Score", "Wert": outlook["adjusted_score"]},
                {"Kennzahl": "Confidence Score", "Wert": confidence["score"]},
                {"Kennzahl": "Drawdown", "Wert": latest["drawdown"]},
                {"Kennzahl": "Volatility 20d", "Wert": latest["volatility_20d"]},
                {"Kennzahl": "RSI", "Wert": latest.get("rsi", np.nan)},
                {"Kennzahl": "MACD Histogramm", "Wert": latest.get("macd_hist", np.nan)},
            ]
        )
        st.dataframe(details, width='stretch')

        conf_df = pd.DataFrame(confidence["components"])
        fig = px.bar(conf_df, x="Faktor", y="Punkte", text="Punkte", title="Confidence Score Breakdown")
        fig.update_traces(textposition="outside")
        fig = apply_chart_theme(fig, theme_mode)
        render_interactive_chart(
            fig,
            data=conf_df,
            title="Confidence Score Breakdown",
            key="confidence_breakdown_chart",
        )

    st.divider()

    st.subheader("Chart")
    price_fig = create_price_chart(feature_data.tail(500), ticker)
    price_fig = apply_chart_theme(price_fig, theme_mode)
    render_interactive_chart(
        price_fig,
        data=feature_data.tail(500),
        title="Historischer Kursverlauf",
        key="wealth_outlook_price_chart",
    )

    st.divider()

    st.subheader("Quellen zum Nachlesen")
    render_analyzed_news_cards(news_intel, app_mode, max_articles=4 if app_mode == "Geführte Ansicht" else 8)

    st.divider()

    report = build_decision_passport(
        ticker=ticker,
        asset_label=asset_label,
        capital=capital,
        allocation_pct=allocation_pct,
        latest_price=latest_price,
        outlook=outlook,
        confidence=confidence,
        protection=protection,
        market_fresh=mf,
        news_fresh=nf,
        model_result=model_result,
    )

    st.download_button(
        "Decision Passport herunterladen",
        data=report,
        file_name=f"wealthscope_decision_passport_{ticker}.md",
        mime="text/markdown",
    )

    with st.expander("Analyse im Entscheidungsjournal speichern"):
        journal_note = st.text_area("Eigene Notiz zur Entscheidung", value="Ich beobachte das Szenario zunächst und investiere nicht vorschnell.")
        decision_choice = st.selectbox("Meine Entscheidung", ["Beobachten", "Simulieren", "Nicht investieren", "Später erneut prüfen", "Weiter analysieren"])

        if st.button("Im Journal speichern"):
            entry = {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "asset_label": asset_label,
                "capital": capital,
                "allocation_pct": allocation_pct,
                "planned_amount": amount,
                "wealth_outlook": outlook["outlook"],
                "outlook_score": outlook["adjusted_score"],
                "confidence": confidence["score"],
                "capital_protection": protection["score"],
                "news_summary": news_intel["summary_label"],
                "decision": decision_choice,
                "note": journal_note,
            }
            saved_path = save_journal_entry(entry)
            st.success(f"Journal gespeichert: {saved_path}")

    st.error("Keine Finanzberatung. Keine Handelsaufforderung. Keine Garantie. Diese App dient Analyse, Lernen und Simulation.")


# =========================================================
# CAPITAL COMPASS
# =========================================================

if page == "Kapital-Kompass":
    st.header("Kapital-Kompass")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Was darf Risiko überhaupt bedeuten?</h2>
        <p>
        Bevor ein Asset analysiert wird, sollte klar sein: Wie viel Kapital steht zur Verfügung?
        Wie viel darf maximal in eine einzelne Idee fließen? Wie viel Verlust wäre emotional und finanziell tragbar?
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    capital = st.number_input("Gesamtkapital", min_value=1000.0, value=250000.0, step=10000.0)
    max_single_allocation = st.slider("Maximale Gewichtung pro Einzelposition (%)", 1.0, 50.0, 10.0, 1.0)
    max_loss_tolerance = st.slider("Gedanklich tolerierbarer Rückgang auf Gesamtportfolio (%)", 1.0, 50.0, 10.0, 1.0)

    amount = capital * max_single_allocation / 100
    tolerated_loss = capital * max_loss_tolerance / 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Gesamtkapital", de_currency(capital))
    c2.metric("Max. Einzelposition", de_currency(amount))
    c3.metric("Tolerierter Rückgang", f"{tolerated_loss:,.2f}")

    st.subheader("Interpretation")

    if max_single_allocation > 25:
        st.error("Eine sehr hohe Gewichtung in ein einzelnes Asset erhöht das Konzentrationsrisiko.")
    elif max_single_allocation > 10:
        st.warning("Die Gewichtung ist spürbar. Für Anfänger besser mit kleineren Bausteinen starten.")
    else:
        st.success("Die Gewichtung wirkt kontrollierter.")

    if max_loss_tolerance < 10:
        st.info("Deine Verlusttoleranz ist eher niedrig. Defensive Allokation und breite Streuung wären wichtiger.")
    else:
        st.info("Auch bei höherer Toleranz sollte Kapitalerhalt zuerst kommen.")

    st.divider()

    st.subheader("Risikoprofil-Fragebogen")

    q1, q2 = st.columns(2)

    with q1:
        horizon_years = st.slider("Anlagehorizont in Jahren", 1, 30, 10)
        loss_tolerance = st.slider("Verlusttoleranz in %", 1, 50, 15)
        experience = st.selectbox("Erfahrung", ["Anfänger", "Fortgeschritten", "Experte"])

    with q2:
        liquidity_need = st.selectbox("Brauchst du das Geld wahrscheinlich in den nächsten 3 Jahren?", ["Nein", "Vielleicht", "Ja, wahrscheinlich"])
        goal = st.selectbox("Hauptziel", ["Kapital erhalten", "Ausgewogen wachsen", "Vermögen stark ausbauen"])

    profile = calculate_risk_profile(horizon_years, loss_tolerance, experience, liquidity_need, goal)

    st.markdown(
        f"""
        <div class="info-card {profile['card']}">
        <h3>Risikoprofil</h3>
        <div class="big-number">{profile['label']}</div>
        <p>Score: <b>{profile['score']} / 100</b></p>
        <p>{profile['explanation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_score_progress("Kapital-Schutz", profile["score"], "Wie gut dein Profil zu schwankungsreichen Anlagen passt.")

    st.divider()

    st.subheader("Was-wäre-wenn-Szenario")

    drop_pct = st.slider("Angenommener Rückgang des Assets (%)", 5, 60, 20)
    monthly_saving = st.number_input("Optional monatliche Sparrate", min_value=0.0, value=500.0, step=100.0)
    years = st.slider("Simulationsdauer in Jahren", 0, 30, 10)
    annual_return = st.slider("Angenommene jährliche Rendite (%)", -10.0, 15.0, 5.0, 0.5) / 100

    sim = simulate_what_if(capital, max_single_allocation, drop_pct, monthly_saving, years, annual_return)

    s1, s2, s3 = st.columns(3)
    s1.metric("Betrag in Einzelposition", de_currency(sim["allocation_amount"]))
    s2.metric("Verlust bei Szenario", f"-{de_currency(sim['loss_on_position'])}")
    s3.metric("Portfolio danach", de_currency(sim["portfolio_after_drop"]))

    if sim["future_value"] is not None:
        st.metric("Hypothetischer Zukunftswert", de_currency(sim["future_value"]))

    st.warning("Der Kapital-Kompass ersetzt keine Anlageberatung. Er dient der Risikoorientierung.")





# =========================================================
# OPERATING STATUS HELPERS
# =========================================================

def status_dot(status):
    if status == "OK":
        return '<span class="status-dot dot-green"></span>'
    if status == "Eingeschränkt":
        return '<span class="status-dot dot-yellow"></span>'
    return '<span class="status-dot dot-red"></span>'


def evaluate_operating_status(market_fresh, news_fresh, model_available, feature_available):
    issues = []

    if market_fresh.get("status") in ["Keine Daten", "Veraltet"]:
        issues.append("Marktdaten sind nicht aktuell oder nicht verfügbar.")

    if news_fresh.get("status") in ["Keine News", "Alt"]:
        issues.append("News sind nicht verfügbar oder nicht aktuell.")

    if not model_available:
        issues.append("ML-Modell ist lokal nicht verfügbar.")

    if not feature_available:
        issues.append("Feature-Datei ist lokal nicht verfügbar.")

    if len(issues) == 0:
        overall = "OK"
        label = "Alle Kernsysteme betriebsbereit"
        card = "good-card"
        dot = "dot-green"
    elif len(issues) <= 2:
        overall = "Eingeschränkt"
        label = "System nutzbar, aber mit Einschränkungen"
        card = "warn-card"
        dot = "dot-yellow"
    else:
        overall = "Kritisch"
        label = "Mehrere Kernsysteme benötigen Prüfung"
        card = "bad-card"
        dot = "dot-red"

    return {
        "overall": overall,
        "label": label,
        "card": card,
        "dot": dot,
        "issues": issues,
        "checked_at": dt.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
    }


def render_status_row(name, status, detail):
    dot = status_dot(status)
    st.markdown(
        f"""
        <div class="status-row">
            <b>{dot}{name}</b><br>
            <span>Status: <b>{status}</b></span><br>
            <span class="muted">{detail}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_operating_status_table(market_fresh, news_fresh, model_available, feature_available):
    rows = []

    market_status = "OK" if market_fresh.get("status") in ["Aktuell", "Leicht verzögert"] else "Eingeschränkt"
    if market_fresh.get("status") in ["Keine Daten", "Veraltet"]:
        market_status = "Kritisch"

    news_status = "OK" if news_fresh.get("status") in ["Sehr aktuell", "Aktuell", "Mittel"] else "Eingeschränkt"
    if news_fresh.get("status") in ["Alt"]:
        news_status = "Kritisch"

    rows.append({
        "Komponente": "Marktdaten",
        "Status": market_status,
        "Details": f"{market_fresh.get('status')} | Alter: {format_age_minutes(market_fresh.get('minutes_old'))}",
    })

    rows.append({
        "Komponente": "News",
        "Status": news_status,
        "Details": f"{news_fresh.get('status')} | Alter: {format_age_hours(news_fresh.get('hours_old'))}",
    })

    rows.append({
        "Komponente": "ML-Modell",
        "Status": "OK" if model_available else "Eingeschränkt",
        "Details": "Lokales Modell geladen." if model_available else "Modell nicht verfügbar. App nutzt dann nur Regeln/Indikatoren.",
    })

    rows.append({
        "Komponente": "Feature-Datei",
        "Status": "OK" if feature_available else "Eingeschränkt",
        "Details": "Lokale ML-Feature-Datei vorhanden." if feature_available else "Feature-Datei fehlt lokal.",
    })

    rows.append({
        "Komponente": "Big-Data-Rohbasis",
        "Status": "OK" if Path("data/raw/price-volume-data-for-all-us-stocks-etfs.zip").exists() else "Eingeschränkt",
        "Details": "Kaggle-Rohdaten lokal vorhanden." if Path("data/raw/price-volume-data-for-all-us-stocks-etfs.zip").exists() else "Rohdaten lokal nicht gefunden.",
    })

    rows.append({
        "Komponente": "Git-Schutz",
        "Status": "OK" if Path(".gitignore").exists() else "Kritisch",
        "Details": ".gitignore vorhanden. Große Daten/Modelle sollten ausgeschlossen sein." if Path(".gitignore").exists() else ".gitignore fehlt.",
    })

    return pd.DataFrame(rows)


# =========================================================
# STABILITY / QA HELPERS
# =========================================================

def file_status(path_str, should_exist=True):
    p = Path(path_str)
    exists = p.exists()
    size = p.stat().st_size if exists and p.is_file() else None

    if should_exist and exists:
        status = "OK"
    elif should_exist and not exists:
        status = "Fehlt"
    elif not should_exist and exists:
        status = "Prüfen"
    else:
        status = "OK"

    return {
        "Pfad": path_str,
        "Existiert": exists,
        "Größe MB": round(size / 1024 / 1024, 2) if size else None,
        "Status": status,
    }


def run_stability_checks(ticker="SPY"):
    import time

# =========================================================
# GUARANTEED PLOTLY THEME HELPER
# =========================================================



def gitignore_quality_check():
    p = Path(".gitignore")
    if not p.exists():
        return pd.DataFrame([{"Regel": ".gitignore vorhanden", "Status": "Fehlt"}])

    content = p.read_text(encoding="utf-8", errors="ignore")

    required_rules = [
        "data/raw/",
        "data/processed/*.csv",
        "models/*.joblib",
        "models/*.pkl",
        "data/journal/",
        ".DS_Store",
    ]

    rows = []
    for rule in required_rules:
        rows.append({
            "Regel": rule,
            "Status": "OK" if rule in content else "Fehlt",
        })

    return pd.DataFrame(rows)


# =========================================================
# PORTFOLIO SIMULATOR
# =========================================================

if page == "Portfolio-Simulator":
    st.header("Portfolio-Simulator")
    render_glossary_box(app_mode)

    st.markdown(
        """
        <div class="hero-card">
        <h2>Mehrere Bausteine statt einzelner Bauchentscheidung</h2>
        <p>
        Der Portfolio-Simulator zeigt, wie sich mehrere Assets gemeinsam verhalten könnten.
        Ziel ist nicht die perfekte Vorhersage, sondern ein besseres Verständnis für Gewichtung, Konzentration und Risiko.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    capital = st.number_input("Portfolio-Kapital", min_value=1000.0, value=250000.0, step=10000.0)

    default_assets = ["ETF – S&P 500 (SPY)", "ETF – Nasdaq 100 (QQQ)", "ETF – Bonds (BND)", "ETF – Gold (GLD)"]
    portfolio_options = unique_asset_labels()
    default_assets = [x for x in deduplicate_selection(default_assets) if x in portfolio_options]

    raw_selected_assets = st.multiselect(
        "Assets auswählen",
        portfolio_options,
        default=default_assets,
        key="portfolio_assets_unique",
    )

    selected_assets = deduplicate_selection(raw_selected_assets)

    if len(raw_selected_assets) != len(selected_assets):
        st.warning("Doppelte Einträge wurden automatisch entfernt.")

    if not selected_assets:
        st.warning("Wähle mindestens ein Asset aus.")
        st.stop()

    st.subheader("Gewichtungen")

    weights = {}
    category_weights = {}

    cols = st.columns(2)
    for i, label in enumerate(selected_assets):
        with cols[i % 2]:
            w = st.slider(label, 0.0, 100.0, 100.0 / len(selected_assets), 1.0)
            weights[label] = w
            cat = ASSETS[label]["category"]
            category_weights[cat] = category_weights.get(cat, 0) + w

    total_weight = sum(weights.values())

    if total_weight == 0:
        st.error("Gesamtgewichtung darf nicht 0 sein.")
        st.stop()

    if abs(total_weight - 100) > 0.1:
        st.warning(f"Die Gewichtungen ergeben aktuell {total_weight:.1f} %. Für Interpretation werden sie normalisiert.")

    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    div = portfolio_diversification_score(category_weights)

    c1, c2, c3 = st.columns(3)
    c1.metric("Anzahl Bausteine", len(selected_assets))
    c2.metric("Gesamtgewichtung", f"{total_weight:.1f} %")
    c3.metric("Diversifikations-Score", f"{div['score']} / 100")

    st.markdown(
        f"""
        <div class="info-card {div['card']}">
        <h3>Diversifikation</h3>
        <div class="big-number">{div['label']}</div>
        <p>{' '.join(div['notes'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Portfolio-Gewichtung")
    weight_df = pd.DataFrame(
        [
            {
                "Asset": label,
                "Ticker": ASSETS[label]["ticker"],
                "Kategorie": ASSETS[label]["category"],
                "Gewichtung": normalized_weights[label] * 100,
                "Betrag": capital * normalized_weights[label],
            }
            for label in selected_assets
        ]
    )

    fig = px.pie(weight_df, names="Asset", values="Gewichtung", title="Portfolio-Aufteilung", hover_data=["Ticker", "Kategorie", "Betrag"])
    fig = apply_chart_theme(fig, theme_mode)
    render_interactive_chart(
        fig,
        data=weight_df,
        title="Portfolio-Gewichtung",
        key="portfolio_weight_chart",
    )
    st.dataframe(weight_df, width='stretch')

    st.subheader("Was-wäre-wenn auf Portfolioebene")

    drop_scenario = st.slider("Angenommener Rückgang des Aktien-/Risikoanteils (%)", 5, 60, 20)

    risk_categories = ["Technologie / Wachstum", "US-Aktienmarkt breit", "US-Gesamtmarkt", "Schwellenländer", "Einzelaktie Technologie", "Einzelaktie Wachstum", "Einzelaktie Banken", "Einzelaktie Gesundheit", "Einzelaktie Energie", "Einzelaktie Konsum / Cloud", "Einzelaktie KI / Halbleiter"]
    risk_weight = weight_df[weight_df["Kategorie"].isin(risk_categories)]["Gewichtung"].sum() / 100

    estimated_loss = capital * risk_weight * drop_scenario / 100
    after = capital - estimated_loss

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Risikoanteil geschätzt", f"{risk_weight:.1%}")
    sc2.metric("Möglicher Verlust", f"-{de_currency(estimated_loss)}")
    sc3.metric("Portfolio danach", de_currency(after))

    st.info("Diese Simulation ist vereinfacht. Sie zeigt Größenordnungen, keine Vorhersage.")


# =========================================================
# WATCHLIST COMPARISON
# =========================================================

if page == "Watchlist-Vergleich":
    st.header("Watchlist-Vergleich")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Mehrere Assets schnell vergleichen</h2>
        <p>
        Diese Ansicht hilft, ETFs und Aktien nach Performance, Volatilität, Drawdown und Trend grob gegenüberzustellen.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_watch = ["ETF – S&P 500 (SPY)", "ETF – Nasdaq 100 (QQQ)", "ETF – Bonds (BND)", "ETF – Gold (GLD)"]
    watch_options = unique_asset_labels()
    default_watch = [x for x in deduplicate_selection(default_watch) if x in watch_options]

    raw_watch_assets = st.multiselect(
        "Watchlist auswählen",
        watch_options,
        default=default_watch,
        key="watchlist_assets_unique",
    )

    watch_assets = deduplicate_selection(raw_watch_assets)

    if len(raw_watch_assets) != len(watch_assets):
        st.warning("Doppelte Einträge wurden automatisch entfernt.")

    rows = []
    for label in watch_assets:
        t = ASSETS[label]["ticker"]
        d = load_market_data(t, "1y", "1d")
        if d.empty:
            continue
        f = prepare_live_features(d)
        latest = f.dropna().tail(1)
        if latest.empty:
            continue
        latest = latest.iloc[0]
        ts_local = trend_score(latest)
        rows.append(
            {
                "Asset": label,
                "Ticker": t,
                "Typ": ASSETS[label]["type"],
                "Kategorie": ASSETS[label]["category"],
                "Performance 1Y": (f["close"].iloc[-1] / f["close"].iloc[0]) - 1,
                "Volatilität 20d": latest.get("volatility_20d", np.nan),
                "Drawdown": latest.get("drawdown", np.nan),
                "Trend Score": ts_local["score"],
                "Letzter Preis": latest["close"],
            }
        )

    if not rows:
        st.warning("Keine Watchlist-Daten verfügbar.")
        st.stop()

    watch_df = pd.DataFrame(rows)

    st.dataframe(watch_df, width='stretch')

    fig_perf = px.bar(
        watch_df,
        x="Ticker",
        y="Performance 1Y",
        text=watch_df["Performance 1Y"].map(lambda x: f"{x:.1%}"),
        hover_data=["Asset", "Kategorie", "Volatilität 20d", "Drawdown", "Trend Score"],
        title="Performance 1 Jahr",
    )
    fig_perf.update_traces(textposition="outside", hovertemplate="<b>%{x}</b><br>Performance: %{y:.2%}<extra></extra>")
    fig_perf = apply_chart_theme(fig_perf, theme_mode)
    render_interactive_chart(
        fig_perf,
        data=comparison_df if "comparison_df" in locals() else None,
        title="Performance 1 Jahr",
        key="watchlist_performance_chart",
    )

    watch_df["Signalstärke"] = watch_df["Trend Score"].abs() + 1

    fig_risk = px.scatter(
        watch_df,
        x="Volatilität 20d",
        y="Performance 1Y",
        size="Signalstärke",
        color="Trend Score",
        hover_name="Asset",
        hover_data=["Ticker", "Kategorie", "Drawdown", "Letzter Preis", "Trend Score"],
        title="Risiko-Rendite-Vergleich",
    )
    fig_risk.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>Volatilität 20d: %{x:.2%}<br>Performance 1Y: %{y:.2%}<extra></extra>"
    )
    fig_risk = apply_chart_theme(fig_risk, theme_mode)
    render_interactive_chart(
        fig_risk,
        data=comparison_df if "comparison_df" in locals() else None,
        title="Risikovergleich",
        key="watchlist_risk_chart",
    )


# =========================================================
# DECISION JOURNAL
# =========================================================

if page == "Entscheidungsjournal":
    st.header("Entscheidungsjournal")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Entscheidungen dokumentieren statt vergessen</h2>
        <p>
        Ein Journal macht sichtbar, warum du eine Entscheidung getroffen oder bewusst nicht getroffen hast.
        Das unterstützt Lernen, Reflexion und professionellere Kapitalentscheidungen.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    journal = load_journal()

    if journal.empty:
        st.info("Noch keine Journal-Einträge vorhanden. Du kannst im Wealth Outlook eine Analyse speichern.")
    else:
        st.dataframe(journal, width='stretch')

        csv = journal.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Journal als CSV herunterladen",
            data=csv,
            file_name="wealthscope_decision_journal.csv",
            mime="text/csv",
        )

        if "decision" in journal.columns:
            decision_counts = journal["decision"].value_counts().reset_index()
            decision_counts.columns = ["Entscheidung", "Anzahl"]
            fig = px.bar(decision_counts, x="Entscheidung", y="Anzahl", text="Anzahl", title="Journal-Entscheidungen")
            fig.update_traces(textposition="outside")
            fig = apply_chart_theme(fig, theme_mode)
            render_interactive_chart(
                fig,
                data=decision_counts,
                title="Journal-Entscheidungen",
                key="journal_decision_chart",
            )




# =========================================================
# NEWS & FRESHNESS
# =========================================================

if page == "News & Aktualität":
    st.header("News & Aktualität")

    api_key = get_news_api_key()
    query = build_news_query(ticker, asset_label, category)

    st.write(f"Suchquery: `{query}`")

    if not api_key:
        st.warning("NEWS_API_KEY fehlt. Lege ihn in `.streamlit/secrets.toml` ab, wenn Live-News geladen werden sollen.")

    result = fetch_news(query, api_key, language="en", page_size=12) if api_key else {"status": "missing_key", "articles": [], "message": "NEWS_API_KEY fehlt."}
    articles = result.get("articles", []) if result.get("status") == "ok" else []

    nf = news_freshness(articles)
    mf = market_data_freshness(market_data, interval)

    c1, c2 = st.columns(2)

    with c1:
        last_ts = mf.get("last_timestamp")
        last_txt = last_ts.strftime("%d.%m.%Y %H:%M") if last_ts else "unbekannt"
        st.markdown(
            f"""
            <div class="info-card {mf['card']}">
            <h3>Marktdaten</h3>
            <div class="big-number">{mf['status']}</div>
            <p>Letzte Kerze: {last_txt}<br>Alter: {format_age_minutes(mf.get('minutes_old'))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        latest_news = nf.get("latest_timestamp")
        news_txt = latest_news.strftime("%d.%m.%Y %H:%M") if latest_news else "unbekannt"
        st.markdown(
            f"""
            <div class="info-card {nf['card']}">
            <h3>News</h3>
            <div class="big-number">{nf['status']}</div>
            <p>Neueste News: {news_txt}<br>Alter: {format_age_hours(nf.get('hours_old'))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    news_intel = analyze_news_intelligence(articles, ticker, asset_type, category)

    st.divider()
    render_news_intelligence_summary(news_intel, app_mode)

    st.subheader("News-Risiko-Radar")
    render_news_risk_radar(news_intel)

    st.subheader("Bewertete Nachrichten")
    render_analyzed_news_cards(news_intel, app_mode, max_articles=12)




# =========================================================
# OPERATING STATUS
# =========================================================


# =========================================================
# ROBUST PLOTLY + BETRIEBSSTATUS HELPERS
# =========================================================



def safe_file_size_mb(path_obj):
    try:
        p = Path(path_obj)
        if p.exists():
            return round(p.stat().st_size / (1024 * 1024), 2)
    except Exception:
        pass
    return 0


def status_level(score):
    if score >= 85:
        return "OK", "🟢"
    if score >= 60:
        return "Eingeschränkt", "🟡"
    return "Kritisch", "🔴"


def build_operational_status(market_data, feature_data):
    now = dt.datetime.now()

    try:
        market_rows = len(market_data)
        latest_market_date = pd.to_datetime(market_data.index.max()) if not market_data.empty else None
    except Exception:
        market_rows = 0
        latest_market_date = None

    try:
        feature_rows = len(feature_data)
    except Exception:
        feature_rows = 0

    processed_path = Path("data/processed/wealthscope_features.csv")
    market_path = Path("data/processed/wealthscope_market_dataset.csv")
    model_path = Path("models/wealthscope_model.joblib")
    model_report_path = Path("models/wealthscope_model_report.txt")

    market_score = 100 if market_rows > 50 else 40
    feature_score = 100 if feature_rows > 0 else 45
    data_score = 100 if processed_path.exists() and market_path.exists() else 50
    model_score = 100 if model_path.exists() else 45
    report_score = 100 if model_report_path.exists() else 55

    try:
        news_key_available = bool(get_news_api_key())
    except Exception:
        news_key_available = False

    news_score = 85 if news_key_available else 55

    app_size_mb = safe_file_size_mb("app.py")
    performance_score = 90 if app_size_mb < 2 else 75 if app_size_mb < 5 else 60

    scores = {
        "Marktdaten": market_score,
        "Feature-Daten": feature_score,
        "Datenbasis": data_score,
        "ML-Modell": model_score,
        "Modellreport": report_score,
        "News API": news_score,
        "Performance": performance_score,
    }

    overall_score = round(sum(scores.values()) / len(scores))
    overall_status, overall_icon = status_level(overall_score)

    return {
        "checked_at": now.strftime("%d.%m.%Y %H:%M:%S"),
        "overall_score": overall_score,
        "overall_status": overall_status,
        "overall_icon": overall_icon,
        "scores": scores,
        "market_rows": market_rows,
        "latest_market_date": str(latest_market_date.date()) if latest_market_date is not None else "Nicht verfügbar",
        "feature_rows": feature_rows,
        "processed_size_mb": safe_file_size_mb(processed_path),
        "market_size_mb": safe_file_size_mb(market_path),
        "model_size_mb": safe_file_size_mb(model_path),
        "report_size_mb": safe_file_size_mb(model_report_path),
        "news_key_available": news_key_available,
        "app_size_mb": app_size_mb,
    }


def render_status_metric_card(title, value, subtitle, score):
    status, icon = status_level(score)

    st.markdown(
        f"""
        <div class="mini-card">
            <h4>{icon} {title}</h4>
            <h2>{value}</h2>
            <p>{subtitle}</p>
            <p><b>Status:</b> {status}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )



def render_clean_table(df, title=None):
    if title:
        st.markdown(f"### {title}")

    html_rows = ""
    cols = list(df.columns)

    header = "".join([f"<th>{c}</th>" for c in cols])

    for _, row in df.iterrows():
        cells = "".join([f"<td>{row[c]}</td>" for c in cols])
        html_rows += f"<tr>{cells}</tr>"

    st.markdown(
        f"""
        <div class="clean-table-wrap">
            <table class="clean-table">
                <thead><tr>{header}</tr></thead>
                <tbody>{html_rows}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_betriebsstatus_page(market_data, feature_data, theme_mode):
    status = build_operational_status(market_data, feature_data)

    st.header("Betriebsstatus")
    st.caption("Statusübersicht für Daten, Modell, News, Performance und lokale Betriebsbereitschaft.")

    st.markdown(
        f"""
        <div class="hero-card">
            <h2>{status["overall_icon"]} Systemstatus: {status["overall_status"]}</h2>
            <p><b>Gesamtscore:</b> {status["overall_score"]} / 100</p>
            <p><b>Letzter Check:</b> {status["checked_at"]}</p>
            <p>
            Diese Seite zeigt, ob die wichtigsten Komponenten der Demo-App verfügbar,
            aktuell und präsentationsbereit sind.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_status_metric_card(
            "Marktdaten",
            status["latest_market_date"],
            f'{status["market_rows"]} Zeilen geladen',
            status["scores"]["Marktdaten"],
        )

    with c2:
        render_status_metric_card(
            "News",
            "Verfügbar" if status["news_key_available"] else "Optional",
            "NEWS_API_KEY vorhanden" if status["news_key_available"] else "Kein NEWS_API_KEY gesetzt",
            status["scores"]["News API"],
        )

    with c3:
        render_status_metric_card(
            "ML-Modell",
            "Verfügbar" if status["scores"]["ML-Modell"] >= 85 else "Fehlt",
            f'{status["model_size_mb"]} MB lokal',
            status["scores"]["ML-Modell"],
        )

    with c4:
        render_status_metric_card(
            "Performance",
            f'{status["scores"]["Performance"]}/100',
            f'App-Datei ca. {status["app_size_mb"]} MB',
            status["scores"]["Performance"],
        )

    st.subheader("Komponentenstatus")

    component_df = pd.DataFrame(
        [
            {
                "Komponente": name,
                "Score": score,
                "Status": status_level(score)[0],
            }
            for name, score in status["scores"].items()
        ]
    )

    fig_status = px.bar(
        component_df,
        x="Komponente",
        y="Score",
        text="Score",
        title="Status-Scores je Komponente",
        range_y=[0, 100],
    )
    fig_status.update_traces(textposition="outside")
    fig_status = apply_chart_theme(fig_status, theme_mode, height=430)
    render_interactive_chart(
        fig_status,
        data=status_df if "status_df" in locals() else None,
        title="Systemstatus",
        key="system_status_chart",
    )

    st.subheader("Systemübersicht")

    left, right = st.columns([1.25, 1])

    with left:
        render_clean_table(component_df)

    with right:
        fig_pie = px.pie(
            component_df,
            names="Komponente",
            values="Score",
            title="Betriebsbereitschaft nach Komponente",
            hole=0.42,
        )
        fig_pie = apply_chart_theme(fig_pie, theme_mode, height=420)
        render_interactive_chart(
        fig_pie,
        data=portfolio_df if "portfolio_df" in locals() else None,
        title="Portfolio-Gewichtung",
        key="portfolio_weight_chart",
    )

    st.subheader("Datenbasis & lokale Artefakte")

    storage_df = pd.DataFrame(
        [
            {"Datei": "wealthscope_market_dataset.csv", "Größe MB": status["market_size_mb"]},
            {"Datei": "wealthscope_features.csv", "Größe MB": status["processed_size_mb"]},
            {"Datei": "wealthscope_model.joblib", "Größe MB": status["model_size_mb"]},
            {"Datei": "wealthscope_model_report.txt", "Größe MB": status["report_size_mb"]},
            {"Datei": "app.py", "Größe MB": status["app_size_mb"]},
        ]
    )

    fig_storage = px.bar(
        storage_df,
        x="Datei",
        y="Größe MB",
        text="Größe MB",
        title="Lokale Dateien und Größe",
    )
    fig_storage.update_traces(textposition="outside")
    fig_storage = apply_chart_theme(fig_storage, theme_mode, height=430)
    render_interactive_chart(
        fig_storage,
        data=storage_df if "storage_df" in locals() else None,
        title="Datenbasis / Speicherstruktur",
        key="storage_chart",
    )

    st.subheader("Erreichbarkeit & Qualität")

    reachability_df = pd.DataFrame(
        [
            {"Prüfpunkt": "Streamlit-App lokal erreichbar", "Ergebnis": "OK"},
            {"Prüfpunkt": "Marktdaten geladen", "Ergebnis": "OK" if status["market_rows"] > 0 else "Fehler"},
            {"Prüfpunkt": "Feature-Daten erzeugt", "Ergebnis": "OK" if status["feature_rows"] > 0 else "Fehler"},
            {"Prüfpunkt": "ML-Modell verfügbar", "Ergebnis": "OK" if status["scores"]["ML-Modell"] >= 85 else "Fehlt"},
            {"Prüfpunkt": "News API konfiguriert", "Ergebnis": "OK" if status["news_key_available"] else "Optional / fehlt"},
            {"Prüfpunkt": "Exportbereich verfügbar", "Ergebnis": "OK"},
            {"Prüfpunkt": "Impressum/Datenschutz vorhanden", "Ergebnis": "OK"},
        ]
    )

    render_clean_table(reachability_df)

    usage_df = pd.DataFrame(
        [
            {"Bereich": "Geführte Ansicht", "Nutzen": "Klartext für nicht-technische Nutzer", "Status": "Umgesetzt"},
            {"Bereich": "Expertenansicht", "Nutzen": "Kennzahlen und Modellherleitung", "Status": "Umgesetzt"},
            {"Bereich": "Bottom-Bar", "Nutzen": "Schneller Zugriff auf Service-Seiten", "Status": "Umgesetzt"},
            {"Bereich": "News-Archiv", "Nutzen": "Nachrichtenbewertung und Nachvollziehbarkeit", "Status": "Optional mit API"},
            {"Bereich": "Export", "Nutzen": "Markdown/PDF für Professor", "Status": "Umgesetzt"},
            {"Bereich": "Datenschutz/Impressum", "Nutzen": "Transparenz für Demo-Betrieb", "Status": "Vorhanden"},
        ]
    )

    st.subheader("Nutzung & Präsentationsreife")
    render_clean_table(usage_df)

    st.info(
        "Der Betriebsstatus bewertet die lokale Demo-Bereitschaft. "
        "Für ein Produktivsystem wären echtes Monitoring, Error Logging, Uptime Checks, API-Health-Checks "
        "und ein Deployment-Konzept zusätzlich notwendig."
    )



if page == "Betriebsstatus":
    render_betriebsstatus_page(market_data, feature_data, theme_mode)


if page == "Stabilität & QA":
    st.header("Stabilität & QA")
    st.caption("Interne Seite für Entwicklung, Prüfung und Präsentationssicherheit.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Interner Systemcheck</h2>
        <p>
        Diese Seite ist nicht für Endnutzer gedacht. Sie prüft lokale Dateien, Datenquellen,
        Modellverfügbarkeit, Datenschutzstatus, Git-Schutzregeln und Ladezeiten.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Cookie- und Datenschutzstatus")
    st.dataframe(privacy_status_table(), width='stretch')

    st.divider()

    st.subheader("Datei- und Modellstatus")
    if st.button("Stabilitätschecks ausführen"):
        df_files, df_perf = run_stability_checks(ticker=ticker)

        st.write("**Dateien**")
        st.dataframe(df_files, width='stretch')

        st.write("**Performance / Ladezeiten**")
        st.dataframe(df_perf, width='stretch')

        ok_count = (df_perf["Status"] == "OK").sum()
        total_count = len(df_perf)

        if ok_count == total_count:
            st.success("Alle Kernchecks sind erfolgreich.")
        else:
            st.warning("Einige Checks sind nicht vollständig erfolgreich. Details prüfen.")

    st.divider()

    st.subheader("Gitignore-Qualität")
    st.dataframe(gitignore_quality_check(), width='stretch')

    st.divider()

    st.subheader("Präsentations-Checkliste")
    checklist = pd.DataFrame(
        [
            {"Punkt": "App startet ohne Fehler", "Status": "manuell prüfen"},
            {"Punkt": "Wealth Outlook funktioniert", "Status": "manuell prüfen"},
            {"Punkt": "Kapital-Kompass funktioniert", "Status": "manuell prüfen"},
            {"Punkt": "Portfolio-Simulator funktioniert", "Status": "manuell prüfen"},
            {"Punkt": "Watchlist-Vergleich funktioniert", "Status": "manuell prüfen"},
            {"Punkt": "News & Aktualität funktioniert", "Status": "abhängig von NEWS_API_KEY"},
            {"Punkt": "Decision Passport Download funktioniert", "Status": "manuell prüfen"},
            {"Punkt": "Journal wird lokal gespeichert", "Status": "manuell prüfen"},
            {"Punkt": "Große Daten sind nicht auf GitHub", "Status": "durch .gitignore abgesichert"},
        ]
    )
    st.dataframe(checklist, width='stretch')



# =========================================================
# EXPERT ANALYSIS
# =========================================================

if page == "Expertenanalyse":
    st.header("Expertenanalyse")

    if feature_data.empty:
        st.error("Keine Daten verfügbar.")
        st.stop()

    latest = feature_data.dropna().tail(1).iloc[0]
    levels = detect_levels(feature_data)
    ts = trend_score(latest)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Letzter Preis", f"{latest['close']:.2f}")
    c2.metric("20d Return", f"{latest.get('return_20d', np.nan):.2%}")
    c3.metric("Drawdown", f"{latest.get('drawdown', np.nan):.2%}")
    c4.metric("Trend Score", ts["score"])

    st.plotly_chart(create_price_chart(feature_data.tail(700), ticker), width='stretch')

    col1, col2 = st.columns(2)

    with col1:
        rsi_fig = px.line(feature_data, x="date", y="rsi", title="RSI")
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        render_interactive_chart(
        rsi_fig,
        data=feature_data.tail(700) if "feature_data" in globals() and feature_data is not None and not feature_data.empty else None,
        title="RSI-Indikator",
        key="rsi_chart",
    )

    with col2:
        macd_fig = px.line(feature_data, x="date", y=["macd", "macd_signal"], title="MACD")
        render_interactive_chart(
        macd_fig,
        data=feature_data.tail(700) if "feature_data" in globals() and feature_data is not None and not feature_data.empty else None,
        title="MACD-Indikator",
        key="macd_chart",
    )

    st.subheader("Technische Kennzahlen")
    metrics_df = pd.DataFrame(
        [
            {"Kennzahl": "1Y High", "Wert": levels.get("one_year_high")},
            {"Kennzahl": "1Y Low", "Wert": levels.get("one_year_low")},
            {"Kennzahl": "Support 60d", "Wert": levels.get("recent_support")},
            {"Kennzahl": "Resistance 60d", "Wert": levels.get("recent_resistance")},
            {"Kennzahl": "MA 20", "Wert": latest.get("ma_20")},
            {"Kennzahl": "MA 50", "Wert": latest.get("ma_50")},
            {"Kennzahl": "MA 200", "Wert": latest.get("ma_200")},
            {"Kennzahl": "Volatility 20d", "Wert": latest.get("volatility_20d")},
        ]
    )
    st.dataframe(metrics_df, width='stretch')


# =========================================================
# MODEL & DATA
# =========================================================

if page == "Modell & Datenbasis":
    st.header("Modell & Datenbasis")

    st.subheader("Big-Data-Rohbasis")
    st.markdown(
        """
        - Kaggle-Dataset: `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`
        - Titel: Huge Stock Market Dataset
        - Lokale `.txt`-Dateien: 17.078
        - Stock-Dateien: 7.195
        - ETF-Dateien: 1.344
        - Lokal gezählte Datenzeilen: 34.906.486
        - Big-Data-Anforderung > 1.000.000: erfüllt
        - Lizenz: CC0-1.0
        """
    )

    st.subheader("Bereinigte Modellbasis")
    st.markdown(
        """
        - `data/processed/wealthscope_market_dataset.csv`
        - 96.017 Zeilen
        - 18 Ticker
        - 9 Spalten
        """
    )

    st.subheader("Feature-Datei")
    st.markdown(
        """
        - `data/processed/wealthscope_features.csv`
        - 92.075 Zeilen
        - 23 Spalten
        - Zielvariable: `target_20d`
        """
    )

    st.subheader("Modellreport")

    if MODEL_REPORT_PATH.exists():
        st.code(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.warning("Modellreport lokal nicht gefunden.")

    st.info("Das Modell ist ein erster Prototyp. Die Accuracy liegt bei ca. 53,25 % und wird daher nur als ergänzender Baustein genutzt.")


# =========================================================
# METHODOLOGY
# =========================================================

if page == "Methodik & Grenzen":
    st.header("Methodik & Grenzen")

    st.subheader("Was die App macht")
    st.write(
        "WealthScope AI kombiniert historische Marktbewegungen, aktuelle Nachrichten, technische Kennzahlen, "
        "Kapital-Kontext und ein ML-Szenario zu einer verständlichen Analyse."
    )

    st.subheader("Was die App nicht macht")
    st.markdown(
        """
        - Keine Finanzberatung
        - Keine Anlageempfehlung
        - Keine sichere Prognose
        - Keine automatische Orderausführung
        - Kein Ersatz für professionelle Beratung
        """
    )

    st.subheader("Warum Guided Mode und Expert Mode?")
    st.write(
        "Nicht-technische Kapitalinhaber brauchen Klartext und nächste Schritte. Experten brauchen Transparenz, Daten und Modellwerte. "
        "Deshalb trennt die App die Darstellung, nutzt aber dieselbe Datenbasis."
    )

    st.subheader("Accuracy vs. Confidence")
    st.write(
        "Accuracy beschreibt die historische Testleistung des Modells. Confidence bewertet die aktuelle Belastbarkeit der Einschätzung "
        "aus Datenfrische, News-Aktualität, technischer Eindeutigkeit, Modellgüte und Volatilität. Beides ist keine Garantie."
    )

    st.subheader("Premium-Erweiterungen")
    st.markdown(
        """
        - Portfolio-Simulator für mehrere Assets
        - Risikoprofil-Fragebogen
        - Was-wäre-wenn-Szenarien
        - Watchlist-Vergleich
        - Entscheidungsjournal
        - News-Risiko-Radar
        - Advisor Explanation
        - Aktualitäts-Timer
        - Guided Mode und Expert Mode mit unterschiedlicher Informationstiefe
        """
    )



# =========================================================
# BOTTOM STATUS BAR
# =========================================================



def render_bottom_status_bar():
    """Feste untere Statusleiste mit stabilen Links inkl. Theme/View."""
    import datetime

    market_status = "Aktuell"
    news_status = "Mittel"
    checked = datetime.datetime.now().strftime("%H:%M")

    news_link = route_link("News-Archiv")
    howto_link = route_link("So funktioniert's")
    project_link = route_link("Über das Projekt")
    export_link = route_link("Professor-Export")
    impressum_link = route_link("Impressum")
    privacy_link = route_link("Datenschutz")
    status_link = route_link("Betriebsstatus")

    bottom_bar_html = f"""
    <nav class="bottom-status-bar">
        <a class="bottom-status-item bottom-status-link" href="{news_link}" target="_self">
            <span class="bottom-status-icon">📰</span>News-Archiv
        </a>
        <a class="bottom-status-item bottom-status-link" href="{howto_link}" target="_self">
            <span class="bottom-status-icon">📖</span>So funktioniert's
        </a>
        <a class="bottom-status-item bottom-status-link" href="{project_link}" target="_self">
            <span class="bottom-status-icon">🎓</span>Projekt
        </a>
        <a class="bottom-status-item bottom-status-link" href="{export_link}" target="_self">
            <span class="bottom-status-icon">📦</span>Export
        </a>
        <a class="bottom-status-item bottom-status-link" href="{impressum_link}" target="_self">
            <span class="bottom-status-icon">⚖️</span>Impressum
        </a>
        <a class="bottom-status-item bottom-status-link" href="{privacy_link}" target="_self">
            <span class="bottom-status-icon">🛡️</span>Datenschutz
        </a>
        <a class="bottom-status-item bottom-status-link" href="{status_link}" target="_self">
            <span class="bottom-status-icon">🟢</span>Betriebsstatus
        </a>

    </nav>
    """

    st.markdown(bottom_bar_html, unsafe_allow_html=True)


def markdown_to_pdf_bytes(title, markdown_text):
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "WealthScopeTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=18,
        alignment=TA_LEFT,
    )

    h1_style = ParagraphStyle(
        "WealthScopeH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=14,
        spaceAfter=8,
    )

    h2_style = ParagraphStyle(
        "WealthScopeH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "WealthScopeBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "WealthScopeBullet",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        leftIndent=12,
        bulletIndent=0,
        spaceAfter=4,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2 * cm))

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()

        if not line:
            story.append(Spacer(1, 0.16 * cm))
            continue

        # Markdown grob in PDF-Struktur übertragen
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), h1_style))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), h2_style))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), h2_style))
        elif line.startswith("- "):
            safe = line[2:].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = safe.replace("**", "")
            story.append(Paragraph(safe, bullet_style, bulletText="•"))
        else:
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = safe.replace("**", "")
            safe = safe.replace("`", "")
            story.append(Paragraph(safe, body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def export_download_card(title, description, filename_base, markdown_content, export_format):
    st.markdown(
        f"""
        <div class="export-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if export_format in ["Markdown", "Beides"]:
        st.download_button(
            f"{title} als Markdown herunterladen",
            data=markdown_content,
            file_name=f"{filename_base}.md",
            mime="text/markdown",
            key=f"download_md_{filename_base}",
        )

    if export_format in ["PDF", "Beides"]:
        pdf_bytes = markdown_to_pdf_bytes(title, markdown_content)
        st.download_button(
            f"{title} als PDF herunterladen",
            data=pdf_bytes,
            file_name=f"{filename_base}.pdf",
            mime="application/pdf",
            key=f"download_pdf_{filename_base}",
        )


# Rückwärtskompatibilität, falls alte Aufrufe noch existieren
def markdown_download_card(title, description, filename, content):
    filename_base = filename.replace(".md", "").replace(".txt", "")
    export_download_card(title, description, filename_base, content, "Markdown")


def render_qua3ck_detail_cards():
    st.subheader("QUA3CK im Detail")

    with st.expander("Q – Question: Problemdefinition"):
        st.markdown(
            """
            **Ziel dieser Phase:**  
            Die zentrale Machine-Learning-Frage wird definiert.

            **Auf WealthScope AI angewendet:**  
            Die Leitfrage lautet:  
            *Wie können historische Aktien-/ETF-Daten, aktuelle Nachrichten und Risikokennzahlen genutzt werden, um Kapitalentscheidungen verständlicher und nachvollziehbarer zu machen?*

            **Warum wichtig:**  
            Ohne klare Fragestellung würde die App nur Charts anzeigen. Durch die Question-Phase wird festgelegt, dass nicht die reine Kursprognose im Mittelpunkt steht, sondern eine verständliche Entscheidungshilfe für nicht-technische Nutzer.

            **Konkretes Ergebnis:**  
            - Zielgruppe: Menschen mit neuem Kapital, z. B. Erbe, Abfindung oder Vermögensaufbau  
            - Ziel: Orientierung statt Finanzberatung  
            - Output: Wealth Outlook, Kapital-Schutz-Score, News-Bewertung und Decision Passport
            """
        )

    with st.expander("U – Understanding Data: Datenverständnis"):
        st.markdown(
            """
            **Ziel dieser Phase:**  
            Die Datenbasis wird geprüft, verstanden und dokumentiert.

            **Auf WealthScope AI angewendet:**  
            Genutzt wird das Kaggle-Dataset `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`.

            **Dokumentierte Datenbasis:**  
            - 17.078 `.txt`-Dateien lokal gefunden  
            - 7.195 Stock-Dateien  
            - 1.344 ETF-Dateien  
            - 34.906.486 lokal gezählte Datenzeilen  
            - Big-Data-Anforderung > 1.000.000 erfüllt  
            - Lizenz: CC0-1.0

            **Warum wichtig:**  
            Diese Phase zeigt, dass die App nicht auf kleinen Beispieldaten basiert, sondern auf einer großen historischen Marktdatenbasis. Gleichzeitig werden Grenzen sichtbar, z. B. Datenalter, mögliche Duplikate und fehlende Live-Rohdaten im Kaggle-Dataset.

            **Konkretes Ergebnis:**  
            - `wealthscope_market_dataset.csv` mit 96.017 Zeilen  
            - `wealthscope_features.csv` mit 92.075 Zeilen und 23 Spalten  
            - kontrollierter Trainings-/Prototyping-Datensatz aus der großen Rohdatenbasis
            """
        )

    with st.expander("A³ – Algorithm, Adaption, Adjustment"):
        st.markdown(
            """
            **Ziel dieser Phase:**  
            In der iterativen A³-Schleife werden Algorithmus, Datenanpassung und Parameter verbessert.

            **Algorithm Selection:**  
            Als erster ML-Prototyp wurde ein `RandomForestClassifier` verwendet. Der Algorithmus ist für tabellarische Daten geeignet und liefert nachvollziehbare Klassifikationen.

            **Adaption / Feature Engineering:**  
            Aus den Kursdaten wurden technische Merkmale erzeugt, z. B.:

            - tägliche Rendite  
            - 5-Tage- und 20-Tage-Rendite  
            - gleitende Durchschnitte: MA20, MA50, MA200  
            - Abstand zum gleitenden Durchschnitt  
            - 20-Tage-Volatilität  
            - Rolling High  
            - Drawdown  
            - zukünftige 20-Tage-Rendite  
            - Zielvariable `target_20d`

            **Adjustment:**  
            Das Modell wurde mit Trainings- und Testdaten geprüft. Die Accuracy lag bei ca. **53,25 %**.

            **Warum wichtig:**  
            Das Ergebnis wird bewusst nicht überverkauft. Die Modellleistung liegt nur leicht über Zufall. Deshalb wird das Modell in der App nicht als sichere Prognose verwendet, sondern nur als ein Baustein im Confidence Score.

            **Konkretes Ergebnis:**  
            - Modell: `models/wealthscope_model.joblib` lokal  
            - Report: `models/wealthscope_model_report.txt`  
            - Modellbewertung transparent in der App dokumentiert
            """
        )

    with st.expander("C – Conclude & Compare: Bewertung und Vergleich"):
        st.markdown(
            """
            **Ziel dieser Phase:**  
            Ergebnisse werden bewertet, verglichen und kritisch eingeordnet.

            **Auf WealthScope AI angewendet:**  
            Die App vergleicht nicht nur Modellwerte, sondern kombiniert mehrere Perspektiven:

            - historische Kursentwicklung  
            - Volatilität  
            - Drawdown  
            - gleitende Durchschnitte  
            - News-Sentiment  
            - News-Relevanz  
            - Kapitalgewichtung  
            - Risikoprofil des Nutzers  
            - Modell-Confidence

            **Wichtige Schlussfolgerung:**  
            Das ML-Modell allein ist nicht stark genug für eine verlässliche Prognose. Der eigentliche Mehrwert entsteht durch die Kombination aus Datenanalyse, News Intelligence, Kapital-Schutz und verständlicher UX.

            **Warum wichtig:**  
            Diese Phase schützt vor falschen Aussagen. Die App macht transparent, dass eine Einschätzung keine Garantie ist.

            **Konkretes Ergebnis:**  
            - klare Hinweise: keine Finanzberatung  
            - Confidence ist keine Trefferwahrscheinlichkeit  
            - Betriebsstatus zeigt Aktualität von Marktdaten und News  
            - Methodik & Grenzen werden offen dargestellt
            """
        )

    with st.expander("K – Knowledge Transfer: Transfer in eine nutzbare Anwendung"):
        st.markdown(
            """
            **Ziel dieser Phase:**  
            Das erarbeitete Wissen wird in eine verständliche Lösung übertragen.

            **Auf WealthScope AI angewendet:**  
            Die Ergebnisse wurden in eine Streamlit-App übertragen, die zwei Zielgruppen berücksichtigt:

            **Geführte Ansicht:**  
            Für nicht-technische Nutzer mit Fokus auf Klartext, Risiko, Kapital-Schutz und nächste Schritte.

            **Expertenansicht:**  
            Für Nutzer mit Vorwissen mit Fokus auf technische Kennzahlen, Modellwerte, Score-Zerlegung und Datenstatus.

            **Transfer-Leistungen in der App:**  
            - Wealth Outlook  
            - Kapital-Kompass  
            - Portfolio-Simulator  
            - Watchlist-Vergleich  
            - News-Archiv  
            - Betriebsstatus  
            - Datenschutz / Impressum  
            - Professor-Export  
            - Decision Passport  
            - Methodik & Grenzen

            **Warum wichtig:**  
            Das Projekt bleibt nicht nur ein Notebook oder Modellreport, sondern wird als nutzerorientierte Anwendung greifbar.
            """
        )


# =========================================================
# BOTTOM BAR PAGES - FINAL CONTENT
# =========================================================

if page == "News-Archiv":
    st.header("News-Archiv")
    st.caption("Nachrichtenübersicht mit Bewertung, Quelle und Relevanz.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Aktuelle Nachrichten nachvollziehbar machen</h2>
        <p>
        Das News-Archiv zeigt, welche Nachrichten in die Einschätzung einfließen können.
        Jede Nachricht wird nach Sentiment, Relevanz und potenziellem Markteinfluss eingeordnet.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_key = get_news_api_key()
    query = build_news_query(ticker, asset_label, category)

    st.markdown(f"**Aktuelle Suchlogik:** `{query}`")

    if not api_key:
        st.warning("NEWS_API_KEY fehlt. Deshalb können aktuell keine Live-News geladen werden.")
        st.info("Für die Präsentation kannst du diese Seite als geplantes News-Archiv mit Bewertungslogik erklären.")
    else:
        result = fetch_news(query, api_key, language="en", page_size=20)
        articles = result.get("articles", []) if result.get("status") == "ok" else []

        if not articles:
            st.warning("Keine News gefunden oder News-Quelle nicht erreichbar.")
        else:
            news_intel = analyze_news_intelligence(articles, ticker, asset_type, category)
            render_news_intelligence_summary(news_intel, app_mode)
            render_analyzed_news_cards(news_intel, app_mode, max_articles=20)


if page == "So funktioniert's":
    st.header("So funktioniert's")
    st.caption("Kurzes Onboarding für neue Nutzer.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>In 5 Schritten zur verständlichen Kapitalanalyse</h2>
        <p>
        WealthScope AI ist bewusst für Menschen gebaut, die Kapital verantwortungsvoll einordnen möchten,
        ohne direkt technische Finanzmodelle verstehen zu müssen.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="decision-step"><b>1. Asset auswählen:</b> Wähle eine Aktie oder einen ETF aus der Liste.</div>
        <div class="decision-step"><b>2. Zeitraum bestimmen:</b> Lege fest, ob kurz-, mittel- oder langfristige Daten betrachtet werden.</div>
        <div class="decision-step"><b>3. Wealth Outlook prüfen:</b> Lies die Ampel, die Confidence und die Kapital-Schutz-Einschätzung.</div>
        <div class="decision-step"><b>4. Nachrichten einordnen:</b> Prüfe, ob aktuelle News die historische Datenlage bestätigen oder widersprechen.</div>
        <div class="decision-step"><b>5. Entscheidung dokumentieren:</b> Nutze Export, Report oder Journal für Nachvollziehbarkeit.</div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Geführte Ansicht vs. Expertenansicht")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="info-card neutral-card">
            <h3>Geführte Ansicht</h3>
            <p>Für nicht-technische Nutzer: klare Einschätzung, einfache Sprache und konkrete Orientierung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card neutral-card">
            <h3>Expertenansicht</h3>
            <p>Für Nutzer mit Vorwissen: Kennzahlen, Modellwerte, Datenstatus und technische Herleitung.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.warning("Keine Finanzberatung. Die App dient Analyse, Lernen und Simulation.")


if page == "Datenschutz":
    st.header("Datenschutz")
    st.caption("Transparenz über Datenverarbeitung in der lokalen Demo-App.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Datenschutz-Hinweis</h2>
        <p>
        Diese lokale Demo-App speichert keine Werbe- oder Tracking-Cookies.
        Eingaben werden lokal verarbeitet. Externe Dienste können für Marktdaten und optional Nachrichten genutzt werden.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Datenverarbeitung")
    st.dataframe(privacy_status_table(), width='stretch')

    st.subheader("Kurz erklärt")
    st.markdown(
        """
        - Es werden keine persönlichen Finanzdaten dauerhaft an externe Systeme gesendet.
        - Journal-Einträge werden lokal gespeichert, sofern diese Funktion genutzt wird.
        - Marktdaten werden über externe Datenquellen geladen.
        - News werden nur geladen, wenn ein API-Key konfiguriert ist.
        - Für einen echten Produktivbetrieb wären vollständiges Impressum, Datenschutztext und Hosting-Konzept erforderlich.
        """
    )


if page == "Impressum":
    st.header("Impressum")
    st.caption("Platzhalter für eine mögliche öffentliche Bereitstellung.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Impressum / Anbieterkennzeichnung</h2>
        <p>
        WealthScope AI ist aktuell eine lokale Demo-Anwendung im Rahmen eines Studienprojekts.
        Die folgenden Angaben sind Platzhalter und müssten vor einer echten Veröffentlichung rechtlich geprüft werden.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Projektangaben")
    st.markdown(
        """
        **Projektname:** WealthScope AI  
        **Art:** Studienprojekt / Demo-Anwendung  
        **Zweck:** Big-Data-, Machine-Learning- und UX-Prototyp zur verständlichen Kapitalanalyse  
        **Hinweis:** Keine Finanzberatung, keine Anlageempfehlung, keine Garantie.
        """
    )

    st.subheader("Für eine echte Veröffentlichung zu ergänzen")
    st.markdown(
        """
        - vollständiger Name oder Organisation
        - ladungsfähige Anschrift
        - Kontaktmöglichkeit
        - verantwortliche Person
        - rechtliche Pflichtangaben
        - Datenschutzbeauftragter, falls erforderlich
        """
    )

    st.warning("Dies ist kein rechtlich vollständiges Impressum.")


if page == "Über das Projekt":
    st.header("Über das Projekt")
    st.caption("Projektidee, wissenschaftlicher Bezug und QUA3CK-Struktur.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>WealthScope AI</h2>
        <p>
        WealthScope AI ist ein Big-Data- und Machine-Learning-Projekt zur verständlichen Kapitalanalyse.
        Es verbindet historische Aktien-/ETF-Daten, aktuelle Nachrichten, technische Indikatoren,
        ML-Prototyping und eine nutzerorientierte Oberfläche.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("QUA3CK-Bezug")

    q1, q2, q3, q4, q5 = st.columns(5)

    with q1:
        st.markdown('<div class="mini-card"><h3>Q</h3><p><b>Question</b><br>Wie können historische Daten und News Kapitalentscheidungen verständlicher machen?</p></div>', unsafe_allow_html=True)
    with q2:
        st.markdown('<div class="mini-card"><h3>U</h3><p><b>Understanding Data</b><br>Analyse von Datenqualität, Tickerstruktur, Zeitreihen und Big-Data-Umfang.</p></div>', unsafe_allow_html=True)
    with q3:
        st.markdown('<div class="mini-card"><h3>A³</h3><p><b>Algorithm, Adaption, Adjustment</b><br>Feature Engineering, Modelltraining und Parameterauswahl.</p></div>', unsafe_allow_html=True)
    with q4:
        st.markdown('<div class="mini-card"><h3>C</h3><p><b>Conclude & Compare</b><br>Ergebnisse kritisch bewerten und Modellgrenzen sichtbar machen.</p></div>', unsafe_allow_html=True)
    with q5:
        st.markdown('<div class="mini-card"><h3>K</h3><p><b>Knowledge Transfer</b><br>Transfer in eine verständliche Streamlit-App mit Guided und Expert Mode.</p></div>', unsafe_allow_html=True)

    st.subheader("Projektkern")
    st.markdown(
        """
        - Big-Data-Rohbasis: 34.906.486 lokal gezählte Datenzeilen
        - Verarbeitete Modellbasis: 96.017 Zeilen
        - Feature-Datei: 92.075 Zeilen und 23 Spalten
        - ML-Modell: RandomForestClassifier
        - Modellgüte: ca. 53,25 % Accuracy
        - Einordnung: transparenter Prototyp, keine sichere Prognose
        """
    )


if page == "Professor-Export":
    st.header("Professor-Export")
    st.caption("Markdown-Export-Bereich für Präsentation, Bewertung und Dokumentation.")

    st.markdown(
        """
        <div class="hero-card">
        <h2>Export-Center für die Projektbewertung</h2>
        <p>
        Hier kannst du die wichtigsten Projektnachweise als Markdown oder PDF herunterladen:
        Projektzusammenfassung, Datenbasis, QUA3CK-Dokumentation, Modellbericht und Präsentationsnotizen.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    export_format = st.radio(
        "Exportformat auswählen",
        ["Markdown", "PDF", "Beides"],
        horizontal=True,
        index=0,
        help="Markdown ist ideal für GitHub und Dokumentation. PDF ist besser für Abgabe, Präsentation oder Weitergabe.",
    )

    st.info(f"Exportformat aktiv: {export_format}")

    project_summary = """# WealthScope AI – Projektzusammenfassung

## 1. Projektziel

WealthScope AI ist eine Streamlit-Anwendung zur verständlichen Analyse von Aktien und ETFs.  
Die App kombiniert historische Marktdaten, aktuelle Nachrichten, technische Kennzahlen, ein erstes Machine-Learning-Modell und Kapital-Schutz-Logik.

Ziel ist nicht, eine sichere Finanzprognose zu geben, sondern eine transparente Entscheidungshilfe zu schaffen.

## 2. Zielgruppe

Die App richtet sich an nicht-technische Nutzer, die Kapital verantwortungsvoll einordnen möchten, zum Beispiel:

- Personen nach einem Erbe
- Personen mit Abfindung
- Menschen mit größeren Rücklagen
- Einsteiger in Aktien und ETFs
- Nutzer, die Märkte verstehen möchten, bevor sie handeln

## 3. Kernfunktionen

- Guided Mode für nicht-technische Nutzer
- Expert Mode für technische Nachvollziehbarkeit
- Wealth Outlook
- Kapital-Kompass
- Portfolio-Simulator
- Watchlist-Vergleich
- News Intelligence
- News-Archiv
- Betriebsstatus
- Datenschutz / Impressum
- Professor-Export
- Methodik & Grenzen

## 4. Big-Data-Basis

Genutztes Kaggle-Dataset:

`borismarjanovic/price-volume-data-for-all-us-stocks-etfs`

Lokale Prüfung:

- 17.078 `.txt`-Dateien
- 7.195 Stock-Dateien
- 1.344 ETF-Dateien
- 34.906.486 lokal gezählte Datenzeilen
- Big-Data-Anforderung > 1.000.000 erfüllt
- Lizenz: CC0-1.0

## 5. Verarbeitete Daten

Für Prototyping und Modelltraining wurde ein kontrollierter Teil der Rohdaten verarbeitet:

- `wealthscope_market_dataset.csv`: 96.017 Zeilen, 9 Spalten, 18 Ticker
- `wealthscope_features.csv`: 92.075 Zeilen, 23 Spalten

## 6. Modell

- Algorithmus: RandomForestClassifier
- Zielvariable: `target_20d`
- Fragestellung: Liegt der Kurs in 20 Handelstagen höher?
- Accuracy: ca. 53,25 %

## 7. Kritische Einordnung

Das Modell liegt nur leicht über Zufall und wird deshalb nicht als sichere Prognose verwendet.  
Es dient als zusätzlicher Baustein im Confidence Score.

## 8. Hinweis

WealthScope AI ist keine Finanzberatung, keine Anlageempfehlung und keine Handelsaufforderung.
"""

    data_basis = """# WealthScope AI – Datenbasis

## 1. Rohdaten

Quelle:

`borismarjanovic/price-volume-data-for-all-us-stocks-etfs`

Dataset-Titel: Huge Stock Market Dataset  
Lizenz: CC0-1.0

## 2. Lokale Datenprüfung

Gezählte lokale Datenbasis:

- `.txt`-Dateien gesamt: 17.078
- Stock-Dateien: 7.195
- ETF-Dateien: 1.344
- Datenzeilen gesamt: 34.906.486

Damit ist die Big-Data-Anforderung von mehr als 1.000.000 Zeilen erfüllt.

## 3. Hinweis zu möglichen Duplikaten

Die Ordnerstruktur enthält sowohl Top-Level-Pfade wie `Stocks/ETFs` als auch `Data/Stocks`.  
Dadurch können Duplikate vorhanden sein. Für die Dokumentation wird deshalb transparent von lokal gezählten Datenzeilen gesprochen.

## 4. Processed Dataset

Datei:

`data/processed/wealthscope_market_dataset.csv`

Eigenschaften:

- 96.017 Zeilen
- 9 Spalten
- 18 Ticker
- Dateigröße ca. 6,8 MB

Spalten:

- date
- open
- high
- low
- close
- volume
- ticker
- asset_type
- source_file

## 5. Feature Dataset

Datei:

`data/processed/wealthscope_features.csv`

Eigenschaften:

- 92.075 Zeilen
- 23 Spalten
- 18 Ticker
- 65.675 Stock-Zeilen
- 26.400 ETF-Zeilen

Beispielfeatures:

- daily_return
- return_5d
- return_20d
- ma_20
- ma_50
- ma_200
- ma_20_distance
- ma_50_distance
- ma_200_distance
- volatility_20d
- rolling_high
- drawdown
- future_return_20d
- target_20d

## 6. Zielvariable

`target_20d`

Bedeutung:

- 1 = Kurs liegt in 20 Handelstagen höher
- 0 = Kurs liegt in 20 Handelstagen nicht höher
"""

    qua3ck_doc = """# WealthScope AI – QUA3CK-Dokumentation

## Q – Question

Die zentrale Frage lautet:

Wie können historische Aktien-/ETF-Daten, aktuelle Nachrichten und Risikokennzahlen genutzt werden, um Kapitalentscheidungen verständlicher und nachvollziehbarer zu machen?

Die App soll keine sichere Finanzprognose liefern, sondern eine transparente Entscheidungshilfe für nicht-technische Nutzer darstellen.

## U – Understanding Data

Die Datenbasis stammt aus dem Kaggle-Dataset `borismarjanovic/price-volume-data-for-all-us-stocks-etfs`.

Lokale Prüfung:

- 17.078 `.txt`-Dateien
- 7.195 Stock-Dateien
- 1.344 ETF-Dateien
- 34.906.486 lokal gezählte Datenzeilen

Für das Modell wurde ein kontrollierter Datenausschnitt erstellt:

- 96.017 Zeilen im Market Dataset
- 92.075 Zeilen im Feature Dataset
- 23 Feature-Spalten

## A³ – Algorithm, Adaption, Adjustment

### Algorithm Selection

Als erster ML-Prototyp wurde ein RandomForestClassifier gewählt.

### Adaption

Aus Kursdaten wurden technische Features erzeugt:

- tägliche Rendite
- 5-Tage- und 20-Tage-Rendite
- gleitende Durchschnitte
- Abstand zu gleitenden Durchschnitten
- Volatilität
- Drawdown
- zukünftige 20-Tage-Rendite
- Zielvariable `target_20d`

### Adjustment

Das Modell wurde trainiert und getestet.  
Die Accuracy lag bei ca. 53,25 %.

## C – Conclude & Compare

Das Modell liegt nur leicht über Zufall. Deshalb wird es nicht als sichere Prognose genutzt.

Die App kombiniert stattdessen:

- historische Daten
- aktuelle News
- Volatilität
- Drawdown
- Kapitalgewichtung
- Risikoprofil
- Modell-Confidence

## K – Knowledge Transfer

Die Erkenntnisse wurden in eine Streamlit-App übertragen.

Umgesetzte Transfer-Elemente:

- Geführte Ansicht
- Expertenansicht
- Wealth Outlook
- Kapital-Kompass
- Portfolio-Simulator
- Watchlist-Vergleich
- News-Archiv
- Betriebsstatus
- Datenschutz und Impressum
- Professor-Export
"""

    model_interpretation = """# WealthScope AI – Modellinterpretation

## Modell

RandomForestClassifier

## Zielvariable

`target_20d`

Bedeutung:

- 1 = Kurs liegt in 20 Handelstagen höher
- 0 = Kurs liegt in 20 Handelstagen nicht höher

## Trainings- und Testdaten

- Gesamtdaten im Feature Dataset: 92.075 Zeilen
- Train Rows: 69.056
- Test Rows: 23.019

## Modellgüte

Accuracy: ca. 53,25 %

## Interpretation

Die Modellleistung liegt nur leicht über Zufall.  
Daraus folgt: Das Modell ist nicht stark genug, um als alleinige Prognosegrundlage genutzt zu werden.

## Konsequenz für die App

Das Modell wird nicht als sichere Prognose dargestellt.  
Es wird nur als zusätzlicher Baustein im Confidence Score genutzt.

## Warum das positiv ist

Die App übertreibt die Modellleistung nicht.  
Sie zeigt transparent, dass Machine Learning im Finanzkontext kritisch eingeordnet werden muss.
"""

    presentation_notes = """# Präsentationsnotizen – WealthScope AI

## Einstieg

Viele Menschen erhalten Kapital, wissen aber nicht, wie sie Märkte, Risiken und aktuelle Nachrichten einordnen sollen.

## Problem

Normale Watchlists zeigen meist nur Kurse.  
Sie beantworten aber nicht:

- Was bedeutet das für mein Kapital?
- Wie riskant ist die Position?
- Sind aktuelle News relevant?
- Wie belastbar ist die Einschätzung?
- Welche Grenzen hat das Modell?

## Lösung

WealthScope AI kombiniert:

- historische Marktdaten
- aktuelle News
- technische Kennzahlen
- Kapital-Schutz-Logik
- ML-Prototyp
- verständliche UX

## Besonderheit

Die App bietet zwei Ansichten:

- Geführte Ansicht für nicht-technische Nutzer
- Expertenansicht für technische Nachvollziehbarkeit

## Big Data

Die Rohdatenbasis umfasst lokal gezählte 34.906.486 Datenzeilen.  
Damit ist die Big-Data-Anforderung erfüllt.

## Machine Learning

Ein RandomForestClassifier wurde als erster Prototyp trainiert.  
Die Accuracy liegt bei ca. 53,25 %.

## Kritische Einordnung

Die Modellleistung wird nicht überverkauft.  
Die App sagt klar: keine sichere Prognose, keine Finanzberatung.

## QUA3CK

Das Projekt folgt der QUA3CK-Logik:

- Question
- Understanding Data
- A³-Prozess
- Conclude & Compare
- Knowledge Transfer

## Abschluss

WealthScope AI zeigt, wie Big Data, Machine Learning, aktuelle News und UX zu einer verständlichen Anwendung kombiniert werden können.
"""

    c1, c2 = st.columns(2)

    with c1:
        export_download_card(
            "Projektzusammenfassung",
            "Kompakter Überblick über Ziel, Zielgruppe, Datenbasis, Modell und Grenzen.",
            "wealthscope_project_summary",
            project_summary,
            export_format,
        )

        export_download_card(
            "QUA3CK-Dokumentation",
            "Wissenschaftliche Einordnung entlang der QUA3CK-Phasen.",
            "wealthscope_qua3ck_documentation",
            qua3ck_doc,
            export_format,
        )

        export_download_card(
            "Präsentationsnotizen",
            "Strukturierte Notizen für die mündliche Vorstellung.",
            "wealthscope_presentation_notes",
            presentation_notes,
            export_format,
        )

    with c2:
        export_download_card(
            "Datenbasis",
            "Dokumentation der Rohdaten, verarbeiteten Daten und Feature-Struktur.",
            "wealthscope_data_basis",
            data_basis,
            export_format,
        )

        export_download_card(
            "Modellinterpretation",
            "Kritische Einordnung des RandomForest-Modells und seiner Grenzen.",
            "wealthscope_model_interpretation",
            model_interpretation,
            export_format,
        )

        if MODEL_REPORT_PATH.exists():
            export_download_card(
                "Originaler Modellreport",
                "Direkter Export des lokal erzeugten Modellreports.",
                "wealthscope_model_report",
                MODEL_REPORT_PATH.read_text(encoding="utf-8"),
                export_format,
            )
        else:
            st.warning("Originaler Modellreport wurde lokal nicht gefunden.")


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="opacity:0.55; font-size:0.85rem; padding-top:1rem; padding-bottom:0.5rem;">
        <b>WealthScope AI</b> · Demo-Anwendung · Keine Finanzberatung
    </div>
    """,
    unsafe_allow_html=True,
)

render_bottom_status_bar()

# Final runtime theme override

# Absolute final visual override

# Finaler Design-Override nach allen Seiteninhalten

# Finaler Design-Override nach allen Seiteninhalten
load_base_css()
render_runtime_theme(theme_mode, app_mode)
