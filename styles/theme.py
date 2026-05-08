from pathlib import Path
import streamlit as st


def load_base_css():
    css_path = Path(__file__).parent / "wealthscope.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_runtime_theme(theme_mode: str, app_mode: str):
    is_light = theme_mode == "Light Mode"
    is_expert = app_mode == "Expertenansicht"

    if is_light:
        values = {
            "APP_BG": "linear-gradient(135deg, #f8fafc 0%, #eef3fb 45%, #ffffff 100%)",
            "SIDEBAR_BG": "linear-gradient(180deg, #ffffff 0%, #f4f7fb 52%, #edf2f8 100%)",
            "CARD_BG": "rgba(255,255,255,0.96)",
            "ELEVATED_BG": "rgba(255,255,255,0.99)",
            "BORDER": "rgba(15,23,42,0.14)",
            "TEXT": "#111827",
            "MUTED": "#475569",
            "INPUT_BG": "rgba(255,255,255,0.98)",
            "BOTTOM_BG": "linear-gradient(180deg, rgba(248,250,255,0.99), rgba(232,237,247,0.99))",
            "SHADOW": "0 18px 45px rgba(15,23,42,0.10)",
            "HEADER_BG": "rgba(248,250,255,0.94)",
            "CHART_PANEL_BG": "rgba(255,255,255,0.96)",
        }
    else:
        values = {
            "APP_BG": "linear-gradient(135deg, #07101f 0%, #0b1020 48%, #12111d 100%)",
            "SIDEBAR_BG": "linear-gradient(180deg, #080d1b 0%, #0b1020 100%)",
            "CARD_BG": "rgba(255,255,255,0.065)",
            "ELEVATED_BG": "rgba(255,255,255,0.085)",
            "BORDER": "rgba(255,255,255,0.15)",
            "TEXT": "#eef3ff",
            "MUTED": "#aeb7c8",
            "INPUT_BG": "rgba(255,255,255,0.08)",
            "BOTTOM_BG": "linear-gradient(180deg, rgba(10,16,30,0.98), rgba(5,9,18,0.99))",
            "SHADOW": "0 18px 45px rgba(0,0,0,0.24)",
            "HEADER_BG": "rgba(5,9,18,0.90)",
            "CHART_PANEL_BG": "rgba(5,9,18,0.55)",
        }

    values["ACCENT"] = "#6f8cff" if is_expert else "#22c55e"

    css = """
    <style id="wealthscope-runtime-theme">
    html,
    body,
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > .main,
    .main,
    .block-container {
        background: __APP_BG__ !important;
        color: __TEXT__ !important;
    }

    header[data-testid="stHeader"] {
        background: __HEADER_BG__ !important;
    }

    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background: __SIDEBAR_BG__ !important;
        color: __TEXT__ !important;
        border-right: 1px solid __BORDER__ !important;
    }

    [data-testid="stSidebar"] *,
    section[data-testid="stSidebar"] * {
        color: __TEXT__ !important;
    }

    [data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] [role="radiogroup"],
    section[data-testid="stSidebar"] [role="radiogroup"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        background: __CARD_BG__ !important;
        border: 1px solid __BORDER__ !important;
        color: __TEXT__ !important;
        box-shadow: __SHADOW__ !important;
        border-radius: 18px !important;
    }

    h1, h2, h3, h4, h5, h6,
    p, li, label, span, div {
        color: __TEXT__ !important;
    }

    .subtitle,
    .muted,
    small,
    .stCaptionContainer,
    [data-testid="stCaptionContainer"] {
        color: __MUTED__ !important;
    }

    .hero-card,
    .mini-card,
    .info-card,
    .export-card,
    .decision-step,
    .status-row,
    .news-card,
    .premium-strip,
    .advisor-box,
    .neutral-card,
    .good-card,
    .warn-card,
    .bad-card,
    .mode-proof-card,
    .clean-table-wrap,
    .clean-table,
    .clean-table td,
    .clean-table th {
        background: __CARD_BG__ !important;
        border-color: __BORDER__ !important;
        color: __TEXT__ !important;
        box-shadow: __SHADOW__ !important;
    }

    .hero-card {
        border-left: 7px solid __ACCENT__ !important;
    }

    div[style*="background: #0b1020"],
    div[style*="background:#0b1020"],
    div[style*="background-color: #0b1020"],
    div[style*="background-color:#0b1020"],
    div[style*="background: #07101f"],
    div[style*="background:#07101f"],
    div[style*="background-color: #07101f"],
    div[style*="background-color:#07101f"],
    div[style*="rgba(5,9,18"],
    div[style*="rgba(5, 9, 18"],
    div[style*="rgba(8,13,27"],
    div[style*="rgba(8, 13, 27"] {
        background: __CARD_BG__ !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
    }

    div[data-testid="stVerticalBlock"],
    div[data-testid="stHorizontalBlock"],
    div[data-testid="column"] {
        background: transparent !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
    }

    div[data-testid="stExpander"],
    details,
    details > div,
    details summary {
        background: __CARD_BG__ !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
        border-radius: 18px !important;
    }

    input,
    textarea,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input,
    [data-baseweb="tag"] {
        background: __INPUT_BG__ !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="popover"] *,
    [role="listbox"],
    [role="option"] {
        background: __ELEVATED_BG__ !important;
        color: __TEXT__ !important;
    }

    [data-testid="stMetric"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        background: __ELEVATED_BG__ !important;
        color: __TEXT__ !important;
        border-color: __BORDER__ !important;
    }

    .js-plotly-plot,
    .plotly,
    .plot-container,
    .svg-container,
    .js-plotly-plot .main-svg {
        background: transparent !important;
    }

    .element-container:has(.js-plotly-plot) {
        background: __CHART_PANEL_BG__ !important;
        border-radius: 20px !important;
        border: 1px solid __BORDER__ !important;
        box-shadow: __SHADOW__ !important;
        padding: 0.4rem !important;
    }

    .bottom-status-bar {
        background: __BOTTOM_BG__ !important;
        border-top: 1px solid __BORDER__ !important;
        box-shadow: 0 -8px 35px rgba(0,0,0,0.10) !important;
    }

    .bottom-status-item,
    .bottom-status-item * {
        color: __TEXT__ !important;
    }

    [data-testid="stToast"],
    div[data-testid="stToast"],
    [data-testid="stNotification"],
    div[data-testid="stNotification"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }
    </style>
    """

    for key, value in values.items():
        css = css.replace(f"__{key}__", value)

    st.markdown(css, unsafe_allow_html=True)


def get_chart_theme(theme_mode: str):
    if theme_mode == "Light Mode":
        return {
            "template": "plotly_white",
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font_color": "#111827",
            "grid_color": "rgba(15,23,42,0.12)",
            "zero_line": "rgba(15,23,42,0.20)",
        }

    return {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font_color": "#eef3ff",
        "grid_color": "rgba(255,255,255,0.12)",
        "zero_line": "rgba(255,255,255,0.20)",
    }


def apply_chart_theme(fig, theme_mode: str, height=None):
    chart_theme = get_chart_theme(theme_mode)

    fig.update_layout(
        template=chart_theme["template"],
        paper_bgcolor=chart_theme["paper_bgcolor"],
        plot_bgcolor=chart_theme["plot_bgcolor"],
        font=dict(color=chart_theme["font_color"]),
        legend=dict(
            font=dict(color=chart_theme["font_color"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=20, r=20, t=60, b=35),
    )

    if height is not None:
        fig.update_layout(height=height)

    try:
        fig.update_xaxes(
            gridcolor=chart_theme["grid_color"],
            zerolinecolor=chart_theme["zero_line"],
            tickfont=dict(color=chart_theme["font_color"]),
            title_font=dict(color=chart_theme["font_color"]),
        )
        fig.update_yaxes(
            gridcolor=chart_theme["grid_color"],
            zerolinecolor=chart_theme["zero_line"],
            tickfont=dict(color=chart_theme["font_color"]),
            title_font=dict(color=chart_theme["font_color"]),
        )
    except Exception:
        pass

    return fig
