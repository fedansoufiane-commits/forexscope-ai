"""Report export: Markdown summary, CSV, and a bundled ZIP (+ optional PDF)."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd

from src.model import AnalysisResult


def analysis_markdown(result: AnalysisResult) -> str:
    return f"""# WealthScope AI — Analyse-Bericht

**Ticker:** {result.ticker}
**Kurs:** {result.price:.2f}
**Datum:** {datetime.now():%Y-%m-%d %H:%M}

## Scores
- Trend: {result.trend_score}/100
- Volatilität: {result.volatility_score}/100
- Drawdown: {result.drawdown_score}/100
- News-Sentiment: {result.news_label} ({result.news_score:+.2f})
- **Confidence: {result.confidence}/100**
- RandomForest-Wahrscheinlichkeit (bullish, 20 Handelstage): {result.rf_proba*100:.1f}%

## Einschätzung
**{result.outlook}**

---
⚠️ Keine Finanzberatung. Ausschließlich zu Lernzwecken (IU-Modul Data Analytics und Big Data, DSDABD072501).
"""


def build_export_zip(result: AnalysisResult, features_df: pd.DataFrame, news_df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("bericht.md", analysis_markdown(result))
        if not features_df.empty:
            zf.writestr("features.csv", features_df.to_csv(index=False))
        if not news_df.empty:
            zf.writestr("news.csv", news_df.to_csv(index=False))
    buf.seek(0)
    return buf.read()


def build_pdf(result: AnalysisResult) -> bytes | None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = [Paragraph("WealthScope AI — Analyse-Bericht", styles["Title"]), Spacer(1, 12)]
    for line in analysis_markdown(result).split("\n"):
        if line.strip():
            flow.append(Paragraph(line.replace("**", ""), styles["Normal"]))
            flow.append(Spacer(1, 4))
    doc.build(flow)
    buf.seek(0)
    return buf.read()
