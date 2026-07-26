from __future__ import annotations

import streamlit as st

from src.context import get_context
from src.export import analysis_markdown, build_export_zip, build_pdf
from src.ui import disclaimer_footer, page_header


def render() -> None:
    ctx = get_context()
    result = ctx["result"]

    page_header("Export", f"Analyse-Bericht für {ctx['ticker']} exportieren")

    st.markdown(analysis_markdown(result))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Markdown", analysis_markdown(result), f"{ctx['ticker']}_bericht.md",
                            icon=":material/description:")
    with c2:
        st.download_button("Feature-Daten (CSV)", ctx["period_df"].to_csv(index=False), f"{ctx['ticker']}_daten.csv",
                            icon=":material/table_chart:")
    with c3:
        zip_bytes = build_export_zip(result, ctx["period_df"], ctx["news"])
        st.download_button("Alles (ZIP)", zip_bytes, f"{ctx['ticker']}_export.zip",
                            icon=":material/folder_zip:")

    pdf_bytes = build_pdf(result)
    if pdf_bytes:
        st.download_button("PDF-Bericht", pdf_bytes, f"{ctx['ticker']}_bericht.pdf",
                            icon=":material/picture_as_pdf:")
    else:
        st.caption("PDF-Export benötigt `reportlab` (`pip install reportlab`).")

    disclaimer_footer()
