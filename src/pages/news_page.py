from __future__ import annotations

import streamlit as st

from src.context import get_context
from src.ui import disclaimer_footer, page_header, section_title


def render() -> None:
    ctx = get_context()
    news_df = ctx["news"]

    page_header("News-Archiv", f"Aktuelle Nachrichten zu {ctx['ticker']} (NewsAPI + Lexikon-Sentiment)")

    if news_df.empty:
        st.info(
            "Keine Nachrichten geladen. Prüfe, ob `NEWS_API_KEY` in `.streamlit/secrets.toml` gesetzt ist, "
            "oder aktiviere „News-Sentiment einbeziehen“ in der Sidebar."
        )
        disclaimer_footer()
        return

    section_title(f"{len(news_df)} Artikel", "newspaper")
    for _, row in news_df.head(15).iterrows():
        title = row.get("title", "(ohne Titel)")
        source = (row.get("source") or {}).get("name", "") if isinstance(row.get("source"), dict) else ""
        url = row.get("url", "")
        st.markdown(f"**[{title}]({url})**  \n<span style='color:var(--ws-text-muted);font-size:0.8rem'>{source} · {row.get('publishedAt','')}</span>",
                    unsafe_allow_html=True)
        st.caption(row.get("description", "") or "")
        st.divider()

    disclaimer_footer()
