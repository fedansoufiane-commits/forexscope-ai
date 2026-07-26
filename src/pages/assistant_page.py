from __future__ import annotations

import streamlit as st

from src.context import get_context
from src.export import analysis_markdown
from src.news import assistant_answer
from src.ui import disclaimer_footer, page_header


def render() -> None:
    ctx = get_context()
    result = ctx["result"]

    page_header("KI-Assistent", "Fragen zur aktuellen Analyse — beantwortet von Gemini (falls API-Key gesetzt)")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, msg in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(msg)

    prompt = st.chat_input(f"Frage zu {ctx['ticker']} stellen ...")
    if prompt:
        st.session_state["chat_history"].append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        context = analysis_markdown(result)
        answer = assistant_answer(prompt, context)
        st.session_state["chat_history"].append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

    disclaimer_footer()
