from __future__ import annotations

import streamlit as st

from src.quiz import QUESTIONS, build_arsnova_quiz, grade_answers
from src.ui import card, disclaimer_footer, page_header


def render() -> None:
    page_header(
        "Lernstudio",
        "Aktives Wiederholen · direktes Feedback · Export für arsnova.eu",
    )
    card(
        "<b>Vom Modell zum Wissen:</b> Acht Fragen verbinden QUA³CK, Datenqualität, "
        "Klassifikation, SVM, Entscheidungsbäume und Random Forests mit der konkreten App.",
        variant="ws-accent",
    )

    with st.form("wealthscope_quiz"):
        selected = []
        for index, question in enumerate(QUESTIONS, start=1):
            st.markdown(f"#### {index}. {question['chapter']}")
            selected.append(st.radio(
                question["question"],
                question["options"],
                index=None,
                key=f"quiz_{index}",
            ))
        submitted = st.form_submit_button(
            "Auswerten", type="primary", icon=":material/check_circle:"
        )

    if submitted:
        answer_indices = [
            question["options"].index(answer) if answer in question["options"] else -1
            for question, answer in zip(QUESTIONS, selected)
        ]
        result = grade_answers(answer_indices)
        st.metric("Ergebnis", f"{result['score']} / {result['total']}", f"{result['percentage']} %")
        for index, (question, detail) in enumerate(zip(QUESTIONS, result["details"]), start=1):
            if detail["correct"]:
                st.success(f"{index}. Richtig — {question['explanation']}")
            else:
                correct = question["options"][question["answer"]]
                separator = "" if correct.endswith((".", "!", "?")) else "."
                st.error(
                    f"{index}. Noch nicht — richtig ist: **{correct}**"
                    f"{separator} {question['explanation']}"
                )

    st.divider()
    st.markdown("#### Peer Instruction")
    st.caption(
        "Den Fragensatz als arsnova.eu-JSON exportieren, dort importieren und live mit "
        "anonymem Feedback oder als Team-Quiz einsetzen."
    )
    st.download_button(
        "arsnova.eu-Quiz herunterladen",
        data=build_arsnova_quiz(),
        file_name="wealthscope-ai-quiz.json",
        mime="application/json",
        icon=":material/download:",
    )
    disclaimer_footer()
