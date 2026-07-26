"""Curated learning questions and arsnova.eu-compatible export."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

QUESTIONS: List[Dict[str, Any]] = [
    {
        "chapter": "QUA³CK",
        "question": "Welche Reihenfolge beschreibt QUA³CK korrekt?",
        "options": [
            "Question → Understanding → A³ → Conclude & Compare → Knowledge Transfer",
            "Quality → Upload → Analyse → Code → Kontrolle",
            "Question → Use Case → Accuracy → Cloud → Knowledge",
        ],
        "answer": 0,
        "explanation": "QUA³CK führt von einer präzisen Frage über Datenverständnis und die iterative A³-Schleife bis zu Vergleich und Wissenstransfer.",
    },
    {
        "chapter": "Datenphase",
        "question": "Warum ist ein zufälliger Train/Test-Split für diese Finanzzeitreihe problematisch?",
        "options": [
            "Er ist nur bei kleinen Datensätzen erlaubt.",
            "Spätere Marktbeobachtungen können im Training landen und überlappende 20-Tage-Ziele erzeugen Leakage.",
            "Random Forest akzeptiert ausschließlich chronologische Daten.",
        ],
        "answer": 1,
        "explanation": "Zeitreihen müssen zeitlich getrennt werden. Zusätzlich verhindert die 20-Tage-Sperrzone überlappende Zielhorizonte.",
    },
    {
        "chapter": "Klassifikation",
        "question": "Welche Kennzahl misst die Trennschärfe über alle möglichen Schwellenwerte?",
        "options": ["ROC-AUC", "Nur Accuracy", "Trainingsdauer"],
        "answer": 0,
        "explanation": "ROC-AUC bewertet das Ranking positiver gegenüber negativer Fälle schwellenwertunabhängig. Bei starker Imbalance ergänzt die PR-Kurve.",
    },
    {
        "chapter": "Modelltraining",
        "question": "Was zeigt eine große Lücke zwischen Trainings- und Validierungsleistung?",
        "options": ["Typischerweise hohe Varianz/Overfitting", "Sichere Generalisierung", "Fehlende Zielvariable"],
        "answer": 0,
        "explanation": "Ein sehr gutes Training bei schwacher Validierung deutet auf Overfitting hin. Regularisierung oder mehr repräsentative Daten können helfen.",
    },
    {
        "chapter": "Support Vector Machines",
        "question": "Welches geometrische Ziel verfolgt eine SVM?",
        "options": ["Maximaler Abstand der Trennfläche zu den nächsten Punkten", "Maximale Baumtiefe", "Minimale Zahl an Klassen"],
        "answer": 0,
        "explanation": "Die SVM sucht eine Maximum-Margin-Hyperebene; die nächstgelegenen Punkte heißen Support Vectors.",
    },
    {
        "chapter": "Entscheidungsbäume",
        "question": "Warum gelten kleine Entscheidungsbäume als gut erklärbar?",
        "options": ["Ihre If-Then-Splits können als Entscheidungsweg gelesen werden.", "Sie benötigen nie Daten.", "Sie liefern immer kausale Aussagen."],
        "answer": 0,
        "explanation": "Der Pfad von der Wurzel zum Blatt ist nachvollziehbar. Das macht den Baum transparent, aber nicht automatisch kausal oder fair.",
    },
    {
        "chapter": "Random Forests",
        "question": "Wodurch reduziert ein Random Forest die Varianz einzelner Bäume?",
        "options": ["Viele diversifizierte Bäume werden aggregiert.", "Alle Bäume erhalten exakt dieselben Splits.", "Die Zielvariable wird entfernt."],
        "answer": 0,
        "explanation": "Bagging und zufällige Feature-Auswahl dekorrelieren Bäume; Mehrheitsentscheid bzw. Mittelung stabilisiert die Vorhersage.",
    },
    {
        "chapter": "KI damals/heute",
        "question": "Welche Aussage trifft den fairen historischen Modellvergleich?",
        "options": [
            "Das neueste Modell gewinnt grundsätzlich.",
            "Alle Modelle werden auf demselben zeitlich späteren Testfenster anhand mehrerer Metriken verglichen.",
            "Nur die Trainings-Accuracy ist relevant.",
        ],
        "answer": 1,
        "explanation": "Fortschritt wird empirisch geprüft. Ein älteres, einfacheres Modell kann bei Generalisierung, Tempo oder Erklärbarkeit überlegen sein.",
    },
]


def grade_answers(answers: List[int]) -> Dict[str, Any]:
    """Grade one answer index per question; -1 means unanswered."""
    if len(answers) != len(QUESTIONS):
        raise ValueError("Exactly one answer slot per question is required.")
    details = []
    for question, selected in zip(QUESTIONS, answers):
        correct = selected == question["answer"]
        details.append({
            "chapter": question["chapter"],
            "selected": selected,
            "correct": correct,
            "explanation": question["explanation"],
        })
    score = sum(item["correct"] for item in details)
    return {
        "score": score,
        "total": len(QUESTIONS),
        "percentage": round(score / len(QUESTIONS) * 100),
        "details": details,
    }


def build_arsnova_quiz() -> bytes:
    """Export the schema used by arsnova.eu course quiz files."""
    questions = []
    for order, question in enumerate(QUESTIONS):
        questions.append({
            "text": f"### {question['chapter']}\n\n{question['question']}",
            "type": "SINGLE_CHOICE",
            "difficulty": "MEDIUM",
            "order": order,
            "answers": [
                {"text": text, "isCorrect": index == question["answer"]}
                for index, text in enumerate(question["options"])
            ],
            "ratingMin": None,
            "ratingMax": None,
            "ratingLabelMin": None,
            "ratingLabelMax": None,
            "enabled": True,
        })
    payload = {
        "exportVersion": 1,
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "quiz": {
            "name": "WealthScope AI 1.0 – ML & QUA³CK",
            "description": "Begleitquiz zum IU-Lernprojekt WealthScope AI.",
            "showLeaderboard": True,
            "allowCustomNicknames": False,
            "defaultTimer": 30,
            "timerScaleByDifficulty": True,
            "enableSoundEffects": True,
            "enableRewardEffects": True,
            "enableMotivationMessages": True,
            "enableEmojiReactions": True,
            "anonymousMode": False,
            "teamMode": False,
            "teamCount": 2,
            "teamAssignment": "AUTO",
            "teamNames": [],
            "backgroundMusic": None,
            "nicknameTheme": "KINDERGARTEN",
            "bonusTokenCount": 1,
            "readingPhaseEnabled": True,
            "questions": questions,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
