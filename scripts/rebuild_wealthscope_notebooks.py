from __future__ import annotations

from pathlib import Path
import nbformat as nbf


PROJECT_ROOT = Path(".")
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write_notebook(filename: str, title: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
        "wealthscope": {
            "project": "WealthScope AI",
            "generated_by": "scripts/rebuild_wealthscope_notebooks.py",
            "purpose": "QUA3CK Big-Data / ML documentation",
        },
    }

    header = md(f"""
# {title}

**Projekt:** WealthScope AI  
**Kontext:** QUA3CK / Big-Data / Machine-Learning / Streamlit-App  
**Datenbasis:** Kaggle Stock/ETF Dataset, lokal verarbeitet  
**Hinweis:** Diese Notebooks dienen der wissenschaftlichen Nachvollziehbarkeit. Sie ersetzen keine Finanzberatung.
""")

    nb["cells"] = [header] + cells

    output_path = NOTEBOOK_DIR / filename
    nbf.write(nb, output_path)
    print(f"created: {output_path}")


COMMON_SETUP = code("""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("..").resolve()
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PARQUET_PATH = DATA_DIR / "wealthscope_features.parquet"
CSV_PATH = DATA_DIR / "wealthscope_features.csv"

def load_features():
    if PARQUET_PATH.exists():
        df = pd.read_parquet(PARQUET_PATH)
        source = "REAL_PARQUET"
    elif CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        source = "REAL_CSV"
    else:
        raise FileNotFoundError("Kein Feature-Datensatz gefunden. Erwartet wealthscope_features.parquet oder .csv")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df, source

df, source = load_features()

print("Datenquelle:", source)
print("Shape:", df.shape)
display(df.head())
""")


# ---------------------------------------------------------
# 00 Project Overview
# ---------------------------------------------------------
write_notebook(
    "00_project_overview.ipynb",
    "00 – Project Overview",
    [
        md("""
## Ziel des Projekts

WealthScope AI ist eine interaktive Finanz-/Marktdaten-App auf Basis von Streamlit.

Die App soll zeigen:

- echte lokale Big-Data-Grundlage
- Feature Engineering
- NewsAPI-Integration
- regelbasiertes Scoring
- Visualisierung
- Export
- methodische Grenzen
- spätere ML-Erweiterbarkeit
"""),
        COMMON_SETUP,
        code("""
summary = {
    "rows": int(len(df)),
    "columns": int(len(df.columns)),
    "tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
    "asset_types": df["asset_type"].value_counts().to_dict() if "asset_type" in df.columns else {},
    "date_min": str(df["date"].min().date()) if "date" in df.columns else None,
    "date_max": str(df["date"].max().date()) if "date" in df.columns else None,
    "target_20d_available": "target_20d" in df.columns,
}
summary
"""),
        code("""
pd.DataFrame([summary]).T.rename(columns={0: "Wert"})
"""),
        md("""
## Wissenschaftlicher Nutzen

Dieses Notebook dokumentiert die Projektidee, die Datenbasis und die geplante methodische Struktur.  
Es ist der Einstiegspunkt für die Bewertung durch Professor, Team oder Prüfer.
"""),
    ],
)


# ---------------------------------------------------------
# 01 Question
# ---------------------------------------------------------
write_notebook(
    "01_question.ipynb",
    "01 – Question",
    [
        md("""
## Forschungs-/Projektfrage

**Wie kann eine interaktive Streamlit-App historische Marktdaten, technische Features und Newsdaten so kombinieren, dass eine nachvollziehbare, reproduzierbare und erklärbare Markteinschätzung entsteht?**

Teilfragen:

1. Welche Datenbasis liegt vor?
2. Welche Features eignen sich für eine erste Bewertung?
3. Wie kann eine Zielvariable wie `target_20d` erklärt werden?
4. Wie können Newsdaten ergänzend genutzt werden?
5. Wie lassen sich Ergebnisse transparent exportieren?
"""),
        COMMON_SETUP,
        code("""
question_frame = pd.DataFrame(
    [
        ["Q1", "Welche Marktdaten stehen zur Verfügung?", "Datenprofil, Ticker, Zeitraum, Asset-Typen"],
        ["Q2", "Welche Features wurden erzeugt?", "Returns, Moving Averages, Volatilität, Drawdown"],
        ["Q3", "Wie ist die Zielvariable definiert?", "target_20d"],
        ["Q4", "Wie werden News genutzt?", "NewsAPI + regelbasierte Einordnung"],
        ["Q5", "Wie wird Wissen übertragen?", "Streamlit-App, Export, Assistent, Methodikdialog"],
    ],
    columns=["ID", "Frage", "Operationalisierung"],
)
question_frame
"""),
        md("""
## Erwartetes Ergebnis

Am Ende soll nicht nur ein Modell oder Chart entstehen, sondern eine **prüfbare Daten-App** mit klarer Methodik.
"""),
    ],
)


# ---------------------------------------------------------
# 02 Understanding Data
# ---------------------------------------------------------
write_notebook(
    "02_understanding_the_data.ipynb",
    "02 – Understanding the Data",
    [
        md("""
## Ziel

Dieses Notebook prüft die Datenbasis:

- Größe
- Spalten
- Zeitraum
- Ticker-Verteilung
- Asset-Typen
- fehlende Werte
- Zielvariable
"""),
        COMMON_SETUP,
        code("""
profile = {
    "rows": len(df),
    "columns": len(df.columns),
    "tickers": df["ticker"].nunique() if "ticker" in df.columns else None,
    "date_min": df["date"].min() if "date" in df.columns else None,
    "date_max": df["date"].max() if "date" in df.columns else None,
    "target_20d_available": "target_20d" in df.columns,
}
pd.DataFrame([profile]).T.rename(columns={0: "Wert"})
"""),
        code("""
missing = (
    df.isna()
    .mean()
    .mul(100)
    .round(2)
    .reset_index()
)
missing.columns = ["Spalte", "Fehlende Werte in %"]
missing.sort_values("Fehlende Werte in %", ascending=False).head(30)
"""),
        code("""
ticker_counts = df["ticker"].value_counts().reset_index()
ticker_counts.columns = ["ticker", "rows"]
ticker_counts
"""),
        code("""
plt.figure(figsize=(12, 5))
plt.bar(ticker_counts["ticker"], ticker_counts["rows"])
plt.title("Datenpunkte je Ticker")
plt.xticks(rotation=45)
plt.ylabel("Zeilen")
plt.show()
"""),
        code("""
if "asset_type" in df.columns:
    asset_counts = df["asset_type"].value_counts()
    display(asset_counts)

    plt.figure(figsize=(6, 4))
    plt.bar(asset_counts.index.astype(str), asset_counts.values)
    plt.title("Asset-Typen")
    plt.ylabel("Zeilen")
    plt.show()
"""),
        md("""
## Interpretation

Diese Analyse belegt, dass die App auf einer echten Datenbasis arbeitet und nicht auf einer Demo-Tabelle.
"""),
    ],
)


# ---------------------------------------------------------
# 03 Feature Engineering
# ---------------------------------------------------------
write_notebook(
    "03_feature_engineering.ipynb",
    "03 – Feature Engineering",
    [
        md("""
## Ziel

Dieses Notebook erklärt und prüft die verwendeten Features:

- `daily_return`
- `return_5d`
- `return_20d`
- `ma_20`, `ma_50`, `ma_200`
- `ma_*_distance`
- `volatility_20d`
- `drawdown`
- `future_return_20d`
- `target_20d`
"""),
        COMMON_SETUP,
        code("""
feature_groups = {
    "Preis": ["open", "high", "low", "close", "volume"],
    "Rendite": ["daily_return", "return_5d", "return_20d"],
    "Trend": ["ma_20", "ma_50", "ma_200", "ma_20_distance", "ma_50_distance", "ma_200_distance"],
    "Risiko": ["volatility_20d", "rolling_high", "drawdown"],
    "Zielvariable": ["future_return_20d", "target_20d"],
}

rows = []
for group, cols in feature_groups.items():
    for col in cols:
        rows.append({
            "Gruppe": group,
            "Feature": col,
            "Vorhanden": col in df.columns,
            "Fehlende Werte %": round(df[col].isna().mean() * 100, 2) if col in df.columns else None,
        })

pd.DataFrame(rows)
"""),
        code("""
sample_ticker = df["ticker"].value_counts().index[0]
d = df[df["ticker"] == sample_ticker].sort_values("date").copy()

plt.figure(figsize=(14, 5))
plt.plot(d["date"], d["close"], label="Close")
for ma in ["ma_20", "ma_50", "ma_200"]:
    if ma in d.columns:
        plt.plot(d["date"], d[ma], label=ma)
plt.title(f"Kurs und Moving Averages – {sample_ticker}")
plt.legend()
plt.show()
"""),
        code("""
if "drawdown" in d.columns:
    plt.figure(figsize=(14, 4))
    plt.plot(d["date"], d["drawdown"] * 100)
    plt.title(f"Drawdown – {sample_ticker}")
    plt.ylabel("Drawdown in %")
    plt.show()
"""),
        code("""
if "volatility_20d" in d.columns:
    plt.figure(figsize=(14, 4))
    plt.plot(d["date"], d["volatility_20d"] * 100)
    plt.title(f"20T Volatilität – {sample_ticker}")
    plt.ylabel("Volatilität in %")
    plt.show()
"""),
        md("""
## Ergebnis

Die Features sind fachlich gruppierbar und können in der App verständlich erklärt werden.
"""),
    ],
)


# ---------------------------------------------------------
# 04 Modeling
# ---------------------------------------------------------
write_notebook(
    "04_modeling_baseline_ml.ipynb",
    "04 – Modeling: KI damals und heute",
    [
        md("""
## Ziel

Dieses Notebook dokumentiert den reproduzierbaren 1.0-Benchmark zur Zielvariable
`target_20d`. Fünf Modellgenerationen werden fair auf denselben Zeitfenstern
verglichen:

- Dummy-Baseline
- logistische Regression
- Entscheidungsbaum
- Linear-SVM
- Random Forest

Finanzdaten sind zeitlich geordnet. Deshalb gibt es keinen zufälligen Split:
Die neuesten 20 % der Handelstage sind der unangetastete Test; eine Sperrzone
von 20 Handelstagen verhindert überlappende Zielhorizonte.
"""),
        COMMON_SETUP,
        code("""
import json
from pathlib import Path

diagnostics = json.loads((BASE_DIR / "models" / "diagnostics.json").read_text())
diagnostics["validation"]
"""),
        code("""
comparison = []
for model in diagnostics["model_comparison"].values():
    metric = model["test_metrics"]
    comparison.append({
        "Epoche": model["year"],
        "Modell": model["label"],
        "Idee": model["family"],
        "Accuracy": metric["accuracy"],
        "Balanced Accuracy": metric["balanced_accuracy"],
        "ROC-AUC": metric["roc_auc"],
        "Walk-forward AUC": model["walk_forward_roc_auc_mean"],
        "Training (s)": metric["fit_seconds"],
    })

comparison_df = pd.DataFrame(comparison)
display(comparison_df)
"""),
        code("""
ax = comparison_df.plot(
    x="Modell",
    y=["ROC-AUC", "Balanced Accuracy"],
    kind="bar",
    figsize=(11, 4),
)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_ylim(0.45, 0.60)
ax.set_title("Identischer Out-of-Time-Test: komplexer ist nicht automatisch besser")
plt.xticks(rotation=25, ha="right")
plt.show()
"""),
        md("""
## Reproduktion

Der ausführbare Single Source of Truth ist `scripts/train_and_diagnose.py`.
Er erzeugt Modell, Testkurven, Walk-forward-Ergebnisse und Lernkurve:

```bash
python scripts/train_and_diagnose.py
```

Das ehrliche Ergebnis ist fachlich wichtiger als eine hohe Zahl: Der Random
Forest liegt bei Out-of-Time-ROC-AUC nur knapp über 0,5. Historische Preise
allein ergeben kein belastbares Handelssignal.
"""),
    ],
)


# ---------------------------------------------------------
# 05 Evaluation / Conclude
# ---------------------------------------------------------
write_notebook(
    "05_conclude_evaluate.ipynb",
    "05 – Conclude & Evaluate",
    [
        md("""
## Ziel

Dieses Notebook bewertet die Ergebnisse methodisch.

Es geht nicht darum, ein perfektes Finanzmodell zu behaupten, sondern:

- Baseline erklären
- Modellgüte kritisch prüfen
- Grenzen offenlegen
- nächsten Ausbau ableiten
"""),
        COMMON_SETUP,
        code("""
evaluation_points = pd.DataFrame(
    [
        ["Datenbasis", "Echte lokale Marktdaten vorhanden", "Stark"],
        ["Feature Engineering", "Returns, Moving Averages, Volatilität, Drawdown", "Stark"],
        ["Zielvariable", "target_20d vorhanden", "Gut erklärbar"],
        ["News", "NewsAPI integriert", "Extern abhängig"],
        ["Scoring", "Regelbasiert", "Transparent, aber noch kein echtes ML-Produkt"],
        ["ML", "Fünf Modelle im purged Out-of-Time-Benchmark", "Reproduzierbar"],
        ["Reproduzierbarkeit", "Export und Notebooks", "Stark"],
    ],
    columns=["Bereich", "Befund", "Bewertung"],
)
evaluation_points
"""),
        code("""
limitations = [
    "Historische Daten garantieren keine zukünftige Kursentwicklung.",
    "target_20d ist eine vereinfachte Zielvariable.",
    "News-Sentiment ist aktuell regelbasiert und nicht semantisch tief.",
    "Ticker-Auswahl ist ein kontrollierter Ausschnitt.",
    "Das Modell ist keine Anlageberatung.",
]

for item in limitations:
    print("-", item)
"""),
        md("""
## Fazit

WealthScope AI ist als datenbasierter, erklärbarer Prototyp geeignet.  
Die App zeigt Datenbasis, Feature Engineering, Visualisierung, News-Kontext und Exportfähigkeit.

Version 1.0 liefert einen dokumentierten Modellvergleich, Confusion Matrix,
ROC/PR, Walk-forward-Lernkurve und Feature Importance. Die geringe
Out-of-Time-Leistung wird als wissenschaftliches Ergebnis gezeigt, nicht kaschiert.
"""),
    ],
)


# ---------------------------------------------------------
# 06 Knowledge Transfer
# ---------------------------------------------------------
write_notebook(
    "06_knowledge_transfer_streamlit.ipynb",
    "06 – Knowledge Transfer: Streamlit App",
    [
        md("""
## Ziel

Dieses Notebook verbindet die Notebook-Arbeit mit der Streamlit-App.

Die App ist der Knowledge-Transfer-Kanal:

- Interaktion
- Visualisierung
- Export
- Erklärung
- Assistent
- Methodikdialog
"""),
        COMMON_SETUP,
        code("""
app_components = pd.DataFrame(
    [
        ["Startseite", "Projekt erklären und Nutzer führen"],
        ["Wealth Outlook", "Analyse und Charts"],
        ["Datenlabor", "Rohdaten, Profil, Fehlwerte"],
        ["ML-Labor", "Features und Modellbezug"],
        ["Lernstudio", "Direktes Feedback und arsnova.eu-Export"],
        ["News-Archiv", "NewsAPI transparent machen"],
        ["Assistent", "Analyse erklären"],
        ["Export", "Reproduzierbarkeit"],
        ["Status", "technischer Zustand"],
    ],
    columns=["App-Bereich", "Funktion"],
)
app_components
"""),
        code("""
print("Start der App lokal:")
print("cd <repository>")
print("python -m streamlit run app.py")
"""),
        md("""
## Transferargument

Die Notebooks zeigen die methodische Herleitung.  
Die Streamlit-App macht die Ergebnisse für Nutzer interaktiv erlebbar.
"""),
    ],
)


# ---------------------------------------------------------
# 07 NewsAPI / Assistant / Export
# ---------------------------------------------------------
write_notebook(
    "07_newsapi_assistant_export.ipynb",
    "07 – NewsAPI, Assistant & Export",
    [
        md("""
## Ziel

Dieses Notebook dokumentiert die Zusatzmodule:

- NewsAPI
- regelbasierte News-Einordnung
- Analyse-Assistent
- Export-Paket
"""),
        COMMON_SETUP,
        code("""
news_design = pd.DataFrame(
    [
        ["NewsAPI", "Externe Nachrichtenquelle"],
        ["Query", "Suchlogik abhängig von Asset und Thema"],
        ["Sentiment", "Regelbasierte Einschätzung"],
        ["News-Karten", "Bessere UX als Tabelle"],
        ["Assistent", "Erklärt Analysekontext"],
        ["Export", "Markdown, CSV, JSON, ZIP"],
    ],
    columns=["Modul", "Zweck"],
)
news_design
"""),
        code("""
assistant_questions = [
    "Was bedeutet Drawdown?",
    "Was ist target_20d?",
    "Welche News wurden berücksichtigt?",
    "Wie funktioniert die Methodik?",
    "Warum ist die Einschätzung neutral?",
]

pd.DataFrame({"Beispielfragen": assistant_questions})
"""),
        md("""
## Wichtig

Der Assistent soll erklären, nicht beraten.  
Er ist ein Analyse- und Methodik-Assistent, kein Finanzberater.
"""),
    ],
)


print("\\nNotebook-Rebuild abgeschlossen.")
