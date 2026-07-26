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
    "04 – Modeling: Baseline & ML",
    [
        md("""
## Ziel

Dieses Notebook erstellt ein erstes ML-Baseline-Modell zur Zielvariable `target_20d`.

Wichtig:

- Das Modell ist eine Demonstration.
- Keine Anlageberatung.
- Ziel ist methodische Nachvollziehbarkeit.
"""),
        COMMON_SETUP,
        code("""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
"""),
        code("""
target = "target_20d"

feature_cols = [
    "daily_return",
    "return_5d",
    "return_20d",
    "ma_20_distance",
    "ma_50_distance",
    "ma_200_distance",
    "volatility_20d",
    "drawdown",
]

available_features = [c for c in feature_cols if c in df.columns]

model_df = df[available_features + [target]].dropna(subset=[target]).copy()
model_df[target] = model_df[target].astype(int)

print("Features:", available_features)
print("Model shape:", model_df.shape)
print("Target distribution:")
display(model_df[target].value_counts(normalize=True).rename("share"))
"""),
        code("""
X = model_df[available_features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

baseline_pred = np.repeat(y_train.mode().iloc[0], len(y_test))

baseline_scores = {
    "model": "Majority Baseline",
    "accuracy": accuracy_score(y_test, baseline_pred),
    "precision": precision_score(y_test, baseline_pred, zero_division=0),
    "recall": recall_score(y_test, baseline_pred, zero_division=0),
    "f1": f1_score(y_test, baseline_pred, zero_division=0),
}

baseline_scores
"""),
        code("""
log_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)

rf_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)),
    ]
)

models = {
    "Logistic Regression": log_model,
    "Random Forest": rf_model,
}

results = [baseline_scores]

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    })

pd.DataFrame(results).sort_values("f1", ascending=False)
"""),
        code("""
best_model = rf_model
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

print(classification_report(y_test, pred, zero_division=0))

cm = confusion_matrix(y_test, pred)
cm
"""),
        code("""
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.show()
"""),
        code("""
rf = best_model.named_steps["model"]
importances = pd.DataFrame({
    "feature": available_features,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)

display(importances)

plt.figure(figsize=(8, 4))
plt.bar(importances["feature"], importances["importance"])
plt.title("Feature Importance – Random Forest")
plt.xticks(rotation=45, ha="right")
plt.show()
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
        ["ML", "Baseline möglich", "Weiterer Ausbau nötig"],
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

Der nächste fachliche Schritt ist ein sauber dokumentiertes ML-Labor mit Modellvergleich, Confusion Matrix und Feature Importance.
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
print("cd ~/UNI/forexscope-ai")
print("python3 -m streamlit run app_max.py")
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
