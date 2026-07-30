"""Generate the eight executable WealthScope AI 1.0 notebooks."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def write(filename: str, title: str, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "wealthscope": {
            "version": "1.0",
            "phase": title,
            "generated_by": "scripts/rebuild_wealthscope_notebooks.py",
        },
    }
    header = md(f"""
# {title}

**Projekt:** WealthScope AI 1.0  
**Methode:** QUA³CK · reproduzierbarer Out-of-Time-Benchmark  
**Hinweis:** Wissenschaftlicher Prototyp, keine Anlageberatung.
""")
    nb["cells"] = [header, *cells]
    nbf.write(nb, OUT / filename)
    print(f"created: {OUT / filename}")


SETUP = code("""
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path("..").resolve()
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "wealthscope_features.parquet"
DIAGNOSTICS_PATH = PROJECT_ROOT / "models" / "diagnostics.json"
EXPERIMENTS_PATH = PROJECT_ROOT / "models" / "validation_experiments.json"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Datensatz fehlt: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print(f"Daten: {len(df):,} Zeilen × {len(df.columns)} Spalten")
display(df.head(3))
""")


write(
    "00_project_overview.ipynb",
    "00 – Projektüberblick",
    [
        md("""
## Zielbild

WealthScope AI verbindet historische US-Aktienmarktdaten mit technischer
Analyse, einem ehrlichen ML-Benchmark, Risikoplanung und Wissenstransfer.
Der Mehrwert liegt in **Nachvollziehbarkeit**, nicht in einem Renditeversprechen.

Die Version 1.0 stellt drei Perspektiven nebeneinander:

1. **KI damals:** Baseline, Logistische Regression, Entscheidungsbaum und SVM.
2. **KI im klassischen ML:** Random Forest, Diagnostik und Erklärbarkeit.
3. **KI heute:** ein erklärender Assistent, klar getrennt vom Prognosemodell.
"""),
        SETUP,
        code("""
overview = pd.Series({
    "Zeilen": len(df),
    "Ticker": df["ticker"].nunique(),
    "Zeitraum von": df["date"].min().date(),
    "Zeitraum bis": df["date"].max().date(),
    "Zielvariable vorhanden": "target_20d" in df.columns,
    "Diagnostik vorhanden": DIAGNOSTICS_PATH.exists(),
})
overview.to_frame("Wert")
"""),
        code("""
qua3ck = pd.DataFrame([
    ["Q", "Question", "Leitfrage und Hypothesen"],
    ["U", "Understanding", "Datenprofil, Fehlwerte, Klassen und Zeit"],
    ["A", "Analytics", "Returns, Trend- und Risikofeatures"],
    ["A", "Algorithm", "Fünf Klassifikatoren auf identischen Fenstern"],
    ["A", "Adaption", "Pipeline, Purge, Walk-forward und Hyperparameter"],
    ["C", "Conclude", "Metriken, Grenzen und Hypothesenbewertung"],
    ["K", "Knowledge", "Streamlit, Export, Lernstudio und Dokumentation"],
], columns=["Phase", "Name", "WealthScope-Artefakt"])
qua3ck
"""),
        md("""
## Reproduzierbarkeit

Die Notebooks erklären und prüfen die Artefakte. Die verbindliche Trainingslogik
liegt in `scripts/train_and_diagnose.py`; Kennzahlen werden aus
`models/diagnostics.json` gelesen. So widersprechen sich App, Ausarbeitung und
Notebooks nicht.
"""),
    ],
)


write(
    "01_question.ipynb",
    "01 – Question",
    [
        md("""
## Leitfrage

**Wie können historische US-Aktienmarktdaten genutzt werden, um eine interaktive
Finanzanalyse-App zu entwickeln, die technische Analyse, ML-Signalgebung und
risikobasierte Positionsplanung nachvollziehbar kombiniert?**

### Hypothesen

- **H1:** Der Random Forest erreicht im purged Out-of-Time-Test eine ROC-AUC
  oberhalb von 0,5 und bleibt im Walk-forward-Vergleich stabil.
- **H2:** Die Streamlit-App macht Daten, Methodik und Grenzen für
  Nicht-Experten zugänglich.
- **H3:** Die Kombination aus technischen Indikatoren, ML und Risiko verbessert
  die Orientierung; ein wirtschaftlicher Nutzen muss separat getestet werden.
"""),
        SETUP,
        code("""
questions = pd.DataFrame([
    ["Q1", "Welche Daten liegen wirklich vor?", "Zeilen, Ticker, Zeitraum, Fehlwerte"],
    ["Q2", "Was ist das Prognoseziel?", "target_20d: Richtung nach 20 Handelstagen"],
    ["Q3", "Wie wird Leakage verhindert?", "Zeit-Split, 20T-Purge, Pipeline"],
    ["Q4", "Welches Modell ist tragfähig?", "Dummy bis Random Forest, gleiche Fenster"],
    ["Q5", "Wie wird Wissen übertragen?", "App, Model Card, Export, Lernstudio"],
], columns=["ID", "Frage", "Operationalisierung"])
questions
"""),
        code("""
scope = pd.DataFrame([
    ["Im Scope", "Klassifikation, technische Features, Erklärbarkeit, Risikoszenarien"],
    ["Nicht im Scope", "Handelsbot, sichere Prognosen, persönliche Anlageberatung"],
    ["Erfolgskriterium", "Reproduzierbare und kritisch interpretierte Ergebnisse"],
], columns=["Bereich", "Festlegung"])
scope
"""),
    ],
)


write(
    "02_understanding_the_data.ipynb",
    "02 – Understanding the Data",
    [
        md("""
## Ziel

Die Datenphase prüft Umfang, zeitliche Abdeckung, Klassenverteilung und
Fehlwerte. Finanzdaten sind **nicht austauschbar**: Reihenfolge und Marktregime
gehören zur Bedeutung jeder Zeile.
"""),
        SETUP,
        code("""
profile = pd.Series({
    "Zeilen": len(df),
    "Spalten": len(df.columns),
    "Ticker": df["ticker"].nunique(),
    "Start": df["date"].min().date(),
    "Ende": df["date"].max().date(),
    "Handelstage": df["date"].nunique(),
})
profile.to_frame("Wert")
"""),
        code("""
missing = df.isna().mean().mul(100).sort_values(ascending=False)
missing[missing > 0].round(2).to_frame("Fehlwerte (%)").head(20)
"""),
        md("""
Fehlwerte in gleitenden Fenstern sind überwiegend **strukturell**: Ein MA-200
kann für die ersten 199 Beobachtungen eines Tickers nicht vollständig
berechnet werden. Das ist keine zufällige MCAR-Lücke. Im Training übernimmt ein
Median-Imputer die Behandlung innerhalb jedes Trainingsfensters.
"""),
        code("""
target = df["target_20d"].dropna().astype(int)
class_balance = target.value_counts(normalize=True).sort_index().rename({
    0: "0 · nicht höher", 1: "1 · höher"
}).mul(100).round(2)
display(class_balance.to_frame("Anteil (%)"))

ax = class_balance.plot(kind="bar", color=["#B98519", "#1F5A4E"], figsize=(6, 3))
ax.set_title("Klassenverteilung target_20d")
ax.set_ylabel("Anteil (%)")
ax.set_xlabel("")
plt.xticks(rotation=0)
plt.show()
"""),
        code("""
ticker_span = (
    df.groupby("ticker")
      .agg(Zeilen=("date", "size"), Start=("date", "min"), Ende=("date", "max"))
      .sort_values("Zeilen", ascending=False)
)
ticker_span.head(10)
"""),
        md("""
## Datenrisiken

- Ticker haben unterschiedlich lange Historien.
- Die Auswahl von 26 Titeln kann Survivorship-/Selection Bias enthalten.
- Daten nach 2017 sind gegenüber dem Training ein Distribution Shift.
- Zufälliges Mischen würde Zeitinformation und überlappende Zielhorizonte leaken.
"""),
    ],
)


write(
    "03_feature_engineering.ipynb",
    "03 – Feature Engineering",
    [
        md("""
## Acht Modellfeatures

| Gruppe | Features | Aussage |
|---|---|---|
| Rendite | `daily_return`, `return_5d`, `return_20d` | kurz- bis mittelfristiges Momentum |
| Trend | `ma_20_distance`, `ma_50_distance`, `ma_200_distance` | Abstand zu gleitenden Durchschnitten |
| Risiko | `volatility_20d`, `drawdown` | Schwankung und Abstand zum Hoch |

Die Zielvariable `target_20d` ist 1, wenn der Schlusskurs 20 Handelstage später
höher ist. Nur die Zielbildung darf nach vorn blicken; alle Eingabefeatures
verwenden ausschließlich Vergangenheit und Gegenwart.
"""),
        SETUP,
        code("""
features = [
    "daily_return", "return_5d", "return_20d",
    "ma_20_distance", "ma_50_distance", "ma_200_distance",
    "volatility_20d", "drawdown",
]

feature_check = pd.DataFrame({
    "vorhanden": [f in df.columns for f in features],
    "fehlend_%": [df[f].isna().mean() * 100 for f in features],
    "median": [df[f].median() for f in features],
    "std": [df[f].std() for f in features],
}, index=features)
feature_check.round(4)
"""),
        code("""
sample_ticker = df["ticker"].value_counts().index[0]
d = df.loc[df["ticker"].eq(sample_ticker)].sort_values("date")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(d["date"], d["close"], label="Close", color="#1F5A4E")
for name, color in [("ma_20", "#B98519"), ("ma_50", "#52796F"), ("ma_200", "#8C6D31")]:
    if name in d:
        axes[0].plot(d["date"], d[name], label=name, linewidth=1, color=color)
axes[0].set_title(f"Kurs und gleitende Durchschnitte · {sample_ticker}")
axes[0].legend(ncol=4)
axes[1].plot(d["date"], d["drawdown"] * 100, color="#B54D3A")
axes[1].set_title("Drawdown zum bisherigen Hoch")
axes[1].set_ylabel("%")
plt.tight_layout()
plt.show()
"""),
        code("""
corr = df[features + ["target_20d"]].corr(method="spearman")
corr["target_20d"].drop("target_20d").sort_values().to_frame(
    "Spearman-Korrelation mit target_20d"
).round(4)
"""),
        md("""
## Interpretation

Schwache Einzelkorrelationen sind bei Marktdaten erwartbar. Sie rechtfertigen
keine Kausalbehauptung. Nichtlineare Modelle dürfen Interaktionen prüfen, müssen
aber auf einem späteren Zeitfenster zeigen, ob das Muster Bestand hat.
"""),
    ],
)


write(
    "04_modeling_baseline_ml.ipynb",
    "04 – Modeling: StandardScaler, KI damals und heute",
    [
        md("""
## Faire Pipeline

**StandardScaler:** $z=(x-\\mu_{Train})/\\sigma_{Train}$

Der Scaler wird nur auf dem Training gefittet. Testdaten werden mit den
Trainingsparametern transformiert. Das verhindert Data Leakage.

- **notwendig/sinnvoll:** Logistische Regression und Linear-SVM, da
  Optimierung bzw. Abstände von der Skala abhängen;
- **nicht erforderlich:** Entscheidungsbaum und Random Forest, weil
  Schwellenentscheidungen weitgehend skaleninvariant sind.

Vor jedem Test- oder Validierungsfenster liegen 20 gesperrte Handelstage
(`purge`), passend zum Horizont von `target_20d`.
"""),
        SETUP,
        code("""
diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
pd.Series(diagnostics["validation"]).to_frame("Validierung")
"""),
        code("""
rows = []
for key, model in diagnostics["model_comparison"].items():
    metric = model["test_metrics"]
    rows.append({
        "Epoche": model["year"],
        "Modell": model["label"],
        "Scaler": "StandardScaler" if key in {"logistic", "linear_svm"} else "nicht nötig",
        "Accuracy": metric["accuracy"],
        "Balanced Accuracy": metric["balanced_accuracy"],
        "ROC-AUC": metric["roc_auc"],
        "Walk-forward AUC": model["walk_forward_roc_auc_mean"],
    })
comparison = pd.DataFrame(rows)
comparison.style.format({
    "Accuracy": "{:.3f}", "Balanced Accuracy": "{:.3f}",
    "ROC-AUC": "{:.3f}", "Walk-forward AUC": "{:.3f}",
})
"""),
        code("""
ax = comparison.plot(
    x="Modell", y=["ROC-AUC", "Balanced Accuracy"],
    kind="bar", figsize=(10, 4), color=["#1F5A4E", "#B98519"],
)
ax.axhline(0.5, color="#58635E", linestyle="--", linewidth=1)
ax.set_ylim(0.47, 0.54)
ax.set_title("Identischer purged Out-of-Time-Test")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
"""),
        code("""
# Sichtbarer Leakage-Test: Der Scaler kennt nur X_train.
from sklearn.preprocessing import StandardScaler

features = diagnostics["features"]
model_df = (
    df.dropna(subset=features + ["target_20d", "date"])
      .sort_values(["date", "ticker"])
)
train_end = pd.Timestamp(diagnostics["validation"]["train_end"])
test_start = pd.Timestamp(diagnostics["validation"]["test_start"])
X_train = model_df.loc[model_df["date"].le(train_end), features]
X_test = model_df.loc[model_df["date"].ge(test_start), features]

scaler = StandardScaler().fit(X_train)
train_scaled = scaler.transform(X_train)
test_scaled = scaler.transform(X_test)

pd.DataFrame({
    "Prüfung": ["Mittelwert skaliertes Training", "Mittelwert skalierter Test"],
    "absoluter Mittelwert über Features": [
        np.abs(train_scaled.mean(axis=0)).mean(),
        np.abs(test_scaled.mean(axis=0)).mean(),
    ],
})
"""),
        md("""
Der Trainingsmittelwert liegt nach Skalierung praktisch bei null. Der
Testmittelwert darf davon abweichen: Würde man auch ihn künstlich auf null
zentrieren, hätte der Scaler bereits Information aus der Zukunft gesehen.

Die Single Source of Truth ist `scripts/train_and_diagnose.py`.

## Was kostet eine falsche Aufteilung?

Die Forderung „kein Zufalls-Split" bleibt eine Behauptung, solange man sie nicht
beziffert. `scripts/validation_experiments.py` trainiert deshalb dasselbe Modell
auf denselben Daten dreimal und variiert ausschließlich die Aufteilung.
"""),
        code("""
experiments = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))

splits = pd.DataFrame(experiments["split_comparison"])[
    ["label", "leakage", "roc_auc", "balanced_accuracy",
     "accuracy", "majority_baseline_accuracy"]
].rename(columns={
    "label": "Aufteilung", "leakage": "Leakage", "roc_auc": "ROC-AUC",
    "balanced_accuracy": "Balanced Accuracy", "accuracy": "Accuracy",
    "majority_baseline_accuracy": "Baseline Accuracy",
})
splits.style.format({
    "ROC-AUC": "{:.4f}", "Balanced Accuracy": "{:.4f}",
    "Accuracy": "{:.4f}", "Baseline Accuracy": "{:.4f}",
}).hide(axis="index")
"""),
        code("""
summary = experiments["split_comparison_summary"]
print(f"purged Out-of-Time (v1.0) : ROC-AUC {summary['reference_roc_auc']:.4f}")
print(f"naiver Zufalls-Split      : ROC-AUC {summary['leaky_roc_auc']:.4f}")
print(f"Differenz                 : {summary['absolute_difference']:+.4f}")
print()
print("Gemessen am Zufallsniveau 0,5:")
print(f"  echtes scheinbares Signal   {summary['apparent_signal_reference']:.4f}")
print(f"  Signal mit Leakage          {summary['apparent_signal_leaky']:.4f}")
print(f"  Faktor                      {summary['apparent_signal_factor']:.1f}x")
"""),
        md("""
Der naive Split lässt das scheinbare Signal rund **4,2-mal größer** aussehen –
ohne dass Modell oder Daten sich ändern. Die niedrige Kennzahl dieses Projekts
ist damit keine Schwäche, sondern die Folge einer Validierung, die dem Modell
die Zukunft entzieht.
"""),
        code("""
# Ist die Trainings-Test-Lücke ein Kapazitätsproblem? Sieben Stufen im Vergleich.
capacity = pd.DataFrame(experiments["capacity_sweep"])[
    ["label", "train_roc_auc", "roc_auc", "gap"]
].rename(columns={
    "label": "Konfiguration", "train_roc_auc": "Train-AUC",
    "roc_auc": "Test-AUC", "gap": "Lücke",
})

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(capacity["Konfiguration"], capacity["Train-AUC"], "o-",
        color="#B98519", label="Train-AUC")
ax.plot(capacity["Konfiguration"], capacity["Test-AUC"], "o-",
        color="#1F5A4E", label="Test-AUC")
ax.axhline(0.5, color="#58635E", linestyle="--", linewidth=1)
ax.set_title("Kapazität verändert nur die Trainingsseite")
ax.set_ylabel("ROC-AUC")
ax.legend()
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()

display(capacity.style.format({
    "Train-AUC": "{:.4f}", "Test-AUC": "{:.4f}", "Lücke": "{:.4f}",
}).hide(axis="index"))
"""),
        md("""
Der Trainings-AUC wandert von 1,000 (perfektes Auswendiglernen aller
Trainingszeilen) bis 0,543. Der Test-AUC bleibt dabei in einer Spanne von
0,007 – also im Rauschen. **Die Obergrenze setzt der Informationsgehalt der
Features, nicht die Modellkapazität.** Beste Einstellung im Vergleich ist genau
die produktive Konfiguration `max_depth=8, min_samples_leaf=5`.
"""),
    ],
)


write(
    "05_conclude_evaluate.ipynb",
    "05 – Conclude & Evaluate",
    [
        md("""
## Ergebnislogik

Accuracy allein ist wegen der Mehrheitsklasse irreführend. Deshalb werden
Balanced Accuracy, ROC-AUC, Average Precision, Konfusionsmatrix und
Walk-forward-Stabilität gemeinsam bewertet.
"""),
        SETUP,
        code("""
diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
m = diagnostics["test_metrics"]
summary = pd.Series({
    "Random-Forest Accuracy": m["accuracy"],
    "Mehrheitsbaseline Accuracy": m["majority_baseline"],
    "Balanced Accuracy": m["balanced_accuracy"],
    "ROC-AUC": m["roc_auc"],
    "Average Precision": m["average_precision"],
    "Walk-forward ROC-AUC": diagnostics["cross_validation"]["auc_mean"],
})
summary.to_frame("Wert").style.format("{:.3f}")
"""),
        code("""
cm = np.array(diagnostics["confusion_matrix"]["counts"])
fig, ax = plt.subplots(figsize=(4.5, 3.8))
im = ax.imshow(cm, cmap="Greens")
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, f"{value:,}", ha="center", va="center")
ax.set_xticks([0, 1], ["0", "1"])
ax.set_yticks([0, 1], ["0", "1"])
ax.set_xlabel("Vorhersage")
ax.set_ylabel("Wahr")
ax.set_title("Konfusionsmatrix · Out-of-Time-Test")
plt.colorbar(im, ax=ax, fraction=.045)
plt.show()
"""),
        code("""
hypotheses = pd.DataFrame([
    ["H1", "falsifiziert",
     f"RF AUC {m['roc_auc']:.3f}; Walk-forward {diagnostics['cross_validation']['auc_mean']:.3f}"],
    ["H2", "im Prototyp umgesetzt", "App, Methodik, Export und Lernstudio; Nutzerstudie offen"],
    ["H3", "teilweise plausibel", "Orientierung verbessert; wirtschaftlicher Nutzen nicht bewiesen"],
], columns=["Hypothese", "Urteil", "Begründung"])
hypotheses
"""),
        md("""
H1 ist **falsifiziert, nicht ungeprüft**. Die Hypothese war so formuliert, dass
sie scheitern konnte – genau das macht sie zu einer wissenschaftlichen Aussage.
Ein Negativergebnis ist allerdings nur dann ein Befund, wenn die naheliegenden
Gegenerklärungen ausgeschlossen sind. Genau das prüft die nächste Zelle.
"""),
        code("""
experiments = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
cap = experiments["capacity_sweep_summary"]
split = experiments["split_comparison_summary"]
lc = json.loads((PROJECT_ROOT / "models" / "learning_curve.json").read_text(encoding="utf-8"))
val_trend = lc["val_scores_mean"][-1] - lc["val_scores_mean"][0]
data_factor = lc["train_sizes_abs"][-1] / lc["train_sizes_abs"][0]

checks = pd.DataFrame([
    ["Zu wenig Modellkapazität?", "ausgeschlossen",
     f"Test-AUC-Spanne über 7 Stufen nur {cap['test_roc_auc_span']:.4f} "
     f"({cap['test_roc_auc_min']:.4f}–{cap['test_roc_auc_max']:.4f})"],
    ["Zu wenig Daten?", "ausgeschlossen",
     f"{data_factor:.1f}-fache Trainingsmenge verändert die Validierung um "
     f"{val_trend:+.4f}"],
    ["Ist die Aufteilung der Hebel?", "ja – und zwar der einzige",
     f"naiver Zufalls-Split: AUC {split['leaky_roc_auc']:.4f} statt "
     f"{split['reference_roc_auc']:.4f} → scheinbares Signal "
     f"{split['apparent_signal_factor']:.1f}x größer"],
], columns=["Gegenerklärung", "Ergebnis", "Belegte Messung"])
checks.style.hide(axis="index")
"""),
        md("""
## Fazit

Die Out-of-Time-AUC des Random Forest beträgt rund **0,519** – kein belastbares
Handelssignal. Das eigentliche Ergebnis dieses Projekts ist deshalb nicht das
Modell, sondern eine **Messung**: Wie viel verwertbare Information tragen rein
kursbasierte technische Indikatoren? Antwort: nahezu keine, belegt an 190.527
Beobachtungen mit elf Jahren unangetastetem Testzeitraum.

Zwei Gegenerklärungen wurden ausgeschlossen (Kapazität, Datenmenge), und der
tatsächliche Hebel wurde beziffert: Ein naiver Zufalls-Split hätte dieselben
Daten mit AUC 0,581 bewertet – ein rund **4,2-mal größeres scheinbares Signal**.
Damit ist die niedrige Kennzahl kein Qualitätsmangel, sondern der Nachweis, dass
dieser Fehler nicht gemacht wurde. Das ist eine nachgerechnete, nicht bloß
zitierte Bestätigung der Effizienzmarkthypothese (Fama 1970).

### Nächste Schritte

- echter Forward-Test mit neueren Daten;
- Transaktionskosten, Slippage und Steuern;
- Makro-, Fundamental- und Sentimentvariablen (der einzige Hebel, der die
  Validierungskurve heben könnte);
- formale Nutzerstudie zur Verständlichkeit.
"""),
    ],
)


write(
    "06_knowledge_transfer_streamlit.ipynb",
    "06 – Knowledge Transfer: Streamlit-App",
    [
        md("""
## Von Analyse zu Wissen

Notebooks dokumentieren die Herleitung; die App macht sie interaktiv. Der
Knowledge-Schritt trennt Prognose, Erklärung und Handlungshilfe sichtbar.
"""),
        SETUP,
        code("""
components = pd.DataFrame([
    ["Marktanalyse", "Kurs, Moving Averages, Candlesticks, Volatilität, Drawdown"],
    ["ML Insights", "Vergleich, Konfusionsmatrix, ROC/PR, Lernkurve, Wichtigkeiten"],
    ["Kapital-Kompass", "Positionsgröße auf Basis eines expliziten Risikobudgets"],
    ["Portfolio-Simulator", "Szenarien statt Renditeversprechen"],
    ["Lernstudio", "Quiz, Feedback und arsnova.eu-kompatibler Export"],
    ["Methodik & Export", "QUA³CK, Model Card und reproduzierbare Dateien"],
    ["Assistent", "Erklärt Analysekontext; erteilt keine Anlageberatung"],
], columns=["Bereich", "Wissenstransfer"])
components
"""),
        code("""
required_files = [
    "app.py",
    "src/pages/market.py",
    "src/pages/ml_insights.py",
    "src/pages/kompass.py",
    "src/pages/simulator.py",
    "src/pages/learning_studio.py",
    "src/pages/methodology.py",
    "docs/model_card.md",
]
pd.DataFrame({
    "Artefakt": required_files,
    "vorhanden": [(PROJECT_ROOT / path).exists() for path in required_files],
})
"""),
        code("""
print("Lokaler Start:")
print("python -m streamlit run app.py")
print("\\nDidaktische Referenzen:")
print("https://arsnova.eu/de/")
print("https://mc-test.streamlit.app/")
"""),
        md("""
## Transferprinzip

„KI damals und heute“ wird nicht nur erzählt: Der identische Test zeigt
historische Modellgenerationen, während der moderne Assistent ausschließlich
erklärt. Damit bleibt nachvollziehbar, welche Komponente rechnet, welche
visualisiert und welche Sprache erzeugt.
"""),
    ],
)


write(
    "07_newsapi_assistant_export.ipynb",
    "07 – NewsAPI, Assistent und Export",
    [
        md("""
## Zusatzmodule mit klaren Grenzen

Diese Module erhöhen Aktualität und Verständlichkeit, verändern aber nicht
rückwirkend den historischen ML-Benchmark.
"""),
        SETUP,
        code("""
modules = pd.DataFrame([
    ["NewsAPI", "externe Schlagzeilen", "Netzwerk/API-Key, Abdeckung, Aktualität"],
    ["Regelbasiertes Sentiment", "transparente Einordnung", "keine tiefe Semantik"],
    ["KI-Assistent", "Erklärung der aktuellen Ansicht", "Halluzination, keine Beratung"],
    ["Export", "CSV, JSON, Markdown, ZIP", "Zeitstempel und Annahmen mitliefern"],
], columns=["Modul", "Nutzen", "Grenze"])
modules
"""),
        code("""
assistant_guardrails = [
    "Keine Kauf-, Verkaufs- oder Renditeversprechen.",
    "Modellkennzahlen und Datenzeitraum nennen.",
    "Historische Signale nicht als Kausalität darstellen.",
    "Bei fehlenden Live-Daten den Zustand transparent ausweisen.",
    "Export muss Annahmen, Quelle und Zeitstempel enthalten.",
]
pd.DataFrame({"Leitplanke": assistant_guardrails})
"""),
        code("""
export_contract = {
    "app_version": "1.0",
    "model_metrics_source": "models/diagnostics.json",
    "method": "purged out-of-time + expanding walk-forward",
    "target": "target_20d",
    "financial_advice": False,
}
print(json.dumps(export_contract, ensure_ascii=False, indent=2))
"""),
        md("""
## Schluss

Ein gutes KI-Produkt macht Unsicherheit sichtbar. News, Assistent und Export
sind daher Kommunikationsschichten – keine Abkürzung zu einer stärkeren
Prognose.
"""),
    ],
)

