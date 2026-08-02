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


def lernziele(*items: str):
    """Opening block: what this notebook lets the reader do afterwards."""
    bullets = "\n".join(f"- {i}" for i in items)
    return md(f"## Lernziele\n\nNach diesem Notebook könnt ihr:\n\n{bullets}")


def checkpoint(text: str):
    """Closing block: the defensible takeaway, in the wording of the module."""
    return md(f"## Management-Checkpoint\n\n{text.strip()}")


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
        lernziele(
            "die acht Notebooks der Reihe nach einordnen",
            "nachvollziehen, welches Artefakt welche Kennzahl verbindlich festlegt",
            "das zentrale Ergebnis des Projekts in einem Satz benennen",
        ),
        md("""
## Leseanleitung

Jedes Notebook beantwortet genau eine Frage. Wer nur das Ergebnis sucht,
springt zu **05**; wer die Methodik prüfen will, liest **03** und **04**.

| Notebook | Beantwortet | Kernbefund |
|---|---|---|
| 00 Überblick | Worum geht es? | Ehrlichkeit vor Renditeversprechen |
| 01 Question | Was wird gefragt, was widerlegt es? | H1 ist falsifizierbar formuliert |
| 02 Understanding | Welche Daten liegen vor? | Strukturelle Fehlwerte, instabile Klassenlage |
| 03 Feature Engineering | Was darf ein Feature sein? | Der Prognosezeitpunkt entscheidet |
| 04 Modeling | Welches Modell trägt? | Keines schlägt die Baseline; Leakage täuscht |
| 05 Conclude | Was bleibt? | H1 falsifiziert, mit Konfidenzintervall |
| 06 Knowledge | Wie wird es zugänglich? | Streamlit, Export, Lernstudio |
| 07 NewsAPI | Was macht die KI-Ebene? | Erklären, nicht prognostizieren |

> **Arbeitsregel des Projekts:** Jede Zahl in App, Ausarbeitung, Poster und
> Notebooks stammt aus `models/*.json`. Wo eine Zahl von Hand getippt wurde, ist
> sie erfahrungsgemäß nach der ersten Änderung falsch.
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
        checkpoint("""
WealthScope AI liefert **kein** Handelssignal. Der Ertrag der Arbeit ist eine
Messung: Wie viel verwertbare Information tragen rein kursbasierte technische
Indikatoren? Antwort — nahezu keine, belegt an 190.527 Beobachtungen mit elf
Jahren unangetastetem Testzeitraum. Wer das Projekt in einem Satz zusammenfasst,
nennt dieses Ergebnis, nicht die App.
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
        lernziele(
            "eine Hypothese so formulieren, dass sie scheitern kann",
            "das Prognoseziel vom Prognosezeitpunkt her denken",
            "Erfolgskriterien vor der ersten Modellzeile festlegen",
        ),
        md("""
## Ausgangslage

Privatanleger stehen vor Datenfülle, widersprüchlichen Signalen und
Werkzeugen, die ihre eigene Treffsicherheit nicht offenlegen. Die naheliegende
Projektidee — „ein ML-Modell, das Kurse vorhersagt" — ist deshalb weniger
interessant als die Frage dahinter: **Wie viel Signal steckt überhaupt in reinen
Kursdaten, wenn man ehrlich misst?**

Diese Umformulierung ist der eigentliche Projektentscheid. Sie macht ein
schwaches Ergebnis zu einem Befund statt zu einem Misserfolg.

## Falsifikationskriterien

Eine Hypothese, die jeder Ausgang bestätigt, ist wertlos. Deshalb steht
**vor** dem Training fest, was sie widerlegen würde:

| Hypothese | Bestätigt, wenn … | Widerlegt, wenn … | Geprüft in |
|---|---|---|---|
| **H1** | ROC-AUC deutlich über 0,5 **und** über die Walk-forward-Folds stabil | AUC nahe 0,5 oder über Zeitfenster instabil | 04, 05 |
| **H2** | Methodik, Kennzahlen und Grenzen sind in der App ohne Vorwissen auffindbar | zentrale Kennzahlen fehlen oder sind unerklärt | 06 |
| **H3** | messbarer Zusatznutzen gegenüber Einzelindikatoren | kein Nutzen nachweisbar oder nicht prüfbar | 05 |

> **Wichtig für H1:** „AUC über 0,5" allein genügt nicht. Bei 69.147 Testfällen
> wird auch ein winziger Vorsprung statistisch signifikant. Notebook 05 prüft
> deshalb zusätzlich per Bootstrap, wie groß der Vorsprung wirklich ist — und
> trennt statistische Signifikanz von praktischer Bedeutung.
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
        checkpoint("""
Das Erfolgskriterium dieses Projekts ist **nicht** eine hohe Kennzahl, sondern
eine Kennzahl, der man glauben kann. Wer H1 vor dem Training so formuliert, dass
sie scheitern kann, darf ein schwaches Ergebnis später als Befund berichten
statt es zu verstecken. Umgekehrt gilt: Wäre H1 als „das Modell findet Muster"
formuliert worden, hätte jedes Ergebnis sie bestätigt — und die Arbeit wäre
wissenschaftlich wertlos.
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
        lernziele(
            "den Fehlwert-Mechanismus belegen statt ihn zu behaupten",
            "aus der Verteilungsform eine Imputationsentscheidung ableiten",
            "Nichtstationarität der Zielvariablen über Dekaden erkennen",
            "einen Distribution Shift zwischen Trainings- und Testfenster messen",
        ),
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
### Den Mechanismus belegen, nicht behaupten

Die übliche Formulierung „die Fehlwerte sind strukturell bedingt" ist eine
Behauptung. Prüfbar wird sie erst, wenn man zeigt, dass ein Fehlwert
**ausschließlich** innerhalb der Warm-up-Phase des jeweiligen Fensters auftritt:
Ein MA-200 kann für die ersten 199 Handelstage eines Tickers gar nicht existieren.

Wenn das stimmt, darf kein fehlender Wert jenseits von Handelstag 199 liegen.
"""),
        code("""
warm = df.sort_values(["ticker", "date"]).copy()
warm["handelstag_index"] = warm.groupby("ticker").cumcount()  # 0-basiert

nachweis = []
for spalte, fenster in [("ma_20_distance", 20), ("ma_50_distance", 50),
                        ("ma_200_distance", 200)]:
    fehlend = warm.loc[warm[spalte].isna(), "handelstag_index"]
    nachweis.append({
        "Feature": spalte,
        "Fenster": fenster,
        "Fehlwerte": len(fehlend),
        "spätester Handelstag": int(fehlend.max()) if len(fehlend) else None,
        "nur im Warm-up?": bool(len(fehlend) == 0 or fehlend.max() < fenster),
    })
pd.DataFrame(nachweis)
"""),
        md("""
Der späteste fehlende Wert liegt bei jedem Feature **genau ein Handelstag vor
dem Fensterende**: Index 18, 48 und 198 bei nullbasierter Zählung, also der 19.,
49. und 199. Handelstag des jeweiligen Tickers. Ab dem 20., 50. bzw. 200. Tag
ist das Fenster vollständig und der Wert existiert. Kein einziger Fehlwert tritt
später auf. Damit
ist der Mechanismus nicht vermutet, sondern nachgewiesen: Die Lücken sind
vollständig durch die Konstruktion erklärt und hängen nicht vom Kurs ab — der
für eine Imputation günstigste Fall.

Für das Modell werden diese Zeilen ohnehin verworfen (192.119 → 190.527). Der
Median-Imputer in der Pipeline ist die zweite Absicherung: Er greift nur
innerhalb des jeweiligen Trainingsfensters, damit keine Testinformation in die
Vorverarbeitung sickert.
"""),
        md("""
### Warum Median und nicht Mittelwert?

Die Wahl ist keine Geschmacksfrage, sondern folgt aus der Verteilungsform.
Schiefe und Kurtosis zeigen, wie stark Ausreißer einen Mittelwert verzerren
würden.
"""),
        code("""
features = [
    "daily_return", "return_5d", "return_20d",
    "ma_20_distance", "ma_50_distance", "ma_200_distance",
    "volatility_20d", "drawdown",
]

form = pd.DataFrame({
    "Schiefe": df[features].skew(),
    "Kurtosis": df[features].kurtosis(),
    "Mittelwert": df[features].mean(),
    "Median": df[features].median(),
})
form["Mittelwert − Median"] = form["Mittelwert"] - form["Median"]
form.round(3)
"""),
        md("""
`daily_return` hat eine Kurtosis von rund **59** — extrem fettschwänzig, wie bei
Finanzrenditen zu erwarten. `volatility_20d` ist mit einer Schiefe von etwa
**3,2** deutlich rechtsschief. In beiden Fällen wird der Mittelwert von wenigen
Krisentagen nach oben gezogen, während der Median stabil bleibt. Genau deshalb
imputiert die Pipeline mit dem Median.
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
        md("""
### Die Klassenlage ist nicht konstant

Ein Gesamtanteil von rund 56 % sagt wenig, wenn er über die Zeit schwankt. Für
ein Modell, das auf alten Daten trainiert und auf neuen geprüft wird, ist genau
diese Schwankung entscheidend: Sie bestimmt, wie stark sich die Baseline
zwischen Trainings- und Testfenster verschiebt.
"""),
        code("""
dek = df.dropna(subset=["target_20d"]).copy()
dek["Dekade"] = (dek["date"].dt.year // 10) * 10

je_dekade = dek.groupby("Dekade")["target_20d"].agg(
    Anteil_bullish="mean", Beobachtungen="size"
)
display(je_dekade.assign(Anteil_bullish=lambda t: (t["Anteil_bullish"] * 100).round(1)))

ax = (je_dekade["Anteil_bullish"] * 100).plot(
    kind="bar", color="#1F5A4E", figsize=(7, 3)
)
ax.axhline(50, color="#B54D3A", linestyle="--", linewidth=1, label="50 % (Zufall)")
ax.set_title("Anteil steigender 20-Tage-Fenster je Dekade")
ax.set_ylabel("% bullish")
ax.set_ylim(40, 65)
ax.legend()
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""),
        md("""
Die Bandbreite reicht von etwa **49 % in den 1970ern bis 60 % in den 2010ern**.
Die Zielvariable ist also nicht stationär — sie folgt dem langfristigen
Aufwärtstrend der Aktienmärkte. Ein Modell, das auf den 1980ern trainiert und in
den 2010ern getestet wird, sieht eine andere Grundgesamtheit als im Training.
Das ist ein Argument **für** den zeitlichen Split und zugleich eine Warnung:
Ein Teil jeder scheinbaren „Trefferquote" ist schlicht Marktdrift.
"""),
        md("""
### Distribution Shift zwischen Training und Test messen

Der Out-of-Time-Test unterstellt, dass Trainings- und Testfenster überhaupt
vergleichbar sind. Prüfen lässt sich das, indem man die Feature-Mittelwerte
beider Fenster in Einheiten der Trainings-Standardabweichung vergleicht.
"""),
        code("""
diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
V = diagnostics["validation"]

modelldaten = df.dropna(subset=features + ["target_20d", "date"])
train = modelldaten[modelldaten["date"] <= pd.Timestamp(V["train_end"])]
test = modelldaten[modelldaten["date"] >= pd.Timestamp(V["test_start"])]

shift = pd.DataFrame({
    "Train": train[features].mean(),
    "Test": test[features].mean(),
})
shift["Differenz in Train-Std"] = (
    (test[features].mean() - train[features].mean()) / train[features].std()
)
display(shift.round(3))
print(f"Trainingsfenster: bis {V['train_end']}  ({len(train):,} Zeilen)")
print(f"Testfenster:      ab {V['test_start']}  ({len(test):,} Zeilen)")
"""),
        md("""
Die meisten Features verschieben sich um weniger als 0,06 Trainings-Standard­
abweichungen — vernachlässigbar. Eine Ausnahme sticht heraus: `volatility_20d`
liegt im Testfenster rund **0,58 Standardabweichungen niedriger**. Das
Testfenster (2006–2017) enthält zwar die Finanzkrise, aber überwiegend die
ruhige Aufwärtsphase danach, während das Trainingsfenster die volatilen
1970er- und 2000er-Jahre mitträgt.

Das ist ein realer Distribution Shift — und er ist **kein Fehler des Aufbaus**,
sondern genau das, was ein ehrlicher Test abbilden muss: Zukünftige Marktregime
sind nie mit den vergangenen identisch.
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
        checkpoint("""
Drei Befunde aus der Datenphase, die jede spätere Kennzahl mitbestimmen:

1. **Die Fehlwerte sind nachgewiesen strukturell**, nicht vermutet — kein
   Fehlwert liegt jenseits der Warm-up-Phase seines Fensters. Die Imputation ist
   damit unkritisch, muss aber trotzdem in der Pipeline liegen.
2. **Die Zielvariable ist nicht stationär** (49 % bis 60 % je Dekade). Ein Teil
   jeder Trefferquote ist Marktdrift, kein Modellverdienst. Deshalb wird gegen
   die Mehrheitsbaseline des jeweiligen Fensters gemessen, nicht gegen 50 %.
3. **Es gibt einen messbaren Distribution Shift** in der Volatilität
   (−0,58 Trainings-Std). Das ist erwünscht, nicht störend: Ein Test, der dem
   Training gleicht, prüft keine Generalisierung.

Entscheidung für die nächste Phase: zeitlicher Split ist gesetzt, Imputation
median-basiert und ausschließlich trainingsseitig gefittet.
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
        lernziele(
            "die Feature-Auswahl aus dem Prognosezeitpunkt herleiten",
            "Pearson und Spearman unterscheiden und die Wahl begründen",
            "Multikollinearität messen und ihre Folgen je Modellklasse einordnen",
            "erkennen, welche Spalten des Datensatzes niemals Feature sein dürfen",
        ),
        md("""
## Die Regel, die über alles andere entscheidet

Vor jeder Korrelation steht eine Frage, die keine Statistik beantwortet:

> **Welche Information liegt zum Prognosezeitpunkt tatsächlich vor?**

Wer heute entscheiden will, ob er in 20 Handelstagen einen höheren Kurs erwartet,
kennt heute den Kursverlauf bis heute — mehr nicht. Jede Spalte, die Wissen von
morgen enthält, ist als Feature verboten, **egal wie gut sie korreliert**.

Der Datensatz enthält absichtlich zwei solcher Spalten:

| Spalte | Rolle | Als Feature? |
|---|---|---|
| `future_return_20d` | Rendite der nächsten 20 Handelstage | **nie** — reine Zukunft |
| `target_20d` | Vorzeichen ebendieser Rendite | **nie** — das ist das Ziel |

Beide bleiben im Datensatz, weil die Zielbildung sie braucht. Ein Modell darf
sie nicht sehen. Notebook 04 zeigt, was passiert, wenn man diese Regel bricht.
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
        md("""
## Korrelation mit dem Ziel: Pearson und Spearman

Pearson misst **lineare** Zusammenhänge und reagiert empfindlich auf die
Ausreißer, die Notebook 02 nachgewiesen hat. Spearman arbeitet auf Rängen und
ist damit robust. Weichen beide stark voneinander ab, liegt der Unterschied an
Ausreißern oder an einem nichtlinearen Zusammenhang — beides ist eine
Information, kein Störgeräusch.
"""),
        code("""
ziel_korr = pd.DataFrame({
    "Pearson": df[features + ["target_20d"]].corr()["target_20d"],
    "Spearman": df[features + ["target_20d"]].corr(method="spearman")["target_20d"],
}).drop("target_20d")
ziel_korr["|Differenz|"] = (ziel_korr["Pearson"] - ziel_korr["Spearman"]).abs()
ziel_korr.sort_values("Spearman").round(4)
"""),
        md("""
Der stärkste Zusammenhang ist `volatility_20d` mit rund **−0,037** — und selbst
der liegt unter jeder Schwelle, ab der Lehrbücher von einem „schwachen"
Zusammenhang sprechen. Pearson und Spearman stimmen dabei fast überall überein;
nennenswert abweichend ist nur `drawdown` (0,023 gegen 0,031), was auf einen
leicht nichtlinearen Zusammenhang hindeutet. Ein einzelnes technisches Merkmal
trägt praktisch keine Richtungsinformation.

Das schließt ein nichtlineares Zusammenspiel mehrerer Merkmale nicht aus —
genau dafür gibt es den Random Forest. Notebook 04 prüft, ob er etwas findet.
"""),
        md("""
## Multikollinearität: wie eigenständig sind die acht Features?

Die drei MA-Abstände messen dieselbe Grundgröße auf unterschiedlichen
Zeitskalen. Der Variance Inflation Factor (VIF) beziffert, wie gut sich ein
Feature aus den übrigen sieben vorhersagen lässt: VIF = 1 bedeutet
eigenständig, Werte über 10 gelten üblicherweise als kritisch.
"""),
        code("""
from sklearn.linear_model import LinearRegression

X = df[features].dropna()
vif = {}
for f in features:
    andere = [c for c in features if c != f]
    r2 = LinearRegression().fit(X[andere], X[f]).score(X[andere], X[f])
    vif[f] = 1.0 / (1.0 - r2)

pd.Series(vif, name="VIF").sort_values(ascending=False).to_frame().round(2)
"""),
        md("""
Die Werte liegen zwischen rund 1,0 und 8,3 — erhöht, aber unter der üblichen
Warnschwelle von 10. Die Rendite- und Trendfeatures überlappen erwartungsgemäß
(`ma_20_distance` und `return_20d` beschreiben teilweise dieselbe Bewegung),
`volatility_20d` und `drawdown` sind dagegen weitgehend eigenständig.

**Folge je Modellklasse — und deshalb steht diese Analyse hier:**

- **Logistische Regression und Linear SVM:** Multikollinearität macht einzelne
  Koeffizienten instabil und ihre Interpretation unzuverlässig. Die
  Vorhersagegüte leidet weniger als die Erklärbarkeit.
- **Entscheidungsbaum und Random Forest:** unkritisch für die Vorhersage, aber
  die Feature Importance verteilt sich willkürlich auf korrelierte Merkmale.
  Wenn in Notebook 04 `drawdown` und `volatility_20d` vorne liegen, ist das auch
  deshalb plausibel, weil sie die geringste Überlappung haben.
"""),
        md("""
## Interpretation

Schwache Einzelkorrelationen sind bei Marktdaten erwartbar. Sie rechtfertigen
keine Kausalbehauptung. Nichtlineare Modelle dürfen Interaktionen prüfen, müssen
aber auf einem späteren Zeitfenster zeigen, ob das Muster Bestand hat.

Umgekehrt gilt: Eine **hohe** Korrelation mit dem Ziel wäre an dieser Stelle
kein Grund zur Freude, sondern ein Anlass, die betreffende Spalte auf Leakage zu
prüfen. In Finanzdaten ist ein starkes Signal fast immer ein Konstruktionsfehler.
"""),
        checkpoint("""
Die Feature-Auswahl folgt nicht der Korrelationsstärke, sondern dem
Prognosezeitpunkt. `future_return_20d` korreliert perfekt mit dem Ziel und ist
trotzdem als Feature ausgeschlossen — Notebook 04 beziffert, was seine
Verwendung vortäuschen würde.

Für die Modellwahl folgt daraus: Da kein Einzelfeature trägt (|ρ| < 0,03) und
die Features sich teilweise überlappen (VIF bis 8,3), ist ein nichtlineares
Ensemble die naheliegende Wahl — mit der Einschränkung, dass seine Feature
Importance bei korrelierten Merkmalen vorsichtig zu lesen ist.
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
        lernziele(
            "eine Baseline vor jedem Modell festlegen und gegen sie messen",
            "Data Leakage nicht nur erklären, sondern seinen Effekt beziffern",
            "den Klassifikationsschwellenwert als Geschäftsentscheidung behandeln",
            "Trainingserfolg von Generalisierung unterscheiden",
        ),
        SETUP,
        code("""
diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
pd.Series(diagnostics["validation"]).to_frame("Validierung")
"""),
        md("""
## Die Baseline zuerst

Bevor irgendein Modell bewertet wird, steht fest, was man ohne Modell erreicht.
Im Testfenster sind rund 59 % der Fenster steigend. Wer immer „steigt" sagt,
liegt also in 59 % der Fälle richtig — **ohne eine einzige Zeile Code**.

Diese Zahl ist der Maßstab. Ein Modell mit 52 % Accuracy ist nicht „etwas besser
als Raten", sondern **schlechter als die triviale Regel**.
"""),
        code("""
m = diagnostics["test_metrics"]
baseline = m["majority_baseline"]

vergleich = pd.DataFrame([
    ["Immer 'steigt' sagen (Baseline)", baseline, 0.5],
    ["Random Forest", m["accuracy"], m["roc_auc"]],
], columns=["Verfahren", "Accuracy", "ROC-AUC"])
vergleich["Accuracy vs. Baseline"] = vergleich["Accuracy"] - baseline
vergleich.style.format({
    "Accuracy": "{:.4f}", "ROC-AUC": "{:.4f}", "Accuracy vs. Baseline": "{:+.4f}",
}).hide(axis="index")
"""),
        md("""
Der Random Forest liegt bei der rohen Accuracy **rund 7 Prozentpunkte unter**
der Baseline. Er ist trotzdem nicht wertlos: Die Baseline hat per Konstruktion
eine ROC-AUC von 0,5 und eine Balanced Accuracy von 0,5 — sie erkennt keine
einzige fallende Phase. Der Random Forest rankt geringfügig besser als Zufall.

Genau deshalb berichtet dieses Projekt **nie Accuracy allein**.
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

## Die Leakage-Falle: das fast perfekte Modell

Notebook 03 hat `future_return_20d` als Feature ausgeschlossen — mit einem
Argument, nicht mit einer Messung. Hier wird die Regel bewusst gebrochen, um
den Effekt zu beziffern.

`target_20d` ist per Konstruktion das Vorzeichen von `future_return_20d`. Ein
Modell, das diese Spalte sehen darf, muss also nur ein Vorzeichen ablesen. Beide
Modelle bekommen dieselben Daten, dieselbe Aufteilung und dieselben
Hyperparameter — der einzige Unterschied ist ein zusätzliches Feature.
"""),
        code("""
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

H = diagnostics["hyperparams"]
features = diagnostics["features"]

# Identische Zeilenbasis für beide Varianten, damit der Vergleich fair bleibt.
basis = (
    df.dropna(subset=features + ["target_20d", "future_return_20d", "date"])
      .sort_values(["date", "ticker"])
)
tr = basis[basis["date"] <= pd.Timestamp(diagnostics["validation"]["train_end"])]
te = basis[basis["date"] >= pd.Timestamp(diagnostics["validation"]["test_start"])]
y_tr = tr["target_20d"].astype(int)
y_te = te["target_20d"].astype(int)


def trainiere(spalten):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("mod", RandomForestClassifier(
            n_estimators=H["n_estimators"], max_depth=H["max_depth"],
            min_samples_leaf=H["min_samples_leaf"], class_weight=H["class_weight"],
            random_state=H["random_state"], n_jobs=H["n_jobs"],
        )),
    ])
    pipe.fit(tr[spalten], y_tr)
    p = pipe.predict_proba(te[spalten])[:, 1]
    return roc_auc_score(y_te, p), accuracy_score(y_te, (p >= 0.5).astype(int)), p


auc_sauber, acc_sauber, p_sauber = trainiere(features)
auc_leak, acc_leak, _ = trainiere(features + ["future_return_20d"])

print(f"Zeilen: {len(tr):,} Training / {len(te):,} Test")
pd.DataFrame([
    ["sauber — 8 Features", auc_sauber, acc_sauber],
    ["mit future_return_20d — LEAKAGE", auc_leak, acc_leak],
], columns=["Variante", "ROC-AUC", "Accuracy"]).style.format({
    "ROC-AUC": "{:.4f}", "Accuracy": "{:.4f}",
}).hide(axis="index")
"""),
        md("""
### Was hier gerade passiert ist

Ein einziges zusätzliches Feature hebt die ROC-AUC von **0,52 auf 0,9999** und
die Accuracy auf praktisch 100 %. Ein solches Ergebnis in einer Präsentation
sähe nach einem Durchbruch aus. Tatsächlich hat das Modell nichts gelernt: Es
liest das Vorzeichen einer Zahl ab, die zum Prognosezeitpunkt **noch nicht
existiert**.

Der entscheidende Punkt ist, dass keine Kennzahl diesen Fehler anzeigt. Accuracy,
AUC, Precision, Recall, die Konfusionsmatrix — alle sind exzellent und alle sind
korrekt berechnet. Auffliegen kann Leakage nur durch die inhaltliche Frage:

> Läge dieses Feature zum Entscheidungszeitpunkt tatsächlich vor?

**Merkregel für künftige Projekte:** Ein unerwartet gutes Ergebnis ist ein
Anlass zur Prüfung, nicht zur Freude. In diesem Projekt war die Reihenfolge
umgekehrt — das schwache Ergebnis war das glaubwürdige.
"""),
        md("""
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
        md("""
## Der Schwellenwert ist eine Geschäftsentscheidung

Die 0,5-Grenze zwischen „bullish" und „bearish" ist eine Konvention, keine
Eigenschaft des Modells. Verschiebt man sie, tauscht man Precision gegen Recall
— und dieser Tausch ist eine fachliche Entscheidung, keine statistische.

Für einen Anleger übersetzt sich das so:

- **Precision** — wenn das Modell „steigt" sagt: Wie oft stimmt das?
- **Recall** — von allen tatsächlich steigenden Phasen: Wie viele erkennt es?
- **Signalanteil** — an wie vielen Tagen wird überhaupt ein Kaufsignal erzeugt?
"""),
        code("""
from sklearn.metrics import precision_recall_fscore_support

basisrate = y_te.mean()
zeilen = []
for schwelle in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
    vorhersage = (p_sauber >= schwelle).astype(int)
    if vorhersage.sum() == 0:
        zeilen.append({"Schwelle": schwelle, "Precision": np.nan, "Recall": 0.0,
                       "F1": 0.0, "Signalanteil": 0.0, "Vorsprung vor Basisrate": np.nan})
        continue
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_te, vorhersage, average="binary", zero_division=0
    )
    zeilen.append({
        "Schwelle": schwelle, "Precision": pr, "Recall": rc, "F1": f1,
        "Signalanteil": vorhersage.mean(), "Vorsprung vor Basisrate": pr - basisrate,
    })

print(f"Basisrate im Testfenster (Anteil steigend): {basisrate:.4f}")
pd.DataFrame(zeilen).style.format({
    "Schwelle": "{:.2f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}",
    "Signalanteil": "{:.3f}", "Vorsprung vor Basisrate": "{:+.4f}",
}).hide(axis="index")
"""),
        md("""
### Was die Tabelle für einen Anleger bedeutet

Bei der Standardschwelle 0,5 liegt die Precision bei rund **0,605** — gegenüber
einer Basisrate von **0,593**. Der Vorsprung beträgt gut **einen Prozentpunkt**.
Man könnte also blind „steigt" sagen und wäre fast genauso treffsicher.

Dreht man die Schwelle auf 0,55 hoch, steigt die Precision auf etwa **0,658** —
das klingt zunächst nach einer Verbesserung. Der Preis: Der Recall bricht auf
rund **0,11** ein, und es entsteht nur noch an gut **10 %** der Tage überhaupt
ein Signal. Man verzichtet auf neun von zehn Gelegenheiten, um einen Vorsprung
von rund sechs Prozentpunkten gegenüber blindem Raten zu erkaufen — vor
Transaktionskosten, Slippage und Steuern.

**Kein Schwellenwert rettet ein Modell, dessen Ranking-Signal fehlt.** Die
Schwellenwertoptimierung ist ein legitimes Werkzeug, aber sie erzeugt keine
Information, die in den Features nicht steckt. Genau das ist der Grund, warum
dieses Projekt keine Handelsempfehlung ausspricht.
"""),
        checkpoint("""
Drei belastbare Aussagen aus der Modellierungsphase:

1. **Die Baseline schlägt das Modell bei der Accuracy.** Wer nur Accuracy
   berichtet, berichtet ein schlechteres Ergebnis als „immer steigt sagen".
2. **Leakage ist nicht an Kennzahlen erkennbar.** Ein zusätzliches Feature hebt
   die AUC auf 0,9999, ohne dass eine einzige Metrik protestiert. Die einzige
   Verteidigung ist die Frage nach dem Prognosezeitpunkt — organisatorisch, nicht
   technisch.
3. **Weder Kapazität noch Schwellenwert heben das Signal.** Sieben
   Regularisierungsstufen bewegen den Test-AUC um 0,007; die
   Schwellenwertverschiebung tauscht nur Precision gegen Recall.

Empfehlung: Das Modell taugt als Lehr- und Analyseinstrument, nicht als
Signalgeber. Der nächste sinnvolle Schritt wären exogene Variablen — nicht ein
größeres Modell auf denselben Features.
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
        lernziele(
            "eine Punktschätzung durch ein Konfidenzintervall ergänzen",
            "statistische Signifikanz von praktischer Bedeutung trennen",
            "ein Negativergebnis gegen seine Gegenerklärungen absichern",
            "die Grenzen der eigenen Arbeit präzise statt pauschal benennen",
        ),
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
"""),
        md("""
## Wie sicher ist die 0,519 überhaupt?

Eine Punktschätzung ohne Streuungsmaß ist unvollständig. `0,519` könnte ein
stabiler kleiner Vorsprung sein — oder Zufall. Ein **Bootstrap** beantwortet
das: Aus dem Testfenster werden wiederholt Stichproben mit Zurücklegen gezogen
und jedes Mal die AUC neu berechnet. Die Streuung dieser Werte schätzt die
Unsicherheit der Kennzahl.
"""),
        code("""
import joblib
from sklearn.metrics import roc_auc_score

features = diagnostics["features"]
V = diagnostics["validation"]

modelldaten = df.dropna(subset=features + ["target_20d", "date"]).sort_values(["date", "ticker"])
test = modelldaten[modelldaten["date"] >= pd.Timestamp(V["test_start"])]

modell = joblib.load(PROJECT_ROOT / "models" / "wealthscope_model.joblib")
p = modell.predict_proba(test[features])[:, 1]
y = test["target_20d"].astype(int).to_numpy()

auc_punkt = roc_auc_score(y, p)
print(f"AUC aus dem Artefakt : {m['roc_auc']:.6f}")
print(f"AUC hier reproduziert: {auc_punkt:.6f}")

rng = np.random.default_rng(42)
B, n = 400, len(y)
werte = []
for _ in range(B):
    idx = rng.integers(0, n, n)
    if y[idx].min() == y[idx].max():
        continue
    werte.append(roc_auc_score(y[idx], p[idx]))
werte = np.array(werte)
lo, hi = np.percentile(werte, [2.5, 97.5])

print(f"\\nBootstrap mit B={len(werte)} Ziehungen über {n:,} Testfälle")
print(f"95 %-Konfidenzintervall: [{lo:.4f}, {hi:.4f}]")
print(f"Anteil der Ziehungen mit AUC <= 0,5: {(werte <= 0.5).mean():.4f}")
"""),
        code("""
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(werte, bins=40, color="#1F5A4E", alpha=.85)
ax.axvline(0.5, color="#B54D3A", linestyle="--", linewidth=1.5, label="Zufallsniveau 0,5")
ax.axvline(lo, color="#B98519", linestyle=":", linewidth=1.5, label="95 %-KI")
ax.axvline(hi, color="#B98519", linestyle=":", linewidth=1.5)
ax.set_title("Bootstrap-Verteilung der Test-ROC-AUC")
ax.set_xlabel("ROC-AUC")
ax.set_ylabel("Häufigkeit")
ax.legend()
plt.tight_layout()
plt.show()
"""),
        md("""
### Signifikant ja, bedeutsam nein

Das Konfidenzintervall liegt bei etwa **[0,515; 0,523]** und schließt das
Zufallsniveau 0,5 aus. Statistisch ist der Vorsprung also **nachweisbar** — bei
69.147 Testfällen wird auch ein winziger Effekt signifikant.

Genau hier liegt die Falle, vor der Notebook 01 gewarnt hat. Ein naives „AUC > 0,5,
also H1 bestätigt" wäre formal vertretbar und inhaltlich falsch. Denn:

- Der Vorsprung beträgt rund **2 AUC-Punkte** gegenüber Münzwurf.
- Er ist über die Walk-forward-Folds **nicht stabil** (0,514 im Mittel, eine
  Standardabweichung von 0,014 — einzelne Folds liegen unter 0,5).
- Er reicht nach Notebook 04 nicht für einen Precision-Vorsprung, der
  Transaktionskosten überstehen würde.

**Statistische Signifikanz ist nicht dasselbe wie praktische Bedeutung.** H1
verlangte einen Vorsprung, der *deutlich* und *stabil* ist. Beides ist nicht
erfüllt — deshalb gilt die Hypothese als falsifiziert, obwohl das Intervall die
0,5 ausschließt. Diese Unterscheidung sauber zu treffen, ist das eigentliche
Ergebnis der Evaluationsphase.
"""),
        md("""
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
        checkpoint("""
**Entscheidungsvorlage in fünf Sätzen.**

Ein Modell auf rein kursbasierten technischen Indikatoren erreicht im ehrlichen
Out-of-Time-Test eine ROC-AUC von 0,519 mit einem 95 %-Konfidenzintervall von
etwa [0,515; 0,523]. Der Vorsprung vor dem Zufall ist damit statistisch
nachweisbar, aber zu klein und zeitlich zu instabil, um Transaktionskosten zu
überstehen — H1 ist falsifiziert. Die beiden naheliegenden Ausreden sind
ausgeschlossen: Weder mehr Modellkapazität (Test-AUC-Spanne 0,007 über sieben
Stufen) noch mehr Daten (−0,005 über die 2,3-fache Trainingsmenge) heben das
Ergebnis. Dass andere Arbeiten hier bessere Zahlen berichten, lässt sich
beziffern: Ein naiver Zufalls-Split hätte auf denselben Daten 0,581 ergeben, ein
4,2-mal größeres scheinbares Signal.

**Empfehlung:** Kein Einsatz als Signalgeber. Investition in exogene Datenquellen
statt in größere Modelle auf denselben Features. Der Wert dieser Arbeit liegt in
der Messmethodik, die jedes Folgeprojekt übernehmen sollte — insbesondere die
Regel, jedes Feature am Prognosezeitpunkt zu prüfen.
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
        lernziele(
            "erklären, warum ein negatives Ergebnis besonders sorgfältig kommuniziert werden muss",
            "die Trennung von Rechnen, Visualisieren und Sprache-Erzeugen benennen",
            "prüfen, ob eine App ihre eigenen Grenzen sichtbar macht",
        ),
        md("""
### Die schwierigste Kommunikationsaufgabe

Ein Prototyp, der ein starkes Signal zeigt, verkauft sich von selbst. Dieser
zeigt keines — und muss trotzdem verständlich machen, warum das die richtige
Antwort ist. Daraus folgen drei Gestaltungsentscheidungen der App:

1. **Die Baseline steht neben jeder Kennzahl.** „51,9 %" allein wäre irreführend,
   „51,9 % gegen eine Baseline von 59,1 %" ist eine Aussage.
2. **Kein Wert ist im Code hartkodiert.** Alle Seiten lesen `models/*.json`.
   Ändert sich das Modell, ändert sich die App — ein Widerspruch zwischen
   Dokumentation und Anzeige kann nicht entstehen.
3. **Der Assistent ist vom Prognosemodell getrennt.** Er erklärt den
   berechneten Kontext, er erzeugt keine Vorhersage. Diese Trennung ist die
   Voraussetzung dafür, dass „KI" im Projekt kein Sammelbegriff bleibt.
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
        checkpoint("""
Wissenstransfer ist bei einem Negativergebnis keine Kür, sondern die eigentliche
Prüfung: Die App muss ein schwaches Modell erklären, ohne es zu beschönigen und
ohne es zu verstecken. Der belastbare Nachweis dafür ist technisch, nicht
rhetorisch — **kein einziger Kennwert ist in `src/` hartkodiert**. Was die App
anzeigt, steht so in den Artefakten, aus denen auch Ausarbeitung, Poster und
diese Notebooks lesen.
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
        lernziele(
            "generative KI von prädiktivem ML sauber trennen",
            "Leitplanken für einen erklärenden Assistenten formulieren",
            "einen Export so gestalten, dass er ohne die App interpretierbar bleibt",
        ),
        md("""
### Warum die Trennung nicht kosmetisch ist

Im Projekt kommen zwei völlig verschiedene Dinge vor, die beide „KI" heißen:

| | Random Forest | Gemini-Assistent |
|---|---|---|
| Aufgabe | Wahrscheinlichkeit schätzen | Kontext in Sprache fassen |
| Trainiert auf | 120.920 Kursbeobachtungen bis 2006 | fremdem Textkorpus |
| Prüfbar durch | Out-of-Time-Test, ROC-AUC | keine projektinterne Metrik |
| Fehlermodus | schwaches Signal | plausibel klingende Falschaussage |
| Im Ergebnis der Arbeit | ja, er *ist* das Ergebnis | nein, reine Bedienhilfe |

Würde der Assistent Prognosen formulieren, wären sie durch **nichts** in diesem
Projekt gedeckt — die gesamte Validierungsarbeit bezieht sich ausschließlich auf
den Random Forest. Die Trennung ist deshalb eine methodische Notwendigkeit.
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
    "app_version": "1.0.5",
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
        checkpoint("""
Der Assistent darf rechnen und formulieren; verantworten muss die Aussage der
Mensch, der sie fachlich begründen kann. Genau deshalb nennt der Export-Vertrag
oben `financial_advice: False` und verweist auf `models/diagnostics.json` als
Quelle der Kennzahlen: Ein exportierter Bericht muss auch dann noch korrekt
interpretierbar sein, wenn niemand mehr weiß, in welcher App er entstanden ist.
"""),
    ],
)

