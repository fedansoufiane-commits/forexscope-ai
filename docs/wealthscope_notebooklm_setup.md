# WealthScope AI — Architektur & Design-Entscheidungen

> Diese Datei ersetzt eine frühere, sehr ausführliche Entwicklungs-Historie des
> alten `app_max.py`-Monolithen (Routing per Query-Parameter, `route_link()`,
> manuelle Sidebar-Funktionen etc.). Diese Architektur wurde am 2026-07-09
> komplett neu aufgebaut; der historische Stand ist bei Bedarf über Commit
> `834ff98` erreichbar. Dieses Dokument beschreibt nur noch den
> **aktuellen** Stand.

## Projektstruktur

```
app.py                    Entrypoint: Theme-CSS, Sidebar, st.navigation-Routing
src/
  config.py               Pfade, Konstanten, Ticker-/Feature-Listen
  state.py                 Session-State-Defaults
  context.py                Baut den Analyse-Kontext pro Rerun
  sidebar.py                 Globale Sidebar-Controls
  theme.py                    Design-Tokens (Hell/Dunkel) + Plotly-Theme
  icons.py                     Eigenes Linien-Icon-Set (kein Emoji)
  ui.py                         Card/Badge/KPI-Grid-Bausteine
  data.py                        Laden & Feature Engineering
  model.py                        Scoring-Engine + Modell-Loader
  diagnostics.py                  Korrelation, Konfusionsmatrix, ROC/PR, Lernkurve, SHAP
  charts.py                        Kurs-/Risiko-/Portfolio-Charts
  news.py                           NewsAPI + Sentiment + Gemini
  export.py                         Markdown/CSV/ZIP/PDF-Export
  pages/                            Eine Datei pro Seite
scripts/train_and_diagnose.py     Training + Diagnostik-Cache (offline, reproduzierbar)
```

## Warum `st.navigation` statt eigenem Routing?

Der Vorgänger baute Routing manuell über `st.query_params` + eine
`route_page()`-Dispatch-Funktion, weil Streamlits native Multipage-API zu
diesem Zeitpunkt nur dateibasierte `pages/`-Ordner unterstützte. Seit
Streamlit 1.36 akzeptiert `st.Page()` auch Funktionen direkt (nicht nur
Dateipfade) — dadurch lässt sich jede Seite als eigenes Modul in
`src/pages/` schreiben und trotzdem die native Navigation, URL-Synchronisation
und Sidebar-Struktur von Streamlit nutzen, ohne eigenes Routing zu pflegen.

**Stolperfalle (in dieser Session live erlebt):** `st.Page(fn, ...)` leitet den
URL-Pfad standardmäßig vom Funktionsnamen ab. Da mehrere Seiten-Module ihre
Hauptfunktion `render()` nennen, kollidierten die URL-Pfade und die App
crashte beim Start (`StreamlitAPIException: Multiple Pages specified with URL
pathname render`). Fix: jedes `st.Page(...)` bekommt einen expliziten
`url_path=`. `tests/test_app_static.py::test_app_registers_unique_url_paths`
verhindert ein Regressions-Comeback.

## Session-State-Schlüssel (aktuell, siehe `src/state.py`)

| Key | Zweck |
|---|---|
| `theme_mode` | „Hell" / „Dunkel" |
| `app_mode` | „Geführte Ansicht" / „Expertenansicht" |
| `ticker` | aktuell gewählter Ticker |
| `period` | Zeitraum-Filter |
| `use_live_data` | yfinance statt Kaggle-Parquet |
| `asset_weight` | Positionsgröße (%) für den Scoring-Kontext |
| `enable_news` | NewsAPI-Sentiment ein/aus |

## Theme-Architektur

Zwei-Schichten-Modell: `styles/base.css` enthält nur strukturelle Regeln gegen
CSS-Variablen (`var(--ws-*)`); `src/theme.py` setzt die konkreten Werte pro
Modus (`THEMES["Hell"]` / `THEMES["Dunkel"]`) und liefert das passende
Plotly-Layout (`apply_chart_theme()`), damit Charts und HTML-Karten nie
auseinanderlaufen.

**Reihenfolge ist wichtig:** Die Sidebar-Widgets (inkl. Hell/Dunkel-Umschalter)
müssen **vor** `inject_theme_vars()` gerendert werden, sonst liest die
Theme-Injection den `session_state`-Wert von *vor* dem Klick — das Theme hinkt
dann einen Rerun hinterher (ebenfalls ein realer Bug dieser Session, siehe
`app.py`: `render_global_sidebar()` läuft vor `inject_theme_vars()`).

## Icons statt Emoji

Karten, Badges und Score-Zeilen nutzen `src/icons.py` (handgezeichnete
Linien-Icons, 24×24, `stroke="currentColor"`) statt Emoji — konsistenter, in
beiden Farbmodi gut lesbar, und lässt sich in Akzentfarbe einfärben. Die
Sidebar-Navigation nutzt Streamlits native `:material/…:`-Icon-Syntax in
`st.Page(icon=...)`.

## Diagnostik-Caching

`learning_curve()` aus scikit-learn fittet für diese App ~40 RandomForest-Modelle
(8 Trainingsgrößen × 5 CV-Folds). Das im laufenden Streamlit-Rerun zu berechnen
wäre inakzeptabel langsam. Deshalb: `scripts/train_and_diagnose.py` läuft
einmalig offline und schreibt `models/diagnostics.json` +
`models/learning_curve.json`; die App liest nur noch JSON.
