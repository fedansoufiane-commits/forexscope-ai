# WealthScope AI – Projektsetup, Arbeitsweise und Entwicklungsdokumentation

## 1. Projektüberblick

WealthScope AI ist eine lokale Streamlit-Webanwendung zur datenbasierten Kapitalanalyse. Die Anwendung kombiniert historische Marktinformationen, technische Kennzahlen, Nachrichtenbewertung, Portfolio-Simulation und erklärbare Visualisierungen.

Ziel ist nicht, eine echte Finanzberatung zu ersetzen, sondern eine verständliche, transparente und wissenschaftlich nachvollziehbare Demo-Anwendung zu bauen.

Der zentrale Anspruch des Projekts lautet:

> Kapital verstehen, Risiken prüfen und Entscheidungen simulieren.

Die App richtet sich an Menschen, die neues Kapital verantwortungsvoll einordnen möchten. Beispiele sind Erbe, Abfindung, Rücklagenaufbau oder eine erste strukturierte Auseinandersetzung mit ETFs und Aktien.

---

## 2. Lokales Setup

Das Projekt liegt lokal im Ordner:

`~/UNI/forexscope-ai`

Die App wird über das Terminal gestartet mit:

`cd ~/UNI/forexscope-ai`

`python3 -m streamlit run app.py`

Danach ist die Anwendung lokal erreichbar unter:

`http://localhost:8501`

Falls Port 8501 bereits belegt ist, kann Streamlit automatisch einen anderen Port verwenden, zum Beispiel 8502.

---

## 3. Projektstruktur

Aktuelle Struktur:

forexscope-ai/
- app.py
- components/
  - __init__.py
  - charts.py
- styles/
  - __init__.py
  - theme.py
  - wealthscope.css
- data/
  - processed/
    - wealthscope_market_dataset.csv
    - wealthscope_features.csv
- requirements.txt
- .gitignore
- docs/

Die Datei `app.py` enthält die Hauptlogik der Anwendung. Dort werden Seiten, Daten, Nutzerinteraktionen und zentrale App-Abläufe gesteuert.

Die Datei `styles/theme.py` enthält dynamische Theme-Logik für Dark Mode, Light Mode und Plotly-Diagramme.

Die Datei `styles/wealthscope.css` enthält statisches CSS-Design für Sidebar, Karten, Bottom-Bar, Tabellen, Abstände, Rundungen und visuelle Harmonie.

Die Datei `components/charts.py` enthält wiederverwendbare Chart-Komponenten, insbesondere für interaktive Plotly-Diagramme mit Datentabelle, CSV-Export und Auswahlfunktion.

---

## 4. Genutzte Technologien

### Python

Python ist die zentrale Programmiersprache des Projekts. Alle Datenverarbeitungsschritte, ML-Logik, UI-Steuerung und Visualisierungen werden aus Python heraus gesteuert.

### Streamlit

Streamlit dient als Web-App-Framework. Die App nutzt Streamlit für Sidebar, Inputs, Buttons, Expander, Tabellen, Downloads und Diagrammeinbindung.

Wichtig: Streamlit führt die Datei bei Interaktionen neu aus. Deshalb mussten UI-Zustände wie Theme, aktuelle Seite und Ansicht über `st.session_state` stabil gehalten werden.

### Session State

Für Zustände wie aktueller Modus, aktuelle Seite, gewähltes Asset, Zeitraum und Ansicht wird `st.session_state` verwendet.

Wichtige Werte:

- `st.session_state["theme_mode"]`
- `st.session_state["app_mode"]`
- `st.session_state["current_page"]`
- `st.session_state["selected_asset"]`
- `st.session_state["ticker"]`
- `st.session_state["period"]`
- `st.session_state["interval"]`

### Query Params

Die Anwendung nutzt URL-Parameter wie:

`?page=Start&theme=Light+Mode&view=Geführte+Ansicht`

Dadurch können Seite, Theme und Ansicht auch über Links in der Bottom-Bar erhalten bleiben.

### Plotly

Plotly wird für interaktive Diagramme genutzt. Die Diagramme unterstützen Hover-Informationen, Auswahlverhalten und Exportmöglichkeiten.

### Git und GitHub

Das Projekt wird mit Git versioniert.

Typische Befehle:

- `git status --short`
- `git add app.py styles/__init__.py styles/theme.py styles/wealthscope.css components/__init__.py components/charts.py`
- `git commit -m "Refactor WealthScope design and chart components"`
- `git push`

---

## 5. Datenbasis

Die ursprüngliche Big-Data-Basis stammt aus dem Kaggle-Dataset:

`borismarjanovic/price-volume-data-for-all-us-stocks-etfs`

Lokal wurden viele `.txt`-Dateien gezählt. Die Rohdatenbasis umfasst lokal über 34 Millionen Datenzeilen. Damit ist die Big-Data-Anforderung deutlich erfüllt.

Für die App wurde eine kontrollierte, verarbeitete Teilmenge erstellt:

`data/processed/wealthscope_market_dataset.csv`

Diese Datei umfasst ungefähr:

- 96.017 Zeilen
- 9 Spalten
- 18 Ticker
- ca. 6,8 MB

Zusätzlich wurde eine Feature-Datei erstellt:

`data/processed/wealthscope_features.csv`

Diese Datei umfasst ungefähr:

- 92.075 Zeilen
- 23 Spalten
- 18 Ticker
- ca. 27 MB

Wichtige Feature-Spalten:

- daily_return
- return_5d
- return_20d
- ma_20
- ma_50
- ma_200
- ma_20_distance
- ma_50_distance
- ma_200_distance
- volatility_20d
- rolling_high
- drawdown
- future_return_20d
- target_20d

Die Zielvariable lautet:

`target_20d`

Sie beschreibt, ob der Preis nach 20 Handelstagen höher liegt.

---

## 6. Machine-Learning-Ansatz

Das Projekt nutzt einen ML-Prototypen, unter anderem mit einem Random-Forest-Klassifikator.

Ziel des Modells ist nicht eine sichere Finanzprognose, sondern eine beispielhafte Bewertung zukünftiger Markttendenzen auf Basis historischer Kennzahlen.

Die Modellgüte wurde ungefähr mit einer Accuracy von rund 53 % eingeordnet. Deshalb wird das Modell bewusst als Prototyp und nicht als verlässliches Prognosesystem dargestellt.

Die App kommuniziert transparent:

- Keine Finanzberatung.
- Keine sichere Prognose.
- Analyse, Lernen und Simulation.

---

## 7. Seitenstruktur der App

### Hauptseiten

- Start
- Wealth Outlook
- Kapital-Kompass
- Portfolio-Simulator
- Watchlist-Vergleich

Diese Seiten befinden sich in der linken Sidebar.

### Service-Seiten

- News-Archiv
- So funktioniert's
- Projekt
- Export
- Impressum
- Datenschutz
- Betriebsstatus

Diese Seiten befinden sich in der festen Bottom-Bar.

Dadurch ist die linke Sidebar nicht überladen. Die Hauptnavigation bleibt fokussiert, während rechtliche, erklärende und technische Seiten unten erreichbar sind.

---

## 8. Navigation und Routing

Die Anwendung verwendet keine klassische Streamlit-Multipage-Struktur, sondern eine eigene Routing-Logik über Query-Parameter.

Beispiel:

`?page=Betriebsstatus&theme=Dark+Mode&view=Geführte+Ansicht`

Dafür wurde eine Funktion `route_link(page_name)` gebaut. Sie erzeugt Links, die den aktuellen Theme- und View-State mitnehmen.

Das war notwendig, weil die Bottom-Bar sonst beim Seitenwechsel den Light Mode oder Dark Mode verlieren konnte.

---

## 9. Dark Mode und Light Mode

Ein wichtiger Entwicklungspunkt war die Umschaltung zwischen Dark Mode und Light Mode.

Anfangs gab es mehrere Probleme:

- Design war nicht konstant.
- Bottom-Bar verlor teilweise den Modus.
- Einige Seiten blieben dunkel trotz Light Mode.
- Diagramme und Tabellen hatten falsche Hintergründe.

Die Lösung war eine zentrale Theme-Architektur:

- `styles/theme.py`
- `styles/wealthscope.css`

In `styles/theme.py` werden dynamische Werte wie Hintergrund, Textfarbe, Border, Card-Farbe und Chart-Theme abhängig vom aktiven Modus gesetzt.

In `styles/wealthscope.css` liegen statische Layout- und Designregeln.

Diese Trennung war ein wichtiger Refactoring-Schritt.

---

## 10. Sidebar-Design

Die Sidebar wurde mehrfach überarbeitet.

Ziel war eine harmonische, nicht überladene linke Navigation.

Finale Idee:

- oben: WealthScope AI als klickbares Branding
- darunter: kurzer Claim
- danach: Darstellung Dark/Light
- danach: Ansicht Geführt/Experte
- danach: Hauptnavigation

Der Webseitname soll anklickbar sein und zur Startseite führen.

Das Design soll nicht kantig wirken, sondern weich und eingebettet:

- runde Cards
- Glassmorphism-Effekt
- dezente Borders
- harmonischer Hintergrund
- klare Lesbarkeit

---

## 11. Analyse-Einstellungen

Die Analyse-Einstellungen wurden bewusst aus der Sidebar in den Hauptbereich verschoben. Dadurch bleibt die Sidebar schlank.

Die Analyse-Einstellungen enthalten:

- Asset auswählen
- Optional eigener Ticker
- Zeitraum
- Intervall

Die Werte werden in `st.session_state` gespeichert, damit sie auf allen Seiten verfügbar sind.

---

## 12. Bottom-Bar

Die Bottom-Bar ist eine feste Leiste am unteren Bildschirmrand. Sie orientiert sich visuell an modernen Web-Apps und an einer Status-/Service-Navigation.

Sie enthält:

- News-Archiv
- So funktioniert's
- Projekt
- Export
- Impressum
- Datenschutz
- Betriebsstatus
- Marktdaten: Aktuell
- News: Mittel
- Check: Uhrzeit

Die Bottom-Bar ist fest positioniert und bleibt beim Scrollen sichtbar.

Sie ist nicht für Hauptnavigation gedacht, sondern für Service-, Transparenz- und Statusseiten.

---

## 13. Betriebsstatus

Die Betriebsstatus-Seite ist eine technische Qualitätsseite. Sie zeigt, ob wichtige Komponenten der App verfügbar sind.

Beispiele:

- Systemstatus
- Marktdaten
- News
- ML-Modell
- Performance
- Komponentenstatus
- Erreichbarkeit
- Präsentationsreife

Dadurch wirkt das Projekt professioneller, weil technische Stabilität sichtbar gemacht wird.

Diese Seite ist besonders nützlich für die Projektbewertung, weil sie zeigt, dass nicht nur Fachlogik, sondern auch Betriebsfähigkeit betrachtet wurde.

---

## 14. Interaktive Diagramme

Ein wichtiger wissenschaftlicher Verbesserungsschritt war die Umstellung der Diagramme auf interaktive Chart-Komponenten.

Dafür wurde die Datei erstellt:

`components/charts.py`

Dort liegt die Funktion:

`render_interactive_chart(...)`

Diese Funktion soll:

- Diagramm anzeigen
- Hover-Informationen ermöglichen
- Datenpunkte auswählbar machen
- zugrunde liegende Datentabelle anzeigen
- CSV-Download anbieten
- ausgewählte Punkte anzeigen

---

## 15. Wissenschaftlicher Mehrwert der Diagramme

Die Diagramme sollen nicht nur dekorativ sein. Sie sollen prüfbar und nachvollziehbar sein.

Geplante beziehungsweise umgesetzte wissenschaftliche Funktionen:

- Hover zeigt konkrete Werte.
- Datenpunkte können angeklickt oder ausgewählt werden.
- Datenbasis kann eingeblendet werden.
- CSV-Export der Chart-Daten.
- Visualisierung und Datentabelle passen zusammen.

Damit kann in der Präsentation argumentiert werden:

> Die Visualisierungen sind nicht nur grafische Elemente, sondern überprüfbare Analysebausteine. Zu zentralen Diagrammen kann die zugrunde liegende Datentabelle eingeblendet und exportiert werden.

Das stärkt Transparenz, Reproduzierbarkeit und Nachvollziehbarkeit.

---

## 16. QUA3CK-Bezug

Das Projekt lässt sich gut in die QUA3CK-Struktur einordnen.

### Q – Question

Zentrale Frage:

Wie können historische Marktdaten, technische Kennzahlen und Nachrichten genutzt werden, um Kapitalentscheidungen verständlicher zu machen?

### U – Understanding Data

Die Datenbasis wurde untersucht, bereinigt und auf eine kontrollierte Teilmenge reduziert. Dabei wurden Ticker, Asset-Typen, Zeiträume und Feature-Strukturen betrachtet.

### A³ – Algorithm, Adaption, Adjustment

Es wurden Features erstellt, ein ML-Modell trainiert und eine App-Struktur aufgebaut. Dazu gehören technische Indikatoren, Renditen, Volatilität und Zielvariable `target_20d`.

### C – Conclude & Compare

Die Ergebnisse werden über Scores, Diagramme, Watchlist-Vergleich, Portfolio-Simulation und Betriebsstatus eingeordnet.

### K – Knowledge Transfer

Die App macht technische und finanzielle Zusammenhänge für Nicht-Experten verständlich. Deshalb gibt es eine geführte Ansicht und eine Expertenansicht.

---

## 17. Geführte Ansicht und Expertenansicht

Die App besitzt zwei Modi:

- Geführte Ansicht
- Expertenansicht

Die geführte Ansicht richtet sich an nicht-technische Nutzer. Sie erklärt Ergebnisse einfacher und stärker handlungsorientiert.

Die Expertenansicht richtet sich an Nutzer mit mehr Vorwissen. Sie zeigt mehr Kennzahlen, Modellwerte, technische Indikatoren und Score-Zerlegungen.

Der Modus wird über `st.session_state["app_mode"]` gespeichert und über die Sidebar gesteuert.

---

## 18. Typische Entwicklungsprobleme und Lösungen

### NameError

Beispiele:

- name `route_link` is not defined
- name `ticker` is not defined
- name `asset_type` is not defined
- name `app_mode` is not defined

Ursache war fast immer, dass Variablen oder Funktionen verwendet wurden, bevor sie definiert wurden.

Lösung:

- zentrale State-Initialisierung
- globaler App-Kontext
- korrekte Reihenfolge im Code
- Funktionen vor Verwendung definieren

### SyntaxError in f-Strings

Ein Fehler entstand durch Backslashes in f-String-Ausdrücken.

Lösung:

- Links vorab in Variablen speichern
- keine komplexen `route_link(...)`-Ausdrücke mit Escapes direkt in HTML-f-Strings

### Inkonsistenter Light Mode

Ein Problem war, dass einige Seiten oder die Bottom-Bar den aktiven Light Mode nicht übernommen haben.

Lösung:

- Theme in `st.session_state` speichern
- Theme in `route_link` mitgeben
- CSS zentral über `theme.py` anwenden
- finaler Runtime-Theme-Override

### Doppelte Sidebar-Inhalte

Die Sidebar enthielt zeitweise doppelte Branding-Blöcke.

Lösung:

- `render_sidebar()` zentral neu bauen
- alte `st.sidebar.title`-/`st.sidebar.markdown`-Blöcke entfernen
- Sidebar-CSS auslagern

---

## 19. Refactoring-Entscheidungen

Vorher:

`app.py` enthielt App-Logik, CSS, Theme, Charts und viele UI-Patches zusammen.

Nachher:

- `app.py` = App-Logik und Seiten
- `styles/theme.py` = dynamisches Theme und Chart-Theme
- `styles/wealthscope.css` = statisches Design
- `components/charts.py` = wiederverwendbare Chart-Komponenten

Das ist eine klare Verbesserung der Wartbarkeit.

---

## 20. Aktueller Entwicklungsstand

Aktuell funktioniert:

- lokaler App-Start
- Dark/Light Mode grundsätzlich
- Sidebar-Routing teilweise
- Bottom-Bar mit Service-Seiten
- Betriebsstatus-Seite
- News-Archiv grundsätzlich
- externe CSS-/Theme-Struktur
- interaktive Chart-Komponente
- Git-Push auf GitHub

Noch offen bzw. weiter zu verbessern:

- Sidebar harmonischer finalisieren
- doppelte alte CSS-Regeln bereinigen
- alle Diagramme vollständig mit DataFrames verbinden
- Light Mode bei allen Tabellen und Charts konsistent halten
- Betriebsstatus weiter ausbauen
- Export-Seite finalisieren
- Impressum/Datenschutz mit finalen Platzhaltern prüfen
- Screenshots für Professor-Export integrieren

---

## 21. Kurzfassung für Präsentation

WealthScope AI ist eine lokale Streamlit-App zur verständlichen Kapitalanalyse. Die App kombiniert historische Aktien- und ETF-Daten, technische Kennzahlen, Nachrichtenbewertung, ML-Prototyping und Portfolio-Simulation. Ziel ist keine Finanzberatung, sondern eine transparente Lern- und Analyseumgebung.

Die App wurde schrittweise verbessert: Zunächst wurden Datenbasis und Features aufgebaut, danach ML- und Analysefunktionen integriert. Anschließend wurde das Design refaktoriert, sodass Theme, CSS und Chart-Komponenten ausgelagert sind. Besonderer Fokus lag auf Dark-/Light-Mode, Routing, Betriebsstatus, Bottom-Bar und interaktiven Diagrammen.

Der wissenschaftliche Mehrwert liegt darin, dass Visualisierungen nicht nur dargestellt, sondern prüfbar gemacht werden. Zu zentralen Diagrammen können die zugrunde liegenden Daten eingeblendet und exportiert werden. Dadurch werden Transparenz, Reproduzierbarkeit und Nachvollziehbarkeit gestärkt.

---

## 22. Geeignete NotebookLM-Fragen

- Fasse mir das Projekt WealthScope AI in 5 Sätzen zusammen.
- Welche technischen Entscheidungen wurden im Projekt getroffen?
- Welche Probleme traten während der Entwicklung auf und wie wurden sie gelöst?
- Wie hängt das Projekt mit QUA3CK zusammen?
- Warum sind interaktive Diagramme wissenschaftlich sinnvoll?
- Welche Architektur hat die App aktuell?
- Was ist der Unterschied zwischen Geführter Ansicht und Expertenansicht?
- Welche Punkte sollte ich in der Präsentation hervorheben?
- Welche Schwächen sind noch offen?
- Formuliere mir daraus einen Präsentationstext für 5 Minuten.
