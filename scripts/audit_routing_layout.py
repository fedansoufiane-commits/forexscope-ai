from pathlib import Path
import ast
import re
import sys

APP = Path("app_max.py")

if not APP.exists():
    print("❌ app_max.py nicht gefunden.")
    sys.exit(1)

text = APP.read_text(encoding="utf-8")

print("1. Syntax prüfen ...")
try:
    tree = ast.parse(text)
    print("   ✅ Syntax OK")
except SyntaxError as e:
    print(f"   ❌ SyntaxError: line {e.lineno}: {e.msg}")
    if e.lineno:
        lines = text.splitlines()
        start = max(1, e.lineno - 5)
        end = min(len(lines), e.lineno + 5)
        for no in range(start, end + 1):
            marker = ">>" if no == e.lineno else "  "
            print(f"{marker} {no:4d}: {lines[no-1]}")
    sys.exit(1)

print("\n2. Funktionen sammeln ...")
funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
page_funcs = sorted(f for f in funcs if f.startswith("page_"))
print(f"   Gefundene page_* Funktionen: {len(page_funcs)}")
for f in page_funcs:
    print(f"   - {f}")

def literal_list(name):
    pattern = rf"{name}\s*=\s*(\[.*?\])"
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception as e:
        print(f"   ❌ {name} konnte nicht gelesen werden: {e}")
        return None

def literal_dict(name):
    pattern = rf"{name}\s*=\s*(\{{.*?\}})"
    m = re.search(pattern, text, re.S)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1))
    except Exception as e:
        print(f"   ❌ {name} konnte nicht gelesen werden: {e}")
        return None

print("\n3. Seitenlisten prüfen ...")
main_pages = literal_list("MAIN_PAGES")
service_pages = literal_list("SERVICE_PAGES")
labels = literal_dict("PAGE_LABELS")
icons = literal_dict("PAGE_ICONS")

for name, obj in [
    ("MAIN_PAGES", main_pages),
    ("SERVICE_PAGES", service_pages),
    ("PAGE_LABELS", labels),
    ("PAGE_ICONS", icons),
]:
    if obj is None:
        print(f"   ❌ {name} fehlt oder ist nicht parsebar.")
    else:
        print(f"   ✅ {name}: {len(obj)} Einträge")

if main_pages is None or service_pages is None or labels is None or icons is None:
    sys.exit(1)

all_pages = main_pages + service_pages

print("\n4. Seitenkonsistenz prüfen ...")
problems = 0

if "Assistent" in all_pages or "💬 Assistent" in all_pages:
    print("   ❌ Assistent ist noch als Seite in MAIN/SERVICE_PAGES vorhanden.")
    problems += 1
else:
    print("   ✅ Assistent ist keine eigene Navigationsseite.")

duplicates = sorted({p for p in all_pages if all_pages.count(p) > 1})
if duplicates:
    print("   ❌ Doppelte Seiten:", duplicates)
    problems += 1
else:
    print("   ✅ Keine doppelten Seiten.")

for p in all_pages:
    if p not in labels:
        print(f"   ❌ PAGE_LABELS fehlt für: {p}")
        problems += 1
    if p not in icons:
        print(f"   ❌ PAGE_ICONS fehlt für: {p}")
        problems += 1

for p in labels:
    if p not in all_pages:
        print(f"   ⚠️ PAGE_LABELS enthält nicht navigierte Seite: {p}")

for p in icons:
    if p not in all_pages:
        print(f"   ⚠️ PAGE_ICONS enthält nicht navigierte Seite: {p}")

if problems == 0:
    print("   ✅ Seitenlisten, Labels und Icons sind konsistent.")

print("\n5. route_page prüfen ...")
route_match = re.search(r"def route_page\(page: str, ctx: Dict\[str, Any\]\) -> None:(.*?)(?:\n# =+|\ndef main\()", text, re.S)
if not route_match:
    print("   ❌ route_page Block nicht gefunden.")
    problems += 1
else:
    route_block = route_match.group(1)
    routes = re.findall(r'page\s*==\s*"([^"]+)"', route_block)
    route_targets = re.findall(r'(page_[a-zA-Z0-9_]+)\(ctx\)', route_block)

    print("   Gefundene Routennamen:")
    for r in routes:
        print(f"   - {r}")

    print("   Gefundene Routenziele:")
    for t in route_targets:
        status = "✅" if t in funcs else "❌"
        print(f"   {status} {t}")

    for t in route_targets:
        if t not in funcs:
            print(f"   ❌ Route ruft nicht vorhandene Funktion auf: {t}")
            problems += 1

    missing_routes = []
    alias_map = {
        "Kompass": ["Kompass", "Kapital-Kompass"],
        "Simulator": ["Simulator", "Portfolio-Simulator"],
        "Watchlist": ["Watchlist", "Watchlist-Vergleich"],
        "Projekt": ["Projekt", "Über das Projekt"],
        "Export": ["Export", "Professor-Export"],
        "Status": ["Status", "Betriebsstatus"],
        "Methodik": ["Methodik", "So funktioniert's"],
    }

    for page in all_pages:
        aliases = alias_map.get(page, [page])
        if not any(a in routes for a in aliases):
            missing_routes.append(page)

    if missing_routes:
        print("   ❌ Seiten ohne Route:", missing_routes)
        problems += 1
    else:
        print("   ✅ Alle Navigationsseiten haben eine Route oder Alias-Route.")

print("\n6. Layout-Basis prüfen ...")
if "st.set_page_config" not in text:
    print("   ❌ st.set_page_config fehlt.")
    problems += 1
else:
    print("   ✅ st.set_page_config vorhanden.")

if 'layout="wide"' in text or "layout='wide'" in text:
    print("   ✅ layout='wide' vorhanden.")
else:
    print("   ⚠️ layout='wide' nicht eindeutig gefunden.")

if 'initial_sidebar_state="expanded"' in text or "initial_sidebar_state='expanded'" in text:
    print("   ✅ initial_sidebar_state='expanded' vorhanden.")
else:
    print("   ⚠️ initial_sidebar_state='expanded' nicht eindeutig gefunden.")

if "st.query_params" in text:
    print("   ✅ st.query_params wird genutzt.")
else:
    print("   ⚠️ st.query_params nicht gefunden.")

print("\n7. Kritische Reste prüfen ...")
bad_patterns = [
    "page ==(ctx)",
    "page == (ctx)",
    "page == ctx",
    "theme-light",
    "theme-dark",
    "ws-fab-marker",
]
for pat in bad_patterns:
    if pat in text:
        print(f"   ❌ Kritischer Rest gefunden: {pat}")
        problems += 1
    else:
        print(f"   ✅ Nicht gefunden: {pat}")

print("\nERGEBNIS")
if problems:
    print(f"❌ Audit mit {problems} Problem(en) abgeschlossen.")
    sys.exit(1)
else:
    print("✅ Routing- und Layout-Audit ohne kritische Probleme abgeschlossen.")
