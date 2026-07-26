from pathlib import Path
import ast
import re
import sys

APP = Path("app_max.py")

if not APP.exists():
    print("❌ app_max.py nicht gefunden.")
    sys.exit(1)

text = APP.read_text(encoding="utf-8", errors="ignore")

problems = 0
warnings = 0

def fail(msg):
    global problems
    problems += 1
    print(f"❌ {msg}")

def warn(msg):
    global warnings
    warnings += 1
    print(f"⚠️ {msg}")

def ok(msg):
    print(f"✅ {msg}")

print("1. Syntax / AST")
try:
    tree = ast.parse(text)
    ok("Syntax OK")
    ok("AST OK")
except SyntaxError as e:
    fail(f"SyntaxError in Zeile {e.lineno}: {e.msg}")
    if e.lineno:
        lines = text.splitlines()
        for no in range(max(1, e.lineno - 5), min(len(lines), e.lineno + 5) + 1):
            marker = ">>" if no == e.lineno else "  "
            print(f"{marker} {no:4d}: {lines[no-1]}")
    sys.exit(1)

funcs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
page_funcs = sorted(f for f in funcs if f.startswith("page_"))

print("\n2. Page-Funktionen")
for f in page_funcs:
    print(f"   - {f}")

def get_assign(name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    try:
                        return ast.literal_eval(node.value)
                    except Exception:
                        return None
    return None

main_pages = get_assign("MAIN_PAGES")
service_pages = get_assign("SERVICE_PAGES")
labels = get_assign("PAGE_LABELS")
icons = get_assign("PAGE_ICONS")

print("\n3. Seitenlisten")
for name, value in [
    ("MAIN_PAGES", main_pages),
    ("SERVICE_PAGES", service_pages),
    ("PAGE_LABELS", labels),
    ("PAGE_ICONS", icons),
]:
    if value is None:
        fail(f"{name} fehlt oder ist nicht parsebar.")
    else:
        ok(f"{name}: {len(value)} Einträge")

if not all([main_pages, service_pages, labels, icons]):
    sys.exit(1)

all_pages = main_pages + service_pages

print("\n4. Navigationskonsistenz")
dupes = sorted({p for p in all_pages if all_pages.count(p) > 1})
if dupes:
    fail(f"Doppelte Seiten gefunden: {dupes}")
else:
    ok("Keine doppelten Seiten.")

for p in all_pages:
    if p not in labels:
        fail(f"PAGE_LABELS fehlt für: {p}")
    if p not in icons:
        fail(f"PAGE_ICONS fehlt für: {p}")

for p in labels:
    if p not in all_pages:
        warn(f"PAGE_LABELS enthält nicht navigierte Seite: {p}")

for p in icons:
    if p not in all_pages:
        warn(f"PAGE_ICONS enthält nicht navigierte Seite: {p}")

if "Assistent" in all_pages or "💬 Assistent" in all_pages:
    warn("Assistent ist noch als eigene Navigationsseite vorhanden.")
else:
    ok("Assistent ist keine eigene Navigationsseite.")

print("\n5. route_page")
route_node = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "route_page":
        route_node = node
        break

if route_node is None:
    fail("route_page fehlt.")
else:
    route_src = ast.get_source_segment(text, route_node) or ""
    routes = re.findall(r'page\s*==\s*"([^"]+)"', route_src)
    targets = re.findall(r'(page_[a-zA-Z0-9_]+)\(ctx\)', route_src)

    print("   Routennamen:")
    for r in routes:
        print(f"   - {r}")

    print("   Routenziele:")
    for t in targets:
        if t in funcs:
            ok(t)
        else:
            fail(f"Route ruft nicht vorhandene Funktion auf: {t}")

    for p in all_pages:
        if p not in routes:
            fail(f"Navigationsseite ohne direkte Route: {p}")

print("\n6. Page Config / Layout")
if "st.set_page_config(" in text:
    ok("st.set_page_config vorhanden.")
else:
    fail("st.set_page_config fehlt.")

if 'layout="wide"' in text or "layout='wide'" in text:
    ok("layout='wide' vorhanden.")
else:
    warn("layout='wide' nicht gefunden.")

if 'initial_sidebar_state="expanded"' in text or "initial_sidebar_state='expanded'" in text:
    ok("initial_sidebar_state='expanded' vorhanden.")
else:
    warn("initial_sidebar_state='expanded' nicht gefunden.")

if "st.query_params" in text:
    ok("st.query_params wird genutzt.")
else:
    warn("st.query_params wird nicht genutzt.")

if "@st.cache_data" in text:
    ok("st.cache_data wird genutzt.")
else:
    warn("st.cache_data wird nicht genutzt.")

print("\n7. Kritische Reste")
bad_patterns = [
    "\\\\nimport",
    "page ==(ctx)",
    "page == (ctx)",
    "page == ctx",
    "theme-light",
    "theme-dark",
    "ws-fab-marker",
]

for pat in bad_patterns:
    if pat in text:
        fail(f"Kritischer Rest gefunden: {pat}")
    else:
        ok(f"Nicht gefunden: {pat}")

print("\n8. Ergebnis")
if problems:
    print(f"❌ Audit mit {problems} Problem(en) und {warnings} Warnung(en).")
    sys.exit(1)

if warnings:
    print(f"⚠️ Audit ohne harte Fehler, aber mit {warnings} Warnung(en).")
    sys.exit(0)

print("✅ Audit vollständig sauber.")
