"""Structural validation for the modular WealthScope AI app.

Supersedes the old audit_app_structure.py / audit_routing_layout.py, which
checked invariants of the pre-rebuild app_max.py monolith (single file,
MAIN_PAGES/SERVICE_PAGES constants, route_page() dispatcher). The rebuild
replaced that with app.py + src/pages/*.py + st.navigation, so this script
checks the equivalent invariants for the new shape:

  1. Every file under src/ and app.py compiles.
  2. Every module imported by src/pages/__init__.py-adjacent app.py exposes
     the render function(s) app.py expects.
  3. st.Page(..., url_path=...) values in app.py are unique (a duplicate
     crashes the app at runtime with StreamlitAPIException — hit once
     during manual testing of this rebuild).
  4. Model artifacts (joblib + diagnostics/learning-curve JSON) exist, since
     src/model.py and src/diagnostics.py assume they're present.

Run: python3 scripts/validate_app.py
"""
from __future__ import annotations

import ast
import importlib
import json
import py_compile
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

PAGE_MODULES = {
    "start": ["render"], "market": ["render"], "ml_insights": ["render"],
    "kompass": ["render"], "simulator": ["render"], "watchlist": ["render"],
    "data_lab": ["render"], "news_page": ["render"], "assistant_page": ["render"],
    "methodology": ["render"], "project": ["render"], "export_page": ["render"],
    "legal": ["render_impressum", "render_datenschutz"], "status": ["render"],
}


def fail(message: str) -> None:
    raise SystemExit(f"❌ {message}")


def main() -> None:
    print("1. Prüfe Python-Syntax (app.py + src/**) ...")
    files = [BASE_DIR / "app.py"] + list((BASE_DIR / "src").rglob("*.py"))
    for f in files:
        py_compile.compile(str(f), doraise=True)
    print(f"   ✅ {len(files)} Dateien compilieren")

    print("2. Prüfe Seiten-Module ...")
    for mod_name, fn_names in PAGE_MODULES.items():
        module = importlib.import_module(f"src.pages.{mod_name}")
        for fn_name in fn_names:
            if not hasattr(module, fn_name):
                fail(f"src/pages/{mod_name}.py fehlt Funktion '{fn_name}()'")
    print(f"   ✅ Alle {len(PAGE_MODULES)} Seiten-Module exponieren ihre render-Funktion(en)")

    print("3. Prüfe eindeutige url_path-Werte in app.py ...")
    tree = ast.parse((BASE_DIR / "app.py").read_text())
    url_paths = [
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "Page"
        for kw in node.keywords
        if kw.arg == "url_path" and isinstance(kw.value, ast.Constant)
    ]
    if not url_paths:
        fail("Keine st.Page(..., url_path=...) Aufrufe in app.py gefunden.")
    dupes = {p for p in url_paths if url_paths.count(p) > 1}
    if dupes:
        fail(f"Doppelte url_path-Werte in app.py: {sorted(dupes)}")
    print(f"   ✅ {len(url_paths)} Seiten, alle url_path-Werte eindeutig")

    print("4. Prüfe Modell-Artefakte ...")
    model_path = BASE_DIR / "models" / "wealthscope_model.joblib"
    diag_path = BASE_DIR / "models" / "diagnostics.json"
    lc_path = BASE_DIR / "models" / "learning_curve.json"
    for p in (model_path, diag_path, lc_path):
        if not p.exists():
            fail(f"Fehlendes Artefakt: {p.relative_to(BASE_DIR)} — führe scripts/train_and_diagnose.py aus.")
    json.loads(diag_path.read_text())
    json.loads(lc_path.read_text())
    print("   ✅ Modell, Diagnostics-Cache und Lernkurven-Cache vorhanden und lesbar")

    print("\n✅ App-Validierung abgeschlossen.")


if __name__ == "__main__":
    main()
