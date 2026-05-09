from __future__ import annotations

import ast
import py_compile
from collections import Counter
from pathlib import Path


APP_FILE = Path("app_max.py")

REQUIRED_FUNCTIONS = [
    "main",
    "route_page",
    "render_sidebar",
    "render_bottom_bar",
    "build_context",
    "page_outlook",
    "page_news",
    "page_export",
    "page_project",
    "page_impressum",
    "page_status",
]

OPTIONAL_UPGRADE_FUNCTIONS = [
    "render_news_cards",
    "build_ticker_ranking",
    "chart_risk_return_scatter",
    "chart_feature_correlation",
    "methodology_dialog",
    "assistant_answer",
    "render_floating_assistant_panel",
]


def fail(message: str) -> None:
    raise SystemExit(f"❌ {message}")


def main() -> None:
    if not APP_FILE.exists():
        fail("app_max.py wurde nicht gefunden.")

    text = APP_FILE.read_text(encoding="utf-8", errors="ignore")

    first_line = text.lstrip().splitlines()[0] if text.strip() else ""
    if "\\n" in first_line:
        fail("app_max.py enthält wahrscheinlich literal \\n in Zeile 1.")

    print("1. Prüfe Python-Syntax ...")
    py_compile.compile(str(APP_FILE), doraise=True)
    print("   ✅ Syntax OK")

    print("2. Parse AST ...")
    tree = ast.parse(text)
    print("   ✅ AST OK")

    print("3. Prüfe Funktionsdefinitionen ...")
    function_names = [
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    counts = Counter(function_names)
    duplicates = {name: count for name, count in counts.items() if count > 1}

    if duplicates:
        print("   ⚠️ Doppelte Funktionsdefinitionen gefunden:")
        for name, count in sorted(duplicates.items()):
            print(f"      - {name}: {count}x")
    else:
        print("   ✅ Keine doppelten Funktionsdefinitionen")

    missing_required = [name for name in REQUIRED_FUNCTIONS if name not in counts]
    if missing_required:
        fail(f"Pflichtfunktionen fehlen: {missing_required}")
    print("   ✅ Pflichtfunktionen vorhanden")

    missing_optional = [name for name in OPTIONAL_UPGRADE_FUNCTIONS if name not in counts]
    if missing_optional:
        print("   ⚠️ Optionale Upgrade-Funktionen fehlen:")
        for name in missing_optional:
            print(f"      - {name}")
    else:
        print("   ✅ Alle Upgrade-Funktionen vorhanden")

    print("4. Prüfe kritische Aufrufe ohne Definition ...")
    critical_calls = [
        "render_news_cards",
        "build_ticker_ranking",
        "methodology_dialog",
        "assistant_answer",
        "render_floating_assistant_panel",
    ]

    for call in critical_calls:
        if f"{call}(" in text and call not in counts:
            fail(f"'{call}' wird aufgerufen, aber nicht definiert.")

    print("   ✅ Keine kritischen fehlenden Helper erkannt")

    print("\n✅ App-Validierung abgeschlossen.")


if __name__ == "__main__":
    main()
