from __future__ import annotations

import ast
import py_compile
from pathlib import Path


def test_app_compiles():
    py_compile.compile("app_max.py", doraise=True)


def test_app_has_main_and_router():
    text = Path("app_max.py").read_text(encoding="utf-8")
    tree = ast.parse(text)

    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "main" in functions
    assert "route_page" in functions
    assert "render_sidebar" in functions
    assert "build_context" in functions


def test_no_literal_newline_damage_at_file_start():
    first_line = Path("app_max.py").read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    assert "\\n" not in first_line
