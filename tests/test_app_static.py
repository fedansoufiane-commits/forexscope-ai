"""Static checks for the modular WealthScope AI app (app.py + src/).

Run: pytest tests/test_app_static.py
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

SRC_MODULES = [
    "config", "state", "context", "sidebar", "theme", "ui",
    "data", "model", "diagnostics", "charts", "news", "export",
]

PAGE_MODULES = {
    "start": ["render"], "market": ["render"], "ml_insights": ["render"],
    "kompass": ["render"], "simulator": ["render"], "watchlist": ["render"],
    "data_lab": ["render"], "news_page": ["render"], "assistant_page": ["render"],
    "methodology": ["render"], "project": ["render"], "export_page": ["render"],
    "legal": ["render_impressum", "render_datenschutz"], "status": ["render"],
}


def _all_py_files():
    yield BASE_DIR / "app.py"
    yield from (BASE_DIR / "src").rglob("*.py")


def test_all_files_compile():
    for path in _all_py_files():
        py_compile.compile(str(path), doraise=True)


def test_app_entrypoint_has_no_literal_newline_damage():
    first_line = (BASE_DIR / "app.py").read_text().splitlines()[0]
    assert "\\n" not in first_line


def test_src_modules_importable():
    for name in SRC_MODULES:
        importlib.import_module(f"src.{name}")


def test_page_modules_expose_render_functions():
    for mod_name, fn_names in PAGE_MODULES.items():
        module = importlib.import_module(f"src.pages.{mod_name}")
        for fn_name in fn_names:
            assert hasattr(module, fn_name), f"src/pages/{mod_name}.py missing {fn_name}()"
            assert callable(getattr(module, fn_name))


def test_app_registers_unique_url_paths():
    """Regression guard: st.Page with duplicate inferred/explicit url_path
    crashes at runtime with StreamlitAPIException. Hit once during manual
    testing (multiple page functions were all named `render`) — this makes
    sure it can't silently regress."""
    tree = ast.parse((BASE_DIR / "app.py").read_text())
    url_paths = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "Page":
            for kw in node.keywords:
                if kw.arg == "url_path" and isinstance(kw.value, ast.Constant):
                    url_paths.append(kw.value.value)
    assert url_paths, "No st.Page(..., url_path=...) calls found in app.py"
    assert len(url_paths) == len(set(url_paths)), f"Duplicate url_path values: {url_paths}"


def test_diagnostics_artifacts_exist_and_are_well_formed():
    diag_path = BASE_DIR / "models" / "diagnostics.json"
    lc_path = BASE_DIR / "models" / "learning_curve.json"
    model_path = BASE_DIR / "models" / "wealthscope_model.joblib"
    assert model_path.exists(), "Run scripts/train_and_diagnose.py first"
    assert diag_path.exists()
    assert lc_path.exists()

    diag = json.loads(diag_path.read_text())
    for key in ("test_metrics", "confusion_matrix", "roc_curve", "pr_curve",
                "cross_validation", "feature_importance"):
        assert key in diag, f"diagnostics.json missing '{key}'"

    lc = json.loads(lc_path.read_text())
    for key in ("train_sizes_abs", "train_scores_mean", "val_scores_mean"):
        assert key in lc, f"learning_curve.json missing '{key}'"
    assert len(lc["train_sizes_abs"]) == len(lc["train_scores_mean"]) == len(lc["val_scores_mean"])
