"""Static checks for the modular WealthScope AI app (app.py + src/).

Run: pytest tests/test_app_static.py
"""
from __future__ import annotations

import ast
import importlib
import json
import math
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
    "learning_studio": ["render"], "legal": ["render_impressum", "render_datenschutz"],
    "status": ["render"],
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
                "cross_validation", "feature_importance", "model_comparison", "validation"):
        assert key in diag, f"diagnostics.json missing '{key}'"
    assert diag["schema_version"] == 2
    validation = diag["validation"]
    assert validation["train_end"] < validation["test_start"] <= validation["test_end"]
    assert validation["purge_trading_days"] == 20
    assert set(diag["model_comparison"]) == {
        "dummy", "logistic", "decision_tree", "linear_svm", "random_forest",
    }
    assert len(diag["roc_curve"]["fpr"]) <= 800
    assert len(diag["pr_curve"]["precision"]) <= 800
    for model in diag["model_comparison"].values():
        assert math.isfinite(model["test_metrics"]["roc_auc"])
        assert len(model["walk_forward"]) == 4

    lc = json.loads(lc_path.read_text())
    for key in ("train_sizes_abs", "train_scores_mean", "val_scores_mean"):
        assert key in lc, f"learning_curve.json missing '{key}'"
    assert len(lc["train_sizes_abs"]) == len(lc["train_scores_mean"]) == len(lc["val_scores_mean"])


def test_validation_experiments_artifact_supports_the_negative_result():
    """The report and notebooks quote these numbers, so they must stay in sync."""
    path = BASE_DIR / "models" / "validation_experiments.json"
    assert path.exists(), "Run scripts/validation_experiments.py first"

    exp = json.loads(path.read_text())
    assert exp["schema_version"] == 1

    # Capacity sweep: training score spans a wide range, test score barely moves.
    sweep = exp["capacity_sweep"]
    assert len(sweep) == 7
    train_span = max(r["train_roc_auc"] for r in sweep) - min(r["train_roc_auc"] for r in sweep)
    summary = exp["capacity_sweep_summary"]
    assert train_span > 0.30, "capacity sweep no longer covers a wide capacity range"
    assert summary["test_roc_auc_span"] < 0.02, (
        "test AUC now varies with capacity - the 'capacity is not the bottleneck' "
        "claim in the report and notebook 04 would no longer hold"
    )

    # Split comparison: the leaky split must inflate the apparent signal.
    splits = exp["split_comparison"]
    assert len(splits) == 3
    assert all(math.isfinite(row["roc_auc"]) for row in splits)
    split_summary = exp["split_comparison_summary"]
    assert split_summary["leaky_roc_auc"] > split_summary["reference_roc_auc"]
    assert split_summary["apparent_signal_factor"] > 2.0, (
        "the leakage demonstration lost its effect - slide 19 and section 6 of the "
        "report quote this factor"
    )


def test_quiz_export_is_arsnova_compatible():
    from src.quiz import QUESTIONS, build_arsnova_quiz

    payload = json.loads(build_arsnova_quiz().decode("utf-8"))
    assert payload["exportVersion"] == 1
    exported = payload["quiz"]["questions"]
    assert len(exported) == len(QUESTIONS)
    assert all(q["type"] == "SINGLE_CHOICE" for q in exported)
    assert all(sum(a["isCorrect"] for a in q["answers"]) == 1 for q in exported)
