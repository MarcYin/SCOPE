from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_suite_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_scope_timeseries_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_scope_timeseries_benchmark_suite", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load time-series benchmark suite script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_discover_default_steps_matches_verification_series_length():
    module = _load_suite_module()
    repo_root = Path(__file__).resolve().parents[1]

    steps = module._discover_default_steps(repo_root)

    assert steps[0] == 1
    assert steps[-1] == 30
    assert len(steps) == 30


def test_matlab_vector_formats_step_list_for_batch_command():
    module = _load_suite_module()

    assert module._matlab_vector([1, 2, 10]) == "[1 2 10]"


def test_available_highlight_keys_skips_metrics_missing_from_filtered_summary():
    module = _load_suite_module()
    summary = {
        "max_abs": {
            "reflectance.refl": {
                "case": "026",
                "max_abs": 1.0,
                "mean_abs": 0.5,
                "max_rel": 0.1,
                "mean_rel": 0.05,
            }
        },
        "max_rel": {
            "reflectance.refl": {
                "case": "026",
                "max_abs": 1.0,
                "mean_abs": 0.5,
                "max_rel": 0.1,
                "mean_rel": 0.05,
            }
        },
    }

    keys = module._available_highlight_keys(
        summary,
        ["reflectance.refl", "energy_balance.Rntot"],
    )

    assert keys == ["reflectance.refl"]
    assert module._SCENE_SUITE._highlight(summary, keys) == {
        "reflectance.refl": {
            "worst_max_abs_case": "026",
            "worst_max_abs": 1.0,
            "worst_max_rel_case": "026",
            "worst_max_rel": 0.1,
            "worst_mean_abs_from_max_abs_case": 0.5,
        }
    }


def test_scene_suite_stable_summary_path_relativizes_repo_paths():
    module = _load_suite_module()
    repo_root = Path(__file__).resolve().parents[1]
    reports_dir = repo_root / "tests" / "data" / "timeseries_benchmark_reports"

    assert module._SCENE_SUITE._stable_summary_path(reports_dir, repo_root) == "tests/data/timeseries_benchmark_reports"


def test_scene_suite_summary_subset_allows_empty_absolute_policy_subset():
    module = _load_suite_module()
    summary = {
        "max_abs": {
            "reflectance.refl": {
                "case": "026",
                "max_abs": 1.0,
                "mean_abs": 0.5,
                "max_rel": 0.1,
                "mean_rel": 0.05,
            }
        },
        "max_rel": {
            "reflectance.refl": {
                "case": "026",
                "max_abs": 1.0,
                "mean_abs": 0.5,
                "max_rel": 0.1,
                "mean_rel": 0.05,
            }
        },
    }

    subset = module._SCENE_SUITE._summary_subset(
        summary,
        sorted(module._SCENE_SUITE.LOW_MAGNITUDE_ABSOLUTE_POLICY_METRICS),
    )

    assert subset == {"max_abs": {}, "max_rel": {}}
    assert module._SCENE_SUITE._highlight(subset, list(subset["max_abs"])) == {}


def test_default_matlab_honours_environment_override(monkeypatch):
    monkeypatch.setenv("MATLAB_BIN", "/custom/matlab")
    module = _load_suite_module()

    assert module.DEFAULT_MATLAB == "/custom/matlab"


def test_scene_suite_policy_helper_is_reused_for_timeseries_policy():
    module = _load_suite_module()

    policy = module._SCENE_SUITE._parity_policy_metadata(
        nonconverged_key="nonconverged_energy_step_rule",
        nonconverged_rule="keep non-converged steps as stress diagnostics",
    )

    assert policy["primary_relative_summary"] == "parity_worst_cases"
    assert policy["absolute_policy_summary"] == "absolute_policy_worst_cases"
    assert policy["stress_summary"] == "stress_worst_cases"
    assert policy["nonconverged_energy_step_rule"] == "keep non-converged steps as stress diagnostics"
