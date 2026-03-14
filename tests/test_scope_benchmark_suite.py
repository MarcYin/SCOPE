from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_suite_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_scope_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_scope_benchmark_suite", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark suite script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metric(max_rel: float) -> dict[str, float]:
    return {"max_abs": max_rel, "mean_abs": max_rel / 2.0, "max_rel": max_rel, "mean_rel": max_rel / 2.0}


def test_summary_subset_skips_missing_metrics():
    module = _load_suite_module()
    summary = {
        "max_abs": {
            "reflectance.refl": {"case": "001", **_metric(1e-6)},
        },
        "max_rel": {
            "reflectance.refl": {"case": "001", **_metric(1e-6)},
        },
    }

    subset = module._summary_subset(
        summary,
        ["reflectance.refl", "energy_balance.Rnhc"],
    )

    assert subset == {
        "max_abs": {
            "reflectance.refl": {"case": "001", **_metric(1e-6)},
        },
        "max_rel": {
            "reflectance.refl": {"case": "001", **_metric(1e-6)},
        },
    }
    assert module._highlight(subset, list(subset["max_abs"])) == {
        "reflectance.refl": {
            "worst_max_abs_case": "001",
            "worst_max_abs": 1e-6,
            "worst_max_rel_case": "001",
            "worst_max_rel": 1e-6,
            "worst_mean_abs_from_max_abs_case": 5e-7,
        }
    }


def test_available_summary_keys_skip_missing_highlights():
    module = _load_suite_module()
    summary = {
        "max_abs": {
            "reflectance.refl": {"case": "042", **_metric(2e-6)},
        },
        "max_rel": {
            "reflectance.refl": {"case": "042", **_metric(2e-6)},
        },
    }

    keys = module._available_summary_keys(
        summary,
        ["reflectance.refl", "energy_iteration_input.sunlit_Cs"],
    )

    assert keys == ["reflectance.refl"]
    assert module._highlight(summary, keys) == {
        "reflectance.refl": {
            "worst_max_abs_case": "042",
            "worst_max_abs": 2e-6,
            "worst_max_rel_case": "042",
            "worst_max_rel": 2e-6,
            "worst_mean_abs_from_max_abs_case": 1e-6,
        }
    }


def test_filtered_worst_cases_excludes_nonconverged_energy_metrics_only():
    module = _load_suite_module()

    per_case = {
        "001": {
            "energy_balance": {"Rnuct": _metric(1e-4)},
            "reflectance": {"refl": _metric(1e-6)},
        },
        "042": {
            "energy_balance": {"Rnuct": _metric(1e-2)},
            "reflectance": {"refl": _metric(2e-6)},
        },
    }

    summary = module._filtered_worst_cases(
        per_case,
        exclude=set(),
        nonconverged_energy_cases={"042"},
    )

    assert summary["max_rel"]["energy_balance.Rnuct"]["case"] == "001"
    assert summary["max_rel"]["reflectance.refl"]["case"] == "042"


def test_subset_worst_cases_keeps_nonconverged_energy_metrics_as_stress_diagnostics():
    module = _load_suite_module()

    per_case = {
        "001": {
            "energy_balance": {"Rnuct": _metric(1e-4)},
            "reflectance": {"refl": _metric(1e-6)},
        },
        "042": {
            "energy_balance": {"Rnuct": _metric(1e-2)},
            "energy_iteration_input": {"sunlit_Cs": _metric(2e-2)},
            "reflectance": {"refl": _metric(2e-6)},
        },
    }

    stress = module._subset_worst_cases(
        per_case,
        include_cases={"042"},
        include_prefixes=module.NONCONVERGED_ENERGY_PREFIXES,
    )

    assert stress["max_rel"]["energy_balance.Rnuct"]["case"] == "042"
    assert stress["max_rel"]["energy_iteration_input.sunlit_Cs"]["case"] == "042"
    assert "reflectance.refl" not in stress["max_rel"]


def test_default_matlab_honours_environment_override(monkeypatch):
    monkeypatch.setenv("MATLAB_BIN", "/custom/matlab")
    module = _load_suite_module()

    assert module.DEFAULT_MATLAB == "/custom/matlab"


def test_parity_policy_metadata_exposes_interpretation_order():
    module = _load_suite_module()

    policy = module._parity_policy_metadata(
        nonconverged_key="nonconverged_energy_case_rule",
        nonconverged_rule="keep stress cases separate",
    )

    assert policy["primary_relative_summary"] == "parity_worst_cases"
    assert policy["absolute_policy_summary"] == "absolute_policy_worst_cases"
    assert policy["stress_summary"] == "stress_worst_cases"
    assert policy["recommended_interpretation_order"] == [
        "parity_worst_cases",
        "absolute_policy_worst_cases",
        "stress_worst_cases",
    ]
    assert "energy_balance.sunlit_A" in policy["phase_lagged_metrics"]
    assert policy["phase_lagged_metric_replacements"]["energy_balance.sunlit_A"] == "leaf_iteration.sunlit_A"
    assert policy["nonconverged_energy_case_rule"] == "keep stress cases separate"
