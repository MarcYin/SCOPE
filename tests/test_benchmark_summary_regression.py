from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_SUMMARY_PATH = REPO_ROOT / "tests" / "data" / "scope_benchmark_suite_summary.json"
TIMESERIES_SUMMARY_PATH = REPO_ROOT / "tests" / "data" / "scope_timeseries_benchmark_summary.json"

SCENE_THRESHOLDS = {
    "reflectance.refl": 1e-6,
    "resistances_direct.raa": 1e-12,
    "resistances_direct.raws": 1e-12,
    "fluorescence_transport.EoutFrc_": 5e-5,
    "fluorescence_transport.sigmaF": 5e-5,
    "thermal_transport.Eoutte_": 1e-10,
    "leaf_iteration.sunlit_A": 1e-9,
    "leaf_iteration.shaded_A": 1e-9,
    "leaf_iteration.sunlit_rcw": 1e-9,
    "leaf_iteration.shaded_rcw": 1e-9,
    "energy_iteration_input.sunlit_Cs": 1e-3,
    "energy_iteration_input.shaded_Cs": 5e-4,
    "energy_balance.Rnuc_sw": 1e-10,
    "energy_balance.Rnhc_sw": 1e-10,
    "energy_balance.Rntot": 1e-4,
    "energy_balance.lEtot": 5e-5,
    "energy_balance.Htot": 5e-4,
    "energy_balance.Tcu": 1e-4,
    "energy_balance.Tch": 1e-4,
    "energy_balance.Tsu": 1e-4,
    "energy_balance.Tsh": 1e-4,
    "energy_balance.canopyemis": 1e-6,
    "energy_balance.L": 1e-3,
}

TIMESERIES_THRESHOLDS = {
    "reflectance.refl": 1e-6,
    "fluorescence_transport.LoF_": 1e-6,
    "thermal_transport.Lot_": 1e-6,
    "energy_balance.Rntot": 1e-3,
    "energy_balance.lEtot": 1e-3,
    "energy_balance.Htot": 1e-3,
    "energy_balance.Tcu": 1e-3,
    "energy_balance.Tch": 1e-3,
}

ABSOLUTE_POLICY_THRESHOLDS = {
    "energy_balance.Rnuc": 5e-4,
    "energy_balance.Rnhc": 5e-4,
    "energy_balance.Rnuct": 5e-4,
    "energy_balance.Rnhct": 5e-4,
}


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _max_rel(summary: dict, key: str) -> float:
    return float(summary["parity_worst_cases"]["max_rel"][key]["max_rel"])


def _assert_highlights_match_parity_summary(summary: dict) -> None:
    for key, values in summary["highlights"].items():
        parity_entry = summary["parity_worst_cases"]["max_rel"][key]
        assert values["worst_max_rel_case"] == parity_entry["case"]
        assert values["worst_max_rel"] == parity_entry["max_rel"]


def _assert_relative_energy_parity_below(summary: dict, threshold: float) -> None:
    relative_energy = {
        key: entry["max_rel"]
        for key, entry in summary["parity_worst_cases"]["max_rel"].items()
        if key.startswith("energy_balance.") or key.startswith("energy_iteration_input.")
    }
    assert max(relative_energy.values()) < threshold


def test_scene_benchmark_summary_matches_committed_policy():
    summary = _load_summary(SCENE_SUMMARY_PATH)

    assert len(summary["cases"]) == 100
    assert summary["cases"][0] == 1
    assert summary["cases"][-1] == 100
    assert summary["nonconverged_energy_cases"] == ["042"]
    assert summary["parity_policy"]["primary_relative_summary"] == "parity_worst_cases"
    assert summary["parity_policy"]["absolute_policy_summary"] == "absolute_policy_worst_cases"
    assert summary["parity_policy"]["stress_summary"] == "stress_worst_cases"
    assert summary["parity_policy"]["recommended_interpretation_order"] == [
        "parity_worst_cases",
        "absolute_policy_worst_cases",
        "stress_worst_cases",
    ]
    assert "leaf_iteration.*" in summary["parity_policy"]["same_state_rule"]
    assert summary["parity_policy"]["always_excluded_metrics"] == [
        "energy_balance.Rnhc",
        "energy_balance.Rnhct",
        "energy_balance.Rnuc",
        "energy_balance.Rnuct",
        "energy_balance.shaded_A",
        "energy_balance.shaded_Ci",
        "energy_balance.shaded_eta",
        "energy_balance.shaded_rcw",
        "energy_balance.sunlit_A",
        "energy_balance.sunlit_Ci",
        "energy_balance.sunlit_eta",
        "energy_balance.sunlit_rcw",
    ]
    assert summary["parity_policy"]["phase_lagged_metrics"] == [
        "energy_balance.shaded_A",
        "energy_balance.shaded_Ci",
        "energy_balance.shaded_eta",
        "energy_balance.shaded_rcw",
        "energy_balance.sunlit_A",
        "energy_balance.sunlit_Ci",
        "energy_balance.sunlit_eta",
        "energy_balance.sunlit_rcw",
    ]
    assert summary["parity_policy"]["nonconverged_energy_metric_prefixes"] == [
        "energy_balance.",
        "energy_iteration_input.",
    ]
    assert summary["parity_policy"]["absolute_policy_metrics"] == [
        "energy_balance.Rnhc",
        "energy_balance.Rnhct",
        "energy_balance.Rnuc",
        "energy_balance.Rnuct",
    ]
    assert "energy_balance.Rnuct" in summary["stress_worst_cases"]["max_rel"]
    assert "reflectance.refl" not in summary["stress_worst_cases"]["max_rel"]
    _assert_highlights_match_parity_summary(summary)
    _assert_relative_energy_parity_below(summary, 1e-3)

    for key, threshold in SCENE_THRESHOLDS.items():
        assert _max_rel(summary, key) < threshold, key
    for key, threshold in ABSOLUTE_POLICY_THRESHOLDS.items():
        assert float(summary["absolute_policy_worst_cases"]["max_abs"][key]["max_abs"]) < threshold, key


def test_timeseries_benchmark_summary_matches_committed_policy():
    summary = _load_summary(TIMESERIES_SUMMARY_PATH)

    assert len(summary["steps"]) == 30
    assert summary["steps"][0] == "001"
    assert summary["steps"][-1] == "030"
    assert summary["nonconverged_energy_steps"] == ["026"]
    assert summary["parity_policy"]["primary_relative_summary"] == "parity_worst_cases"
    assert summary["parity_policy"]["absolute_policy_summary"] == "absolute_policy_worst_cases"
    assert summary["parity_policy"]["stress_summary"] == "stress_worst_cases"
    assert summary["parity_policy"]["recommended_interpretation_order"] == [
        "parity_worst_cases",
        "absolute_policy_worst_cases",
        "stress_worst_cases",
    ]
    assert "leaf_iteration.*" in summary["parity_policy"]["same_state_rule"]
    assert summary["parity_policy"]["always_excluded_metrics"] == [
        "energy_balance.Rnhc",
        "energy_balance.Rnhct",
        "energy_balance.Rnuc",
        "energy_balance.Rnuct",
        "energy_balance.shaded_A",
        "energy_balance.shaded_Ci",
        "energy_balance.shaded_eta",
        "energy_balance.shaded_rcw",
        "energy_balance.sunlit_A",
        "energy_balance.sunlit_Ci",
        "energy_balance.sunlit_eta",
        "energy_balance.sunlit_rcw",
    ]
    assert summary["parity_policy"]["phase_lagged_metrics"] == [
        "energy_balance.shaded_A",
        "energy_balance.shaded_Ci",
        "energy_balance.shaded_eta",
        "energy_balance.shaded_rcw",
        "energy_balance.sunlit_A",
        "energy_balance.sunlit_Ci",
        "energy_balance.sunlit_eta",
        "energy_balance.sunlit_rcw",
    ]
    assert summary["parity_policy"]["nonconverged_energy_metric_prefixes"] == [
        "energy_balance.",
        "energy_iteration_input.",
    ]
    assert summary["parity_policy"]["absolute_policy_metrics"] == [
        "energy_balance.Rnhc",
        "energy_balance.Rnhct",
        "energy_balance.Rnuc",
        "energy_balance.Rnuct",
    ]
    assert "energy_balance.Rnuct" in summary["stress_worst_cases"]["max_rel"]
    assert "reflectance.refl" not in summary["stress_worst_cases"]["max_rel"]
    _assert_highlights_match_parity_summary(summary)
    _assert_relative_energy_parity_below(summary, 1e-3)

    for key, threshold in TIMESERIES_THRESHOLDS.items():
        assert _max_rel(summary, key) < threshold, key
    for key, threshold in ABSOLUTE_POLICY_THRESHOLDS.items():
        assert float(summary["absolute_policy_worst_cases"]["max_abs"][key]["max_abs"]) < threshold, key
