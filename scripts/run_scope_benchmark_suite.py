from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MATLAB = os.environ.get("MATLAB_BIN", "/Applications/MATLAB_R2025b.app/bin/matlab")
NONCONVERGED_ENERGY_PREFIXES = ("energy_balance.", "energy_iteration_input.")
PHASE_LAGGED_ENERGY_METRICS = {
    "energy_balance.sunlit_A",
    "energy_balance.shaded_A",
    "energy_balance.sunlit_Ci",
    "energy_balance.shaded_Ci",
    "energy_balance.sunlit_eta",
    "energy_balance.shaded_eta",
    "energy_balance.sunlit_rcw",
    "energy_balance.shaded_rcw",
}
LOW_MAGNITUDE_ABSOLUTE_POLICY_METRICS = {
    "energy_balance.Rnuc",
    "energy_balance.Rnhc",
    "energy_balance.Rnuct",
    "energy_balance.Rnhct",
}
RELATIVE_PARITY_EXCLUDE = PHASE_LAGGED_ENERGY_METRICS | LOW_MAGNITUDE_ABSOLUTE_POLICY_METRICS
PHASE_LAGGED_METRIC_REPLACEMENTS = {
    "energy_balance.sunlit_A": "leaf_iteration.sunlit_A",
    "energy_balance.shaded_A": "leaf_iteration.shaded_A",
    "energy_balance.sunlit_Ci": "leaf_iteration.sunlit_Ci",
    "energy_balance.shaded_Ci": "leaf_iteration.shaded_Ci",
    "energy_balance.sunlit_eta": "leaf_iteration.sunlit_eta",
    "energy_balance.shaded_eta": "leaf_iteration.shaded_eta",
    "energy_balance.sunlit_rcw": "leaf_iteration.sunlit_rcw",
    "energy_balance.shaded_rcw": "leaf_iteration.shaded_rcw",
}


def _parity_policy_metadata(
    *,
    nonconverged_rule: str,
    nonconverged_key: str,
) -> dict[str, Any]:
    return {
        "primary_relative_summary": "parity_worst_cases",
        "absolute_policy_summary": "absolute_policy_worst_cases",
        "stress_summary": "stress_worst_cases",
        "recommended_interpretation_order": [
            "parity_worst_cases",
            "absolute_policy_worst_cases",
            "stress_worst_cases",
        ],
        "same_state_rule": (
            "Use parity_worst_cases for same-state relative parity. "
            "For leaf biochemistry quantities, prefer leaf_iteration.* entries over "
            "phase-lagged energy_balance.sunlit_* and energy_balance.shaded_* diagnostics."
        ),
        "phase_lagged_metrics": sorted(PHASE_LAGGED_ENERGY_METRICS),
        "always_excluded_metrics": sorted(RELATIVE_PARITY_EXCLUDE),
        "phase_lagged_metric_replacements": PHASE_LAGGED_METRIC_REPLACEMENTS,
        "phase_lagged_rule": (
            "Exclude phase-lagged iterate diagnostics from relative parity gating and "
            "replace them with the same-state leaf_iteration.* metrics."
        ),
        "absolute_policy_metrics": sorted(LOW_MAGNITUDE_ABSOLUTE_POLICY_METRICS),
        "absolute_policy_rule": (
            "Use max_abs rather than max_rel for low-magnitude canopy thermal component "
            "terms whose absolute errors stay negligible while relative errors are unstable."
        ),
        "nonconverged_energy_metric_prefixes": list(NONCONVERGED_ENERGY_PREFIXES),
        nonconverged_key: nonconverged_rule,
    }


def _discover_default_cases(repo_root: Path) -> list[int]:
    latin_hypercube = repo_root / "upstream" / "SCOPE" / "input" / "input_data_latin_hypercube.csv"
    with latin_hypercube.open(newline="", encoding="utf-8") as handle:
        max_cols = max(len(row) for row in csv.reader(handle))
    return list(range(1, max_cols))


def _case_id(case_index: int) -> str:
    return f"{case_index:03d}"


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_summary_path(path: Path, repo_root: Path) -> str | None:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return None


def _available_summary_keys(summary: dict[str, dict[str, dict[str, Any]]], keys: list[str]) -> list[str]:
    return [
        key
        for key in keys
        if key in summary["max_abs"] and key in summary["max_rel"]
    ]


def _summary_subset(summary: dict[str, dict[str, dict[str, Any]]], keys: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    selected = _available_summary_keys(summary, keys)
    return {
        "max_abs": {key: summary["max_abs"][key] for key in selected},
        "max_rel": {key: summary["max_rel"][key] for key in selected},
    }


def _worst_cases(per_case: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, dict[str, dict[str, Any]]]:
    worst_abs: dict[str, dict[str, Any]] = {}
    worst_rel: dict[str, dict[str, Any]] = {}
    for case_id, report in per_case.items():
        for section, metrics in report.items():
            for name, values in metrics.items():
                key = f"{section}.{name}"
                max_abs = float(values["max_abs"])
                max_rel = float(values["max_rel"])
                if key not in worst_abs or max_abs > worst_abs[key]["max_abs"]:
                    worst_abs[key] = {"case": case_id, **values}
                if key not in worst_rel or max_rel > worst_rel[key]["max_rel"]:
                    worst_rel[key] = {"case": case_id, **values}
    return {"max_abs": worst_abs, "max_rel": worst_rel}


def _filtered_worst_cases(
    per_case: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    exclude: set[str],
    nonconverged_energy_cases: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    nonconverged_energy_cases = set() if nonconverged_energy_cases is None else nonconverged_energy_cases
    filtered: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for case_id, report in per_case.items():
        case_report: dict[str, dict[str, dict[str, float]]] = {}
        for section, metrics in report.items():
            kept = {
                name: values
                for name, values in metrics.items()
                if f"{section}.{name}" not in exclude
                and not (
                    case_id in nonconverged_energy_cases
                    and any(f"{section}.{name}".startswith(prefix) for prefix in NONCONVERGED_ENERGY_PREFIXES)
                )
            }
            if kept:
                case_report[section] = kept
        filtered[case_id] = case_report
    return _worst_cases(filtered)


def _subset_worst_cases(
    per_case: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    include_cases: set[str],
    include_prefixes: tuple[str, ...] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    filtered: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for case_id, report in per_case.items():
        if case_id not in include_cases:
            continue
        case_report: dict[str, dict[str, dict[str, float]]] = {}
        for section, metrics in report.items():
            kept = {
                name: values
                for name, values in metrics.items()
                if include_prefixes is None or any(f"{section}.{name}".startswith(prefix) for prefix in include_prefixes)
            }
            if kept:
                case_report[section] = kept
        filtered[case_id] = case_report
    return _worst_cases(filtered)


def _highlight(summary: dict[str, dict[str, dict[str, Any]]], keys: list[str]) -> dict[str, dict[str, Any]]:
    highlights: dict[str, dict[str, Any]] = {}
    for key in keys:
        abs_entry = summary["max_abs"][key]
        rel_entry = summary["max_rel"][key]
        highlights[key] = {
            "worst_max_abs_case": abs_entry["case"],
            "worst_max_abs": abs_entry["max_abs"],
            "worst_max_rel_case": rel_entry["case"],
            "worst_max_rel": rel_entry["max_rel"],
            "worst_mean_abs_from_max_abs_case": abs_entry["mean_abs"],
        }
    return highlights


def _export_benchmarks(
    *,
    matlab: str,
    repo_root: Path,
    benchmark_dir: Path,
    cases: list[int],
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    statements = [f"addpath('{_matlab_quote(str(repo_root / 'scripts'))}')"]
    for case_index in cases:
        output_path = benchmark_dir / f"scope_case_{_case_id(case_index)}.mat"
        statements.append(
            f"export_scope_benchmark('{_matlab_quote(str(output_path))}', {case_index})"
        )
    command = "; ".join(statements)
    subprocess.run(
        [matlab, "-batch", command],
        check=True,
        cwd=repo_root,
    )


def _compare_benchmarks(
    *,
    repo_root: Path,
    benchmark_dir: Path,
    reports_dir: Path,
    cases: list[int],
    device: str,
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, dict[str, Any]]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    compare_script = repo_root / "scripts" / "compare_scope_benchmark.py"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"

    per_case: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    case_status: dict[str, dict[str, Any]] = {}
    for case_index in cases:
        case_id = _case_id(case_index)
        benchmark_path = benchmark_dir / f"scope_case_{case_id}.mat"
        report_path = reports_dir / f"scope_case_{case_id}_report.json"
        subprocess.run(
            [
                sys.executable,
                str(compare_script),
                "--benchmark",
                str(benchmark_path),
                "--report-json",
                str(report_path),
                "--device",
                device,
            ],
            check=True,
            cwd=repo_root,
            env=env,
            stdout=subprocess.DEVNULL,
        )
        loaded = _load_json(report_path)
        case_status[case_id] = loaded.pop("benchmark_status", {})
        per_case[case_id] = loaded
    return per_case, case_status


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_cases = _discover_default_cases(repo_root)
    parser = argparse.ArgumentParser(description="Export and compare multiple MATLAB SCOPE benchmark scenes.")
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=default_cases,
        help="Case indices to export and compare.",
    )
    parser.add_argument(
        "--matlab",
        default=DEFAULT_MATLAB,
        help="Path to the MATLAB executable.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "scope_torch_benchmark_suite",
        help="Directory to store exported MAT benchmark fixtures.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("tests/data/benchmark_suite_reports"),
        help="Directory to store per-case JSON comparison reports.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("tests/data/scope_benchmark_suite_summary.json"),
        help="JSON file for the aggregate benchmark suite summary.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use for the Python comparison runs.",
    )
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else repo_root / args.reports_dir
    summary_json = args.summary_json if args.summary_json.is_absolute() else repo_root / args.summary_json
    cases = sorted(set(args.cases))

    _export_benchmarks(
        matlab=args.matlab,
        repo_root=repo_root,
        benchmark_dir=benchmark_dir,
        cases=cases,
    )
    per_case, case_status = _compare_benchmarks(
        repo_root=repo_root,
        benchmark_dir=benchmark_dir,
        reports_dir=reports_dir,
        cases=cases,
        device=args.device,
    )
    worst = _worst_cases(per_case)
    nonconverged_energy_cases = {
        case_id
        for case_id, status in case_status.items()
        if status and not status.get("energy_converged", True)
    }
    parity_exclude = set(RELATIVE_PARITY_EXCLUDE)
    parity_worst = _filtered_worst_cases(
        per_case,
        exclude=parity_exclude,
        nonconverged_energy_cases=nonconverged_energy_cases,
    )
    absolute_policy_source = _filtered_worst_cases(
        per_case,
        exclude=set(PHASE_LAGGED_ENERGY_METRICS),
        nonconverged_energy_cases=nonconverged_energy_cases,
    )
    absolute_policy_worst = _summary_subset(
        absolute_policy_source,
        sorted(LOW_MAGNITUDE_ABSOLUTE_POLICY_METRICS),
    )
    stress_worst = _subset_worst_cases(
        per_case,
        include_cases=nonconverged_energy_cases,
        include_prefixes=NONCONVERGED_ENERGY_PREFIXES,
    )
    highlights = _highlight(
        parity_worst,
        keys=_available_summary_keys(
            parity_worst,
            [
                "reflectance.refl",
                "resistances_direct.raa",
                "resistances_direct.raws",
                "fluorescence_transport.EoutFrc_",
                "fluorescence_transport.sigmaF",
                "thermal_transport.Eoutte_",
                "leaf_iteration.sunlit_A",
                "leaf_iteration.shaded_A",
                "leaf_iteration.sunlit_rcw",
                "leaf_iteration.shaded_rcw",
                "energy_iteration_input.sunlit_Cs",
                "energy_iteration_input.shaded_Cs",
                "energy_balance.Rnuc_sw",
                "energy_balance.Rnhc_sw",
                "energy_balance.Rntot",
                "energy_balance.lEtot",
                "energy_balance.Htot",
                "energy_balance.Tcu",
                "energy_balance.Tch",
                "energy_balance.Tsu",
                "energy_balance.Tsh",
                "energy_balance.canopyemis",
                "energy_balance.L",
            ],
        ),
    )
    absolute_highlights = _highlight(
        absolute_policy_worst,
        keys=list(absolute_policy_worst["max_abs"]),
    )
    summary = {
        "cases": cases,
        "benchmark_dir": _stable_summary_path(benchmark_dir, repo_root),
        "reports_dir": _stable_summary_path(reports_dir, repo_root),
        "case_status": case_status,
        "nonconverged_energy_cases": sorted(nonconverged_energy_cases),
        "parity_policy": _parity_policy_metadata(
            nonconverged_key="nonconverged_energy_case_rule",
            nonconverged_rule=(
                "Exclude energy-balance and energy-iteration parity metrics for upstream "
                "scenes that hit ebal max iterations; retain them as stress diagnostics."
            ),
        ),
        "per_case": per_case,
        "worst_cases": worst,
        "parity_exclude": sorted(parity_exclude),
        "parity_worst_cases": parity_worst,
        "absolute_policy_worst_cases": absolute_policy_worst,
        "stress_worst_cases": stress_worst,
        "highlights": highlights,
        "absolute_highlights": absolute_highlights,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Compared cases: {', '.join(_case_id(case) for case in cases)}")
    for key, values in highlights.items():
        print(
            f"{key:<36} "
            f"worst_max_abs={values['worst_max_abs']:.6e} (case {values['worst_max_abs_case']}) "
            f"worst_max_rel={values['worst_max_rel']:.6e} (case {values['worst_max_rel_case']})"
        )
    if absolute_highlights:
        print("\nAbsolute-policy energy components:")
        for key, values in absolute_highlights.items():
            print(
                f"{key:<36} "
                f"worst_max_abs={values['worst_max_abs']:.6e} (case {values['worst_max_abs_case']}) "
                f"worst_max_rel={values['worst_max_rel']:.6e} (case {values['worst_max_rel_case']})"
            )
    print(f"\nWrote suite summary to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
