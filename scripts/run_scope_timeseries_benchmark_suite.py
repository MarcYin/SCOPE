from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MATLAB = os.environ.get("MATLAB_BIN", "/Applications/MATLAB_R2025b.app/bin/matlab")
NONCONVERGED_ENERGY_PREFIXES = ("energy_balance.", "energy_iteration_input.")


def _load_scene_suite_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "run_scope_benchmark_suite.py"
    spec = importlib.util.spec_from_file_location("run_scope_benchmark_suite", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scene benchmark suite script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SCENE_SUITE = _load_scene_suite_module()


def _discover_default_steps(repo_root: Path) -> list[int]:
    dataset = repo_root / "upstream" / "SCOPE" / "input" / "dataset for_verification" / "input_data_latin_hypercube_ts.csv"
    with dataset.open(newline="", encoding="utf-8") as handle:
        nrows = sum(1 for _ in csv.reader(handle)) - 1
    return list(range(1, nrows + 1))


def _step_id(step_index: int) -> str:
    return f"{step_index:03d}"


def _matlab_quote(value: str) -> str:
    return value.replace("'", "''")


def _matlab_vector(values: list[int]) -> str:
    return "[" + " ".join(str(value) for value in values) + "]"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_highlight_keys(
    summary: dict[str, dict[str, dict[str, Any]]],
    keys: list[str],
) -> list[str]:
    return _SCENE_SUITE._available_summary_keys(summary, keys)


def _export_benchmarks(
    *,
    matlab: str,
    repo_root: Path,
    benchmark_dir: Path,
    steps: list[int],
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    command = "; ".join(
        [
            f"addpath('{_matlab_quote(str(repo_root / 'scripts'))}')",
            f"export_scope_timeseries_benchmarks('{_matlab_quote(str(benchmark_dir))}', {_matlab_vector(steps)})",
        ]
    )
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
    steps: list[int],
    device: str,
) -> tuple[dict[str, dict[str, dict[str, dict[str, float]]]], dict[str, dict[str, Any]]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    compare_script = repo_root / "scripts" / "compare_scope_benchmark.py"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"

    per_step: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    step_status: dict[str, dict[str, Any]] = {}
    for step_index in steps:
        step_id = _step_id(step_index)
        benchmark_path = benchmark_dir / f"scope_ts_step_{step_id}.mat"
        report_path = reports_dir / f"scope_ts_step_{step_id}_report.json"
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
        step_status[step_id] = loaded.pop("benchmark_status", {})
        per_step[step_id] = loaded
    return per_step, step_status


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_steps = _discover_default_steps(repo_root)
    parser = argparse.ArgumentParser(description="Export and compare MATLAB SCOPE time-series timesteps.")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=default_steps,
        help="Time-series step indices to export and compare.",
    )
    parser.add_argument(
        "--matlab",
        default=DEFAULT_MATLAB,
        help="Path to the MATLAB executable.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "scope_torch_timeseries_benchmark_suite",
        help="Directory to store exported MAT benchmark fixtures.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("tests/data/timeseries_benchmark_reports"),
        help="Directory to store per-step JSON comparison reports.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("tests/data/scope_timeseries_benchmark_summary.json"),
        help="JSON file for the aggregate time-series benchmark summary.",
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
    steps = sorted(set(args.steps))

    _export_benchmarks(
        matlab=args.matlab,
        repo_root=repo_root,
        benchmark_dir=benchmark_dir,
        steps=steps,
    )
    per_step, step_status = _compare_benchmarks(
        repo_root=repo_root,
        benchmark_dir=benchmark_dir,
        reports_dir=reports_dir,
        steps=steps,
        device=args.device,
    )

    worst = _SCENE_SUITE._worst_cases(per_step)
    nonconverged_energy_steps = {
        step_id
        for step_id, status in step_status.items()
        if status and not status.get("energy_converged", True)
    }
    parity_exclude = set(_SCENE_SUITE.RELATIVE_PARITY_EXCLUDE)
    parity_worst = _SCENE_SUITE._filtered_worst_cases(
        per_step,
        exclude=parity_exclude,
        nonconverged_energy_cases=nonconverged_energy_steps,
    )
    absolute_policy_source = _SCENE_SUITE._filtered_worst_cases(
        per_step,
        exclude=set(_SCENE_SUITE.PHASE_LAGGED_ENERGY_METRICS),
        nonconverged_energy_cases=nonconverged_energy_steps,
    )
    absolute_policy_worst = _SCENE_SUITE._summary_subset(
        absolute_policy_source,
        sorted(_SCENE_SUITE.LOW_MAGNITUDE_ABSOLUTE_POLICY_METRICS),
    )
    stress_worst = _SCENE_SUITE._subset_worst_cases(
        per_step,
        include_cases=nonconverged_energy_steps,
        include_prefixes=NONCONVERGED_ENERGY_PREFIXES,
    )

    highlight_keys = [
        "reflectance.refl",
        "fluorescence_transport.LoF_",
        "thermal_transport.Lot_",
        "energy_balance.Rntot",
        "energy_balance.lEtot",
        "energy_balance.Htot",
        "energy_balance.Tcu",
        "energy_balance.Tch",
    ]
    absolute_highlights = _SCENE_SUITE._highlight(
        absolute_policy_worst,
        list(absolute_policy_worst["max_abs"]),
    )
    summary = {
        "steps": [_step_id(step) for step in steps],
        "benchmark_dir": _SCENE_SUITE._stable_summary_path(benchmark_dir, repo_root),
        "reports_dir": _SCENE_SUITE._stable_summary_path(reports_dir, repo_root),
        "step_status": step_status,
        "nonconverged_energy_steps": sorted(nonconverged_energy_steps),
        "parity_policy": _SCENE_SUITE._parity_policy_metadata(
            nonconverged_key="nonconverged_energy_step_rule",
            nonconverged_rule=(
                "Exclude energy-balance and energy-iteration parity metrics for upstream "
                "time-series steps that hit ebal max iterations; retain them as stress diagnostics."
            ),
        ),
        "worst_cases": worst,
        "parity_worst_cases": parity_worst,
        "absolute_policy_worst_cases": absolute_policy_worst,
        "stress_worst_cases": stress_worst,
        "highlights": _SCENE_SUITE._highlight(parity_worst, _available_highlight_keys(parity_worst, highlight_keys)),
        "absolute_highlights": absolute_highlights,
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote time-series benchmark summary to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
