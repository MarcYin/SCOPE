from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_MATLAB = "/Applications/MATLAB_R2025b.app/bin/matlab"


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
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    compare_script = repo_root / "scripts" / "compare_scope_benchmark.py"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"

    per_case: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
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
        per_case[case_id] = _load_json(report_path)
    return per_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Export and compare multiple MATLAB SCOPE benchmark scenes.")
    parser.add_argument(
        "--cases",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
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

    repo_root = Path(__file__).resolve().parents[1]
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
    per_case = _compare_benchmarks(
        repo_root=repo_root,
        benchmark_dir=benchmark_dir,
        reports_dir=reports_dir,
        cases=cases,
        device=args.device,
    )
    worst = _worst_cases(per_case)
    highlights = _highlight(
        worst,
        keys=[
            "reflectance.refl",
            "fluorescence_transport.EoutFrc_",
            "fluorescence_transport.sigmaF",
            "thermal_transport.Eoutte_",
            "energy_balance.Rnuc_sw",
            "energy_balance.Rnhc_sw",
            "energy_balance.Rnuct",
            "energy_balance.Rnhct",
            "energy_balance.Rntot",
            "energy_balance.lEtot",
            "energy_balance.Htot",
            "energy_balance.Tcu",
            "energy_balance.Tch",
            "energy_balance.Tsu",
            "energy_balance.Tsh",
            "energy_balance.canopyemis",
            "energy_balance.sunlit_A",
            "energy_balance.shaded_A",
            "energy_balance.sunlit_rcw",
            "energy_balance.shaded_rcw",
            "energy_balance.L",
        ],
    )
    summary = {
        "cases": cases,
        "benchmark_dir": _stable_summary_path(benchmark_dir, repo_root),
        "reports_dir": _stable_summary_path(reports_dir, repo_root),
        "per_case": per_case,
        "worst_cases": worst,
        "highlights": highlights,
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
    print(f"\nWrote suite summary to {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
