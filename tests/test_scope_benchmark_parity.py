from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_scope_benchmark_locked_subsystems(tmp_path):
    if os.environ.get("SCOPE_BENCHMARK") != "1":
        pytest.skip("Set SCOPE_BENCHMARK=1 to run the MATLAB/Python benchmark parity check.")

    repo_root = Path(__file__).resolve().parents[1]
    benchmark = repo_root / "tests" / "data" / "scope_case_001.mat"
    if not benchmark.exists():
        pytest.skip(f"Benchmark fixture not found: {benchmark}")

    report_path = tmp_path / "scope_case_001_report.test.json"
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = "src" if not pythonpath else f"src{os.pathsep}{pythonpath}"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compare_scope_benchmark.py"),
            "--benchmark",
            str(benchmark),
            "--report-json",
            str(report_path),
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    leaf = report["leaf"]
    for name in ("refl", "tran", "Mb", "Mf"):
        assert leaf[name]["max_rel"] < 1e-9

    fluorescence_source = report["fluorescence_source"]
    for name in ("MpluEsun", "MminEsun", "piLs", "piLd", "Femmin", "Femplu", "Fmin_", "Fplu_"):
        assert fluorescence_source[name]["max_rel"] < 1e-9

    fluorescence_transport = report["fluorescence_transport"]
    assert fluorescence_transport["PoutFrc"]["max_rel"] < 1e-9
    for name in ("LoF_", "EoutF_", "Femleaves_", "LoF_sunlit", "LoF_shaded", "LoF_scattered", "LoF_soil"):
        assert fluorescence_transport[name]["max_rel"] < 1e-9
    for name in ("EoutFrc_", "sigmaF"):
        assert fluorescence_transport[name]["max_rel"] < 5e-5

    resistances = report["resistances_direct"]
    for name in ("ustar", "Kh", "rai", "rar", "rac", "rws", "raa", "raws"):
        assert resistances[name]["max_rel"] < 5e-3
    assert resistances["uz0"]["max_rel"] < 1e-2
    assert resistances["rawc"]["max_abs"] == 0.0
