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

    reflectance = report["reflectance"]
    for name in ("rsd", "rdd", "rdo", "rso", "refl"):
        assert reflectance[name]["max_rel"] < 1e-9

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

    thermal_transport = report["thermal_transport"]
    for name in ("Lot_", "Eoutte_", "Loutt", "Eoutt"):
        assert thermal_transport[name]["max_rel"] < 1e-9

    energy = report["energy_balance"]
    for name in ("Rnuc_sw", "Rnhc_sw", "Rnus_sw", "Rnhs_sw"):
        assert energy[name]["max_rel"] < 5e-3
    assert energy["canopyemis"]["max_rel"] < 1e-6

    for name in ("sunlit_eta", "shaded_eta", "sunlit_Ci", "shaded_Ci", "sunlit_rcw", "shaded_rcw"):
        assert energy[name]["max_rel"] < 1e-3
    assert energy["sunlit_A"]["max_rel"] < 1e-3
    assert energy["shaded_A"]["max_abs"] < 1e-3
    assert energy["Tcu"]["max_abs"] < 2e-2
    assert energy["Tch"]["max_abs"] < 1e-2
    assert energy["Tsu"]["max_abs"] < 5e-2
    assert energy["Tsh"]["max_abs"] < 5e-3

    assert energy["Rnuct"]["max_rel"] < 2e-2
    assert energy["Rnhct"]["max_rel"] < 1e-1
    assert energy["Rnuc"]["max_rel"] < 1e-2
    assert energy["Rnhc"]["max_rel"] < 2e-2
    assert energy["Rnus"]["max_rel"] < 5e-3
    assert energy["Rnhs"]["max_rel"] < 1e-3

    assert energy["L"]["max_abs"] < 1e-1
    assert energy["counter"]["max_abs"] == 0.0
    for name in ("Rnctot", "lEctot", "Hctot", "Actot", "Tcave", "Rnstot", "lEstot", "Hstot", "Gtot", "Tsave", "Rntot", "lEtot", "Htot", "raa", "raws", "ustar"):
        assert energy[name]["max_rel"] < 5e-3
