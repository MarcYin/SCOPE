from __future__ import annotations

import subprocess

from tests._matlab_benchmark_helpers import REPO_ROOT, compare_benchmark_fixture, has_live_matlab, matlab_bin


def test_scope_benchmark_locked_subsystems(tmp_path):
    benchmark = REPO_ROOT / "tests" / "data" / "scope_case_001.mat"
    if has_live_matlab():
        benchmark = tmp_path / "scope_case_001.live.mat"
        scripts_path = str(REPO_ROOT / "scripts").replace("'", "''")
        benchmark_path = str(benchmark).replace("'", "''")
        subprocess.run(
            [
                matlab_bin(),
                "-batch",
                f"addpath('{scripts_path}'); export_scope_benchmark('{benchmark_path}', 1)",
            ],
            check=True,
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
        )
    else:
        assert benchmark.exists(), f"Pregenerated benchmark fixture not found: {benchmark}"

    report_path = tmp_path / "scope_case_001_report.test.json"
    report = compare_benchmark_fixture(benchmark, report_path)
    status = report["benchmark_status"]
    assert status["energy_converged"] is True
    assert status["energy_hit_max_iterations"] is False

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
        assert resistances[name]["max_rel"] < 1e-9
    assert resistances["uz0"]["max_rel"] < 1e-9
    assert resistances["rawc"]["max_abs"] == 0.0

    iteration_inputs = report["energy_iteration_input"]
    for name in ("sunlit_Cs", "shaded_Cs", "sunlit_eb", "shaded_eb", "sunlit_T", "shaded_T"):
        assert iteration_inputs[name]["max_rel"] < 1e-3

    leaf_iteration = report["leaf_iteration"]
    for name in (
        "sunlit_A",
        "shaded_A",
        "sunlit_Ci",
        "shaded_Ci",
        "sunlit_rcw",
        "shaded_rcw",
        "sunlit_eta",
        "shaded_eta",
    ):
        assert leaf_iteration[name]["max_rel"] < 1e-9

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
    for name in (
        "Rnctot",
        "lEctot",
        "Hctot",
        "Actot",
        "Tcave",
        "Rnstot",
        "lEstot",
        "Hstot",
        "Gtot",
        "Tsave",
        "Rntot",
        "lEtot",
        "Htot",
        "raa",
        "raws",
        "ustar",
    ):
        assert energy[name]["max_rel"] < 5e-3

    scope_workflow = report["scope_workflow"]
    for name in ("rso", "rsd", "rdd", "rdo", "refl"):
        assert scope_workflow[name]["max_rel"] < 1e-9
    for name in (
        "LoF_",
        "EoutF_",
        "Femleaves_",
        "LoF_sunlit",
        "LoF_shaded",
        "LoF_scattered",
        "LoF_soil",
        "Fmin_",
        "Fplu_",
    ):
        assert scope_workflow[name]["max_rel"] < 1e-4
    for name in ("EoutFrc_", "sigmaF"):
        assert scope_workflow[name]["max_rel"] < 5e-5
    for name in ("Lot_", "Eoutte_", "Loutt", "Eoutt"):
        assert scope_workflow[name]["max_rel"] < 1e-5
    for name in ("sunlit_eta", "shaded_eta", "sunlit_Ci", "shaded_Ci", "sunlit_rcw", "shaded_rcw"):
        assert scope_workflow[name]["max_rel"] < 1e-3
    assert scope_workflow["sunlit_A"]["max_rel"] < 1e-3
    assert scope_workflow["shaded_A"]["max_abs"] < 1e-3
    assert scope_workflow["Tcu"]["max_abs"] < 2e-2
    assert scope_workflow["Tch"]["max_abs"] < 1e-2
    assert scope_workflow["Tsu"]["max_abs"] < 5e-2
    assert scope_workflow["Tsh"]["max_abs"] < 5e-3
    for name in ("Rntot", "lEtot", "Htot"):
        assert scope_workflow[name]["max_rel"] < 5e-3
    assert scope_workflow["counter"]["max_abs"] == 0.0
