from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scope.variables import render_variable_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_ROOT = REPO_ROOT / "upstream" / "SCOPE"


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def test_fetch_upstream_wrapper_bootstraps_src_path():
    script = REPO_ROOT / "scripts" / "fetch_upstream_scope.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Fetch the pinned upstream MATLAB SCOPE repository" in completed.stdout


def test_prepare_scope_input_wrapper_bootstraps_src_path():
    script = REPO_ROOT / "prepare_scope_input.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Build a runner-ready SCOPE input dataset" in completed.stdout


def test_render_variable_glossary_wrapper_bootstraps_src_path():
    script = REPO_ROOT / "scripts" / "render_variable_glossary.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    glossary_path = REPO_ROOT / "docs" / "variable-glossary.md"
    assert str(glossary_path) in completed.stdout
    assert glossary_path.read_text(encoding="utf-8") == render_variable_markdown()


def test_scope_module_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "scope", "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Top-level command-line interface for SCOPE-RTM." in completed.stdout
    assert "fetch-upstream" in completed.stdout
    assert "prepare" in completed.stdout
    assert "run" in completed.stdout
    assert "vars" in completed.stdout


def test_scope_module_cli_subcommand_help():
    completed = subprocess.run(
        [sys.executable, "-m", "scope", "fetch-upstream", "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Fetch the pinned upstream MATLAB SCOPE repository" in completed.stdout


def test_scope_run_entry_point_help():
    scope_run = Path(sys.executable).parent / "scope-run"
    completed = subprocess.run(
        [str(scope_run), "--help"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Run a prepared SCOPE input dataset" in completed.stdout


def test_scope_module_variable_search_cli():
    completed = subprocess.run(
        [sys.executable, "-m", "scope", "vars", "Rntot"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Rntot" in completed.stdout
    assert "Total net radiation" in completed.stdout


def test_scope_module_run_cli_executes_minimal_reflectance_workflow(tmp_path: Path):
    if not SCOPE_ROOT.exists():
        pytest.skip("Upstream SCOPE assets are not available")

    dataset = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.array([[[45.0]]])),
            "Cw": (("y", "x", "time"), np.array([[[0.010]]])),
            "Cdm": (("y", "x", "time"), np.array([[[0.012]]])),
            "LAI": (("y", "x", "time"), np.array([[[2.2]]])),
            "tts": (("y", "x", "time"), np.array([[[30.0]]])),
            "tto": (("y", "x", "time"), np.array([[[20.0]]])),
            "psi": (("y", "x", "time"), np.array([[[15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0]]])),
        },
        coords={"y": [0.0], "x": [0.0], "time": pd.date_range("2020-07-01T12:00:00", periods=1, freq="h")},
        attrs={"example": "cli_run_reflectance"},
    )
    input_path = tmp_path / "scope_inputs.nc"
    output_path = tmp_path / "scope_outputs.nc"
    summary_path = tmp_path / "scope_summary.json"
    dataset.to_netcdf(input_path, engine="scipy")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scope",
            "run",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-json",
            str(summary_path),
            "--scope-root",
            str(SCOPE_ROOT),
            "--workflow",
            "reflectance",
            "--dtype",
            "float64",
            "--chunk-size",
            "1",
            "--netcdf-engine",
            "scipy",
            "--no-compression",
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(output_path.resolve()) in completed.stdout
    with xr.open_dataset(output_path, engine="scipy") as outputs:
        assert outputs.attrs["scope_product"] == "reflectance"
        assert "rsot" in outputs
        assert outputs["rsot"].sizes["wavelength"] > 100
    assert summary_path.exists()
