from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
