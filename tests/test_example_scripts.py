from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_ROOT = REPO_ROOT / "upstream" / "SCOPE"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_json_close(actual: object, expected: object) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key, expected_value in expected.items():
            _assert_json_close(actual[key], expected_value)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_json_close(actual_item, expected_item)
        return
    if isinstance(expected, float):
        assert isinstance(actual, (int, float))
        assert math.isclose(float(actual), expected, rel_tol=1e-6, abs_tol=1e-8)
        return
    assert actual == expected


def _run_example(script_name: str, output_path: Path) -> object:
    if not SCOPE_ROOT.exists():
        pytest.skip("Upstream SCOPE assets are not available")
    command = [
        sys.executable,
        str(REPO_ROOT / "examples" / script_name),
        "--scope-root",
        str(SCOPE_ROOT),
        "--output",
        str(output_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True)
    return _load_json(output_path)


def test_basic_scene_reflectance_example_matches_checked_output(tmp_path: Path) -> None:
    actual = _run_example("basic_scene_reflectance.py", tmp_path / "basic_scene_reflectance.json")
    expected = _load_json(REPO_ROOT / "examples" / "output" / "basic_scene_reflectance.json")
    _assert_json_close(actual, expected)


def test_scope_workflow_demo_matches_checked_output(tmp_path: Path) -> None:
    actual = _run_example("scope_workflow_demo.py", tmp_path / "scope_workflow_demo.json")
    expected = _load_json(REPO_ROOT / "examples" / "output" / "scope_workflow_demo.json")
    _assert_json_close(actual, expected)


def test_render_example_visuals_script_bootstraps_src_path() -> None:
    if not SCOPE_ROOT.exists():
        pytest.skip("Upstream SCOPE assets are not available")

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_example_visuals.py"),
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=True, capture_output=True, text=True)

    assert "docs/assets/basic_scene_reflectance.svg" in completed.stdout
    assert "docs/assets/scope_workflow_fluorescence.svg" in completed.stdout
    assert "docs/assets/scope_workflow_profile.svg" in completed.stdout
