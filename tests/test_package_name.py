from __future__ import annotations

import importlib
from importlib.metadata import version

import pytest


def test_scope_is_the_public_package_name() -> None:
    scope = importlib.import_module("scope")
    assert scope.__name__ == "scope"
    assert hasattr(scope, "ScopeGridRunner")


def test_distribution_name_is_scope_rtm() -> None:
    scope = importlib.import_module("scope")
    assert version("SCOPE-RTM") == scope.__version__


def test_scope_torch_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("scope_torch")
