from __future__ import annotations

from pathlib import Path

from dataclasses import fields

from scope.canopy.fluorescence import (
    CanopyDirectionalFluorescenceResult,
    CanopyFluorescenceProfileResult,
    CanopyFluorescenceResult,
)
from scope.canopy.reflectance import (
    CanopyDirectionalReflectanceResult,
    CanopyRadiationProfileResult,
    CanopyReflectanceResult,
)
from scope.canopy.thermal import (
    CanopyDirectionalThermalResult,
    CanopyThermalProfileResult,
    CanopyThermalRadianceResult,
)
from scope.energy import CanopyEnergyBalanceResult
from scope.io.prepare import DEFAULT_SCOPE_OPTIONS
from scope.variables import render_variable_markdown, search_variables


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_search_variables_finds_exact_output() -> None:
    matches = search_variables("Rntot")

    assert matches
    assert matches[0].name == "Rntot"
    assert "Total net radiation" in matches[0].meaning


def test_search_variables_handles_prefixed_leaf_outputs() -> None:
    matches = search_variables("sunlit_rcw")
    names = {match.name for match in matches}

    assert "sunlit_rcw" in names
    assert "sunlit_*" in names


def test_variable_glossary_doc_is_generated_from_registry() -> None:
    expected = render_variable_markdown()
    actual = (REPO_ROOT / "docs" / "variable-glossary.md").read_text(encoding="utf-8")

    assert actual == expected


def test_variable_registry_covers_public_result_dataclass_fields() -> None:
    registry_names = {match.name for match in search_variables()}
    result_classes = (
        CanopyReflectanceResult,
        CanopyRadiationProfileResult,
        CanopyDirectionalReflectanceResult,
        CanopyFluorescenceResult,
        CanopyFluorescenceProfileResult,
        CanopyDirectionalFluorescenceResult,
        CanopyThermalRadianceResult,
        CanopyThermalProfileResult,
        CanopyDirectionalThermalResult,
        CanopyEnergyBalanceResult,
    )
    public_names = {
        field.name
        for cls in result_classes
        for field in fields(cls)
    }

    assert public_names.issubset(registry_names)


def test_variable_registry_covers_default_scope_option_names() -> None:
    registry_names = {match.name for match in search_variables(kind="option")}

    assert set(DEFAULT_SCOPE_OPTIONS).issubset(registry_names)
