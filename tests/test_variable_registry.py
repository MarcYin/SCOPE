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
import xarray as xr

from scope.variables import (
    annotate_dataset,
    get_variable_definition,
    render_variable_markdown,
    render_workflow_variable_markdown,
    search_variables,
    variable_attrs,
)


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


def test_search_variables_exposes_relationship_notes() -> None:
    matches = search_variables("Rntot")

    assert matches
    assert matches[0].relationship == "Rntot = Rnctot + Rnstot"


def test_search_variables_filters_by_workflow_and_related_terms() -> None:
    fluorescence = search_variables(workflow="fluorescence")
    related = search_variables(related_to="Rntot")

    assert any(match.name == "LoF_" for match in fluorescence)
    assert any(match.name == "Rnctot" for match in related)


def test_variable_glossary_doc_is_generated_from_registry() -> None:
    expected = render_variable_markdown()
    actual = (REPO_ROOT / "docs" / "variable-glossary.md").read_text(encoding="utf-8")

    assert actual == expected


def test_workflow_variable_guides_are_generated_from_registry() -> None:
    for workflow in ("reflectance", "fluorescence", "thermal", "energy-balance"):
        expected = render_workflow_variable_markdown(workflow)
        actual = (REPO_ROOT / "docs" / "workflow-variables" / f"{workflow}.md").read_text(encoding="utf-8")

        assert actual == expected


def test_variable_attrs_and_dataset_annotation_include_registry_metadata() -> None:
    attrs = variable_attrs("Rntot")
    dataset = annotate_dataset(xr.Dataset({"Rntot": (("time",), [123.0])}, coords={"time": [0]}))

    assert attrs["long_name"] == "Total net radiation."
    assert attrs["scope_relationship"] == "Rntot = Rnctot + Rnstot"
    assert dataset["Rntot"].attrs["scope_source_doc"]
    assert dataset["time"].attrs["long_name"] == "Time axis for scenes or time-series runs."


def test_public_result_docstrings_are_registry_backed() -> None:
    assert "Fields:" in CanopyReflectanceResult.__doc__
    assert "`rsot`" in CanopyReflectanceResult.__doc__
    assert get_variable_definition("rsot") is not None


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
