from __future__ import annotations

from typing import Mapping

import xarray as xr

from ..variables import get_variable_definition


_BASE_REFLECTANCE_REQUIRED = ("Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi")
_SOIL_GROUPS = (("soil_refl",), ("soil_spectrum",), ("BSMBrightness", "BSMlat", "BSMlon", "SMC"))
_FLUORESCENCE_REQUIRED = _BASE_REFLECTANCE_REQUIRED + ("fqe", "Esun_", "Esky_")
_THERMAL_REQUIRED = ("LAI", "tts", "tto", "psi", "Tcu", "Tch", "Tsu", "Tsh")
_ENERGY_REQUIRED = (
    *_BASE_REFLECTANCE_REQUIRED,
    "Ta",
    "ea",
    "Ca",
    "Oa",
    "p",
    "z",
    "u",
    "Cd",
    "rwc",
    "z0m",
    "d",
    "h",
    "rss",
    "rbs",
    "Esun_sw",
    "Esky_sw",
)
_DIRECTIONAL_REQUIRED = ("directional_tto", "directional_psi")


def validate_scope_dataset(
    dataset: xr.Dataset,
    *,
    workflow: str = "scope",
    scope_options: Mapping[str, object] | None = None,
) -> None:
    """Validate a runner-ready dataset for a given workflow."""

    errors: list[str] = []
    required = set(_BASE_REFLECTANCE_REQUIRED)
    require_soil = workflow not in {"thermal", "directional-thermal", "thermal-profiles"}
    directional = workflow in {"directional-reflectance", "directional-fluorescence", "directional-thermal"}

    if workflow == "scope":
        options = {key: dataset.attrs.get(key) for key in ("calc_fluor", "calc_planck", "calc_directional", "calc_vert_profiles")}
        if scope_options:
            options.update(scope_options)
        if _as_bool(options.get("calc_fluor"), default=False):
            required.update(_FLUORESCENCE_REQUIRED)
        if _as_bool(options.get("calc_planck"), default=False):
            required.update(_THERMAL_REQUIRED)
        if _as_bool(options.get("calc_directional"), default=False):
            directional = True
    elif workflow in {"reflectance", "directional-reflectance", "reflectance-profiles"}:
        required.update(_BASE_REFLECTANCE_REQUIRED)
    elif workflow in {"fluorescence", "layered-fluorescence", "directional-fluorescence", "fluorescence-profiles"}:
        required.update(_FLUORESCENCE_REQUIRED)
    elif workflow in {"thermal", "directional-thermal", "thermal-profiles"}:
        required = set(_THERMAL_REQUIRED)
    elif workflow in {"biochemical-fluorescence", "energy-balance-fluorescence"}:
        required.update(_ENERGY_REQUIRED)
        required.update(_FLUORESCENCE_REQUIRED)
    elif workflow == "energy-balance-thermal":
        required.update(_ENERGY_REQUIRED)
    else:
        raise ValueError(f"Unsupported workflow '{workflow}'")

    for name in sorted(required):
        if name not in dataset:
            errors.append(f"Missing required variable {name}: {_meaning(name)}")

    if require_soil and not _has_any_group(dataset, _SOIL_GROUPS):
        soil_groups = " or ".join("+".join(group) for group in _SOIL_GROUPS)
        errors.append(f"Missing soil description. Provide one of: {soil_groups}.")
    _validate_partial_group(dataset, errors, ("BSMBrightness", "BSMlat", "BSMlon", "SMC"))

    if "Esun_" in dataset and "excitation_wavelength" not in dataset["Esun_"].dims:
        errors.append("Esun_ must include the 'excitation_wavelength' dimension.")
    if "Esky_" in dataset and "excitation_wavelength" not in dataset["Esky_"].dims:
        errors.append("Esky_ must include the 'excitation_wavelength' dimension.")
    for name in ("Esun_sw", "Esky_sw", "soil_refl"):
        if name in dataset and "wavelength" not in dataset[name].dims:
            errors.append(f"{name} must include the 'wavelength' dimension.")
    for name in ("etau", "etah", "fV"):
        if name in dataset and "layer" not in dataset[name].dims:
            errors.append(f"{name} must include the 'layer' dimension.")
    if directional:
        for name in _DIRECTIONAL_REQUIRED:
            if name not in dataset.coords and name not in dataset:
                errors.append(f"Directional workflow requires '{name}' as a 1D coordinate or variable.")
            else:
                source = dataset.coords[name] if name in dataset.coords else dataset[name]
                if source.ndim != 1 or "direction" not in source.dims:
                    errors.append(f"{name} must be one-dimensional on the 'direction' dimension.")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))


def _has_any_group(dataset: xr.Dataset, groups: tuple[tuple[str, ...], ...]) -> bool:
    return any(all(name in dataset for name in group) for group in groups)


def _validate_partial_group(dataset: xr.Dataset, errors: list[str], group: tuple[str, ...]) -> None:
    present = [name for name in group if name in dataset]
    if present and len(present) != len(group):
        missing = [name for name in group if name not in dataset]
        errors.append(
            f"Incomplete grouped input {'+'.join(group)}. Present: {', '.join(present)}. Missing: {', '.join(missing)}."
        )


def _meaning(name: str) -> str:
    item = get_variable_definition(name)
    return item.meaning if item is not None else "No glossary entry available."


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)
