from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scope import validate_scope_dataset


def _base_dataset() -> xr.Dataset:
    return xr.Dataset(
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
        coords={"y": [0], "x": [0], "time": pd.date_range("2020-07-01T12:00:00", periods=1, freq="h")},
        attrs={"calc_fluor": 0, "calc_planck": 0, "calc_directional": 0, "calc_vert_profiles": 0},
    )


def _coupled_energy_dataset() -> xr.Dataset:
    return (
        _base_dataset()
        .assign(
            Esun_sw=(("y", "x", "time", "wavelength"), np.full((1, 1, 1, 3), 900.0)),
            Esky_sw=(("y", "x", "time", "wavelength"), np.full((1, 1, 1, 3), 120.0)),
            Ta=(("y", "x", "time"), np.array([[[25.0]]])),
            ea=(("y", "x", "time"), np.array([[[20.0]]])),
            Ca=(("y", "x", "time"), np.array([[[390.0]]])),
            Oa=(("y", "x", "time"), np.array([[[209.0]]])),
            p=(("y", "x", "time"), np.array([[[970.0]]])),
            z=(("y", "x", "time"), np.array([[[10.0]]])),
            u=(("y", "x", "time"), np.array([[[2.0]]])),
            Cd=(("y", "x", "time"), np.array([[[0.2]]])),
            rwc=(("y", "x", "time"), np.array([[[0.5]]])),
            z0m=(("y", "x", "time"), np.array([[[0.15]]])),
            d=(("y", "x", "time"), np.array([[[1.3]]])),
            h=(("y", "x", "time"), np.array([[[2.0]]])),
            rss=(("y", "x", "time"), np.array([[[120.0]]])),
            rbs=(("y", "x", "time"), np.array([[[12.0]]])),
        )
        .assign_coords(wavelength=np.arange(3))
    )


def test_validate_scope_dataset_accepts_minimal_reflectance_case() -> None:
    validate_scope_dataset(_base_dataset(), workflow="reflectance")


def test_validate_scope_dataset_rejects_missing_required_variable() -> None:
    dataset = _base_dataset().drop_vars("Cab")

    with pytest.raises(ValueError, match="Missing required variable Cab"):
        validate_scope_dataset(dataset, workflow="reflectance")


def test_validate_scope_dataset_rejects_incomplete_bsm_group() -> None:
    dataset = _base_dataset().drop_vars("soil_spectrum")
    dataset["BSMBrightness"] = (("y", "x", "time"), np.array([[[0.5]]]))

    with pytest.raises(ValueError, match="Incomplete grouped input BSMBrightness\\+BSMlat\\+BSMlon\\+SMC"):
        validate_scope_dataset(dataset, workflow="reflectance")


def test_validate_scope_dataset_checks_scope_directional_requirements() -> None:
    dataset = _base_dataset()
    dataset.attrs["calc_directional"] = 1

    with pytest.raises(ValueError, match="directional_tto"):
        validate_scope_dataset(dataset, workflow="scope")


def test_validate_scope_dataset_checks_spectral_dimensions() -> None:
    dataset = _base_dataset()
    dataset["fqe"] = (("y", "x", "time"), np.array([[[0.01]]]))
    dataset["Esun_"] = (("y", "x", "time"), np.array([[[1.0]]]))
    dataset["Esky_"] = (("y", "x", "time"), np.array([[[0.2]]]))

    with pytest.raises(ValueError, match="excitation_wavelength"):
        validate_scope_dataset(dataset, workflow="fluorescence")


def test_validate_scope_dataset_ignores_longwave_dimensions_for_reflectance_only() -> None:
    dataset = _base_dataset()
    dataset["Esun_lw"] = (("y", "x", "time"), np.array([[[1.0]]]))

    validate_scope_dataset(dataset, workflow="reflectance")


def test_validate_scope_dataset_checks_longwave_dimensions_for_coupled_workflows() -> None:
    dataset = _coupled_energy_dataset()
    dataset["Esun_lw"] = (("y", "x", "time"), np.array([[[1.0]]]))
    dataset.attrs["calc_ebal"] = 1

    with pytest.raises(ValueError, match="thermal_wavelength"):
        validate_scope_dataset(dataset, workflow="scope")


def test_validate_scope_dataset_scope_calc_ebal_uses_coupled_requirements() -> None:
    dataset = _coupled_energy_dataset()
    dataset.attrs.update({"calc_ebal": 1, "calc_planck": 1})

    validate_scope_dataset(dataset, workflow="scope")


def test_validate_scope_dataset_scope_coupled_fluorescence_requires_fqe() -> None:
    dataset = _coupled_energy_dataset()
    dataset.attrs.update({"calc_ebal": 1, "calc_fluor": 1})

    with pytest.raises(ValueError, match="Missing required variable fqe"):
        validate_scope_dataset(dataset, workflow="scope")

    dataset["fqe"] = (("y", "x", "time"), np.array([[[0.01]]]))
    validate_scope_dataset(dataset, workflow="scope")


def test_validate_scope_dataset_energy_balance_fluorescence_requires_fqe_but_not_excitation_inputs() -> None:
    dataset = _coupled_energy_dataset()

    with pytest.raises(ValueError, match="Missing required variable fqe"):
        validate_scope_dataset(dataset, workflow="energy-balance-fluorescence")

    dataset["fqe"] = (("y", "x", "time"), np.array([[[0.01]]]))
    validate_scope_dataset(dataset, workflow="energy-balance-fluorescence")
