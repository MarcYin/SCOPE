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
