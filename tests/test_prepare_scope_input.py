from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scope_torch.io.prepare as prepare_module
from scope_torch.io import ScopeInputFiles, derive_observation_time_grid, prepare_scope_input_dataset, read_s2_bio_inputs


def test_read_s2_bio_inputs_scales_and_deduplicates_times(tmp_path: Path):
    mask = np.ones((2, 1), dtype=np.uint8)
    doys = np.array([150, 150, 151], dtype=np.int32)
    post_bio_tensor = np.arange(2 * 1 * 7 * 3, dtype=np.float64)
    scale_params = np.arange(2 * 1 * 15, dtype=np.float64)
    geotransform = np.array([100.0, 10.0, 0.0, 200.0, 0.0, -10.0], dtype=np.float64)
    npz_path = tmp_path / "bio.npz"
    np.savez(
        npz_path,
        doys=doys,
        post_bio_tensor=post_bio_tensor,
        mask=mask,
        geotransform=geotransform,
        dat=scale_params,
        crs="EPSG:32616",
    )

    post_bio_da, post_bio_scale_da = read_s2_bio_inputs(npz_path, year=2020)

    assert post_bio_da.dims == ("y", "x", "band", "time")
    assert post_bio_da.sizes["time"] == 2
    assert pd.DatetimeIndex(post_bio_da["time"].values).tolist() == [
        pd.Timestamp("2020-05-29"),
        pd.Timestamp("2020-05-30"),
    ]
    assert np.isclose(post_bio_da.sel(y=200.0, x=100.0, band="N", time="2020-05-29").item(), 0.0)
    assert np.isclose(post_bio_da.sel(y=200.0, x=100.0, band="cab", time="2020-05-29").item(), 0.03)
    assert np.isclose(post_bio_da.sel(y=190.0, x=100.0, band="cm", time="2020-05-30").item(), 0.0029)
    assert post_bio_scale_da.dims == ("y", "x", "band")
    assert np.array_equal(post_bio_scale_da["band"].values, np.array(["N", "cab", "cm", "cw", "lai", "ala", "cbrown", "n0", "m0", "n1", "m1", "BSMBrightness", "BSMlat", "BSMlon", "SMC"], dtype=object))
    assert np.isclose(post_bio_scale_da.sel(y=190.0, x=100.0, band="SMC").item(), 29.0)


def test_derive_observation_time_grid_drops_missing_delta_time():
    times = pd.date_range("2020-06-01", periods=3, freq="h")
    values = np.array(
        [
            [
                [np.datetime64("2020-06-01T00:15:00"), np.datetime64("NaT"), np.datetime64("2020-06-01T02:15:00")],
                [np.datetime64("2020-06-01T00:45:00"), np.datetime64("NaT"), np.datetime64("2020-06-01T02:45:00")],
            ]
        ],
        dtype="datetime64[ns]",
    )
    observation = xr.Dataset(
        {"delta_time": (("y", "x", "time"), values)},
        coords={"y": [0], "x": [0, 1], "time": times},
    )

    time_grid = derive_observation_time_grid(observation)

    assert time_grid.dims == ("time",)
    assert pd.DatetimeIndex(time_grid.values).tolist() == [
        pd.Timestamp("2020-06-01T00:30:00"),
        pd.Timestamp("2020-06-01T02:30:00"),
    ]
    assert pd.DatetimeIndex(time_grid["time"].values).tolist() == [
        pd.Timestamp("2020-06-01T00:00:00"),
        pd.Timestamp("2020-06-01T02:00:00"),
    ]


def test_prepare_scope_input_dataset_builds_runner_ready_dataset():
    weather_times = pd.date_range("2020-06-01", periods=3, freq="h")
    observation_times = pd.date_range("2020-06-01", periods=3, freq="h")
    target_times = pd.to_datetime(["2020-06-01T00:30:00", "2020-06-01T01:30:00"])
    y = np.array([10.0, 20.0])
    x = np.array([100.0, 110.0])

    weather = xr.Dataset(
        {
            "Rin": (("y", "x", "time"), np.stack([np.full((2, 2), 100.0), np.full((2, 2), 200.0), np.full((2, 2), 300.0)], axis=-1)),
            "Rli": (("y", "x", "time"), np.stack([np.full((2, 2), 10.0), np.full((2, 2), 20.0), np.full((2, 2), 30.0)], axis=-1)),
            "Ta": (("y", "x", "time"), np.stack([np.full((2, 2), 290.0), np.full((2, 2), 292.0), np.full((2, 2), 294.0)], axis=-1)),
            "ea": (("y", "x", "time"), np.stack([np.full((2, 2), 1000.0), np.full((2, 2), 1100.0), np.full((2, 2), 1200.0)], axis=-1)),
            "p": (("y", "x", "time"), np.stack([np.full((2, 2), 101000.0), np.full((2, 2), 101100.0), np.full((2, 2), 101200.0)], axis=-1)),
            "u": (("y", "x", "time"), np.stack([np.full((2, 2), 2.0), np.full((2, 2), 3.0), np.full((2, 2), 4.0)], axis=-1)),
        },
        coords={"y": y, "x": x, "time": weather_times},
    )
    observation = xr.Dataset(
        {
            "delta_time": (
                ("y", "x", "time"),
                np.array(
                    [
                        [
                            [np.datetime64("2020-06-01T00:15:00"), np.datetime64("NaT"), np.datetime64("2020-06-01T01:15:00")],
                            [np.datetime64("2020-06-01T00:45:00"), np.datetime64("NaT"), np.datetime64("2020-06-01T01:45:00")],
                        ],
                        [
                            [np.datetime64("2020-06-01T00:15:00"), np.datetime64("NaT"), np.datetime64("2020-06-01T01:15:00")],
                            [np.datetime64("2020-06-01T00:45:00"), np.datetime64("NaT"), np.datetime64("2020-06-01T01:45:00")],
                        ],
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            "solar_zenith_angle": (("y", "x", "time"), np.full((2, 2, 3), 30.0)),
            "viewing_zenith_angle": (("y", "x", "time"), np.full((2, 2, 3), 20.0)),
            "solar_azimuth_angle": (("y", "x", "time"), np.full((2, 2, 3), 35.0)),
            "viewing_azimuth_angle": (("y", "x", "time"), np.full((2, 2, 3), 95.0)),
        },
        coords={"y": y, "x": x, "time": observation_times},
    )
    post_bio = xr.DataArray(
        np.array(
            [
                [
                    [[1.5, 1.7], [40.0, 44.0], [0.010, 0.011], [0.020, 0.021], [2.0, 2.2], [50.0, 52.0], [0.1, 0.2]],
                    [[1.6, 1.8], [42.0, 46.0], [0.012, 0.013], [0.022, 0.023], [2.4, 2.6], [48.0, 50.0], [0.2, 0.3]],
                ],
                [
                    [[1.4, 1.6], [38.0, 40.0], [0.011, 0.012], [0.019, 0.020], [1.8, 2.0], [47.0, 49.0], [0.0, 0.1]],
                    [[1.7, 1.9], [43.0, 47.0], [0.013, 0.014], [0.024, 0.025], [2.7, 2.9], [51.0, 53.0], [0.3, 0.4]],
                ],
            ],
            dtype=np.float64,
        ),
        dims=("y", "x", "band", "time"),
        coords={"y": y, "x": x, "band": ["N", "cab", "cm", "cw", "lai", "ala", "cbrown"], "time": target_times},
    )
    post_bio_scale = xr.DataArray(
        np.array(
            [
                [[1.5, 40.0, 0.010, 0.020, 2.0, 50.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.6, 45.0, 90.0, 0.2], [1.6, 42.0, 0.012, 0.022, 2.4, 48.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.7, 46.0, 91.0, 0.3]],
                [[1.4, 38.0, 0.011, 0.019, 1.8, 47.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 47.0, 92.0, 0.4], [1.7, 43.0, 0.013, 0.024, 2.7, 51.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.9, 48.0, 93.0, 0.5]],
            ],
            dtype=np.float64,
        ),
        dims=("y", "x", "band"),
        coords={"y": y, "x": x, "band": ["N", "cab", "cm", "cw", "lai", "ala", "cbrown", "n0", "m0", "n1", "m1", "BSMBrightness", "BSMlat", "BSMlon", "SMC"]},
    )

    prepared = prepare_scope_input_dataset(weather, observation, post_bio, post_bio_scale)

    expected_vars = {
        "Rin",
        "Rli",
        "Ta",
        "ea",
        "p",
        "u",
        "LAI",
        "N",
        "Cab",
        "Cca",
        "Cdm",
        "Cw",
        "ala",
        "Cs",
        "BSMBrightness",
        "BSMlat",
        "BSMlon",
        "SMC",
        "tts",
        "tto",
        "psi",
    }
    assert expected_vars.issubset(prepared.data_vars)
    assert prepared["Rin"].dims == ("y", "x", "time")
    assert pd.DatetimeIndex(prepared["time"].values).tolist() == target_times.tolist()
    assert np.allclose(prepared["psi"].values, 60.0)
    assert np.allclose(prepared["Cca"].values, prepared["Cab"].values * 0.25)
    assert np.allclose(prepared["BSMBrightness"].isel(time=0).values, np.array([[0.6, 0.7], [0.8, 0.9]]))
    assert np.allclose(prepared["Rin"].isel(time=0).values, 150.0)
    assert np.allclose(prepared["Rin"].isel(time=1).values, 250.0)
    assert prepared.attrs["calc_fluor"] == 1
    assert prepared.attrs["soil_heat_method"] == 2
    assert prepared.attrs["optipar_file"].endswith(".mat")
    assert prepared.attrs["soil_file"].endswith(".txt")
    assert prepared.attrs["atmos_file"].endswith(".atm")


def test_scope_input_files_requires_nonempty_default_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "input" / "fluspect_parameters").mkdir(parents=True)

    monkeypatch.setattr(
        prepare_module,
        "load_scope_filenames",
        lambda root: {"optipar_file": "", "soil_file": "soilnew.txt", "atmos_file": "FLEX-S3_std.atm"},
    )

    with pytest.raises(FileNotFoundError, match="optipar_file"):
        ScopeInputFiles().resolve(scope_root_path=tmp_path)
