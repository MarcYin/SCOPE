from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scope.io import (
    NetCDFWriteOptions,
    available_netcdf_engines,
    build_netcdf_encoding,
    resolve_netcdf_engine,
    write_netcdf_dataset,
)


def test_resolve_netcdf_engine_returns_available_backend():
    available = available_netcdf_engines()

    assert available
    assert "scipy" in available
    assert resolve_netcdf_engine() in available
    assert resolve_netcdf_engine("scipy") == "scipy"


def test_build_netcdf_encoding_skips_compression_for_scipy():
    dataset = xr.Dataset(
        {"foo": (("time",), np.array([1.0, 2.0], dtype=np.float64))},
        coords={"time": pd.date_range("2020-01-01", periods=2, freq="D")},
    )

    encoding = build_netcdf_encoding(
        dataset,
        options=NetCDFWriteOptions(engine="scipy", compression=True),
    )

    assert encoding == {}


def test_write_netcdf_dataset_roundtrips_and_sanitises_attrs(tmp_path: Path):
    dataset = xr.Dataset(
        {"foo": (("time",), np.array([1.0, 2.0], dtype=np.float64))},
        coords={"time": pd.date_range("2020-01-01", periods=2, freq="D")},
        attrs={
            "path_attr": Path("/tmp/demo.nc"),
            "tuple_attr": ("alpha", 1),
            "enabled": True,
            "unused": None,
        },
    )
    dataset["foo"].attrs["meta"] = {"source": "unit-test"}

    output_path = write_netcdf_dataset(
        dataset,
        tmp_path / "roundtrip.nc",
        options=NetCDFWriteOptions(engine="scipy"),
    )

    with xr.open_dataset(output_path) as roundtrip:
        assert np.allclose(roundtrip["foo"].values, dataset["foo"].values)
        assert pd.DatetimeIndex(roundtrip["time"].values).tolist() == pd.DatetimeIndex(dataset["time"].values).tolist()
        assert roundtrip.attrs["Conventions"] == "CF-1.10"
        assert roundtrip.attrs["title"] == "SCOPE-RTM dataset"
        assert roundtrip.attrs["source"].startswith("SCOPE-RTM ")
        assert "scope.write_netcdf_dataset" in roundtrip.attrs["history"]
        assert "scope-model.readthedocs.io" in roundtrip.attrs["references"]
        assert roundtrip.attrs["path_attr"] == "/tmp/demo.nc"
        assert roundtrip.attrs["tuple_attr"] == '["alpha", 1]'
        assert roundtrip.attrs["enabled"] == 1
        assert "unused" not in roundtrip.attrs
        assert roundtrip["foo"].attrs["meta"] == '{"source": "unit-test"}'
        assert roundtrip["time"].attrs["standard_name"] == "time"
        assert roundtrip["time"].attrs["axis"] == "T"


def test_write_netcdf_dataset_appends_to_array_history_attr(tmp_path: Path):
    dataset = xr.Dataset(
        {"foo": (("time",), np.array([1.0, 2.0], dtype=np.float64))},
        coords={"time": pd.date_range("2020-01-01", periods=2, freq="D")},
        attrs={"history": np.array(["prepared", "validated"])},
    )

    output_path = write_netcdf_dataset(
        dataset,
        tmp_path / "history_roundtrip.nc",
        options=NetCDFWriteOptions(engine="scipy"),
    )

    with xr.open_dataset(output_path, engine="scipy") as roundtrip:
        assert roundtrip.attrs["history"].startswith('["prepared", "validated"]\n')
        assert "scope.write_netcdf_dataset" in roundtrip.attrs["history"]


def test_write_netcdf_dataset_scipy_omits_unlimited_dims_for_runner_style_layout(tmp_path: Path):
    dataset = xr.Dataset(
        {
            "rsot": (
                ("y", "x", "time", "wavelength"),
                np.ones((1, 1, 1, 4), dtype=np.float64),
            )
        },
        coords={
            "y": np.array([0.0]),
            "x": np.array([0.0]),
            "time": pd.date_range("2020-01-01", periods=1, freq="D"),
            "wavelength": np.array([500.0, 650.0, 865.0, 1600.0]),
        },
    )

    output_path = write_netcdf_dataset(
        dataset,
        tmp_path / "runner_style_scipy.nc",
        options=NetCDFWriteOptions(engine="scipy"),
    )

    with xr.open_dataset(output_path, engine="scipy") as roundtrip:
        assert np.allclose(roundtrip["rsot"].values, dataset["rsot"].values)
        assert tuple(roundtrip["rsot"].dims) == ("y", "x", "time", "wavelength")
        assert roundtrip["x"].attrs["axis"] == "X"
        assert roundtrip["y"].attrs["axis"] == "Y"
