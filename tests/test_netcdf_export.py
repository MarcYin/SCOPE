from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from scope_torch.io import (
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
        assert roundtrip.attrs["path_attr"] == "/tmp/demo.nc"
        assert roundtrip.attrs["tuple_attr"] == '["alpha", 1]'
        assert roundtrip.attrs["enabled"] == 1
        assert "unused" not in roundtrip.attrs
        assert roundtrip["foo"].attrs["meta"] == '{"source": "unit-test"}'
