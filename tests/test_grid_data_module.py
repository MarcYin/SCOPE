import numpy as np
import pandas as pd
import torch
import xarray as xr

from scope_torch.config import SimulationConfig
from scope_torch.data import ScopeGridDataModule


def test_grid_module_batches():
    times = pd.date_range("2020-06-01", periods=4, freq="h")
    x = np.arange(2)
    y = np.arange(3)
    data = xr.Dataset(
        {
            "Rin": (("y", "x", "time"), np.random.rand(3, 2, 4)),
            "LAI": (("y", "x", "time"), np.random.rand(3, 2, 4)),
        },
        coords={"y": y, "x": x, "time": times},
    )
    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=3)
    module = ScopeGridDataModule(data, cfg, required_vars=["Rin", "LAI"])
    batches = list(module.iter_batches())
    total = data.sizes["y"] * data.sizes["x"] * data.sizes["time"]
    assert len(batches) == int(np.ceil(total / cfg.chunk_size))
    assert batches[0]["Rin"].shape[0] == 3
    assert batches[-1]["LAI"].ndim == 1


def test_grid_module_assemble_dataset_preserves_coords_and_attrs():
    times = pd.date_range("2020-06-01", periods=2, freq="h")
    x = np.arange(2)
    y = np.arange(2)
    data = xr.Dataset(
        {
            "Rin": (("y", "x", "time"), np.arange(8).reshape(2, 2, 2)),
            "LAI": (("y", "x", "time"), np.arange(8, 16).reshape(2, 2, 2)),
        },
        coords={"y": y, "x": x, "time": times},
        attrs={"site": "test-site"},
    )
    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=3)
    module = ScopeGridDataModule(data, cfg, required_vars=["Rin", "LAI"])

    outputs = {
        "scalar": torch.arange(8, dtype=torch.float64),
        "spectral": torch.arange(16, dtype=torch.float64).reshape(8, 2),
    }
    assembled = module.assemble_dataset(
        outputs,
        variable_dims={"spectral": ("wavelength",)},
        variable_coords={"wavelength": np.array([400.0, 500.0])},
        attrs={"scope_torch_product": "unit-test"},
    )

    assert assembled["scalar"].dims == ("y", "x", "time")
    assert assembled["spectral"].dims == ("y", "x", "time", "wavelength")
    assert np.array_equal(assembled["scalar"].values, np.arange(8).reshape(2, 2, 2))
    assert np.array_equal(assembled["wavelength"].values, np.array([400.0, 500.0]))
    assert assembled.attrs["site"] == "test-site"
    assert assembled.attrs["scope_torch_product"] == "unit-test"
