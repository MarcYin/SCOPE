import numpy as np
import pandas as pd
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
    total = data.dims["y"] * data.dims["x"] * data.dims["time"]
    assert len(batches) == int(np.ceil(total / cfg.chunk_size))
    assert batches[0]["Rin"].shape[0] == 3
    assert batches[-1]["LAI"].ndim == 1
