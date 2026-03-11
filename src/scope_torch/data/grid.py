from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import xarray as xr

from ..config import SimulationConfig


@dataclass(slots=True)
class ScopeGridDataModule:
    """Turn ROI x time cubes into torch batches ready for SCOPE Torch."""

    dataset: xr.Dataset
    config: SimulationConfig
    required_vars: Sequence[str]
    stack_dims: Sequence[str] = ("y", "x", "time")

    def _stack_dataset(self) -> xr.Dataset:
        missing = [var for var in self.required_vars if var not in self.dataset]
        if missing:
            raise KeyError(f"Dataset missing required variables: {missing}")
        stacked = self.dataset.stack(batch=self.stack_dims).transpose("batch", ...)
        return stacked

    def _to_tensor(self, data_array: xr.DataArray) -> torch.Tensor:
        values = data_array.values
        tensor = torch.as_tensor(values, dtype=self.config.dtype, device=self.config.torch_device())
        if not tensor.isfinite().all():
            tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor

    def iter_batches(self) -> Iterable[Mapping[str, torch.Tensor]]:
        stacked = self._stack_dataset()
        total = stacked.dims["batch"]
        tensor_map = {var: self._to_tensor(stacked[var]) for var in self.required_vars}
        for chunk in self.config.chunks(total):
            yield {var: tensor[chunk] for var, tensor in tensor_map.items()}
