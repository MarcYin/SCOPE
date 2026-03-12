from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
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
        stacked = self.dataset[self.required_vars].stack(batch=self.stack_dims).transpose("batch", ...)
        return stacked

    def _to_tensor(self, data_array: xr.DataArray) -> torch.Tensor:
        values = data_array.values
        tensor = torch.as_tensor(values, dtype=self.config.dtype, device=self.config.torch_device())
        if not tensor.isfinite().all():
            tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor

    def _to_numpy(self, value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def batch_size(self) -> int:
        return int(self._stack_dataset().sizes["batch"])

    def iter_batches(self) -> Iterable[Mapping[str, torch.Tensor]]:
        stacked = self._stack_dataset()
        total = stacked.sizes["batch"]
        for chunk in self.config.chunks(total):
            batch = stacked.isel(batch=chunk)
            yield {var: self._to_tensor(batch[var]) for var in self.required_vars}

    def assemble_dataset(
        self,
        outputs: Mapping[str, torch.Tensor | np.ndarray | Sequence[float]],
        *,
        variable_dims: Mapping[str, Sequence[str]] | None = None,
        variable_coords: Mapping[str, torch.Tensor | np.ndarray | Sequence[float] | xr.DataArray] | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> xr.Dataset:
        stacked = self._stack_dataset()
        total = stacked.sizes["batch"]
        variable_dims = {} if variable_dims is None else dict(variable_dims)
        variable_coords = {} if variable_coords is None else dict(variable_coords)

        data_vars: dict[str, xr.DataArray] = {}
        for name, value in outputs.items():
            array = self._to_numpy(value)
            if array.ndim == 0:
                raise ValueError(f"Output '{name}' must include the stacked batch dimension")
            if array.shape[0] != total:
                raise ValueError(f"Output '{name}' has batch size {array.shape[0]}, expected {total}")

            extra_dims = tuple(variable_dims.get(name, tuple(f"{name}_dim_{idx}" for idx in range(1, array.ndim))))
            if len(extra_dims) != array.ndim - 1:
                raise ValueError(f"Output '{name}' expected {array.ndim - 1} extra dims, got {len(extra_dims)}")

            coords: dict[str, object] = {"batch": stacked.coords["batch"]}
            for axis, dim_name in enumerate(extra_dims, start=1):
                coords[dim_name] = self._coord_values(dim_name, array.shape[axis], variable_coords)

            data_array = xr.DataArray(array, dims=("batch", *extra_dims), coords=coords)
            data_array = data_array.unstack("batch")
            ordered_dims = [dim for dim in (*self.stack_dims, *extra_dims) if dim in data_array.dims]
            data_vars[name] = data_array.transpose(*ordered_dims)

        dataset = xr.Dataset(data_vars=data_vars, attrs=dict(self.dataset.attrs))
        if attrs:
            dataset.attrs.update(attrs)
        for coord_name, coord in self.dataset.coords.items():
            if coord_name not in dataset.coords and set(coord.dims).issubset(dataset.dims):
                dataset = dataset.assign_coords({coord_name: coord})
        return dataset

    def _coord_values(
        self,
        name: str,
        size: int,
        variable_coords: Mapping[str, torch.Tensor | np.ndarray | Sequence[float] | xr.DataArray],
    ) -> np.ndarray | xr.DataArray:
        if name in variable_coords:
            coord = variable_coords[name]
            if isinstance(coord, xr.DataArray):
                if coord.size != size:
                    raise ValueError(f"Coordinate '{name}' has size {coord.size}, expected {size}")
                return coord
            values = self._to_numpy(coord)
            if values.shape[0] != size:
                raise ValueError(f"Coordinate '{name}' has size {values.shape[0]}, expected {size}")
            return values

        if name in self.dataset.coords and self.dataset.coords[name].size == size:
            return self.dataset.coords[name]

        return np.arange(size)
