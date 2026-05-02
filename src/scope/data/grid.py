from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import xarray as xr

from ..config import SimulationConfig
from ..variables import annotate_dataset


@dataclass(slots=True)
class ScopeGridDataModule:
    """Turn ROI x time cubes into torch batches ready for SCOPE."""

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

    def _to_tensor(self, data_array: xr.DataArray, *, chunk: slice | None = None) -> torch.Tensor:
        missing_dims = [dim for dim in self.stack_dims if dim not in data_array.dims]
        if missing_dims:
            raise ValueError(f"Variable '{data_array.name}' is missing stack dimensions: {missing_dims}")

        extra_dims = tuple(dim for dim in data_array.dims if dim not in self.stack_dims)
        ordered_dims = (*self.stack_dims, *extra_dims)
        raw = self._raw_array(data_array)

        if isinstance(raw, torch.Tensor):
            raw = self._fresh_autograd_view(raw)
            axis_order = tuple(data_array.dims.index(dim) for dim in ordered_dims)
            tensor = raw.permute(axis_order).reshape(self.batch_size(), *self._extra_shape(data_array, extra_dims))
        else:
            values = data_array.transpose(*ordered_dims).data
            tensor = torch.as_tensor(values, dtype=self.config.dtype, device=self.config.torch_device())
            tensor = tensor.reshape(self.batch_size(), *self._extra_shape(data_array, extra_dims))

        if chunk is not None:
            tensor = tensor[chunk]
        if tensor.is_floating_point() or tensor.is_complex():
            tensor = torch.nan_to_num(tensor, nan=0.0)
        return tensor

    def _to_numpy(self, value: torch.Tensor | np.ndarray | Sequence[float]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _raw_array(self, data_array: xr.DataArray) -> object:
        raw = getattr(data_array.variable, "_data", None)
        if isinstance(raw, torch.Tensor):
            return raw
        return data_array.data

    def _fresh_autograd_view(self, tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.requires_grad or tensor.is_leaf:
            return tensor

        base = tensor
        while isinstance(getattr(base, "_base", None), torch.Tensor):
            base = base._base
        if base is tensor:
            return tensor
        return base.as_strided(tensor.shape, tensor.stride(), tensor.storage_offset())

    def _carries_grad(self, data_array: xr.DataArray) -> bool:
        raw = getattr(data_array.variable, "_data", None)
        return isinstance(raw, torch.Tensor) and raw.requires_grad

    def _extra_shape(self, data_array: xr.DataArray, extra_dims: Sequence[str]) -> tuple[int, ...]:
        return tuple(int(data_array.sizes[dim]) for dim in extra_dims)

    def _stacked_batch_template(self) -> xr.Dataset:
        coords: dict[str, object] = {}
        for dim in self.stack_dims:
            if dim in self.dataset.coords:
                coords[dim] = self.dataset.coords[dim]
            else:
                coords[dim] = np.arange(self.dataset.sizes[dim])
        return xr.Dataset(coords=coords).stack(batch=self.stack_dims)

    def batch_size(self) -> int:
        total = 1
        for dim in self.stack_dims:
            total *= int(self.dataset.sizes[dim])
        return total

    def iter_batches(self) -> Iterable[Mapping[str, torch.Tensor]]:
        missing = [var for var in self.required_vars if var not in self.dataset]
        if missing:
            raise KeyError(f"Dataset missing required variables: {missing}")
        cached = {
            var: self._to_tensor(self.dataset[var])
            for var in self.required_vars
            if not self._carries_grad(self.dataset[var])
        }
        total = self.batch_size()
        for chunk in self.config.chunks(total):
            yield {
                var: cached[var][chunk] if var in cached else self._to_tensor(self.dataset[var], chunk=chunk)
                for var in self.required_vars
            }

    def assemble_dataset(
        self,
        outputs: Mapping[str, torch.Tensor | np.ndarray | Sequence[float]],
        *,
        variable_dims: Mapping[str, Sequence[str]] | None = None,
        variable_coords: Mapping[str, torch.Tensor | np.ndarray | Sequence[float] | xr.DataArray] | None = None,
        attrs: Mapping[str, object] | None = None,
    ) -> xr.Dataset:
        stacked = self._stacked_batch_template()
        total = self.batch_size()
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
        return annotate_dataset(dataset)

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
