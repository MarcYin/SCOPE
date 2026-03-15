from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

_ENGINE_MODULES = {
    "netcdf4": "netCDF4",
    "h5netcdf": "h5netcdf",
    "scipy": "scipy",
}
_ENGINE_PREFERENCE = ("netcdf4", "h5netcdf", "scipy")
_HDF5_ENGINES = {"netcdf4", "h5netcdf"}


@dataclass(frozen=True, slots=True)
class NetCDFWriteOptions:
    """Options for writing assembled xarray datasets to NetCDF."""

    engine: str | None = None
    format: str | None = None
    compression: bool = True
    compression_level: int = 4
    unlimited_dims: Sequence[str] = ("time",)


def available_netcdf_engines() -> tuple[str, ...]:
    """Return the NetCDF backends available in the current environment."""

    return tuple(
        engine
        for engine in _ENGINE_PREFERENCE
        if find_spec(_ENGINE_MODULES[engine]) is not None
    )


def resolve_netcdf_engine(preferred: str | None = None) -> str:
    """Pick a supported NetCDF engine, optionally honoring a preferred backend."""

    available = available_netcdf_engines()
    if preferred is not None:
        engine = preferred.lower()
        if engine not in _ENGINE_MODULES:
            raise ValueError(
                f"Unsupported NetCDF engine '{preferred}'. Expected one of {sorted(_ENGINE_MODULES)}."
            )
        if engine not in available:
            raise RuntimeError(
                f"NetCDF engine '{preferred}' is not available. Installed engines: {list(available)}."
            )
        return engine

    if not available:
        raise RuntimeError(
            "No supported NetCDF engine is available. Install one of "
            f"{', '.join(_ENGINE_MODULES[module] for module in _ENGINE_PREFERENCE)}."
        )
    return available[0]


def build_netcdf_encoding(
    dataset: xr.Dataset,
    *,
    options: NetCDFWriteOptions | None = None,
) -> dict[str, dict[str, object]]:
    """Build NetCDF encoding settings for the given dataset."""

    resolved = options or NetCDFWriteOptions()
    engine = resolve_netcdf_engine(resolved.engine)
    if not resolved.compression or engine not in _HDF5_ENGINES:
        return {}

    encoding: dict[str, dict[str, object]] = {}
    for name in dataset.variables:
        variable = dataset[name]
        if variable.dtype.kind in {"O", "S", "U"}:
            continue
        variable_encoding: dict[str, object] = {
            "zlib": True,
            "complevel": int(resolved.compression_level),
        }
        if name in dataset.coords:
            variable_encoding["_FillValue"] = None
        encoding[name] = variable_encoding
    return encoding


def write_netcdf_dataset(
    dataset: xr.Dataset,
    output_path: str | Path,
    *,
    options: NetCDFWriteOptions | None = None,
) -> Path:
    """Write a dataset to NetCDF with backend selection and safe metadata handling."""

    resolved = options or NetCDFWriteOptions()
    engine = resolve_netcdf_engine(resolved.engine)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sanitized = _sanitize_netcdf_dataset(_cf_enrich_dataset(dataset))
    kwargs: dict[str, object] = {
        "path": path,
        "engine": engine,
        "encoding": build_netcdf_encoding(sanitized, options=resolved),
    }
    if resolved.format is not None:
        kwargs["format"] = resolved.format
    unlimited_dims = ()
    if engine in _HDF5_ENGINES:
        unlimited_dims = tuple(dim for dim in resolved.unlimited_dims if dim in sanitized.dims)
    if unlimited_dims:
        kwargs["unlimited_dims"] = unlimited_dims
    sanitized.to_netcdf(**kwargs)
    return path


def _sanitize_netcdf_dataset(dataset: xr.Dataset) -> xr.Dataset:
    sanitized = dataset.copy(deep=False)
    sanitized.attrs = _sanitize_attr_mapping(dataset.attrs)
    for name in sanitized.variables:
        sanitized[name].attrs = _sanitize_attr_mapping(dataset[name].attrs)
    return sanitized


def _cf_enrich_dataset(dataset: xr.Dataset) -> xr.Dataset:
    enriched = dataset.copy(deep=False)
    enriched.attrs = _cf_global_attrs(enriched)
    for name in enriched.coords:
        attrs = dict(enriched.coords[name].attrs)
        attrs.update(_cf_coord_attrs(name))
        enriched.coords[name].attrs = attrs
    return enriched


def _cf_global_attrs(dataset: xr.Dataset) -> dict[str, Any]:
    attrs = dict(dataset.attrs)
    attrs.setdefault("Conventions", "CF-1.10")
    product = str(attrs.get("scope_product", "dataset")).replace("_", " ")
    default_title = "SCOPE-RTM dataset" if product == "dataset" else f"SCOPE-RTM {product} dataset"
    attrs.setdefault("title", default_title)
    attrs.setdefault("source", f"SCOPE-RTM {_scope_version()}")
    attrs.setdefault(
        "references",
        "https://scope-model.readthedocs.io/en/master/; https://github.com/Christiaanvandertol/SCOPE",
    )
    history_entry = (
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}: "
        "wrote NetCDF with scope.write_netcdf_dataset"
    )
    existing_history = _sanitize_history_attr(attrs.get("history"))
    if existing_history not in {None, ""}:
        attrs["history"] = f"{existing_history}\n{history_entry}"
    else:
        attrs["history"] = history_entry
    return attrs


def _scope_version() -> str:
    try:
        return package_version("SCOPE-RTM")
    except Exception:  # pragma: no cover
        return "unknown"


def _cf_coord_attrs(name: str) -> dict[str, str]:
    if name == "time":
        return {"standard_name": "time", "axis": "T"}
    if name == "x":
        return {"axis": "X"}
    if name == "y":
        return {"axis": "Y"}
    if name in {"layer", "layer_interface"}:
        return {"positive": "down"}
    return {}


def _sanitize_history_attr(value: Any) -> str | None:
    sanitized = _sanitize_attr_value(value)
    if sanitized is None:
        return None
    if isinstance(sanitized, bytes):
        return sanitized.decode("utf-8", errors="replace")
    return str(sanitized)


def _sanitize_attr_mapping(attrs: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in attrs.items():
        normalised = _sanitize_attr_value(value)
        if normalised is not None:
            sanitized[str(key)] = normalised
    return sanitized


def _sanitize_attr_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, bytes, int, float)):
        return value
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _sanitize_attr_value(value.item())
        return _json_attr_value(value.tolist())
    if isinstance(value, (list, tuple, set, dict)):
        return _json_attr_value(value)
    return str(value)


def _json_attr_value(value: Any) -> str:
    return json.dumps(value, default=_json_default, sort_keys=isinstance(value, dict))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
