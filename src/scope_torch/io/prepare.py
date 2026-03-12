from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

try:  # pragma: no cover - import activates rio accessors when installed
    import rioxarray  # noqa: F401
except Exception:  # pragma: no cover
    rioxarray = None

from ..spectral.loaders import load_scope_filenames, scope_root

BIO_BANDS = ("N", "cab", "cm", "cw", "lai", "ala", "cbrown")
BIO_SCALES = (1 / 100.0, 1 / 100.0, 1 / 10000.0, 1 / 10000.0, 1 / 100.0, 1 / 100.0, 1 / 1000.0)
SCALE_BANDS = ("N", "cab", "cm", "cw", "lai", "ala", "cbrown", "n0", "m0", "n1", "m1", "BSMBrightness", "BSMlat", "BSMlon", "SMC")
DEFAULT_WEATHER_VAR_MAP = {
    "Rin": "Rin",
    "Rli": "Rli",
    "Ta": "Ta",
    "ea": "ea",
    "p": "p",
    "u": "u",
}
DEFAULT_OBSERVATION_VAR_MAP = {
    "tts": "solar_zenith_angle",
    "tto": "viewing_zenith_angle",
    "solar_azimuth": "solar_azimuth_angle",
    "viewing_azimuth": "viewing_azimuth_angle",
}
DEFAULT_SCOPE_OPTIONS = {
    "lite": 1,
    "calc_fluor": 1,
    "calc_planck": 0,
    "calc_xanthophyllabs": 0,
    "soilspectrum": 1,
    "Fluorescence_model": 0,
    "apply_T_corr": 1,
    "verify": 0,
    "mSCOPE": 0,
    "calc_directional": 0,
    "calc_vert_profiles": 0,
    "soil_heat_method": 2,
    "calc_rss_rbs": 1,
    "MoninObukhov": 1,
    "save_spectral": 0,
}


@dataclass(slots=True)
class ScopeInputFiles:
    """Resolvable SCOPE file references stored on prepared datasets."""

    optipar_file: str | Path | None = None
    soil_file: str | Path | None = None
    atmos_file: str | Path | None = None

    def resolve(self, scope_root_path: str | Path | None = None) -> dict[str, str]:
        return {
            "optipar_file": _resolve_scope_file(self.optipar_file, scope_root_path=scope_root_path, subdir="fluspect_parameters", filename_key="optipar_file"),
            "soil_file": _resolve_scope_file(self.soil_file, scope_root_path=scope_root_path, subdir="soil_spectra", filename_key="soil_file"),
            "atmos_file": _resolve_scope_file(self.atmos_file, scope_root_path=scope_root_path, subdir="radiationdata", filename_key="atmos_file"),
        }


def read_s2_bio_inputs(
    npz_path: str | Path,
    *,
    year: int,
    reference_dataset: xr.Dataset | xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Load Sentinel-2 canopy retrieval inputs from the historic NPZ bundle format."""

    npz_file = Path(npz_path)
    with np.load(npz_file, allow_pickle=True) as payload:
        doys = np.asarray(payload["doys"])
        mask = np.asarray(payload["mask"])
        geotransform = np.asarray(payload["geotransform"], dtype=np.float64)
        post_bio_tensor = np.asarray(payload["post_bio_tensor"], dtype=np.float64)
        scale_params = np.asarray(payload["dat"], dtype=np.float64)
        crs_value = payload.get("crs")

    ny, nx = mask.shape[:2]
    nt = int(doys.size)
    post_bio_tensor = post_bio_tensor.reshape(ny, nx, len(BIO_BANDS), nt)
    post_bio_scale = scale_params.reshape(ny, nx, len(SCALE_BANDS))

    times = pd.to_datetime([f"{year}{int(doy):03d}" for doy in doys], format="%Y%j")
    _, unique_index = np.unique(times.values, return_index=True)
    unique_index = np.sort(unique_index)

    x_coords = geotransform[0] + np.arange(nx) * geotransform[1]
    y_coords = geotransform[3] + np.arange(ny) * geotransform[5]
    scales = np.asarray(BIO_SCALES, dtype=np.float64).reshape(1, 1, len(BIO_BANDS), 1)

    post_bio_da = xr.DataArray(
        post_bio_tensor[:, :, :, unique_index] * scales,
        dims=("y", "x", "band", "time"),
        coords={"y": y_coords, "x": x_coords, "band": list(BIO_BANDS), "time": times[unique_index]},
        name="post_bio",
    )
    post_bio_scale_da = xr.DataArray(
        post_bio_scale,
        dims=("y", "x", "band"),
        coords={"y": y_coords, "x": x_coords, "band": list(SCALE_BANDS)},
        name="post_bio_scale",
    )

    crs = _normalise_crs(crs_value)
    post_bio_da = _write_spatial_metadata(post_bio_da, reference_dataset=reference_dataset, crs=crs)
    post_bio_scale_da = _write_spatial_metadata(post_bio_scale_da, reference_dataset=reference_dataset, crs=crs)
    return post_bio_da, post_bio_scale_da


def derive_observation_time_grid(
    observation_ds: xr.Dataset,
    *,
    delta_time_var: str = "delta_time",
    spatial_dims: Sequence[str] = ("y", "x"),
) -> xr.DataArray:
    """Collapse per-pixel observation times into a single valid time grid."""

    if delta_time_var not in observation_ds:
        raise KeyError(f"Observation dataset missing '{delta_time_var}'")

    delta_time = observation_ds[delta_time_var]
    reduce_dims = [dim for dim in spatial_dims if dim in delta_time.dims]
    if np.issubdtype(delta_time.dtype, np.datetime64):
        mean_numeric = _integer_time_mean(delta_time, reduce_dims, dtype="datetime64[ns]")
        mean_numeric = _drop_null_along_primary_dim(mean_numeric)
        values = np.asarray(mean_numeric.values, dtype=np.int64).astype("datetime64[ns]")
        return xr.DataArray(values, dims=mean_numeric.dims, coords=mean_numeric.coords, name="time")

    if np.issubdtype(delta_time.dtype, np.timedelta64):
        mean_numeric = _integer_time_mean(delta_time, reduce_dims, dtype="timedelta64[ns]")
        mean_numeric = _drop_null_along_primary_dim(mean_numeric)
        values = np.asarray(mean_numeric.values, dtype=np.int64).astype("timedelta64[ns]")
        return xr.DataArray(values, dims=mean_numeric.dims, coords=mean_numeric.coords, name="time")

    time_grid = delta_time.mean(dim=reduce_dims, skipna=True)
    return _drop_null_along_primary_dim(time_grid).rename("time")


def prepare_scope_input_dataset(
    weather_ds: xr.Dataset,
    observation_ds: xr.Dataset,
    post_bio_da: xr.DataArray,
    post_bio_scale_da: xr.DataArray,
    *,
    scope_root_path: str | Path | None = None,
    scope_files: ScopeInputFiles | None = None,
    scope_options: Mapping[str, object] | None = None,
    weather_var_map: Mapping[str, str] | None = None,
    observation_var_map: Mapping[str, str] | None = None,
    transpose_dims: Sequence[str] = ("y", "x", "time"),
    interp_method: str = "linear",
    delta_time_var: str = "delta_time",
    bounds: Sequence[float] | None = None,
    time_slice: slice | None = None,
    resampling: int = 0,
) -> xr.Dataset:
    """Build a runner-ready xarray dataset from weather, observation, and bio inputs."""

    weather_map = dict(DEFAULT_WEATHER_VAR_MAP)
    if weather_var_map:
        weather_map.update(weather_var_map)
    observation_map = dict(DEFAULT_OBSERVATION_VAR_MAP)
    if observation_var_map:
        observation_map.update(observation_var_map)

    post_bio_da = _clip_box(post_bio_da, bounds)
    post_bio_scale_da = _clip_box(post_bio_scale_da, bounds)
    if time_slice is not None and "time" in post_bio_da.dims:
        post_bio_da = post_bio_da.sel(time=time_slice)

    observation = _clip_box(observation_ds, bounds)
    if time_slice is not None and "time" in observation.dims:
        observation = observation.sel(time=time_slice)
    time_grid = derive_observation_time_grid(observation, delta_time_var=delta_time_var)
    observation = observation.sel(time=time_grid.coords[time_grid.dims[0]].values).assign_coords(time=time_grid.values)
    if delta_time_var in observation:
        observation = observation.drop_vars(delta_time_var)

    weather = _clip_box(weather_ds, bounds)
    if "time" in weather.dims:
        if time_slice is not None:
            weather = weather.sel(time=time_slice)
        weather = _interp_time(weather, time_grid.values, method=interp_method)
    if "time" in post_bio_da.dims:
        post_bio_da = _interp_time(post_bio_da, time_grid.values, method=interp_method)

    target = _spatial_target(post_bio_da)
    weather = _match_spatial_grid(weather, target, method="nearest", resampling=resampling)
    observation = _match_spatial_grid(observation, target, method="nearest", resampling=resampling)
    post_bio_scale_da = _match_spatial_grid(post_bio_scale_da, target, method="nearest", resampling=resampling)

    dataset = xr.Dataset()
    for output_name, source_name in weather_map.items():
        if source_name not in weather:
            raise KeyError(f"Weather dataset missing '{source_name}' for '{output_name}'")
        dataset[output_name] = weather[source_name]

    dataset["LAI"] = post_bio_da.sel(band="lai", drop=True)
    dataset["N"] = post_bio_da.sel(band="N", drop=True)
    dataset["Cab"] = post_bio_da.sel(band="cab", drop=True)
    dataset["Cca"] = dataset["Cab"] * 0.25
    dataset["Cdm"] = post_bio_da.sel(band="cm", drop=True)
    dataset["Cw"] = post_bio_da.sel(band="cw", drop=True)
    dataset["ala"] = post_bio_da.sel(band="ala", drop=True)
    dataset["Cs"] = post_bio_da.sel(band="cbrown", drop=True)

    soil = post_bio_scale_da.sel(band=["BSMBrightness", "BSMlat", "BSMlon", "SMC"])
    soil = soil.expand_dims(time=time_grid.values).transpose("y", "x", "band", "time")
    dataset["BSMBrightness"] = soil.sel(band="BSMBrightness", drop=True)
    dataset["BSMlat"] = soil.sel(band="BSMlat", drop=True)
    dataset["BSMlon"] = soil.sel(band="BSMlon", drop=True)
    dataset["SMC"] = soil.sel(band="SMC", drop=True)

    if "tts" not in observation_map or observation_map["tts"] not in observation:
        raise KeyError("Observation dataset must provide a solar zenith variable for 'tts'")
    if "tto" not in observation_map or observation_map["tto"] not in observation:
        raise KeyError("Observation dataset must provide a viewing zenith variable for 'tto'")
    dataset["tts"] = observation[observation_map["tts"]]
    dataset["tto"] = observation[observation_map["tto"]]
    if "psi" in observation_map and observation_map["psi"] in observation:
        dataset["psi"] = observation[observation_map["psi"]] % 360.0
    else:
        solar_azimuth = observation_map.get("solar_azimuth")
        viewing_azimuth = observation_map.get("viewing_azimuth")
        if solar_azimuth not in observation or viewing_azimuth not in observation:
            raise KeyError("Observation dataset must provide either 'psi' or both solar and viewing azimuth variables")
        dataset["psi"] = (observation[viewing_azimuth] - observation[solar_azimuth]) % 360.0

    dataset = _drop_nonstandard_coords(dataset)
    if transpose_dims:
        ordered_dims = [dim for dim in transpose_dims if dim in dataset.dims]
        dataset = dataset.transpose(*ordered_dims)

    dataset.attrs.update(DEFAULT_SCOPE_OPTIONS)
    if scope_options:
        dataset.attrs.update(scope_options)
    dataset.attrs.update((scope_files or ScopeInputFiles()).resolve(scope_root_path))
    return dataset


def _resolve_scope_file(
    path: str | Path | None,
    *,
    scope_root_path: str | Path | None,
    subdir: str,
    filename_key: str,
) -> str:
    root = scope_root(scope_root_path)
    default_name = load_scope_filenames(root).get(filename_key, "")
    if path is None:
        if not default_name:
            raise FileNotFoundError(f"No default value for '{filename_key}' found under {root / 'input'}")
        candidate = Path(default_name)
    else:
        candidate = Path(path)
    if not str(candidate) or str(candidate) == ".":
        raise FileNotFoundError(f"No default value for '{filename_key}' found under {root / 'input'}")

    search_paths = [candidate] if candidate.is_absolute() else [root / "input" / subdir / candidate, root / "input" / candidate, root / candidate]
    for resolved in search_paths:
        if resolved.exists():
            return resolved.as_posix()
    raise FileNotFoundError(f"Could not resolve {candidate} under {root / 'input'}")


def _normalise_crs(value: object | None) -> object | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8")
        return scalar
    return value


def _write_spatial_metadata(
    data: xr.DataArray,
    *,
    reference_dataset: xr.Dataset | xr.DataArray | None,
    crs: object | None,
) -> xr.DataArray:
    if not hasattr(data, "rio"):
        return data

    try:
        data = data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    except Exception:
        return data

    reference_crs = None
    if reference_dataset is not None and hasattr(reference_dataset, "rio"):
        try:
            reference_crs = reference_dataset.rio.crs
        except Exception:
            reference_crs = None
    target_crs = reference_crs or crs
    if target_crs is None:
        return data
    try:
        return data.rio.write_crs(target_crs, inplace=False)
    except Exception:
        return data


def _drop_null_along_primary_dim(data: xr.DataArray) -> xr.DataArray:
    if not data.dims:
        return data
    primary_dim = data.dims[0]
    mask = data.notnull()
    if mask.ndim != 1:
        mask = mask.any(dim=[dim for dim in mask.dims if dim != primary_dim])
    return data.isel({primary_dim: mask.values}, drop=True)


def _integer_time_mean(delta_time: xr.DataArray, reduce_dims: Sequence[str], *, dtype: str) -> xr.DataArray:
    valid = delta_time.notnull()
    numeric = xr.where(valid, delta_time.astype(dtype).astype(np.int64), 0)
    if reduce_dims:
        summed = numeric.sum(dim=reduce_dims, skipna=False)
        counts = valid.sum(dim=reduce_dims)
    else:
        summed = numeric
        counts = valid.astype(np.int64)
    counts_safe = xr.where(counts > 0, counts, 1)
    mean_numeric = (summed + counts_safe // 2) // counts_safe
    return mean_numeric.where(counts > 0)


def _spatial_target(data: xr.DataArray) -> xr.DataArray:
    target = data
    for dim in ("band", "time"):
        if dim in target.dims:
            target = target.isel({dim: 0}, drop=True)
    return target


def _clip_box(data: xr.Dataset | xr.DataArray, bounds: Sequence[float] | None) -> xr.Dataset | xr.DataArray:
    if bounds is None or "x" not in data.coords or "y" not in data.coords:
        return data
    minx, miny, maxx, maxy = map(float, bounds)
    if hasattr(data, "rio"):
        try:
            return data.rio.clip_box(minx, miny, maxx, maxy, allow_one_dimensional_raster=True, auto_expand=True)
        except Exception:
            pass

    x = data.coords["x"]
    y = data.coords["y"]
    x_sel = x[(x >= minx) & (x <= maxx)]
    y_sel = y[(y >= miny) & (y <= maxy)]
    return data.sel(x=x_sel, y=y_sel)


def _match_spatial_grid(
    data: xr.Dataset | xr.DataArray,
    target: xr.DataArray,
    *,
    method: str,
    resampling: int,
) -> xr.Dataset | xr.DataArray:
    if "x" not in target.coords or "y" not in target.coords:
        return data
    if "x" in data.coords and "y" in data.coords:
        same_x = np.array_equal(np.asarray(data.coords["x"].values), np.asarray(target.coords["x"].values))
        same_y = np.array_equal(np.asarray(data.coords["y"].values), np.asarray(target.coords["y"].values))
        if same_x and same_y:
            return data
    if hasattr(data, "rio") and hasattr(target, "rio"):
        try:
            if data.rio.crs is not None and target.rio.crs is not None:
                return data.rio.reproject_match(target, resampling=resampling)
        except Exception:
            pass
    if "x" in data.coords and "y" in data.coords:
        return data.interp(x=target.coords["x"], y=target.coords["y"], method=method)
    return data


def _drop_nonstandard_coords(dataset: xr.Dataset) -> xr.Dataset:
    keep = {"time", "x", "y", "spatial_ref"}
    drop = [name for name in dataset.coords if name not in keep]
    if drop:
        dataset = dataset.drop_vars(drop)
    return dataset


def _interp_time(data: xr.Dataset | xr.DataArray, target_time: Sequence[object], *, method: str) -> xr.Dataset | xr.DataArray:
    canonical_target = np.asarray(target_time)
    source_time = np.asarray(data.coords["time"].values)
    target_index = canonical_target.astype(source_time.dtype, copy=False)
    if np.array_equal(source_time, target_index):
        return data.assign_coords(time=canonical_target)
    exact = data.reindex(time=target_index)
    interpolated = data.interp(time=target_index, method=method)
    return exact.combine_first(interpolated).assign_coords(time=canonical_target)
