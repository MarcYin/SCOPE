from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pandas as pd
import torch
import xarray as xr

from .. import ScopeGridRunner, campbell_lidf
from ..config import SimulationConfig
from ..data import ScopeGridDataModule
from ..io import NetCDFWriteOptions, validate_scope_dataset, write_netcdf_dataset


_WORKFLOW_RUNNERS: Mapping[str, str] = {
    "scope": "run_scope_dataset",
    "reflectance": "run_dataset",
    "directional-reflectance": "run_directional_reflectance_dataset",
    "reflectance-profiles": "run_reflectance_profiles_dataset",
    "fluorescence": "run_fluorescence_dataset",
    "layered-fluorescence": "run_layered_fluorescence_dataset",
    "directional-fluorescence": "run_directional_fluorescence_dataset",
    "fluorescence-profiles": "run_fluorescence_profiles_dataset",
    "thermal": "run_thermal_dataset",
    "directional-thermal": "run_directional_thermal_dataset",
    "thermal-profiles": "run_thermal_profiles_dataset",
    "biochemical-fluorescence": "run_biochemical_fluorescence_dataset",
    "energy-balance-fluorescence": "run_energy_balance_fluorescence_dataset",
    "energy-balance-thermal": "run_energy_balance_thermal_dataset",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a prepared SCOPE input dataset through a selected workflow and write NetCDF outputs."
    )
    parser.add_argument("--input", required=True, help="Prepared input NetCDF dataset path.")
    parser.add_argument("--output", required=True, help="Output NetCDF path for simulated results.")
    parser.add_argument(
        "--workflow",
        choices=tuple(_WORKFLOW_RUNNERS),
        default="scope",
        help="Workflow to execute. 'scope' uses dataset attrs / option overrides for high-level dispatch.",
    )
    parser.add_argument("--scope-root", help="Optional upstream SCOPE root. Used when dataset attrs do not carry absolute asset paths.")
    parser.add_argument("--optipar-file", help="Optional FLUSPECT parameter MAT file override.")
    parser.add_argument("--soil-file", help="Optional soil spectra file override.")
    parser.add_argument("--device", default="cpu", help="Torch device for execution.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32", help="Torch dtype for execution.")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Batch chunk size for ROI/time execution.")
    parser.add_argument("--lidfa", type=float, default=57.0, help="Campbell mean leaf angle used to build the LIDF.")
    parser.add_argument("--default-hotspot", type=float, default=0.2, help="Fallback hotspot value when no hotspot variable is present.")
    parser.add_argument("--nlayers", type=int, help="Optional layer count override for layered workflows.")
    parser.add_argument("--soil-heat-method", type=int, default=2, help="Soil heat method for coupled energy-balance workflows.")
    parser.add_argument("--calc-fluor", choices=(0, 1), type=int, help="Override calc_fluor for workflow='scope'.")
    parser.add_argument("--calc-planck", choices=(0, 1), type=int, help="Override calc_planck for workflow='scope'.")
    parser.add_argument("--calc-directional", choices=(0, 1), type=int, help="Override calc_directional for workflow='scope'.")
    parser.add_argument("--calc-vert-profiles", choices=(0, 1), type=int, help="Override calc_vert_profiles for workflow='scope'.")
    parser.add_argument(
        "--netcdf-engine",
        choices=("netcdf4", "h5netcdf", "scipy"),
        help="Optional NetCDF backend override for writing outputs.",
    )
    parser.add_argument("--compression-level", type=int, default=4, help="Compression level for HDF5-backed NetCDF engines.")
    parser.add_argument("--no-compression", action="store_true", help="Disable NetCDF compression on output.")
    parser.add_argument("--summary-json", help="Optional JSON path for a small run summary.")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> Path:
    dataset = xr.open_dataset(args.input)
    try:
        _validate_input_dataset(args, dataset)
        runner = _build_runner(args, dataset)
        data_module = _build_data_module(args, dataset)
        outputs = _run_workflow(args, runner, data_module)
        output_path = write_netcdf_dataset(
            outputs,
            args.output,
            options=NetCDFWriteOptions(
                engine=args.netcdf_engine,
                compression=not args.no_compression,
                compression_level=args.compression_level,
            ),
        )
        if args.summary_json:
            summary_path = Path(args.summary_json)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(_summarize(outputs), indent=2) + "\n", encoding="utf-8")
        return output_path
    finally:
        dataset.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    resolved = run(args)
    print(resolved.resolve())


def _build_runner(args: argparse.Namespace, dataset: xr.Dataset) -> ScopeGridRunner:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    lidf = campbell_lidf(args.lidfa, device=device, dtype=dtype)
    return ScopeGridRunner.from_scope_assets(
        lidf=lidf,
        fluspect_path=args.optipar_file or dataset.attrs.get("optipar_file"),
        soil_path=args.soil_file or dataset.attrs.get("soil_file"),
        scope_root_path=args.scope_root,
        device=device,
        dtype=dtype,
        default_hotspot=args.default_hotspot,
    )


def _build_data_module(args: argparse.Namespace, dataset: xr.Dataset) -> ScopeGridDataModule:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    time_values = pd.to_datetime(dataset.coords["time"].values)
    config = SimulationConfig(
        roi_bounds=_infer_roi_bounds(dataset),
        start_time=pd.Timestamp(time_values[0]).to_pydatetime(),
        end_time=pd.Timestamp(time_values[-1]).to_pydatetime(),
        device=args.device,
        dtype=dtype,
        chunk_size=args.chunk_size,
    )
    return ScopeGridDataModule(dataset, config, required_vars=list(dataset.data_vars))


def _run_workflow(args: argparse.Namespace, runner: ScopeGridRunner, data_module: ScopeGridDataModule) -> xr.Dataset:
    workflow = args.workflow
    varmap = {name: name for name in data_module.dataset.data_vars}
    method_name = _WORKFLOW_RUNNERS[workflow]
    method: Callable[..., xr.Dataset] = getattr(runner, method_name)
    kwargs: dict[str, object] = {"varmap": varmap}

    if workflow == "scope":
        scope_options = {
            key: value
            for key, value in {
                "calc_fluor": args.calc_fluor,
                "calc_planck": args.calc_planck,
                "calc_directional": args.calc_directional,
                "calc_vert_profiles": args.calc_vert_profiles,
            }.items()
            if value is not None
        }
        if scope_options:
            kwargs["scope_options"] = scope_options
    if workflow in {
        "scope",
        "reflectance",
        "directional-reflectance",
        "reflectance-profiles",
        "layered-fluorescence",
        "directional-fluorescence",
        "fluorescence-profiles",
        "thermal",
        "directional-thermal",
        "thermal-profiles",
        "biochemical-fluorescence",
        "energy-balance-fluorescence",
        "energy-balance-thermal",
    } and args.nlayers is not None:
        kwargs["nlayers"] = args.nlayers
    if workflow in {"energy-balance-fluorescence", "energy-balance-thermal"}:
        kwargs["soil_heat_method"] = args.soil_heat_method

    return method(data_module, **kwargs)


def _validate_input_dataset(args: argparse.Namespace, dataset: xr.Dataset) -> None:
    scope_options = {
        key: value
        for key, value in {
            "calc_fluor": args.calc_fluor,
            "calc_planck": args.calc_planck,
            "calc_directional": args.calc_directional,
            "calc_vert_profiles": args.calc_vert_profiles,
        }.items()
        if value is not None
    }
    validate_scope_dataset(
        dataset,
        workflow=args.workflow,
        scope_options=scope_options if args.workflow == "scope" else None,
    )


def _infer_roi_bounds(dataset: xr.Dataset) -> tuple[float, float, float, float]:
    x = dataset.coords["x"].values.astype(float)
    y = dataset.coords["y"].values.astype(float)
    return (float(x.min()), float(y.min()), float(x.max()), float(y.max()))


def _summarize(dataset: xr.Dataset) -> dict[str, object]:
    summary = {
        "product": dataset.attrs.get("scope_product", ""),
        "components": dataset.attrs.get("scope_components", "").split(",") if dataset.attrs.get("scope_components") else [],
        "dims": {name: int(size) for name, size in dataset.sizes.items()},
        "variables": sorted(dataset.data_vars),
    }
    return summary


if __name__ == "__main__":
    main()
