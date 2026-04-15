from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from scope import ScopeGridRunner, SimulationConfig, campbell_lidf
from scope.data import ScopeGridDataModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal single-scene reflectance example with scope.")
    parser.add_argument(
        "--scope-root", help="Optional upstream SCOPE root. Defaults to ./upstream/SCOPE when available."
    )
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64", help="Torch dtype.")
    parser.add_argument("--output", help="Optional JSON output path for the example summary.")
    return parser.parse_args()


def _default_scope_root() -> str | None:
    candidate = Path(__file__).resolve().parents[1] / "upstream" / "SCOPE"
    return str(candidate) if candidate.exists() else None


def build_dataset() -> xr.Dataset:
    times = pd.date_range("2020-07-01T12:00:00", periods=1, freq="h")
    return xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.array([[[45.0]]])),
            "Cw": (("y", "x", "time"), np.array([[[0.010]]])),
            "Cdm": (("y", "x", "time"), np.array([[[0.012]]])),
            "LAI": (("y", "x", "time"), np.array([[[2.2]]])),
            "tts": (("y", "x", "time"), np.array([[[30.0]]])),
            "tto": (("y", "x", "time"), np.array([[[20.0]]])),
            "psi": (("y", "x", "time"), np.array([[[15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0]]])),
        },
        coords={"y": [0], "x": [0], "time": times},
        attrs={"example": "basic_scene_reflectance"},
    )


def summarize(outputs: xr.Dataset) -> dict[str, object]:
    rsot = outputs["rsot"].isel(y=0, x=0, time=0)
    summary = {
        "product": outputs.attrs["scope_product"],
        "dims": {name: int(size) for name, size in outputs.sizes.items()},
        "variables": sorted(outputs.data_vars),
        "rsot_650nm": float(rsot.sel(wavelength=650.0, method="nearest")),
        "rsot_865nm": float(rsot.sel(wavelength=865.0, method="nearest")),
        "rsot_1600nm": float(rsot.sel(wavelength=1600.0, method="nearest")),
        "rdd_mean": float(outputs["rdd"].mean().item()),
        "rso_mean": float(outputs["rso"].mean().item()),
    }
    return summary


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    scope_root = args.scope_root or _default_scope_root()

    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(
        lidf=lidf,
        device=device,
        dtype=dtype,
        scope_root_path=scope_root,
    )

    dataset = build_dataset()
    config = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=pd.Timestamp(dataset.time.values[0]),
        end_time=pd.Timestamp(dataset.time.values[-1]),
        device=str(device),
        dtype=dtype,
        chunk_size=1,
    )
    module = ScopeGridDataModule(dataset, config, required_vars=list(dataset.data_vars))
    outputs = runner.run_dataset(module, varmap={name: name for name in dataset.data_vars})
    summary = summarize(outputs)
    text = json.dumps(summary, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
