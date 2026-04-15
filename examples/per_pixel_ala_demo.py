"""Demonstrate per-pixel Average Leaf Angle (ALA) integration in SCOPE.

This example shows how spatially-varying ALA values from Sentinel-2
biophysical retrievals flow through the Campbell ellipsoidal LIDF into
the 4SAIL radiative transfer model, producing per-pixel reflectance that
accounts for local canopy architecture.

Five ALA values spanning planophile (20 deg) to erectophile (80 deg) are
simulated with all other parameters held constant so the isolated effect
of leaf angle on reflectance is clearly visible.
"""

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

ALA_VALUES = [20.0, 35.0, 50.0, 57.0, 70.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-pixel ALA demo for SCOPE.")
    parser.add_argument("--scope-root", help="Optional upstream SCOPE root.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def _default_scope_root() -> str | None:
    candidate = Path(__file__).resolve().parents[1] / "upstream" / "SCOPE"
    return str(candidate) if candidate.exists() else None


def build_dataset() -> xr.Dataset:
    """Build a dataset with 5 time steps, each with a different ALA value."""
    n = len(ALA_VALUES)
    times = pd.date_range("2020-07-01T12:00:00", periods=n, freq="h")
    return xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, n), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, n), 0.010)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, n), 0.012)),
            "LAI": (("y", "x", "time"), np.full((1, 1, n), 3.0)),
            "tts": (("y", "x", "time"), np.full((1, 1, n), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, n), 10.0)),
            "psi": (("y", "x", "time"), np.full((1, 1, n), 0.0)),
            "soil_spectrum": (("y", "x", "time"), np.full((1, 1, n), 1.0)),
            "ala": (("y", "x", "time"), np.array([[[*ALA_VALUES]]])),
        },
        coords={"y": [0], "x": [0], "time": times},
        attrs={"example": "per_pixel_ala_demo"},
    )


def summarize(outputs: xr.Dataset) -> dict[str, object]:
    """Extract key reflectance values for each ALA angle."""
    per_ala: list[dict[str, object]] = []
    for i, ala in enumerate(ALA_VALUES):
        rsot = outputs["rsot"].isel(y=0, x=0, time=i)
        per_ala.append(
            {
                "ala_deg": ala,
                "rsot_650nm": float(rsot.sel(wavelength=650.0, method="nearest")),
                "rsot_865nm": float(rsot.sel(wavelength=865.0, method="nearest")),
                "rsot_1600nm": float(rsot.sel(wavelength=1600.0, method="nearest")),
                "rdd_mean": float(outputs["rdd"].isel(y=0, x=0, time=i).mean().item()),
                "rso_mean": float(outputs["rso"].isel(y=0, x=0, time=i).mean().item()),
            }
        )
    return {
        "product": outputs.attrs["scope_product"],
        "n_pixels": len(ALA_VALUES),
        "ala_values": ALA_VALUES,
        "per_ala": per_ala,
    }


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    scope_root = args.scope_root or _default_scope_root()

    # Build runner with a default LIDF (57 deg) and default_lidfa for fallback
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(
        lidf=lidf,
        device=device,
        dtype=dtype,
        scope_root_path=scope_root,
        default_lidfa=57.0,
    )

    dataset = build_dataset()
    config = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=pd.Timestamp(dataset.time.values[0]),
        end_time=pd.Timestamp(dataset.time.values[-1]),
        device=str(device),
        dtype=dtype,
        chunk_size=len(ALA_VALUES),
    )
    module = ScopeGridDataModule(dataset, config, required_vars=list(dataset.data_vars))

    # Run with per-pixel ALA -- varmap includes "ala" so _resolve_lidf activates
    varmap = {name: name for name in dataset.data_vars}
    outputs = runner.run_dataset(module, varmap=varmap)
    summary = summarize(outputs)
    text = json.dumps(summary, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
