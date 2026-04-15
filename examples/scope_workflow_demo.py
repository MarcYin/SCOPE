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
    parser = argparse.ArgumentParser(
        description="Run the high-level run_scope_dataset(...) workflow with reflectance and fluorescence outputs."
    )
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


def build_dataset(n_wlp: int, n_wle: int) -> xr.Dataset:
    times = pd.date_range("2020-07-01T10:00:00", periods=2, freq="h")
    layers = np.array([1, 2, 3])
    return xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.010)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.010)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, n_wlp), 900.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, n_wlp), 120.0)),
            "etau": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 0.010)),
            "etah": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 0.008)),
        },
        coords={
            "y": [0],
            "x": [0],
            "time": times,
            "layer": layers,
            "wavelength": np.arange(n_wlp),
            "excitation_wavelength": np.arange(n_wle),
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([15.0, 35.0])),
            "directional_psi": ("direction", np.array([10.0, 60.0])),
        },
        attrs={"example": "scope_workflow_demo"},
    )


def summarize(outputs: xr.Dataset) -> dict[str, object]:
    rsot = outputs["rsot"].isel(y=0, x=0, time=0)
    lof = outputs["LoF_"].isel(y=0, x=0, time=0)
    directional = outputs["reflectance_directional_refl_"]
    profile = outputs["fluorescence_profile_Fmin_"]
    summary = {
        "product": outputs.attrs["scope_product"],
        "components": outputs.attrs["scope_components"].split(","),
        "dims": {name: int(size) for name, size in outputs.sizes.items()},
        "rsot_650nm_t0": float(rsot.sel(wavelength=650.0, method="nearest")),
        "LoF_peak_t0": float(lof.max().item()),
        "LoF_peak_wavelength_t0": float(
            lof["fluorescence_wavelength"][int(lof.argmax("fluorescence_wavelength"))].item()
        ),
        "directional_refl_shape": [int(size) for size in directional.shape],
        "fluorescence_profile_shape": [int(size) for size in profile.shape],
        "sample_variables": sorted(list(outputs.data_vars))[:12],
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

    dataset = build_dataset(
        int(runner.fluspect.spectral.wlP.numel()),
        int(runner.fluspect.spectral.wlE.numel()),
    )
    config = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=pd.Timestamp(dataset.time.values[0]),
        end_time=pd.Timestamp(dataset.time.values[-1]),
        device=str(device),
        dtype=dtype,
        chunk_size=2,
    )
    module = ScopeGridDataModule(dataset, config, required_vars=list(dataset.data_vars))
    outputs = runner.run_scope_dataset(
        module,
        varmap={name: name for name in dataset.data_vars},
        scope_options={
            "calc_fluor": 1,
            "calc_planck": 0,
            "calc_directional": 1,
            "calc_vert_profiles": 1,
        },
        nlayers=3,
    )
    summary = summarize(outputs)
    text = json.dumps(summary, indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
