from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if src.exists():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)


_ensure_src_on_path()

from scope import SimulationConfig, ScopeGridRunner, campbell_lidf
from scope.data import ScopeGridDataModule


def _default_scope_root() -> str:
    candidate = Path(__file__).resolve().parents[1] / "upstream" / "SCOPE"
    if not candidate.exists():
        raise FileNotFoundError(f"Expected upstream SCOPE assets at {candidate}")
    return str(candidate)


def _basic_scene_dataset() -> xr.Dataset:
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
    )


def _workflow_dataset(n_wlp: int, n_wle: int) -> xr.Dataset:
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
    )


def _runner() -> ScopeGridRunner:
    dtype = torch.float64
    device = torch.device("cpu")
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    return ScopeGridRunner.from_scope_assets(
        lidf=lidf,
        device=device,
        dtype=dtype,
        scope_root_path=_default_scope_root(),
    )


def _module(dataset: xr.Dataset) -> ScopeGridDataModule:
    config = SimulationConfig(
        roi_bounds=(0.0, 0.0, 1.0, 1.0),
        start_time=pd.Timestamp(dataset.time.values[0]),
        end_time=pd.Timestamp(dataset.time.values[-1]),
        device="cpu",
        dtype=torch.float64,
        chunk_size=max(1, int(dataset.sizes.get("time", 1))),
    )
    return ScopeGridDataModule(dataset, config, required_vars=list(dataset.data_vars))


def _line_svg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    x_label: str,
    y_label: str,
    stroke: str,
) -> str:
    width, height = 720, 420
    left, right, top, bottom = 64, 24, 36, 52
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (width - left - right)

    def sy(value: float) -> float:
        return height - bottom - (value - y_min) / (y_max - y_min) * (height - top - bottom)

    points = " ".join(f"{sx(float(xv)):.2f},{sy(float(yv)):.2f}" for xv, yv in zip(x, y))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffdf8"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222" stroke-width="1.5"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222" stroke-width="1.5"/>
  <polyline fill="none" stroke="{stroke}" stroke-width="3" points="{points}"/>
  <text x="{left}" y="24" font-family="Georgia, serif" font-size="20" fill="#111">{title}</text>
  <text x="{width / 2:.1f}" y="{height - 14}" text-anchor="middle" font-family="Georgia, serif" font-size="15" fill="#333">{x_label}</text>
  <text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="Georgia, serif" font-size="15" fill="#333">{y_label}</text>
  <text x="{left}" y="{height - 30}" font-family="monospace" font-size="12" fill="#555">{x_min:.0f}</text>
  <text x="{width - right}" y="{height - 30}" text-anchor="end" font-family="monospace" font-size="12" fill="#555">{x_max:.0f}</text>
  <text x="{left - 8}" y="{height - bottom}" text-anchor="end" dominant-baseline="middle" font-family="monospace" font-size="12" fill="#555">{y_min:.3g}</text>
  <text x="{left - 8}" y="{top}" text-anchor="end" dominant-baseline="middle" font-family="monospace" font-size="12" fill="#555">{y_max:.3g}</text>
</svg>
"""


def _bar_svg(
    x_labels: list[str],
    y: np.ndarray,
    *,
    title: str,
    y_label: str,
    fill: str,
) -> str:
    width, height = 640, 420
    left, right, top, bottom = 64, 24, 36, 52
    y_max = float(np.max(y))
    if y_max == 0.0:
        y_max = 1.0
    bar_width = (width - left - right) / max(len(x_labels), 1)

    def sy(value: float) -> float:
        return height - bottom - value / y_max * (height - top - bottom)

    bars = []
    for index, (label, value) in enumerate(zip(x_labels, y)):
        x0 = left + index * bar_width + 12
        x1 = left + (index + 1) * bar_width - 12
        y0 = sy(float(value))
        bars.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(x1 - x0, 8):.2f}" height="{height - bottom - y0:.2f}" fill="{fill}"/>'
            f'<text x="{(x0 + x1) / 2:.2f}" y="{height - 28}" text-anchor="middle" font-family="monospace" font-size="12" fill="#555">{label}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffdf8"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#222" stroke-width="1.5"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#222" stroke-width="1.5"/>
  {''.join(bars)}
  <text x="{left}" y="24" font-family="Georgia, serif" font-size="20" fill="#111">{title}</text>
  <text x="18" y="{height / 2:.1f}" transform="rotate(-90 18 {height / 2:.1f})" text-anchor="middle" font-family="Georgia, serif" font-size="15" fill="#333">{y_label}</text>
  <text x="{left - 8}" y="{height - bottom}" text-anchor="end" dominant-baseline="middle" font-family="monospace" font-size="12" fill="#555">0</text>
  <text x="{left - 8}" y="{top}" text-anchor="end" dominant-baseline="middle" font-family="monospace" font-size="12" fill="#555">{y_max:.3g}</text>
</svg>
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assets_dir = repo_root / "docs" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    runner = _runner()

    reflectance_outputs = runner.run_dataset(_module(_basic_scene_dataset()), varmap={name: name for name in _basic_scene_dataset().data_vars})
    rsot = reflectance_outputs["rsot"].isel(y=0, x=0, time=0)
    (assets_dir / "basic_scene_reflectance.svg").write_text(
        _line_svg(
            rsot["wavelength"].values.astype(float),
            rsot.values.astype(float),
            title="Basic Scene Reflectance Spectrum",
            x_label="Wavelength (nm)",
            y_label="rsot (-)",
            stroke="#14532d",
        ),
        encoding="utf-8",
    )

    workflow_dataset = _workflow_dataset(
        int(runner.fluspect.spectral.wlP.numel()),
        int(runner.fluspect.spectral.wlE.numel()),
    )
    workflow_outputs = runner.run_scope_dataset(
        _module(workflow_dataset),
        varmap={name: name for name in workflow_dataset.data_vars},
        scope_options={"calc_fluor": 1, "calc_planck": 0, "calc_directional": 1, "calc_vert_profiles": 1},
        nlayers=3,
    )
    lof = workflow_outputs["LoF_"].isel(y=0, x=0, time=0)
    (assets_dir / "scope_workflow_fluorescence.svg").write_text(
        _line_svg(
            lof["fluorescence_wavelength"].values.astype(float),
            lof.values.astype(float),
            title="Directional Fluorescence Spectrum",
            x_label="Fluorescence wavelength (nm)",
            y_label="LoF_ (W m-2 um-1 sr-1)",
            stroke="#9a3412",
        ),
        encoding="utf-8",
    )

    profile = workflow_outputs["fluorescence_profile_layer_fluorescence"].isel(y=0, x=0, time=0)
    labels = [str(int(layer)) for layer in workflow_outputs["layer"].values]
    (assets_dir / "scope_workflow_profile.svg").write_text(
        _bar_svg(
            labels,
            profile.values.astype(float),
            title="Layer Fluorescence Profile",
            y_label="layer_fluorescence (W m-2)",
            fill="#1d4ed8",
        ),
        encoding="utf-8",
    )

    for path in sorted(assets_dir.glob("*.svg")):
        print(path)


if __name__ == "__main__":
    main()
