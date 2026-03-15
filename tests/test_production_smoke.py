from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

from scope import ScopeGridDataModule, ScopeGridRunner, SimulationConfig, write_netcdf_dataset
from scope.canopy.foursail import FourSAILModel, campbell_lidf
from scope.io import NetCDFWriteOptions
from scope.spectral.fluspect import FluspectModel, OptiPar, SpectralGrids


def _spectral(device, dtype):
    wlP = torch.linspace(400.0, 2400.0, 96, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 850.0, 24, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 750.0, 24, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


def _optipar(spectral):
    wl = spectral.wlP
    base = torch.linspace(0.0, 1.0, wl.numel(), dtype=wl.dtype, device=wl.device)
    return OptiPar(
        nr=1.4 + 0.05 * torch.sin(base),
        Kab=0.01 + 0.005 * torch.cos(base),
        Kca=0.008 + 0.003 * torch.sin(base * 2.0),
        KcaV=0.008 + 0.003 * torch.sin(base * 2.0) * 0.95,
        KcaZ=0.008 + 0.003 * torch.sin(base * 2.0) * 1.05,
        Kdm=0.005 + 0.002 * torch.cos(base * 3.0),
        Kw=0.002 + 0.001 * torch.sin(base * 4.0),
        Ks=0.001 + 0.0005 * torch.cos(base * 5.0),
        Kant=0.0002 + 0.0001 * torch.sin(base * 6.0),
        phi=torch.full_like(wl, 0.5),
    )


def _build_realworld_like_dataset(spectral: SpectralGrids) -> xr.Dataset:
    rng = np.random.default_rng(42)
    times = pd.date_range("2020-07-01T09:00:00", periods=6, freq="2h")
    y = np.linspace(51.40, 51.55, 4)
    x = np.linspace(-0.25, -0.05, 3)
    layer = np.arange(4)
    nwl = int(spectral.wlP.numel())
    nwl_e = int(spectral.wlE.numel())

    shape = (y.size, x.size, times.size)
    layer_shape = (*shape, layer.size)

    dataset = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), rng.uniform(35.0, 60.0, size=shape)),
            "Cw": (("y", "x", "time"), rng.uniform(0.008, 0.020, size=shape)),
            "Cdm": (("y", "x", "time"), rng.uniform(0.008, 0.018, size=shape)),
            "fqe": (("y", "x", "time"), rng.uniform(0.008, 0.018, size=shape)),
            "LAI": (("y", "x", "time"), rng.uniform(1.0, 5.5, size=shape)),
            "tts": (("y", "x", "time"), rng.uniform(20.0, 50.0, size=shape)),
            "tto": (("y", "x", "time"), rng.uniform(5.0, 35.0, size=shape)),
            "psi": (("y", "x", "time"), rng.uniform(0.0, 90.0, size=shape)),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.uniform(0.10, 0.28, size=(*shape, nwl))),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), rng.uniform(0.8, 1.8, size=(*shape, nwl_e))),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), rng.uniform(0.1, 0.5, size=(*shape, nwl_e))),
            "etau": (("y", "x", "time", "layer"), rng.uniform(0.010, 0.018, size=layer_shape)),
            "etah": (("y", "x", "time", "layer"), rng.uniform(0.008, 0.015, size=layer_shape)),
            "Tcu": (("y", "x", "time", "layer"), rng.uniform(24.0, 31.0, size=layer_shape)),
            "Tch": (("y", "x", "time", "layer"), rng.uniform(22.0, 28.0, size=layer_shape)),
            "Tsu": (("y", "x", "time"), rng.uniform(23.0, 32.0, size=shape)),
            "Tsh": (("y", "x", "time"), rng.uniform(20.0, 27.0, size=shape)),
        },
        coords={
            "y": y,
            "x": x,
            "time": times,
            "wavelength": spectral.wlP.cpu().numpy(),
            "excitation_wavelength": spectral.wlE.cpu().numpy(),
            "layer": layer,
        },
        attrs={
            "site": "production-smoke",
            "calc_fluor": 1,
            "calc_planck": 1,
        },
    )
    return dataset


def test_scope_workflow_realworld_like_smoke(tmp_path: Path):
    device = torch.device("cpu")
    dtype = torch.float32
    spectral = _spectral(device, dtype)
    fluspect = FluspectModel(spectral, _optipar(spectral), dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner(fluspect, FourSAILModel(lidf=lidf), lidf=lidf)

    dataset = _build_realworld_like_dataset(spectral)
    config = SimulationConfig(
        roi_bounds=(float(dataset.x.min()), float(dataset.y.min()), float(dataset.x.max()), float(dataset.y.max())),
        start_time=pd.Timestamp(dataset.time.values[0]),
        end_time=pd.Timestamp(dataset.time.values[-1]),
        device="cpu",
        dtype=dtype,
        chunk_size=7,
    )
    module = ScopeGridDataModule(dataset, config, required_vars=list(dataset.data_vars))
    outputs = runner.run_scope_dataset(module, varmap={name: name for name in dataset.data_vars}, nlayers=4)

    assert outputs.attrs["scope_product"] == "scope_workflow"
    assert outputs.attrs["scope_components"] == "reflectance,fluorescence,thermal"
    assert outputs["rsot"].dims == ("y", "x", "time", "wavelength")
    assert outputs["LoF_"].dims == ("y", "x", "time", "fluorescence_wavelength")
    assert outputs["Lot_"].dims == ("y", "x", "time", "thermal_wavelength")
    assert bool(np.isfinite(outputs["rsot"].values).all())
    assert bool(np.isfinite(outputs["LoF_"].values).all())
    assert bool(np.isfinite(outputs["Lot_"].values).all())

    output_path = write_netcdf_dataset(
        outputs,
        tmp_path / "production_smoke.nc",
        options=NetCDFWriteOptions(engine="scipy", compression=False),
    )
    with xr.open_dataset(output_path, engine="scipy") as roundtrip:
        assert roundtrip.attrs["Conventions"] == "CF-1.10"
        assert roundtrip.attrs["scope_product"] == "scope_workflow"
        assert roundtrip["time"].attrs["standard_name"] == "time"
        assert roundtrip["x"].attrs["axis"] == "X"
        assert roundtrip["y"].attrs["axis"] == "Y"
