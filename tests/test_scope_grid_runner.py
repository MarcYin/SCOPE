import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from scope.biochem import LeafBiochemistryInputs, LeafBiochemistryResult
from scope.canopy.foursail import FourSAILModel, campbell_lidf
from scope.canopy.fluorescence import CanopyFluorescenceModel, CanopyFluorescenceResult
from scope.canopy.reflectance import CanopyReflectanceModel
from scope.energy import (
    CanopyEnergyBalanceResult,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
)
from scope.canopy.thermal import CanopyThermalRadianceModel, CanopyThermalRadianceResult, default_thermal_wavelengths
from scope.config import SimulationConfig
from scope.data import ScopeGridDataModule
from scope.runners.grid import ScopeGridRunner
from scope.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids
from scope.spectral.loaders import load_soil_spectra


CANOPY_KEYS = [
    "rdd",
    "tdd",
    "rsd",
    "tsd",
    "rdo",
    "tdo",
    "rso",
    "rsos",
    "rsod",
    "rddt",
    "rsdt",
    "rdot",
    "rsodt",
    "rsost",
    "rsot",
    "tss",
    "too",
    "tsstoo",
    "gammasdf",
    "gammasdb",
    "gammaso",
]


def _spectral(device, dtype):
    wlP = torch.linspace(400.0, 700.0, 32, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 740.0, 16, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 700.0, 16, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


def _optipar(spectral):
    wl = spectral.wlP
    base = torch.linspace(0, 1, wl.numel(), dtype=wl.dtype, device=wl.device)
    return OptiPar(
        nr=1.4 + 0.05 * torch.sin(base),
        Kab=0.01 + 0.005 * torch.cos(base),
        Kca=0.008 + 0.003 * torch.sin(base * 2),
        KcaV=0.008 + 0.003 * torch.sin(base * 2) * 0.95,
        KcaZ=0.008 + 0.003 * torch.sin(base * 2) * 1.05,
        Kdm=0.005 + 0.002 * torch.cos(base * 3),
        Kw=0.002 + 0.001 * torch.sin(base * 4),
        Ks=0.001 + 0.0005 * torch.cos(base * 5),
        Kant=0.0002 + 0.0001 * torch.sin(base * 6),
        phi=torch.full_like(wl, 0.5),
    )


def _build_execution_mode_runner(*, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64) -> tuple[ScopeGridRunner, SpectralGrids]:
    device = torch.device(device)
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    return ScopeGridRunner(fluspect, sail, lidf=lidf), spectral


def _build_execution_mode_dataset(spectral: SpectralGrids) -> xr.Dataset:
    times = pd.date_range("2020-07-01", periods=3, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = int(spectral.wlP.numel())
    nwl_e = int(spectral.wlE.numel())
    return xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.array([[[45.0, 41.0, 38.0]]])),
            "Cw": (("y", "x", "time"), np.array([[[0.010, 0.012, 0.014]]])),
            "Cdm": (("y", "x", "time"), np.array([[[0.012, 0.014, 0.016]]])),
            "fqe": (("y", "x", "time"), np.array([[[0.010, 0.012, 0.014]]])),
            "LAI": (("y", "x", "time"), np.array([[[2.1, 2.8, 3.4]]])),
            "tts": (("y", "x", "time"), np.array([[[30.0, 34.0, 38.0]]])),
            "tto": (("y", "x", "time"), np.array([[[12.0, 18.0, 24.0]]])),
            "psi": (("y", "x", "time"), np.array([[[10.0, 22.0, 35.0]]])),
            "soil_refl": (
                ("y", "x", "time", "wavelength"),
                np.linspace(0.12, 0.28, 3 * nwl, dtype=np.float64).reshape(1, 1, 3, nwl),
            ),
            "excitation": (
                ("y", "x", "time", "excitation_wavelength"),
                np.linspace(300.0, 420.0, 3 * nwl_e, dtype=np.float64).reshape(1, 1, 3, nwl_e),
            ),
            "Esun_": (
                ("y", "x", "time", "excitation_wavelength"),
                np.linspace(850.0, 1200.0, 3 * nwl_e, dtype=np.float64).reshape(1, 1, 3, nwl_e),
            ),
            "Esky_": (
                ("y", "x", "time", "excitation_wavelength"),
                np.linspace(120.0, 220.0, 3 * nwl_e, dtype=np.float64).reshape(1, 1, 3, nwl_e),
            ),
            "Vcmax25": (("y", "x", "time"), np.array([[[72.0, 66.0, 60.0]]])),
            "BallBerrySlope": (("y", "x", "time"), np.array([[[9.0, 8.2, 7.4]]])),
            "Csu": (("y", "x", "time"), np.array([[[392.0, 397.0, 402.0]]])),
            "Csh": (("y", "x", "time"), np.array([[[388.0, 392.0, 396.0]]])),
            "ebu": (("y", "x", "time"), np.array([[[17.0, 18.0, 19.0]]])),
            "ebh": (("y", "x", "time"), np.array([[[15.5, 16.5, 17.5]]])),
            "Oa": (("y", "x", "time"), np.full((1, 1, 3), 209.0)),
            "p": (("y", "x", "time"), np.array([[[970.0, 968.0, 966.0]]])),
            "Tcu": (("y", "x", "time"), np.array([[[25.0, 25.8, 26.6]]])),
            "Tch": (("y", "x", "time"), np.array([[[24.2, 24.8, 25.4]]])),
            "Tsu": (("y", "x", "time"), np.array([[[26.0, 26.5, 27.0]]])),
            "Tsh": (("y", "x", "time"), np.array([[[23.5, 24.0, 24.5]]])),
            "rho_thermal": (("y", "x", "time"), np.array([[[0.010, 0.012, 0.014]]])),
            "tau_thermal": (("y", "x", "time"), np.array([[[0.011, 0.013, 0.015]]])),
            "rs_thermal": (("y", "x", "time"), np.array([[[0.060, 0.058, 0.056]]])),
        },
        coords={
            "y": y,
            "x": x,
            "time": times,
            "wavelength": np.arange(nwl),
            "excitation_wavelength": np.arange(nwl_e),
        },
    )


def _build_execution_mode_module(
    dataset: xr.Dataset,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    chunk_size: int,
) -> ScopeGridDataModule:
    cfg = SimulationConfig(
        roi_bounds=(0, 0, 1, 1),
        start_time=pd.Timestamp(dataset["time"].values[0]),
        end_time=pd.Timestamp(dataset["time"].values[-1]),
        device=str(device),
        dtype=dtype,
        chunk_size=chunk_size,
    )
    return ScopeGridDataModule(dataset, cfg, required_vars=list(dataset.data_vars))


def _run_execution_mode_workflow(
    runner: ScopeGridRunner,
    module: ScopeGridDataModule,
    workflow: str,
) -> dict[str, torch.Tensor]:
    varmap = {name: name for name in module.dataset.data_vars}
    if workflow == "reflectance":
        return runner.run(module, varmap=varmap, nlayers=4)
    if workflow == "fluorescence":
        return runner.run_fluorescence(module, varmap=varmap)
    if workflow == "layered_fluorescence":
        return runner.run_layered_fluorescence(module, varmap=varmap, nlayers=4)
    if workflow == "biochemical_fluorescence":
        return runner.run_biochemical_fluorescence(module, varmap=varmap, nlayers=4)
    if workflow == "thermal":
        return runner.run_thermal(module, varmap=varmap, nlayers=4)
    raise ValueError(f"Unsupported workflow '{workflow}'")


def _build_coupled_execution_mode_dataset(spectral: SpectralGrids) -> xr.Dataset:
    times = pd.date_range("2020-07-01", periods=3, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = int(spectral.wlP.numel())
    return xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.array([[[45.0, 41.0, 38.0]]])),
            "Cw": (("y", "x", "time"), np.array([[[0.010, 0.012, 0.014]]])),
            "Cdm": (("y", "x", "time"), np.array([[[0.012, 0.014, 0.016]]])),
            "fqe": (("y", "x", "time"), np.array([[[0.010, 0.012, 0.014]]])),
            "LAI": (("y", "x", "time"), np.array([[[2.3, 2.8, 3.2]]])),
            "tts": (("y", "x", "time"), np.array([[[30.0, 33.0, 37.0]]])),
            "tto": (("y", "x", "time"), np.array([[[15.0, 18.0, 22.0]]])),
            "psi": (("y", "x", "time"), np.array([[[10.0, 20.0, 30.0]]])),
            "soil_refl": (
                ("y", "x", "time", "wavelength"),
                np.linspace(0.18, 0.24, 3 * nwl, dtype=np.float64).reshape(1, 1, 3, nwl),
            ),
            "Esun_sw": (
                ("y", "x", "time", "wavelength"),
                np.linspace(1050.0, 1300.0, 3 * nwl, dtype=np.float64).reshape(1, 1, 3, nwl),
            ),
            "Esky_sw": (
                ("y", "x", "time", "wavelength"),
                np.linspace(140.0, 220.0, 3 * nwl, dtype=np.float64).reshape(1, 1, 3, nwl),
            ),
            "Ta": (("y", "x", "time"), np.array([[[25.0, 26.0, 27.0]]])),
            "ea": (("y", "x", "time"), np.array([[[20.0, 20.8, 21.5]]])),
            "Ca": (("y", "x", "time"), np.array([[[390.0, 397.0, 404.0]]])),
            "Oa": (("y", "x", "time"), np.full((1, 1, 3), 209.0)),
            "p": (("y", "x", "time"), np.array([[[970.0, 967.5, 965.0]]])),
            "z": (("y", "x", "time"), np.full((1, 1, 3), 10.0)),
            "u": (("y", "x", "time"), np.array([[[2.0, 2.6, 3.2]]])),
            "Cd": (("y", "x", "time"), np.full((1, 1, 3), 0.2)),
            "rwc": (("y", "x", "time"), np.full((1, 1, 3), 0.5)),
            "z0m": (("y", "x", "time"), np.full((1, 1, 3), 0.15)),
            "d": (("y", "x", "time"), np.full((1, 1, 3), 1.3)),
            "h": (("y", "x", "time"), np.full((1, 1, 3), 2.0)),
            "kV": (("y", "x", "time"), np.full((1, 1, 3), 0.15)),
            "rss": (("y", "x", "time"), np.full((1, 1, 3), 120.0)),
            "rbs": (("y", "x", "time"), np.full((1, 1, 3), 12.0)),
            "Vcmax25": (("y", "x", "time"), np.array([[[70.0, 64.0, 58.0]]])),
            "BallBerrySlope": (("y", "x", "time"), np.array([[[9.0, 8.25, 7.5]]])),
        },
        coords={
            "y": y,
            "x": x,
            "time": times,
            "wavelength": np.arange(nwl),
        },
        attrs={"site": "coupled-execution-mode"},
    )


def _run_coupled_execution_mode_workflow(
    runner: ScopeGridRunner,
    module: ScopeGridDataModule,
    workflow: str,
) -> dict[str, torch.Tensor]:
    varmap = {name: name for name in module.dataset.data_vars}
    if workflow == "energy_balance_fluorescence":
        return runner.run_energy_balance_fluorescence(
            module,
            varmap=varmap,
            energy_options=EnergyBalanceOptions(max_iter=50),
            nlayers=4,
        )
    if workflow == "energy_balance_thermal":
        return runner.run_energy_balance_thermal(
            module,
            varmap=varmap,
            energy_options=EnergyBalanceOptions(max_iter=50),
            nlayers=4,
        )
    raise ValueError(f"Unsupported coupled workflow '{workflow}'")


def _assert_tensor_mapping_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    assert set(actual) == set(expected)
    for name in actual:
        lhs = actual[name].detach().cpu()
        rhs = expected[name].detach().cpu()
        assert lhs.shape == rhs.shape, name
        assert torch.allclose(lhs, rhs, atol=atol, rtol=rtol), name


def _assert_coupled_runner_outputs_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    radiative_keys: set[str],
    atol: float,
    rtol: float,
) -> None:
    stable_energy_keys = {
        "Pnu_Cab",
        "Pnh_Cab",
        "Rnuc_sw",
        "Rnhc_sw",
        "Rnus_sw",
        "Rnhs_sw",
        "canopyemis",
        "Csu",
        "Csh",
        "ebu",
        "ebh",
        "Tcu",
        "Tch",
        "Tsu",
        "Tsh",
        "L",
        "converged",
    }
    compare_keys = radiative_keys | stable_energy_keys
    assert compare_keys.issubset(actual)
    assert compare_keys.issubset(expected)
    _assert_tensor_mapping_close(
        {name: actual[name] for name in sorted(compare_keys)},
        {name: expected[name] for name in sorted(compare_keys)},
        atol=atol,
        rtol=rtol,
    )


def test_scope_grid_runner_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=3, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(0)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 3), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 3), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 3), 0.012)),
            "LAI": (("y", "x", "time"), np.linspace(2.0, 3.0, 3).reshape(1, 1, 3)),
            "tts": (("y", "x", "time"), np.full((1, 1, 3), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 3), 20.0)),
            "psi": (("y", "x", "time"), np.linspace(5.0, 15.0, 3).reshape(1, 1, 3)),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 3, nwl)) * 0.2 + 0.1),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(nwl)},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "soil_refl"])
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)
    outputs = runner.run(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
        },
    )

    stacked = data.stack(batch=("y", "x", "time"))
    reflectance_model = CanopyReflectanceModel(fluspect, sail, lidf=lidf, default_hotspot=runner.default_hotspot)
    manual_outputs: dict[str, list[torch.Tensor]] = {"leaf_refl": [], "leaf_tran": [], **{key: [] for key in CANOPY_KEYS}}
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_tensor = torch.tensor(stacked["soil_refl"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0)
        reflectance_out = reflectance_model(
            leafbio,
            soil_tensor,
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(reflectance_out, key))

    expected_keys = {"leaf_refl", "leaf_tran", *CANOPY_KEYS}
    assert set(outputs) == expected_keys
    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_run_dataset_preserves_xarray_metadata():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(2)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 3.0]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 2, nwl)) * 0.2 + 0.1),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(nwl)},
        attrs={"site": "demo"},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=1)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "soil_refl"])
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)

    flat_outputs = runner.run(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
        },
    )
    dataset_outputs = runner.run_dataset(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
        },
    )

    assert dataset_outputs["rso"].dims == ("y", "x", "time", "wavelength")
    assert dataset_outputs.attrs["site"] == "demo"
    assert dataset_outputs.attrs["scope_product"] == "reflectance"
    assert np.allclose(dataset_outputs["wavelength"].values, spectral.wlP.cpu().numpy())
    expected_rso = flat_outputs["rso"].cpu().numpy().reshape(1, 1, 2, nwl)
    assert np.allclose(dataset_outputs["rso"].values, expected_rso)


def test_scope_grid_runner_directional_reflectance_dataset_preserves_direction_coord():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(12)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 3.0]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 2, nwl)) * 0.2 + 0.1),
            "Esun_": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, nwl), 900.0)),
            "Esky_": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, nwl), 120.0)),
        },
        coords={
            "y": y,
            "x": x,
            "time": times,
            "wavelength": np.arange(nwl),
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([10.0, 30.0])),
            "directional_psi": ("direction", np.array([5.0, 45.0])),
        },
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "soil_refl", "Esun_", "Esky_"])
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)

    flat_outputs = runner.run_directional_reflectance(
        module,
        varmap={"Cab": "Cab", "Cw": "Cw", "Cdm": "Cdm", "LAI": "LAI", "tts": "tts", "soil_refl": "soil_refl", "Esun_": "Esun_", "Esky_": "Esky_"},
    )
    dataset_outputs = runner.run_directional_reflectance_dataset(
        module,
        varmap={"Cab": "Cab", "Cw": "Cw", "Cdm": "Cdm", "LAI": "LAI", "tts": "tts", "soil_refl": "soil_refl", "Esun_": "Esun_", "Esky_": "Esky_"},
    )

    assert dataset_outputs.attrs["scope_product"] == "directional_reflectance"
    assert dataset_outputs["refl_"].dims == ("y", "x", "time", "direction", "wavelength")
    assert dataset_outputs["rso_"].dims == ("y", "x", "time", "direction", "wavelength")
    assert np.allclose(dataset_outputs["directional_tto"].values, np.array([10.0, 30.0]))
    assert np.allclose(dataset_outputs["directional_psi"].values, np.array([5.0, 45.0]))
    expected_refl = flat_outputs["refl_"].cpu().numpy().reshape(1, 1, 2, 2, nwl)
    assert np.allclose(dataset_outputs["refl_"].values, expected_refl)


def test_scope_grid_runner_reflectance_profiles_dataset_preserves_layer_interface():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(7)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 2, nwl)) * 0.2 + 0.1),
            "Esun_": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, nwl), 900.0)),
            "Esky_": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, nwl), 120.0)),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(nwl)},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "soil_refl", "Esun_", "Esky_"])
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)

    flat_outputs = runner.run_reflectance_profiles(
        module,
        varmap={"Cab": "Cab", "Cw": "Cw", "Cdm": "Cdm", "LAI": "LAI", "tts": "tts", "tto": "tto", "psi": "psi", "soil_refl": "soil_refl", "Esun_": "Esun_", "Esky_": "Esky_"},
        nlayers=3,
    )
    dataset_outputs = runner.run_reflectance_profiles_dataset(
        module,
        varmap={"Cab": "Cab", "Cw": "Cw", "Cdm": "Cdm", "LAI": "LAI", "tts": "tts", "tto": "tto", "psi": "psi", "soil_refl": "soil_refl", "Esun_": "Esun_", "Esky_": "Esky_"},
        nlayers=3,
    )

    assert dataset_outputs.attrs["scope_product"] == "reflectance_profiles"
    assert dataset_outputs["Ps"].dims == ("y", "x", "time", "layer_interface")
    assert dataset_outputs["Es_"].dims == ("y", "x", "time", "layer_interface", "wavelength")
    assert np.array_equal(dataset_outputs["layer_interface"].values, np.arange(4))
    expected_ps = flat_outputs["Ps"].cpu().numpy().reshape(1, 1, 2, 4)
    expected_es = flat_outputs["Es_"].cpu().numpy().reshape(1, 1, 2, 4, nwl)
    assert np.allclose(dataset_outputs["Ps"].values, expected_ps)
    assert np.allclose(dataset_outputs["Es_"].values, expected_es)


def test_scope_grid_runner_reflectance_respects_explicit_nlayers():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.full((1, 1, 2), 6.5)),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 2, nwl)) * 0.2 + 0.1),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(nwl)},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "soil_refl"])
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)
    outputs = runner.run(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
        },
        nlayers=8,
    )

    stacked = data.stack(batch=("y", "x", "time"))
    reflectance_model = CanopyReflectanceModel(fluspect, sail, lidf=lidf, default_hotspot=runner.default_hotspot)
    manual_outputs: dict[str, list[torch.Tensor]] = {"leaf_refl": [], "leaf_tran": [], **{key: [] for key in CANOPY_KEYS}}
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_tensor = torch.tensor(stacked["soil_refl"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0)
        reflectance_out = reflectance_model(
            leafbio,
            soil_tensor,
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
            nlayers=8,
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(reflectance_out, key))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_energy_balance_thermal_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.3, 2.6]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 15.0)),
            "psi": (("y", "x", "time"), np.array([[[10.0, 20.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 0.2)),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 1200.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 180.0)),
            "Ta": (("y", "x", "time"), np.full((1, 1, 2), 25.0)),
            "ea": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "Ca": (("y", "x", "time"), np.full((1, 1, 2), 390.0)),
            "Oa": (("y", "x", "time"), np.full((1, 1, 2), 209.0)),
            "p": (("y", "x", "time"), np.full((1, 1, 2), 970.0)),
            "z": (("y", "x", "time"), np.full((1, 1, 2), 10.0)),
            "u": (("y", "x", "time"), np.full((1, 1, 2), 2.0)),
            "Cd": (("y", "x", "time"), np.full((1, 1, 2), 0.2)),
            "rwc": (("y", "x", "time"), np.full((1, 1, 2), 0.5)),
            "z0m": (("y", "x", "time"), np.full((1, 1, 2), 0.15)),
            "d": (("y", "x", "time"), np.full((1, 1, 2), 1.3)),
            "h": (("y", "x", "time"), np.full((1, 1, 2), 2.0)),
            "kV": (("y", "x", "time"), np.full((1, 1, 2), 0.15)),
            "rss": (("y", "x", "time"), np.full((1, 1, 2), 120.0)),
            "rbs": (("y", "x", "time"), np.full((1, 1, 2), 12.0)),
            "Vcmax25": (("y", "x", "time"), np.full((1, 1, 2), 70.0)),
            "BallBerrySlope": (("y", "x", "time"), np.full((1, 1, 2), 9.0)),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(spectral.wlP.numel())},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(
        data,
        cfg,
        required_vars=[
            "Cab",
            "Cw",
            "Cdm",
            "LAI",
            "tts",
            "tto",
            "psi",
            "soil_refl",
            "Esun_sw",
            "Esky_sw",
            "Ta",
            "ea",
            "Ca",
            "Oa",
            "p",
            "z",
            "u",
            "Cd",
            "rwc",
            "z0m",
            "d",
            "h",
            "kV",
            "rss",
            "rbs",
            "Vcmax25",
            "BallBerrySlope",
        ],
    )
    outputs = runner.run_energy_balance_thermal(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
            "Esun_sw": "Esun_sw",
            "Esky_sw": "Esky_sw",
            "Ta": "Ta",
            "ea": "ea",
            "Ca": "Ca",
            "Oa": "Oa",
            "p": "p",
            "z": "z",
            "u": "u",
            "Cd": "Cd",
            "rwc": "rwc",
            "z0m": "z0m",
            "d": "d",
            "h": "h",
            "kV": "kV",
            "rss": "rss",
            "rbs": "rbs",
            "Vcmax25": "Vcmax25",
            "BallBerrySlope": "BallBerrySlope",
        },
        energy_options=EnergyBalanceOptions(max_iter=50),
        nlayers=4,
    )

    physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
    energy_fields = [name for name in CanopyEnergyBalanceResult.__dataclass_fields__ if name not in {"sunlit", "shaded", "Tsold"}]
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {
        **{name: [] for name in CanopyThermalRadianceResult.__dataclass_fields__},
        **{name: [] for name in energy_fields},
        **{f"sunlit_{name}": [] for name in physiology_fields},
        **{f"shaded_{name}": [] for name in physiology_fields},
    }
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
        )
        biochem = LeafBiochemistryInputs(
            Vcmax25=torch.tensor([float(stacked["Vcmax25"].sel(**idx))], device=device, dtype=dtype),
            BallBerrySlope=torch.tensor([float(stacked["BallBerrySlope"].sel(**idx))], device=device, dtype=dtype),
        )
        result = runner.energy_balance_model.solve_thermal(
            leafbio,
            biochem,
            torch.tensor(stacked["soil_refl"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor(stacked["Esun_sw"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor(stacked["Esky_sw"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            meteo=EnergyBalanceMeteo(
                Ta=torch.tensor([float(stacked["Ta"].sel(**idx))], device=device, dtype=dtype),
                ea=torch.tensor([float(stacked["ea"].sel(**idx))], device=device, dtype=dtype),
                Ca=torch.tensor([float(stacked["Ca"].sel(**idx))], device=device, dtype=dtype),
                Oa=torch.tensor([float(stacked["Oa"].sel(**idx))], device=device, dtype=dtype),
                p=torch.tensor([float(stacked["p"].sel(**idx))], device=device, dtype=dtype),
                z=torch.tensor([float(stacked["z"].sel(**idx))], device=device, dtype=dtype),
                u=torch.tensor([float(stacked["u"].sel(**idx))], device=device, dtype=dtype),
            ),
            canopy=EnergyBalanceCanopy(
                Cd=torch.tensor([float(stacked["Cd"].sel(**idx))], device=device, dtype=dtype),
                rwc=torch.tensor([float(stacked["rwc"].sel(**idx))], device=device, dtype=dtype),
                z0m=torch.tensor([float(stacked["z0m"].sel(**idx))], device=device, dtype=dtype),
                d=torch.tensor([float(stacked["d"].sel(**idx))], device=device, dtype=dtype),
                h=torch.tensor([float(stacked["h"].sel(**idx))], device=device, dtype=dtype),
                kV=torch.tensor([float(stacked["kV"].sel(**idx))], device=device, dtype=dtype),
            ),
            soil=EnergyBalanceSoil(
                rss=torch.tensor([float(stacked["rss"].sel(**idx))], device=device, dtype=dtype),
                rbs=torch.tensor([float(stacked["rbs"].sel(**idx))], device=device, dtype=dtype),
            ),
            options=EnergyBalanceOptions(max_iter=50),
            nlayers=4,
        )
        for name in CanopyThermalRadianceResult.__dataclass_fields__:
            manual_outputs[name].append(getattr(result.thermal, name))
        for name in energy_fields:
            manual_outputs[name].append(getattr(result.energy, name))
        for name in physiology_fields:
            manual_outputs[f"sunlit_{name}"].append(getattr(result.energy.sunlit, name))
            manual_outputs[f"shaded_{name}"].append(getattr(result.energy.shaded, name))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_fluorescence_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "excitation": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
        },
        coords={"y": y, "x": x, "time": times, "excitation_wavelength": np.arange(n_wle)},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(
        data,
        cfg,
        required_vars=["Cab", "Cw", "Cdm", "fqe", "LAI", "tts", "tto", "psi", "soil_spectrum", "excitation"],
    )
    outputs = runner.run_fluorescence(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "fqe": "fqe",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_spectrum": "soil_spectrum",
            "excitation": "excitation",
        },
    )

    fluorescence_model = CanopyFluorescenceModel(runner.reflectance_model)
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {key: [] for key in outputs}
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
            fqe=torch.tensor([float(stacked["fqe"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_idx = torch.tensor([float(stacked["soil_spectrum"].sel(**idx))], device=device, dtype=dtype)
        excitation = torch.tensor(stacked["excitation"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0)
        fluorescence_out = fluorescence_model(
            leafbio,
            runner.soil_spectra.batch(soil_idx),
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            excitation,
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(fluorescence_out, key))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_layered_fluorescence_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.arange(3)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
            "etau": (("y", "x", "time", "layer"), np.ones((1, 1, 2, 3))),
            "etah": (("y", "x", "time", "layer"), np.ones((1, 1, 2, 3))),
        },
        coords={"y": y, "x": x, "time": times, "excitation_wavelength": np.arange(n_wle), "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(
        data,
        cfg,
        required_vars=["Cab", "Cw", "Cdm", "fqe", "LAI", "tts", "tto", "psi", "soil_spectrum", "Esun_", "Esky_", "etau", "etah"],
    )
    outputs = runner.run_layered_fluorescence(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "fqe": "fqe",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_spectrum": "soil_spectrum",
            "Esun_": "Esun_",
            "Esky_": "Esky_",
            "etau": "etau",
            "etah": "etah",
        },
    )

    fluorescence_model = CanopyFluorescenceModel(runner.reflectance_model)
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {key: [] for key in outputs}
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
            fqe=torch.tensor([float(stacked["fqe"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_idx = torch.tensor([float(stacked["soil_spectrum"].sel(**idx))], device=device, dtype=dtype)
        fluorescence_out = fluorescence_model.layered(
            leafbio,
            runner.soil_spectra.batch(soil_idx),
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor(stacked["Esun_"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor(stacked["Esky_"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            etau=torch.tensor(stacked["etau"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            etah=torch.tensor(stacked["etah"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(fluorescence_out, key))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_thermal_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.arange(3)
    data = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Tsu": (("y", "x", "time"), np.full((1, 1, 2), 27.0)),
            "Tsh": (("y", "x", "time"), np.full((1, 1, 2), 21.0)),
        },
        coords={"y": y, "x": x, "time": times, "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["LAI", "tts", "tto", "psi", "Tcu", "Tch", "Tsu", "Tsh"])
    outputs = runner.run_thermal(
        module,
        varmap={
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "Tcu": "Tcu",
            "Tch": "Tch",
            "Tsu": "Tsu",
            "Tsh": "Tsh",
        },
    )

    thermal_model = CanopyThermalRadianceModel(runner.reflectance_model)
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {key: [] for key in outputs}
    for label in stacked["LAI"].indexes["batch"]:
        idx = dict(batch=label)
        thermal_out = thermal_model(
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor(stacked["Tcu"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor(stacked["Tch"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor([float(stacked["Tsu"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["Tsh"].sel(**idx))], device=device, dtype=dtype),
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(thermal_out, key))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_thermal_dataset_preserves_layered_dims():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Tsu": (("y", "x", "time"), np.full((1, 1, 2), 27.0)),
            "Tsh": (("y", "x", "time"), np.full((1, 1, 2), 21.0)),
        },
        coords={"y": y, "x": x, "time": times, "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["LAI", "tts", "tto", "psi", "Tcu", "Tch", "Tsu", "Tsh"])
    flat_outputs = runner.run_thermal(
        module,
        varmap={
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "Tcu": "Tcu",
            "Tch": "Tch",
            "Tsu": "Tsu",
            "Tsh": "Tsh",
        },
    )
    dataset_outputs = runner.run_thermal_dataset(
        module,
        varmap={
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "Tcu": "Tcu",
            "Tch": "Tch",
            "Tsu": "Tsu",
            "Tsh": "Tsh",
        },
    )

    thermal_wavelengths = default_thermal_wavelengths(device=device, dtype=dtype).cpu().numpy()
    assert dataset_outputs["Lot_"].dims == ("y", "x", "time", "thermal_wavelength")
    assert dataset_outputs["Emint_"].dims == ("y", "x", "time", "layer_interface", "thermal_wavelength")
    assert np.array_equal(dataset_outputs["layer_interface"].values, np.arange(layers.size + 1))
    assert np.allclose(dataset_outputs["thermal_wavelength"].values, thermal_wavelengths)
    expected_lot = flat_outputs["Lot_"].cpu().numpy().reshape(1, 1, 2, thermal_wavelengths.size)
    expected_emint = flat_outputs["Emint_"].cpu().numpy().reshape(1, 1, 2, layers.size + 1, thermal_wavelengths.size)
    assert np.allclose(dataset_outputs["Lot_"].values, expected_lot)
    assert np.allclose(dataset_outputs["Emint_"].values, expected_emint)


def test_scope_grid_runner_directional_thermal_dataset_preserves_direction_coord():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Tsu": (("y", "x", "time"), np.full((1, 1, 2), 27.0)),
            "Tsh": (("y", "x", "time"), np.full((1, 1, 2), 21.0)),
        },
        coords={
            "y": y,
            "x": x,
            "time": times,
            "layer": layers,
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([15.0, 35.0])),
            "directional_psi": ("direction", np.array([10.0, 60.0])),
        },
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["LAI", "tts", "Tcu", "Tch", "Tsu", "Tsh"])
    flat_outputs = runner.run_directional_thermal(
        module,
        varmap={"LAI": "LAI", "tts": "tts", "Tcu": "Tcu", "Tch": "Tch", "Tsu": "Tsu", "Tsh": "Tsh"},
    )
    dataset_outputs = runner.run_directional_thermal_dataset(
        module,
        varmap={"LAI": "LAI", "tts": "tts", "Tcu": "Tcu", "Tch": "Tch", "Tsu": "Tsu", "Tsh": "Tsh"},
    )

    thermal_wavelengths = default_thermal_wavelengths(device=device, dtype=dtype).cpu().numpy()
    assert dataset_outputs.attrs["scope_product"] == "directional_thermal"
    assert dataset_outputs["Lot_"].dims == ("y", "x", "time", "direction", "thermal_wavelength")
    assert dataset_outputs["BrightnessT"].dims == ("y", "x", "time", "direction")
    assert np.allclose(dataset_outputs["directional_tto"].values, np.array([15.0, 35.0]))
    assert np.allclose(dataset_outputs["directional_psi"].values, np.array([10.0, 60.0]))
    expected_lot = flat_outputs["Lot_"].cpu().numpy().reshape(1, 1, 2, 2, thermal_wavelengths.size)
    assert np.allclose(dataset_outputs["Lot_"].values, expected_lot)


def test_scope_grid_runner_thermal_profiles_dataset_preserves_layer_dims():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Tsu": (("y", "x", "time"), np.full((1, 1, 2), 27.0)),
            "Tsh": (("y", "x", "time"), np.full((1, 1, 2), 21.0)),
        },
        coords={"y": y, "x": x, "time": times, "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["LAI", "tts", "tto", "psi", "Tcu", "Tch", "Tsu", "Tsh"])
    outputs = runner.run_thermal_profiles(
        module,
        varmap={"LAI": "LAI", "tts": "tts", "tto": "tto", "psi": "psi", "Tcu": "Tcu", "Tch": "Tch", "Tsu": "Tsu", "Tsh": "Tsh"},
    )
    dataset_outputs = runner.run_thermal_profiles_dataset(
        module,
        varmap={"LAI": "LAI", "tts": "tts", "tto": "tto", "psi": "psi", "Tcu": "Tcu", "Tch": "Tch", "Tsu": "Tsu", "Tsh": "Tsh"},
    )

    thermal_wavelengths = default_thermal_wavelengths(device=device, dtype=dtype).cpu().numpy()
    assert dataset_outputs.attrs["scope_product"] == "thermal_profiles"
    assert dataset_outputs["Ps"].dims == ("y", "x", "time", "layer_interface")
    assert dataset_outputs["Emint_"].dims == ("y", "x", "time", "layer_interface", "thermal_wavelength")
    assert dataset_outputs["layer_thermal_upward"].dims == ("y", "x", "time", "layer")
    assert np.array_equal(dataset_outputs["layer"].values, layers)
    assert np.array_equal(dataset_outputs["layer_interface"].values, np.arange(layers.size + 1))
    expected_emint = outputs["Emint_"].cpu().numpy().reshape(1, 1, 2, layers.size + 1, thermal_wavelengths.size)
    expected_layer_up = outputs["layer_thermal_upward"].cpu().numpy().reshape(1, 1, 2, layers.size)
    assert np.allclose(dataset_outputs["Emint_"].values, expected_emint)
    assert np.allclose(dataset_outputs["layer_thermal_upward"].values, expected_layer_up)


def test_scope_grid_runner_biochemical_fluorescence_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.arange(3)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Vcmax25": (("y", "x", "time"), np.full((1, 1, 2), 60.0)),
            "BallBerrySlope": (("y", "x", "time"), np.full((1, 1, 2), 8.0)),
            "BallBerry0": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "RdPerVcmax25": (("y", "x", "time"), np.full((1, 1, 2), 0.015)),
            "Kn0": (("y", "x", "time"), np.full((1, 1, 2), 2.48)),
            "Knalpha": (("y", "x", "time"), np.full((1, 1, 2), 2.83)),
            "Knbeta": (("y", "x", "time"), np.full((1, 1, 2), 0.114)),
            "stressfactor": (("y", "x", "time"), np.full((1, 1, 2), 1.0)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
            "Csu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 390.0)),
            "Csh": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 390.0)),
            "ebu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 20.0)),
            "ebh": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 20.0)),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Oa": (("y", "x", "time"), np.full((1, 1, 2), 209.0)),
            "p": (("y", "x", "time"), np.full((1, 1, 2), 970.0)),
            "fV": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 1.0)),
        },
        coords={"y": y, "x": x, "time": times, "excitation_wavelength": np.arange(n_wle), "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(
        data,
        cfg,
        required_vars=[
            "Cab",
            "Cw",
            "Cdm",
            "fqe",
            "Vcmax25",
            "BallBerrySlope",
            "BallBerry0",
            "RdPerVcmax25",
            "Kn0",
            "Knalpha",
            "Knbeta",
            "stressfactor",
            "LAI",
            "tts",
            "tto",
            "psi",
            "soil_spectrum",
            "Esun_",
            "Esky_",
            "Csu",
            "Csh",
            "ebu",
            "ebh",
            "Tcu",
            "Tch",
            "Oa",
            "p",
            "fV",
        ],
    )
    outputs = runner.run_biochemical_fluorescence(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "fqe": "fqe",
            "Vcmax25": "Vcmax25",
            "BallBerrySlope": "BallBerrySlope",
            "BallBerry0": "BallBerry0",
            "RdPerVcmax25": "RdPerVcmax25",
            "Kn0": "Kn0",
            "Knalpha": "Knalpha",
            "Knbeta": "Knbeta",
            "stressfactor": "stressfactor",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_spectrum": "soil_spectrum",
            "Esun_": "Esun_",
            "Esky_": "Esky_",
            "Csu": "Csu",
            "Csh": "Csh",
            "ebu": "ebu",
            "ebh": "ebh",
            "Tcu": "Tcu",
            "Tch": "Tch",
            "Oa": "Oa",
            "p": "p",
            "fV": "fV",
        },
    )

    fluorescence_model = CanopyFluorescenceModel(runner.reflectance_model)
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {key: [] for key in outputs}
    physiology_fields = [name for name in outputs if name.startswith("sunlit_") or name.startswith("shaded_")]
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
            fqe=torch.tensor([float(stacked["fqe"].sel(**idx))], device=device, dtype=dtype),
        )
        biochem = LeafBiochemistryInputs(
            Type="C3",
            Vcmax25=torch.tensor([float(stacked["Vcmax25"].sel(**idx))], device=device, dtype=dtype),
            BallBerrySlope=torch.tensor([float(stacked["BallBerrySlope"].sel(**idx))], device=device, dtype=dtype),
            BallBerry0=torch.tensor([float(stacked["BallBerry0"].sel(**idx))], device=device, dtype=dtype),
            RdPerVcmax25=torch.tensor([float(stacked["RdPerVcmax25"].sel(**idx))], device=device, dtype=dtype),
            Kn0=torch.tensor([float(stacked["Kn0"].sel(**idx))], device=device, dtype=dtype),
            Knalpha=torch.tensor([float(stacked["Knalpha"].sel(**idx))], device=device, dtype=dtype),
            Knbeta=torch.tensor([float(stacked["Knbeta"].sel(**idx))], device=device, dtype=dtype),
            stressfactor=torch.tensor([float(stacked["stressfactor"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_idx = torch.tensor([float(stacked["soil_spectrum"].sel(**idx))], device=device, dtype=dtype)
        coupled = fluorescence_model.layered_biochemical(
            leafbio,
            biochem,
            runner.soil_spectra.batch(soil_idx),
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor(stacked["Esun_"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor(stacked["Esky_"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            Csu=torch.tensor(stacked["Csu"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            Csh=torch.tensor(stacked["Csh"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            ebu=torch.tensor(stacked["ebu"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            ebh=torch.tensor(stacked["ebh"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            Tcu=torch.tensor(stacked["Tcu"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            Tch=torch.tensor(stacked["Tch"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            Oa=torch.tensor([float(stacked["Oa"].sel(**idx))], device=device, dtype=dtype),
            p=torch.tensor([float(stacked["p"].sel(**idx))], device=device, dtype=dtype),
            fV=torch.tensor(stacked["fV"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
        )
        for key in CanopyFluorescenceResult.__dataclass_fields__:
            manual_outputs[key].append(getattr(coupled.fluorescence, key))
        manual_outputs["Pnu_Cab"].append(coupled.Pnu_Cab)
        manual_outputs["Pnh_Cab"].append(coupled.Pnh_Cab)
        for key in physiology_fields:
            prefix, field = key.split("_", 1)
            source = coupled.sunlit if prefix == "sunlit" else coupled.shaded
            manual_outputs[key].append(getattr(source, field))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_biochemical_fluorescence_dataset_uses_layer_dims():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Vcmax25": (("y", "x", "time"), np.full((1, 1, 2), 60.0)),
            "BallBerrySlope": (("y", "x", "time"), np.full((1, 1, 2), 8.0)),
            "BallBerry0": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "RdPerVcmax25": (("y", "x", "time"), np.full((1, 1, 2), 0.015)),
            "Kn0": (("y", "x", "time"), np.full((1, 1, 2), 2.48)),
            "Knalpha": (("y", "x", "time"), np.full((1, 1, 2), 2.83)),
            "Knbeta": (("y", "x", "time"), np.full((1, 1, 2), 0.114)),
            "stressfactor": (("y", "x", "time"), np.full((1, 1, 2), 1.0)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
            "Csu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 390.0)),
            "Csh": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 390.0)),
            "ebu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 20.0)),
            "ebh": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 20.0)),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Oa": (("y", "x", "time"), np.full((1, 1, 2), 209.0)),
            "p": (("y", "x", "time"), np.full((1, 1, 2), 970.0)),
            "fV": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 1.0)),
        },
        coords={"y": y, "x": x, "time": times, "excitation_wavelength": np.arange(n_wle), "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    outputs = runner.run_biochemical_fluorescence(module, varmap={name: name for name in data.data_vars})
    dataset_outputs = runner.run_biochemical_fluorescence_dataset(module, varmap={name: name for name in data.data_vars})

    assert dataset_outputs["Pnu_Cab"].dims == ("y", "x", "time", "layer", "Pnu_Cab_dim_2")
    assert dataset_outputs["sunlit_A"].dims == ("y", "x", "time", "layer", "sunlit_A_dim_2")
    assert dataset_outputs["Fmin_"].dims == ("y", "x", "time", "layer_interface", "fluorescence_wavelength")
    assert dataset_outputs["leaf_fluor_back"].dims == ("y", "x", "time", "fluorescence_wavelength")
    assert dataset_outputs["LoF_"].dims == ("y", "x", "time", "fluorescence_wavelength")
    assert np.array_equal(dataset_outputs["layer"].values, layers)
    assert np.array_equal(dataset_outputs["layer_interface"].values, np.arange(layers.size + 1))
    assert np.allclose(
        dataset_outputs["Pnu_Cab"].values,
        outputs["Pnu_Cab"].cpu().numpy().reshape(1, 1, 2, 3, outputs["Pnu_Cab"].shape[-1]),
    )
    assert np.allclose(
        dataset_outputs["sunlit_A"].values,
        outputs["sunlit_A"].cpu().numpy().reshape(1, 1, 2, 3, outputs["sunlit_A"].shape[-1]),
    )


def test_scope_grid_runner_directional_fluorescence_dataset_preserves_direction_coord():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
        },
        coords={
            "y": y,
            "x": x,
            "time": times,
            "excitation_wavelength": np.arange(n_wle),
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([15.0, 35.0])),
            "directional_psi": ("direction", np.array([10.0, 60.0])),
        },
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    flat_outputs = runner.run_directional_fluorescence(module, varmap={name: name for name in data.data_vars})
    dataset_outputs = runner.run_directional_fluorescence_dataset(module, varmap={name: name for name in data.data_vars})

    assert dataset_outputs.attrs["scope_product"] == "directional_fluorescence"
    assert dataset_outputs["LoF_"].dims == ("y", "x", "time", "direction", "fluorescence_wavelength")
    assert np.allclose(dataset_outputs["directional_tto"].values, np.array([15.0, 35.0]))
    assert np.allclose(dataset_outputs["directional_psi"].values, np.array([10.0, 60.0]))
    expected_lof = flat_outputs["LoF_"].cpu().numpy().reshape(1, 1, 2, 2, flat_outputs["LoF_"].shape[-1])
    assert np.allclose(dataset_outputs["LoF_"].values, expected_lof)


def test_scope_grid_runner_fluorescence_profiles_dataset_preserves_layer_dims():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
            "etau": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 0.01)),
            "etah": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 0.008)),
        },
        coords={"y": y, "x": x, "time": times, "excitation_wavelength": np.arange(n_wle), "layer": layers},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    outputs = runner.run_fluorescence_profiles(module, varmap={name: name for name in data.data_vars})
    dataset_outputs = runner.run_fluorescence_profiles_dataset(module, varmap={name: name for name in data.data_vars})

    assert dataset_outputs.attrs["scope_product"] == "fluorescence_profiles"
    assert dataset_outputs["Ps"].dims == ("y", "x", "time", "layer_interface")
    assert dataset_outputs["Fmin_"].dims == ("y", "x", "time", "layer_interface", "fluorescence_wavelength")
    assert dataset_outputs["layer_fluorescence"].dims == ("y", "x", "time", "layer")
    assert np.array_equal(dataset_outputs["layer"].values, layers)
    assert np.array_equal(dataset_outputs["layer_interface"].values, np.arange(layers.size + 1))
    expected_fmin = outputs["Fmin_"].cpu().numpy().reshape(1, 1, 2, layers.size + 1, outputs["Fmin_"].shape[-1])
    expected_layer = outputs["layer_fluorescence"].cpu().numpy().reshape(1, 1, 2, layers.size)
    assert np.allclose(dataset_outputs["Fmin_"].values, expected_fmin)
    assert np.allclose(dataset_outputs["layer_fluorescence"].values, expected_layer)


def test_scope_grid_runner_run_scope_dataset_honours_reflectance_option_attrs():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    nwl = spectral.wlP.numel()
    rng = np.random.default_rng(19)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), rng.random((1, 1, 2, nwl)) * 0.2 + 0.1),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, nwl), 900.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, nwl), 120.0)),
        },
        coords={
            "y": np.arange(1),
            "x": np.arange(1),
            "time": times,
            "wavelength": np.arange(nwl),
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([10.0, 30.0])),
            "directional_psi": ("direction", np.array([5.0, 45.0])),
        },
        attrs={"calc_directional": 1, "calc_vert_profiles": 1, "calc_fluor": 0, "calc_planck": 0},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)
    varmap = {name: name for name in data.data_vars}

    scope_outputs = runner.run_scope_dataset(module, varmap=varmap, nlayers=3)
    reflectance_outputs = runner.run_dataset(module, varmap=varmap, nlayers=3)
    directional_outputs = runner.run_directional_reflectance_dataset(module, varmap=varmap, nlayers=3)
    profile_outputs = runner.run_reflectance_profiles_dataset(module, varmap=varmap, nlayers=3)

    assert scope_outputs.attrs["scope_product"] == "scope_workflow"
    assert scope_outputs.attrs["scope_components"] == "reflectance,reflectance_directional,reflectance_profile"
    assert np.allclose(scope_outputs["rsot"].values, reflectance_outputs["rsot"].values)
    assert np.allclose(scope_outputs["reflectance_directional_refl_"].values, directional_outputs["refl_"].values)
    assert np.allclose(scope_outputs["reflectance_profile_Ps"].values, profile_outputs["Ps"].values)


def test_scope_grid_runner_run_scope_dataset_honours_fluorescence_option_attrs():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wle = runner.fluspect.spectral.wlE.numel()
    n_wlp = runner.fluspect.spectral.wlP.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 1.0)),
            "Esky_": (("y", "x", "time", "excitation_wavelength"), np.full((1, 1, 2, n_wle), 0.2)),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, n_wlp), 900.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, n_wlp), 120.0)),
            "etau": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 0.01)),
            "etah": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 0.008)),
        },
        coords={
            "y": np.arange(1),
            "x": np.arange(1),
            "time": times,
            "layer": layers,
            "wavelength": np.arange(n_wlp),
            "excitation_wavelength": np.arange(n_wle),
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([15.0, 35.0])),
            "directional_psi": ("direction", np.array([10.0, 60.0])),
        },
        attrs={"calc_directional": 1, "calc_vert_profiles": 1, "calc_fluor": 1, "calc_planck": 0},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    varmap = {name: name for name in data.data_vars}

    scope_outputs = runner.run_scope_dataset(module, varmap=varmap, nlayers=3)
    fluorescence_outputs = runner.run_layered_fluorescence_dataset(module, varmap=varmap, nlayers=3)
    directional_outputs = runner.run_directional_fluorescence_dataset(module, varmap=varmap, nlayers=3)
    profile_outputs = runner.run_fluorescence_profiles_dataset(module, varmap=varmap, nlayers=3)

    assert "fluorescence" in scope_outputs.attrs["scope_components"]
    assert np.allclose(scope_outputs["LoF_"].values, fluorescence_outputs["LoF_"].values)
    assert np.allclose(scope_outputs["fluorescence_directional_LoF_"].values, directional_outputs["LoF_"].values)
    assert np.allclose(scope_outputs["fluorescence_profile_Fmin_"].values, profile_outputs["Fmin_"].values)
    assert np.allclose(scope_outputs["fluorescence_gammasdf"].values, fluorescence_outputs["gammasdf"].values)


def test_scope_grid_runner_run_scope_dataset_scope_options_override_dataset_attrs_for_thermal():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    n_wlp = runner.fluspect.spectral.wlP.numel()

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    layers = np.array([10.0, 20.0, 30.0])
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 2.0]]])),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, n_wlp), 900.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, n_wlp), 120.0)),
            "Tcu": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 25.0)),
            "Tch": (("y", "x", "time", "layer"), np.full((1, 1, 2, 3), 23.0)),
            "Tsu": (("y", "x", "time"), np.full((1, 1, 2), 27.0)),
            "Tsh": (("y", "x", "time"), np.full((1, 1, 2), 21.0)),
        },
        coords={
            "y": np.arange(1),
            "x": np.arange(1),
            "time": times,
            "layer": layers,
            "wavelength": np.arange(n_wlp),
            "direction": np.arange(2),
            "directional_tto": ("direction", np.array([15.0, 35.0])),
            "directional_psi": ("direction", np.array([10.0, 60.0])),
        },
        attrs={"calc_directional": 0, "calc_vert_profiles": 0, "calc_fluor": 0, "calc_planck": 0},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    varmap = {name: name for name in data.data_vars}

    scope_outputs = runner.run_scope_dataset(
        module,
        varmap=varmap,
        scope_options={"calc_planck": 1, "calc_directional": 1, "calc_vert_profiles": 1},
        nlayers=3,
    )
    thermal_outputs = runner.run_thermal_dataset(module, varmap=varmap, nlayers=3)
    directional_outputs = runner.run_directional_thermal_dataset(module, varmap=varmap, nlayers=3)
    profile_outputs = runner.run_thermal_profiles_dataset(module, varmap=varmap, nlayers=3)

    assert "thermal" in scope_outputs.attrs["scope_components"]
    assert scope_outputs.attrs["calc_planck"] == 1
    assert np.allclose(scope_outputs["Lot_"].values, thermal_outputs["Lot_"].values)
    assert np.allclose(scope_outputs["thermal_directional_Lot_"].values, directional_outputs["Lot_"].values)
    assert np.allclose(scope_outputs["thermal_profile_Emint_"].values, profile_outputs["Emint_"].values)


def test_scope_grid_runner_from_scope_assets_resolves_soil_spectrum():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "soil_spectrum": (("y", "x", "time"), np.array([[[1.0, 3.0]]])),
        },
        coords={"y": y, "x": x, "time": times},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "soil_spectrum"])
    outputs = runner.run(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_spectrum": "soil_spectrum",
        },
    )

    soil_library = load_soil_spectra(device=device, dtype=dtype)
    reflectance_model = CanopyReflectanceModel(runner.fluspect, runner.sail, lidf=lidf, default_hotspot=runner.default_hotspot)
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {"leaf_refl": [], "leaf_tran": [], **{key: [] for key in CANOPY_KEYS}}
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_idx = torch.tensor([float(stacked["soil_spectrum"].sel(**idx))], device=device, dtype=dtype)
        reflectance_out = reflectance_model(
            leafbio,
            soil_library.batch(soil_idx),
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(reflectance_out, key))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_from_scope_assets_resolves_bsm_soil_parameters():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    runner = ScopeGridRunner.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "LAI": (("y", "x", "time"), np.array([[[2.0, 2.5]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "psi": (("y", "x", "time"), np.array([[[5.0, 15.0]]])),
            "BSMBrightness": (("y", "x", "time"), np.array([[[0.5, 0.8]]])),
            "BSMlat": (("y", "x", "time"), np.array([[[25.0, 35.0]]])),
            "BSMlon": (("y", "x", "time"), np.array([[[45.0, 60.0]]])),
            "SMC": (("y", "x", "time"), np.array([[[0.10, 0.25]]])),
        },
        coords={"y": y, "x": x, "time": times},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(
        data,
        cfg,
        required_vars=["Cab", "Cw", "Cdm", "LAI", "tts", "tto", "psi", "BSMBrightness", "BSMlat", "BSMlon", "SMC"],
    )
    outputs = runner.run(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "BSMBrightness": "BSMBrightness",
            "BSMlat": "BSMlat",
            "BSMlon": "BSMlon",
            "SMC": "SMC",
        },
    )

    reflectance_model = CanopyReflectanceModel(
        runner.fluspect,
        runner.sail,
        lidf=lidf,
        default_hotspot=runner.default_hotspot,
        soil_spectra=runner.soil_spectra,
        soil_bsm=runner.soil_bsm,
        soil_index_base=runner.soil_index_base,
    )
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {"leaf_refl": [], "leaf_tran": [], **{key: [] for key in CANOPY_KEYS}}
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
        )
        soil_tensor = reflectance_model.soil_reflectance(
            BSMBrightness=torch.tensor([float(stacked["BSMBrightness"].sel(**idx))], device=device, dtype=dtype),
            BSMlat=torch.tensor([float(stacked["BSMlat"].sel(**idx))], device=device, dtype=dtype),
            BSMlon=torch.tensor([float(stacked["BSMlon"].sel(**idx))], device=device, dtype=dtype),
            SMC=torch.tensor([float(stacked["SMC"].sel(**idx))], device=device, dtype=dtype),
        )
        reflectance_out = reflectance_model(
            leafbio,
            soil_tensor,
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            hotspot=torch.tensor([runner.default_hotspot], device=device, dtype=dtype),
        )
        for key in manual_outputs:
            manual_outputs[key].append(getattr(reflectance_out, key))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_energy_balance_fluorescence_matches_manual():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.3, 2.6]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 15.0)),
            "psi": (("y", "x", "time"), np.array([[[10.0, 20.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 0.2)),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 1200.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 180.0)),
            "Ta": (("y", "x", "time"), np.full((1, 1, 2), 25.0)),
            "ea": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "Ca": (("y", "x", "time"), np.full((1, 1, 2), 390.0)),
            "Oa": (("y", "x", "time"), np.full((1, 1, 2), 209.0)),
            "p": (("y", "x", "time"), np.full((1, 1, 2), 970.0)),
            "z": (("y", "x", "time"), np.full((1, 1, 2), 10.0)),
            "u": (("y", "x", "time"), np.full((1, 1, 2), 2.0)),
            "Cd": (("y", "x", "time"), np.full((1, 1, 2), 0.2)),
            "rwc": (("y", "x", "time"), np.full((1, 1, 2), 0.5)),
            "z0m": (("y", "x", "time"), np.full((1, 1, 2), 0.15)),
            "d": (("y", "x", "time"), np.full((1, 1, 2), 1.3)),
            "h": (("y", "x", "time"), np.full((1, 1, 2), 2.0)),
            "kV": (("y", "x", "time"), np.full((1, 1, 2), 0.15)),
            "rss": (("y", "x", "time"), np.full((1, 1, 2), 120.0)),
            "rbs": (("y", "x", "time"), np.full((1, 1, 2), 12.0)),
            "Vcmax25": (("y", "x", "time"), np.full((1, 1, 2), 70.0)),
            "BallBerrySlope": (("y", "x", "time"), np.full((1, 1, 2), 9.0)),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(spectral.wlP.numel())},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(
        data,
        cfg,
        required_vars=[
            "Cab",
            "Cw",
            "Cdm",
            "fqe",
            "LAI",
            "tts",
            "tto",
            "psi",
            "soil_refl",
            "Esun_sw",
            "Esky_sw",
            "Ta",
            "ea",
            "Ca",
            "Oa",
            "p",
            "z",
            "u",
            "Cd",
            "rwc",
            "z0m",
            "d",
            "h",
            "kV",
            "rss",
            "rbs",
            "Vcmax25",
            "BallBerrySlope",
        ],
    )
    outputs = runner.run_energy_balance_fluorescence(
        module,
        varmap={
            "Cab": "Cab",
            "Cw": "Cw",
            "Cdm": "Cdm",
            "fqe": "fqe",
            "LAI": "LAI",
            "tts": "tts",
            "tto": "tto",
            "psi": "psi",
            "soil_refl": "soil_refl",
            "Esun_sw": "Esun_sw",
            "Esky_sw": "Esky_sw",
            "Ta": "Ta",
            "ea": "ea",
            "Ca": "Ca",
            "Oa": "Oa",
            "p": "p",
            "z": "z",
            "u": "u",
            "Cd": "Cd",
            "rwc": "rwc",
            "z0m": "z0m",
            "d": "d",
            "h": "h",
            "kV": "kV",
            "rss": "rss",
            "rbs": "rbs",
            "Vcmax25": "Vcmax25",
            "BallBerrySlope": "BallBerrySlope",
        },
        energy_options=EnergyBalanceOptions(max_iter=50),
        nlayers=4,
    )

    physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
    energy_fields = [name for name in CanopyEnergyBalanceResult.__dataclass_fields__ if name not in {"sunlit", "shaded", "Tsold"}]
    stacked = data.stack(batch=("y", "x", "time"))
    manual_outputs: dict[str, list[torch.Tensor]] = {
        **{name: [] for name in CanopyFluorescenceResult.__dataclass_fields__},
        **{name: [] for name in energy_fields},
        **{f"sunlit_{name}": [] for name in physiology_fields},
        **{f"shaded_{name}": [] for name in physiology_fields},
    }
    for label in stacked["Cab"].indexes["batch"]:
        idx = dict(batch=label)
        leafbio = LeafBioBatch(
            Cab=torch.tensor([float(stacked["Cab"].sel(**idx))], device=device, dtype=dtype),
            Cw=torch.tensor([float(stacked["Cw"].sel(**idx))], device=device, dtype=dtype),
            Cdm=torch.tensor([float(stacked["Cdm"].sel(**idx))], device=device, dtype=dtype),
            fqe=torch.tensor([float(stacked["fqe"].sel(**idx))], device=device, dtype=dtype),
        )
        biochem = LeafBiochemistryInputs(
            Vcmax25=torch.tensor([float(stacked["Vcmax25"].sel(**idx))], device=device, dtype=dtype),
            BallBerrySlope=torch.tensor([float(stacked["BallBerrySlope"].sel(**idx))], device=device, dtype=dtype),
        )
        result = runner.energy_balance_model.solve_fluorescence(
            leafbio,
            biochem,
            torch.tensor(stacked["soil_refl"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor([float(stacked["LAI"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tts"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["tto"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor([float(stacked["psi"].sel(**idx))], device=device, dtype=dtype),
            torch.tensor(stacked["Esun_sw"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            torch.tensor(stacked["Esky_sw"].sel(**idx).values, device=device, dtype=dtype).unsqueeze(0),
            meteo=EnergyBalanceMeteo(
                Ta=torch.tensor([float(stacked["Ta"].sel(**idx))], device=device, dtype=dtype),
                ea=torch.tensor([float(stacked["ea"].sel(**idx))], device=device, dtype=dtype),
                Ca=torch.tensor([float(stacked["Ca"].sel(**idx))], device=device, dtype=dtype),
                Oa=torch.tensor([float(stacked["Oa"].sel(**idx))], device=device, dtype=dtype),
                p=torch.tensor([float(stacked["p"].sel(**idx))], device=device, dtype=dtype),
                z=torch.tensor([float(stacked["z"].sel(**idx))], device=device, dtype=dtype),
                u=torch.tensor([float(stacked["u"].sel(**idx))], device=device, dtype=dtype),
            ),
            canopy=EnergyBalanceCanopy(
                Cd=torch.tensor([float(stacked["Cd"].sel(**idx))], device=device, dtype=dtype),
                rwc=torch.tensor([float(stacked["rwc"].sel(**idx))], device=device, dtype=dtype),
                z0m=torch.tensor([float(stacked["z0m"].sel(**idx))], device=device, dtype=dtype),
                d=torch.tensor([float(stacked["d"].sel(**idx))], device=device, dtype=dtype),
                h=torch.tensor([float(stacked["h"].sel(**idx))], device=device, dtype=dtype),
                kV=torch.tensor([float(stacked["kV"].sel(**idx))], device=device, dtype=dtype),
            ),
            soil=EnergyBalanceSoil(
                rss=torch.tensor([float(stacked["rss"].sel(**idx))], device=device, dtype=dtype),
                rbs=torch.tensor([float(stacked["rbs"].sel(**idx))], device=device, dtype=dtype),
            ),
            options=EnergyBalanceOptions(max_iter=50),
            nlayers=4,
        )
        for name in CanopyFluorescenceResult.__dataclass_fields__:
            manual_outputs[name].append(getattr(result.fluorescence, name))
        for name in energy_fields:
            manual_outputs[name].append(getattr(result.energy, name))
        for name in physiology_fields:
            manual_outputs[f"sunlit_{name}"].append(getattr(result.energy.sunlit, name))
            manual_outputs[f"shaded_{name}"].append(getattr(result.energy.shaded, name))

    for key, values in manual_outputs.items():
        assert torch.allclose(outputs[key], torch.cat(values, dim=0))


def test_scope_grid_runner_energy_balance_fluorescence_dataset_infers_layer_dims():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    runner = ScopeGridRunner(fluspect, sail, lidf=lidf)

    times = pd.date_range("2020-07-01", periods=2, freq="h")
    y = np.arange(1)
    x = np.arange(1)
    data = xr.Dataset(
        {
            "Cab": (("y", "x", "time"), np.full((1, 1, 2), 45.0)),
            "Cw": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "Cdm": (("y", "x", "time"), np.full((1, 1, 2), 0.012)),
            "fqe": (("y", "x", "time"), np.full((1, 1, 2), 0.01)),
            "LAI": (("y", "x", "time"), np.array([[[2.3, 2.6]]])),
            "tts": (("y", "x", "time"), np.full((1, 1, 2), 30.0)),
            "tto": (("y", "x", "time"), np.full((1, 1, 2), 15.0)),
            "psi": (("y", "x", "time"), np.array([[[10.0, 20.0]]])),
            "soil_refl": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 0.2)),
            "Esun_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 1200.0)),
            "Esky_sw": (("y", "x", "time", "wavelength"), np.full((1, 1, 2, spectral.wlP.numel()), 180.0)),
            "Ta": (("y", "x", "time"), np.full((1, 1, 2), 25.0)),
            "ea": (("y", "x", "time"), np.full((1, 1, 2), 20.0)),
            "Ca": (("y", "x", "time"), np.full((1, 1, 2), 390.0)),
            "Oa": (("y", "x", "time"), np.full((1, 1, 2), 209.0)),
            "p": (("y", "x", "time"), np.full((1, 1, 2), 970.0)),
            "z": (("y", "x", "time"), np.full((1, 1, 2), 10.0)),
            "u": (("y", "x", "time"), np.full((1, 1, 2), 2.0)),
            "Cd": (("y", "x", "time"), np.full((1, 1, 2), 0.2)),
            "rwc": (("y", "x", "time"), np.full((1, 1, 2), 0.5)),
            "z0m": (("y", "x", "time"), np.full((1, 1, 2), 0.15)),
            "d": (("y", "x", "time"), np.full((1, 1, 2), 1.3)),
            "h": (("y", "x", "time"), np.full((1, 1, 2), 2.0)),
            "kV": (("y", "x", "time"), np.full((1, 1, 2), 0.15)),
            "rss": (("y", "x", "time"), np.full((1, 1, 2), 120.0)),
            "rbs": (("y", "x", "time"), np.full((1, 1, 2), 12.0)),
            "Vcmax25": (("y", "x", "time"), np.full((1, 1, 2), 70.0)),
            "BallBerrySlope": (("y", "x", "time"), np.full((1, 1, 2), 9.0)),
        },
        coords={"y": y, "x": x, "time": times, "wavelength": np.arange(spectral.wlP.numel())},
    )

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], device=str(device), dtype=dtype, chunk_size=2)
    module = ScopeGridDataModule(data, cfg, required_vars=list(data.data_vars))
    outputs = runner.run_energy_balance_fluorescence(
        module,
        varmap={name: name for name in data.data_vars},
        energy_options=EnergyBalanceOptions(max_iter=50),
        nlayers=4,
    )
    dataset_outputs = runner.run_energy_balance_fluorescence_dataset(
        module,
        varmap={name: name for name in data.data_vars},
        energy_options=EnergyBalanceOptions(max_iter=50),
        nlayers=4,
    )

    assert dataset_outputs["Pnu_Cab"].dims == ("y", "x", "time", "layer")
    assert dataset_outputs["sunlit_Cs_input"].dims == ("y", "x", "time", "layer")
    assert np.array_equal(dataset_outputs["layer"].values, np.arange(4))
    assert np.allclose(dataset_outputs["Pnu_Cab"].values, outputs["Pnu_Cab"].cpu().numpy().reshape(1, 1, 2, 4))
    assert np.allclose(dataset_outputs["sunlit_Cs_input"].values, outputs["sunlit_Cs_input"].cpu().numpy().reshape(1, 1, 2, 4))


def test_scope_grid_runner_energy_balance_thermal_dataset_preserves_stable_outputs():
    runner, spectral = _build_execution_mode_runner(dtype=torch.float64)
    dataset = _build_coupled_execution_mode_dataset(spectral)
    module = _build_execution_mode_module(dataset, dtype=torch.float64, chunk_size=3)

    outputs = runner.run_energy_balance_thermal(
        module,
        varmap={name: name for name in dataset.data_vars},
        energy_options=EnergyBalanceOptions(max_iter=50),
        nlayers=4,
    )
    dataset_outputs = runner.run_energy_balance_thermal_dataset(
        module,
        varmap={name: name for name in dataset.data_vars},
        energy_options=EnergyBalanceOptions(max_iter=50),
        nlayers=4,
    )

    assert dataset_outputs.attrs["site"] == "coupled-execution-mode"
    assert dataset_outputs.attrs["scope_product"] == "energy_balance_thermal"
    assert dataset_outputs["Lot_"].dims == ("y", "x", "time", "thermal_wavelength")
    assert dataset_outputs["Tcu"].dims == ("y", "x", "time", "layer")
    assert np.allclose(dataset_outputs["Lot_"].values, outputs["Lot_"].cpu().numpy().reshape(1, 1, 3, -1))
    assert np.allclose(dataset_outputs["Tcu"].values, outputs["Tcu"].cpu().numpy().reshape(1, 1, 3, -1))


@pytest.mark.parametrize(
    ("workflow", "dtype", "atol", "rtol"),
    [
        ("reflectance", torch.float32, 1e-5, 5e-5),
        ("reflectance", torch.float64, 1e-8, 1e-8),
        ("fluorescence", torch.float32, 1e-5, 5e-5),
        ("fluorescence", torch.float64, 1e-8, 1e-8),
        ("layered_fluorescence", torch.float32, 2e-5, 8e-5),
        ("layered_fluorescence", torch.float64, 1e-8, 1e-8),
        ("biochemical_fluorescence", torch.float32, 3e-5, 1e-4),
        ("biochemical_fluorescence", torch.float64, 1e-8, 1e-8),
        ("thermal", torch.float32, 2e-5, 8e-5),
        ("thermal", torch.float64, 1e-8, 1e-8),
    ],
)
def test_scope_grid_runner_workflows_match_across_chunk_sizes(workflow, dtype, atol, rtol):
    runner, spectral = _build_execution_mode_runner(dtype=dtype)
    dataset = _build_execution_mode_dataset(spectral)
    batched_module = _build_execution_mode_module(dataset, dtype=dtype, chunk_size=3)
    single_module = _build_execution_mode_module(dataset, dtype=dtype, chunk_size=1)

    batched = _run_execution_mode_workflow(runner, batched_module, workflow)
    single = _run_execution_mode_workflow(runner, single_module, workflow)

    _assert_tensor_mapping_close(batched, single, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("workflow", "atol", "rtol"),
    [
        ("reflectance", 1e-5, 5e-5),
        ("fluorescence", 2e-5, 8e-5),
        ("thermal", 2e-5, 8e-5),
    ],
)
def test_scope_grid_runner_workflows_cpu_match_cuda(workflow, atol, rtol):
    cpu_runner, spectral = _build_execution_mode_runner(device="cpu", dtype=torch.float32)
    dataset = _build_execution_mode_dataset(spectral)
    cpu_module = _build_execution_mode_module(dataset, device="cpu", dtype=torch.float32, chunk_size=3)
    cuda_runner, _ = _build_execution_mode_runner(device="cuda", dtype=torch.float32)
    cuda_module = _build_execution_mode_module(dataset, device="cuda", dtype=torch.float32, chunk_size=3)

    cpu_outputs = _run_execution_mode_workflow(cpu_runner, cpu_module, workflow)
    cuda_outputs = _run_execution_mode_workflow(cuda_runner, cuda_module, workflow)

    _assert_tensor_mapping_close(cpu_outputs, cuda_outputs, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("workflow", "dtype", "atol", "rtol"),
    [
        ("energy_balance_fluorescence", torch.float32, 5e-5, 2e-4),
        ("energy_balance_fluorescence", torch.float64, 1e-8, 1e-8),
        ("energy_balance_thermal", torch.float32, 5e-5, 2e-4),
        ("energy_balance_thermal", torch.float64, 1e-8, 1e-8),
    ],
)
def test_scope_grid_runner_coupled_workflows_match_across_chunk_sizes(workflow, dtype, atol, rtol):
    runner, spectral = _build_execution_mode_runner(dtype=dtype)
    dataset = _build_coupled_execution_mode_dataset(spectral)
    batched_module = _build_execution_mode_module(dataset, dtype=dtype, chunk_size=3)
    single_module = _build_execution_mode_module(dataset, dtype=dtype, chunk_size=1)

    batched = _run_coupled_execution_mode_workflow(runner, batched_module, workflow)
    single = _run_coupled_execution_mode_workflow(runner, single_module, workflow)

    radiative_keys = (
        set(CanopyFluorescenceResult.__dataclass_fields__)
        if workflow == "energy_balance_fluorescence"
        else set(CanopyThermalRadianceResult.__dataclass_fields__)
    )
    _assert_coupled_runner_outputs_close(
        batched,
        single,
        radiative_keys=radiative_keys,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("workflow", "atol", "rtol"),
    [
        ("energy_balance_fluorescence", 5e-5, 2e-4),
        ("energy_balance_thermal", 5e-5, 2e-4),
    ],
)
def test_scope_grid_runner_coupled_workflows_cpu_match_cuda(workflow, atol, rtol):
    cpu_runner, spectral = _build_execution_mode_runner(device="cpu", dtype=torch.float32)
    dataset = _build_coupled_execution_mode_dataset(spectral)
    cpu_module = _build_execution_mode_module(dataset, device="cpu", dtype=torch.float32, chunk_size=3)
    cuda_runner, _ = _build_execution_mode_runner(device="cuda", dtype=torch.float32)
    cuda_module = _build_execution_mode_module(dataset, device="cuda", dtype=torch.float32, chunk_size=3)

    cpu_outputs = _run_coupled_execution_mode_workflow(cpu_runner, cpu_module, workflow)
    cuda_outputs = _run_coupled_execution_mode_workflow(cuda_runner, cuda_module, workflow)

    radiative_keys = (
        set(CanopyFluorescenceResult.__dataclass_fields__)
        if workflow == "energy_balance_fluorescence"
        else set(CanopyThermalRadianceResult.__dataclass_fields__)
    )
    _assert_coupled_runner_outputs_close(
        cpu_outputs,
        cuda_outputs,
        radiative_keys=radiative_keys,
        atol=atol,
        rtol=rtol,
    )
