import numpy as np
import pandas as pd
import torch
import xarray as xr

from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf
from scope_torch.canopy.fluorescence import CanopyFluorescenceModel
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.config import SimulationConfig
from scope_torch.data import ScopeGridDataModule
from scope_torch.runners.grid import ScopeGridRunner
from scope_torch.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids
from scope_torch.spectral.loaders import load_soil_spectra


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

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=2)
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

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=2)
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

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=2)
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

    cfg = SimulationConfig(roi_bounds=(0, 0, 1, 1), start_time=times[0], end_time=times[-1], chunk_size=2)
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
