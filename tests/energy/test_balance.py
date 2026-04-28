from dataclasses import fields, is_dataclass

import pytest
import torch

from scope.biochem import LeafBiochemistryInputs
from scope.canopy.fluorescence import CanopyFluorescenceResult
from scope.canopy.foursail import FourSAILModel, campbell_lidf
from scope.canopy.reflectance import CanopyReflectanceModel
from scope.canopy.thermal import CanopyThermalRadianceResult
from scope.energy import (
    CanopyEnergyBalanceModel,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
)
from scope.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids


def _spectral(device, dtype):
    wlP = torch.linspace(400.0, 2500.0, 128, device=device, dtype=dtype)
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


def _setup_energy_case(*, device: torch.device | str = "cpu", batch: int = 1):
    device = torch.device(device)
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    reflectance = CanopyReflectanceModel(fluspect, sail, lidf=lidf)
    model = CanopyEnergyBalanceModel(reflectance)

    Cab = torch.linspace(45.0, 38.0, batch, device=device, dtype=dtype)
    Cw = torch.linspace(0.01, 0.014, batch, device=device, dtype=dtype)
    Cdm = torch.linspace(0.012, 0.017, batch, device=device, dtype=dtype)
    fqe = torch.linspace(0.01, 0.015, batch, device=device, dtype=dtype)
    leafbio = LeafBioBatch(
        Cab=Cab,
        Cw=Cw,
        Cdm=Cdm,
        fqe=fqe,
    )
    biochemistry = LeafBiochemistryInputs(
        Vcmax25=torch.linspace(70.0, 58.0, batch, device=device, dtype=dtype),
        BallBerrySlope=torch.linspace(9.0, 7.5, batch, device=device, dtype=dtype),
    )
    lai = torch.linspace(2.5, 3.4, batch, device=device, dtype=dtype)
    tts = torch.linspace(30.0, 38.0, batch, device=device, dtype=dtype)
    tto = torch.linspace(15.0, 22.0, batch, device=device, dtype=dtype)
    psi = torch.linspace(20.0, 35.0, batch, device=device, dtype=dtype)
    soil_refl = torch.linspace(
        0.18,
        0.24,
        batch * spectral.wlP.numel(),
        device=device,
        dtype=dtype,
    ).reshape(batch, spectral.wlP.numel())
    Esun_sw = torch.linspace(
        1050.0,
        1300.0,
        batch * spectral.wlP.numel(),
        device=device,
        dtype=dtype,
    ).reshape(batch, spectral.wlP.numel())
    Esky_sw = torch.linspace(
        140.0,
        220.0,
        batch * spectral.wlP.numel(),
        device=device,
        dtype=dtype,
    ).reshape(batch, spectral.wlP.numel())
    meteo = EnergyBalanceMeteo(
        Ta=torch.linspace(25.0, 27.0, batch, device=device, dtype=dtype),
        ea=torch.linspace(20.0, 21.5, batch, device=device, dtype=dtype),
        Ca=torch.linspace(390.0, 404.0, batch, device=device, dtype=dtype),
        Oa=torch.full((batch,), 209.0, device=device, dtype=dtype),
        p=torch.linspace(970.0, 965.0, batch, device=device, dtype=dtype),
        z=10.0,
        u=torch.linspace(2.0, 3.2, batch, device=device, dtype=dtype),
    )
    canopy = EnergyBalanceCanopy(Cd=0.2, rwc=0.5, z0m=0.15, d=1.3, h=2.0, kV=0.15)
    soil = EnergyBalanceSoil(rss=120.0, rbs=12.0)
    options = EnergyBalanceOptions(max_iter=50)
    return model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options


def _index_dataclass(value, index: int):
    if is_dataclass(value):
        kwargs = {field.name: _index_dataclass(getattr(value, field.name), index) for field in fields(value)}
        return type(value)(**kwargs)
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > index:
        return value[index : index + 1]
    return value


def _assert_result_close(actual, expected, *, atol: float = 1e-8, rtol: float = 1e-8):
    if is_dataclass(actual):
        assert is_dataclass(expected)
        for field in fields(actual):
            if field.name == "fcount":
                continue
            _assert_result_close(
                getattr(actual, field.name),
                getattr(expected, field.name),
                atol=atol,
                rtol=rtol,
            )
        return

    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        lhs = actual.detach().cpu()
        rhs = expected.detach().cpu()
        assert lhs.shape == rhs.shape
        if lhs.dtype == torch.bool or rhs.dtype == torch.bool:
            assert torch.equal(lhs, rhs)
        elif torch.is_floating_point(lhs) or torch.is_floating_point(rhs):
            assert torch.allclose(lhs, rhs, atol=atol, rtol=rtol)
        else:
            assert torch.equal(lhs, rhs)
        return

    assert actual == expected


def _assert_stable_energy_thermal_close(actual, expected, *, atol: float = 1e-8, rtol: float = 1e-8):
    # The raw energy dataclass contains phase-lagged diagnostics and a global
    # iteration counter. Batch runs can carry those diagnostics through extra
    # iterations after a scene has already converged. Compare the stable solved
    # state and the downstream thermal product instead.
    stable_energy_fields = (
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
        "Tsold",
    )
    for name in stable_energy_fields:
        _assert_result_close(
            getattr(actual.energy, name),
            getattr(expected.energy, name),
            atol=atol,
            rtol=rtol,
        )
    _assert_result_close(actual.thermal, expected.thermal, atol=atol, rtol=rtol)


def test_energy_balance_converges_and_closes_fluxes():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = (
        _setup_energy_case()
    )

    result = model.solve(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=options,
        nlayers=4,
    )

    assert torch.all(result.converged)
    assert torch.all(result.max_error <= options.max_energy_error)
    assert torch.all(result.Pnu_Cab >= result.Pnh_Cab)
    assert torch.allclose(result.Rntot, result.Rnctot + result.Rnstot, atol=1e-10, rtol=1e-10)
    assert torch.allclose(result.lEtot, result.lEctot + result.lEstot, atol=1e-10, rtol=1e-10)
    assert torch.allclose(result.Htot, result.Hctot + result.Hstot, atol=1e-10, rtol=1e-10)


def test_energy_balance_fixed_iterations_runs_to_max_iter():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, _ = (
        _setup_energy_case()
    )

    early = model.solve(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=EnergyBalanceOptions(max_iter=4, max_energy_error=1e30),
        nlayers=4,
    )
    fixed = model.solve(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=EnergyBalanceOptions(max_iter=4, max_energy_error=1e30, fixed_iterations=True),
        nlayers=4,
    )

    assert torch.all(early.counter == 1)
    assert torch.all(fixed.counter == 4)
    assert torch.all(fixed.converged)


def test_energy_balance_fv_profile_uses_upper_layer_edges():
    model, leafbio, _, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, _, canopy, _, _ = _setup_energy_case()

    leafopt = model.reflectance_model.fluspect(leafbio)
    shortwave = model._shortwave_radiation(
        leafopt=leafopt,
        soil_refl=soil_refl,
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Esun_sw=Esun_sw,
        Esky_sw=Esky_sw,
        hotspot=torch.full_like(lai, model.reflectance_model.default_hotspot),
        lidf=None,
        nlayers=4,
    )

    fV = model._fV_profile(canopy, shortwave.transfer, batch=1)
    kV = torch.as_tensor(canopy.kV, device=fV.device, dtype=fV.dtype).view(1, 1)
    expected = torch.exp(kV * shortwave.transfer.xl[:-1].view(1, -1))

    assert torch.allclose(fV, expected, atol=1e-12, rtol=1e-10)
    assert torch.allclose(fV[:, 0], torch.ones_like(fV[:, 0]), atol=1e-12, rtol=1e-10)


def test_energy_balance_zero_longwave_forcing_is_a_no_op():
    model, leafbio, _, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, _, _, soil, _ = _setup_energy_case()

    leafopt = model.reflectance_model.fluspect(leafbio)
    hotspot = torch.full_like(lai, model.reflectance_model.default_hotspot)
    baseline = model._shortwave_radiation(
        leafopt=leafopt,
        soil_refl=soil_refl,
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Esun_sw=Esun_sw,
        Esky_sw=Esky_sw,
        hotspot=hotspot,
        lidf=None,
        nlayers=4,
    )
    zero_lw = model._shortwave_radiation(
        leafopt=leafopt,
        soil_refl=soil_refl,
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Esun_sw=Esun_sw,
        Esky_sw=Esky_sw,
        Esun_lw=torch.zeros((1, 5), device=leafopt.refl.device, dtype=leafopt.refl.dtype),
        Esky_lw=torch.zeros((1, 5), device=leafopt.refl.device, dtype=leafopt.refl.dtype),
        thermal_optics=soil.thermal_optics,
        hotspot=hotspot,
        lidf=None,
        nlayers=4,
        wlT=torch.linspace(8000.0, 12000.0, 5, device=leafopt.refl.device, dtype=leafopt.refl.dtype),
    )

    for name in ("Rnuc", "Rnhc", "Rnus", "Rnhs", "Pnu_Cab", "Pnh_Cab"):
        assert torch.allclose(getattr(baseline, name), getattr(zero_lw, name), atol=1e-12, rtol=1e-10)


def test_energy_balance_fluorescence_matches_manual_eta_transport():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = (
        _setup_energy_case()
    )

    result = model.solve_fluorescence(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=options,
        nlayers=4,
    )

    wlP = model.reflectance_model.fluspect.spectral.wlP
    wlE = model.reflectance_model.fluspect.spectral.wlE
    Esun_e = model.fluorescence_model._sample_spectrum(Esun_sw, wlP, wlE)
    Esky_e = model.fluorescence_model._sample_spectrum(Esky_sw, wlP, wlE)
    manual = model.fluorescence_model.layered(
        leafbio,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_e,
        Esky_e,
        etau=result.energy.sunlit.eta,
        etah=result.energy.shaded.eta,
        Pnu_Cab=result.energy.Pnu_Cab,
        Pnh_Cab=result.energy.Pnh_Cab,
        nlayers=4,
    )

    for name in CanopyFluorescenceResult.__dataclass_fields__:
        assert torch.allclose(getattr(result.fluorescence, name), getattr(manual, name), atol=1e-10, rtol=1e-10)


def test_energy_balance_thermal_matches_manual_solved_temperature_path():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = (
        _setup_energy_case()
    )

    result = model.solve_thermal(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=options,
        nlayers=4,
    )

    manual = model.thermal_model(
        lai,
        tts,
        tto,
        psi,
        result.energy.Tcu,
        result.energy.Tch,
        result.energy.Tsu,
        result.energy.Tsh,
        thermal_optics=soil.thermal_optics,
        nlayers=4,
    )

    for name in CanopyThermalRadianceResult.__dataclass_fields__:
        assert torch.allclose(getattr(result.thermal, name), getattr(manual, name), atol=1e-10, rtol=1e-10)


def test_energy_balance_soil_history_accepts_batched_dt_seconds():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = (
        _setup_energy_case()
    )
    device = leafbio.Cab.device
    dtype = leafbio.Cab.dtype

    soil.soil_heat_method = 1
    soil.GAM = torch.tensor([120.0], device=device, dtype=dtype)
    soil.Tsold = torch.tensor([[24.0, 26.0], [23.5, 25.5]], device=device, dtype=dtype)
    soil.dt_seconds = torch.tensor([3600.0], device=device, dtype=dtype)

    result = model.solve(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=options,
        nlayers=4,
    )

    assert torch.all(result.converged)
    assert result.Tsold is not None
    assert result.Tsold.shape == (1, 2, 2)
    assert torch.allclose(result.Tsold[:, 0, 0], result.Tsh, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.Tsold[:, 0, 1], result.Tsu, atol=1e-12, rtol=1e-10)


def test_energy_balance_thermal_batch_matches_single_scene_solves():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = (
        _setup_energy_case(batch=2)
    )

    batched = model.solve_thermal(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        Esun_sw,
        Esky_sw,
        meteo=meteo,
        canopy=canopy,
        soil=soil,
        options=options,
        nlayers=4,
    )

    for index in range(2):
        single = model.solve_thermal(
            _index_dataclass(leafbio, index),
            _index_dataclass(biochemistry, index),
            soil_refl[index : index + 1],
            lai[index : index + 1],
            tts[index : index + 1],
            tto[index : index + 1],
            psi[index : index + 1],
            Esun_sw[index : index + 1],
            Esky_sw[index : index + 1],
            meteo=_index_dataclass(meteo, index),
            canopy=canopy,
            soil=soil,
            options=options,
            nlayers=4,
        )
        _assert_stable_energy_thermal_close(_index_dataclass(batched, index), single)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_energy_balance_thermal_cpu_matches_cuda():
    cpu_case = _setup_energy_case(device="cpu", batch=2)
    cuda_case = _setup_energy_case(device="cuda", batch=2)

    cpu_result = cpu_case[0].solve_thermal(
        cpu_case[1],
        cpu_case[2],
        cpu_case[3],
        cpu_case[4],
        cpu_case[5],
        cpu_case[6],
        cpu_case[7],
        cpu_case[8],
        cpu_case[9],
        meteo=cpu_case[10],
        canopy=cpu_case[11],
        soil=cpu_case[12],
        options=cpu_case[13],
        nlayers=4,
    )
    cuda_result = cuda_case[0].solve_thermal(
        cuda_case[1],
        cuda_case[2],
        cuda_case[3],
        cuda_case[4],
        cuda_case[5],
        cuda_case[6],
        cuda_case[7],
        cuda_case[8],
        cuda_case[9],
        meteo=cuda_case[10],
        canopy=cuda_case[11],
        soil=cuda_case[12],
        options=cuda_case[13],
        nlayers=4,
    )

    _assert_stable_energy_thermal_close(cpu_result, cuda_result, atol=1e-8, rtol=1e-8)
