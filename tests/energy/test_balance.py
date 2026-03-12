import torch

from scope_torch.biochem import LeafBiochemistryInputs
from scope_torch.canopy.fluorescence import CanopyFluorescenceResult
from scope_torch.canopy.thermal import CanopyThermalRadianceResult
from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.energy import (
    CanopyEnergyBalanceModel,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
)
from scope_torch.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids


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


def _setup_energy_case():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _spectral(device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    reflectance = CanopyReflectanceModel(fluspect, sail, lidf=lidf)
    model = CanopyEnergyBalanceModel(reflectance)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        fqe=torch.tensor([0.01], device=device, dtype=dtype),
    )
    biochemistry = LeafBiochemistryInputs(Vcmax25=70.0, BallBerrySlope=9.0)
    lai = torch.tensor([2.5], device=device, dtype=dtype)
    tts = torch.tensor([30.0], device=device, dtype=dtype)
    tto = torch.tensor([15.0], device=device, dtype=dtype)
    psi = torch.tensor([20.0], device=device, dtype=dtype)
    soil_refl = torch.full((1, spectral.wlP.numel()), 0.2, device=device, dtype=dtype)
    Esun_sw = torch.full((1, spectral.wlP.numel()), 1200.0, device=device, dtype=dtype)
    Esky_sw = torch.full((1, spectral.wlP.numel()), 180.0, device=device, dtype=dtype)
    meteo = EnergyBalanceMeteo(Ta=25.0, ea=20.0, Ca=390.0, Oa=209.0, p=970.0, z=10.0, u=2.0)
    canopy = EnergyBalanceCanopy(Cd=0.2, rwc=0.5, z0m=0.15, d=1.3, h=2.0, kV=0.15)
    soil = EnergyBalanceSoil(rss=120.0, rbs=12.0)
    options = EnergyBalanceOptions(max_iter=50)
    return model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options


def test_energy_balance_converges_and_closes_fluxes():
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = _setup_energy_case()

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
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = _setup_energy_case()

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
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = _setup_energy_case()

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
    model, leafbio, biochemistry, soil_refl, lai, tts, tto, psi, Esun_sw, Esky_sw, meteo, canopy, soil, options = _setup_energy_case()
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
