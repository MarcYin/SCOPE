import torch

from scope_torch.canopy.fluorescence import CanopyFluorescenceModel
from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf, scope_lidf
from scope_torch.spectral.fluspect import LeafBioBatch
from scope_torch.spectral.soil import SoilEmpiricalParams


def test_canopy_fluorescence_model_outputs_consistent_sif_fields():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyFluorescenceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        fqe=torch.tensor([0.01], device=device, dtype=dtype),
    )
    excitation = torch.full((1, model.reflectance_model.fluspect.spectral.wlE.numel()), 1.0, device=device, dtype=dtype)
    soil = model.reflectance_model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))

    result = model(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        excitation,
    )

    n_wlf = model.reflectance_model.fluspect.spectral.wlF.numel()
    assert result.LoF_.shape == (1, n_wlf)
    assert result.EoutF_.shape == (1, n_wlf)
    assert result.Femleaves_.shape == (1, n_wlf)
    assert torch.all(result.LoF_ >= 0)
    assert torch.all(result.EoutF_ >= 0)
    assert torch.all(result.EoutFrc_ >= 0)
    assert torch.count_nonzero(result.EoutFrc_) > 0
    assert torch.allclose(result.LoF_ * torch.pi, result.sigmaF * result.EoutFrc_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.F684, result.LoF_[:, torch.argmin(torch.abs(model.reflectance_model.fluspect.spectral.wlF - 684.0))])
    assert torch.allclose(result.F761, result.LoF_[:, torch.argmin(torch.abs(model.reflectance_model.fluspect.spectral.wlF - 761.0))])

    leafopt = model.reflectance_model.fluspect(leafbio)
    wlP = model.reflectance_model.fluspect.spectral.wlP
    wlE = model.reflectance_model.fluspect.spectral.wlE
    wlF = model.reflectance_model.fluspect.spectral.wlF
    rho_e = model._sample_spectrum(leafopt.refl, wlP, wlE)
    tau_e = model._sample_spectrum(leafopt.tran, wlP, wlE)
    kchl_e = model._sample_spectrum(leafopt.kChlrel, wlP, wlE)
    epsc_e = (1.0 - rho_e - tau_e).clamp(min=0.0)
    absorbed_cab = 0.001 * torch.trapz(model._e2phot(wlE, excitation * epsc_e * kchl_e), wlE, dim=-1)
    poutfrc = leafbio.fqe * torch.tensor([3.0], device=device, dtype=dtype) * absorbed_cab
    phi_em = model._sample_spectrum(model.reflectance_model.fluspect.optipar.phi.unsqueeze(0), wlP, wlF)
    expected_eoutfrc = 1e-3 * model._ephoton(wlF).unsqueeze(0) * poutfrc.unsqueeze(-1) * phi_em
    assert torch.allclose(result.EoutFrc_, expected_eoutfrc, atol=1e-12, rtol=1e-10)


def test_canopy_fluorescence_model_zero_excitation_returns_zero():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyFluorescenceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        fqe=torch.tensor([0.01], device=device, dtype=dtype),
    )
    excitation = torch.zeros((1, model.reflectance_model.fluspect.spectral.wlE.numel()), device=device, dtype=dtype)
    soil = model.reflectance_model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))

    result = model(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        excitation,
    )

    assert torch.count_nonzero(result.LoF_) == 0
    assert torch.count_nonzero(result.EoutF_) == 0
    assert torch.count_nonzero(result.Femleaves_) == 0
    assert torch.count_nonzero(result.EoutFrc_) == 0


def test_canopy_fluorescence_factory_accepts_reflectance_configuration():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    empirical = SoilEmpiricalParams(SMC=30.0, film=0.02)

    model = CanopyFluorescenceModel.from_scope_assets(
        lidf=lidf,
        sail=sail,
        device=device,
        dtype=dtype,
        soil_empirical=empirical,
    )

    assert model.reflectance_model.sail is sail
    assert float(model.reflectance_model.soil_bsm.empirical.SMC) == 30.0
    assert float(model.reflectance_model.soil_bsm.empirical.film) == 0.02


def test_canopy_fluorescence_layered_outputs_are_consistent():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyFluorescenceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        fqe=torch.tensor([0.01], device=device, dtype=dtype),
    )
    n_wle = model.reflectance_model.fluspect.spectral.wlE.numel()
    Esun_ = torch.full((1, n_wle), 1.0, device=device, dtype=dtype)
    Esky_ = torch.full((1, n_wle), 0.2, device=device, dtype=dtype)
    soil = model.reflectance_model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))

    result = model.layered(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun_,
        Esky_,
        nlayers=4,
    )

    n_wlf = model.reflectance_model.fluspect.spectral.wlF.numel()
    assert result.LoF_.shape == (1, n_wlf)
    assert result.Fmin_.shape == (1, 5, n_wlf)
    assert result.Fplu_.shape == (1, 5, n_wlf)
    assert torch.allclose(result.LoF_, result.LoF_sunlit + result.LoF_shaded + result.LoF_scattered + result.LoF_soil)
    assert torch.allclose(result.EoutF_, result.Fplu_[:, 0, :])
    assert torch.all(result.EoutFrc_ >= 0)
    assert torch.count_nonzero(result.EoutFrc_) > 0
    assert torch.allclose(result.LoF_ * torch.pi, result.sigmaF * result.EoutFrc_, atol=1e-12, rtol=1e-10)


def test_canopy_fluorescence_layered_accepts_orientation_resolved_efficiencies():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = scope_lidf(0.0, 0.0, device=device, dtype=dtype)
    model = CanopyFluorescenceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        fqe=torch.tensor([0.01], device=device, dtype=dtype),
    )
    n_wle = model.reflectance_model.fluspect.spectral.wlE.numel()
    Esun_ = torch.full((1, n_wle), 1.0, device=device, dtype=dtype)
    Esky_ = torch.full((1, n_wle), 0.2, device=device, dtype=dtype)
    soil = model.reflectance_model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))

    layer_constant = model.layered(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun_,
        Esky_,
        etau=torch.ones((1, 4), device=device, dtype=dtype),
        etah=torch.ones((1, 4), device=device, dtype=dtype),
        nlayers=4,
    )
    oriented = model.layered(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun_,
        Esky_,
        etau=torch.ones((1, 4, 13, 36), device=device, dtype=dtype),
        etah=torch.ones((1, 4, 13, 36), device=device, dtype=dtype),
        nlayers=4,
    )

    assert torch.allclose(oriented.LoF_, layer_constant.LoF_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(oriented.EoutF_, layer_constant.EoutF_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(oriented.Femleaves_, layer_constant.Femleaves_, atol=1e-12, rtol=1e-10)
