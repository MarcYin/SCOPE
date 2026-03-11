import torch

from scope_torch.canopy.fluorescence import CanopyFluorescenceModel
from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf
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
    assert torch.allclose(result.EoutFrc_, result.Femleaves_)
    assert torch.allclose(result.LoF_ * torch.pi, result.sigmaF * result.EoutFrc_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.F684, result.LoF_[:, torch.argmin(torch.abs(model.reflectance_model.fluspect.spectral.wlF - 684.0))])
    assert torch.allclose(result.F761, result.LoF_[:, torch.argmin(torch.abs(model.reflectance_model.fluspect.spectral.wlF - 761.0))])


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
