import torch

from scope_torch.canopy.foursail import campbell_lidf
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.spectral.fluspect import LeafBioBatch


def test_canopy_reflectance_outputs_remain_internally_consistent():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyReflectanceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
    )
    soil = model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))

    result = model(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
    )

    assert torch.allclose(result.rso, result.rsos + result.rsod, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rsot, result.rsost + result.rsodt, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rddt, result.rdd, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rsdt, result.rsd, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rdot, result.rdo, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rsot, result.rso, atol=1e-12, rtol=1e-10)


def test_canopy_reflectance_profiles_are_consistent_with_layered_fluxes():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyReflectanceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
    )
    soil = model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))
    Esun = torch.full((1, model.fluspect.spectral.wlP.numel()), 900.0, device=device, dtype=dtype)
    Esky = torch.full((1, model.fluspect.spectral.wlP.numel()), 120.0, device=device, dtype=dtype)

    profiles = model.profiles(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun,
        Esky,
        nlayers=4,
    )

    assert profiles.Ps.shape == (1, 5)
    assert profiles.Po.shape == (1, 5)
    assert profiles.Pso.shape == (1, 5)
    assert profiles.Es_.shape == (1, 5, model.fluspect.spectral.wlP.numel())
    assert torch.allclose(profiles.Es_, profiles.Es_direct_ + profiles.Es_diffuse_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(profiles.Emin_, profiles.Emin_direct_ + profiles.Emin_diffuse_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(profiles.Eplu_, profiles.Eplu_direct_ + profiles.Eplu_diffuse_, atol=1e-12, rtol=1e-10)


def test_canopy_reflectance_directional_matches_single_angle_solution():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyReflectanceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
    )
    soil = model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))
    Esun = torch.full((1, model.fluspect.spectral.wlP.numel()), 900.0, device=device, dtype=dtype)
    Esky = torch.full((1, model.fluspect.spectral.wlP.numel()), 120.0, device=device, dtype=dtype)

    single = model(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        nlayers=4,
    )
    directional = model.directional(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun,
        Esky,
        nlayers=4,
    )

    expected_refl = model._directional_reflectance(single.rso, single.rdo, Esun, Esky)
    assert directional.refl_.shape == (1, 1, model.fluspect.spectral.wlP.numel())
    assert torch.allclose(directional.rso_[:, 0, :], single.rso, atol=1e-12, rtol=1e-10)
    assert torch.allclose(directional.refl_[:, 0, :], expected_refl, atol=1e-12, rtol=1e-10)
