import torch

from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf, scope_lidf
from scope_torch.canopy.thermal import CanopyThermalRadianceModel, ThermalOptics, default_thermal_wavelengths
from scope_torch.spectral.soil import SoilEmpiricalParams


def test_canopy_thermal_radiance_model_outputs_positive_spectra():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    result = model(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 4), 25.0, device=device, dtype=dtype),
        torch.full((1, 4), 23.0, device=device, dtype=dtype),
        torch.tensor([27.0], device=device, dtype=dtype),
        torch.tensor([21.0], device=device, dtype=dtype),
        nlayers=4,
    )

    assert result.Lot_.shape[-1] == result.Eoutte_.shape[-1]
    assert result.Emint_.shape[1] == 5
    assert result.Eplut_.shape[1] == 5
    assert torch.all(result.Lot_ >= 0)
    assert torch.all(result.Eoutte_ >= 0)
    assert torch.all(result.Loutt > 0)
    assert torch.all(result.Eoutt > 0)


def test_canopy_thermal_radiance_model_zero_kelvin_offset_behaves():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    cold = model(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 2), -273.15, device=device, dtype=dtype),
        torch.full((1, 2), -273.15, device=device, dtype=dtype),
        torch.tensor([-273.15], device=device, dtype=dtype),
        torch.tensor([-273.15], device=device, dtype=dtype),
        nlayers=2,
    )

    assert torch.count_nonzero(cold.Lot_) == 0
    assert torch.count_nonzero(cold.Eoutte_) == 0


def test_canopy_thermal_bottom_boundary_matches_soil_emission_and_reflection():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    thermal_optics = ThermalOptics(rs_thermal=0.12)
    soil_temperature = 27.0

    result = model(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 3), 25.0, device=device, dtype=dtype),
        torch.full((1, 3), 23.0, device=device, dtype=dtype),
        torch.tensor([soil_temperature], device=device, dtype=dtype),
        torch.tensor([soil_temperature], device=device, dtype=dtype),
        thermal_optics=thermal_optics,
        nlayers=3,
    )

    wlT = default_thermal_wavelengths(device=device, dtype=dtype)
    soil = model._broadcast_scalar_spectrum(thermal_optics.rs_thermal, 1, wlT)
    emissivity = 1.0 - soil
    Hsoil = torch.pi * model._planck(
        wlT,
        torch.full((1, 1), soil_temperature + 273.15, device=device, dtype=dtype),
        emissivity,
    )

    expected_bottom = soil * result.Emint_[:, -1, :] + Hsoil
    assert torch.allclose(result.Eplut_[:, -1, :], expected_bottom, atol=1e-10, rtol=1e-8)


def test_canopy_thermal_factory_accepts_reflectance_configuration():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    empirical = SoilEmpiricalParams(SMC=30.0, film=0.02)

    model = CanopyThermalRadianceModel.from_scope_assets(
        lidf=lidf,
        sail=sail,
        device=device,
        dtype=dtype,
        soil_empirical=empirical,
    )

    assert model.reflectance_model.sail is sail
    assert float(model.reflectance_model.soil_bsm.empirical.SMC) == 30.0
    assert float(model.reflectance_model.soil_bsm.empirical.film) == 0.02


def test_canopy_thermal_radiance_accepts_orientation_resolved_leaf_temperatures():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = scope_lidf(0.0, 0.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    layer_mean = model(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 3), 25.0, device=device, dtype=dtype),
        torch.full((1, 3), 23.0, device=device, dtype=dtype),
        torch.tensor([27.0], device=device, dtype=dtype),
        torch.tensor([21.0], device=device, dtype=dtype),
        nlayers=3,
    )
    oriented = model(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 3, 13, 36), 25.0, device=device, dtype=dtype),
        torch.full((1, 3, 13, 36), 23.0, device=device, dtype=dtype),
        torch.tensor([27.0], device=device, dtype=dtype),
        torch.tensor([21.0], device=device, dtype=dtype),
        nlayers=3,
    )

    assert torch.allclose(oriented.Lot_, layer_mean.Lot_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(oriented.Eoutte_, layer_mean.Eoutte_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(oriented.LotBB_, layer_mean.LotBB_, atol=1e-12, rtol=1e-10)
