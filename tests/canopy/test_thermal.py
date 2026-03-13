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


def test_canopy_thermal_profiles_match_layered_outputs():
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
    profiles = model.profiles(
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

    wlT = default_thermal_wavelengths(device=device, dtype=dtype)
    assert profiles.Ps.shape == (1, 5)
    assert torch.allclose(profiles.Emint_, result.Emint_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(profiles.Eplut_, result.Eplut_, atol=1e-12, rtol=1e-10)
    expected = 0.001 * torch.trapz(result.Eplut_[:, :-1, :], wlT, dim=-1)
    assert torch.allclose(profiles.layer_thermal_upward, expected, atol=1e-12, rtol=1e-10)


def test_canopy_thermal_directional_matches_single_angle_solution():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    single = model(
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
    directional = model.directional(
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

    sigma_sb = torch.as_tensor(5.67e-8, device=device, dtype=dtype)
    expected_bt = torch.clamp(torch.pi * single.Loutt / sigma_sb, min=0.0) ** 0.25
    assert directional.Lot_.shape == (1, 1, single.Lot_.shape[-1])
    assert torch.allclose(directional.Lot_[:, 0, :], single.Lot_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(directional.BrightnessT[:, 0], expected_bt, atol=1e-12, rtol=1e-10)


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


def test_canopy_thermal_integrated_balance_terms_are_consistent():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)
    optics = ThermalOptics(rho_thermal=0.01, tau_thermal=0.01, rs_thermal=0.06)

    result = model.integrated_balance(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 4), 25.0, device=device, dtype=dtype),
        torch.full((1, 4), 23.0, device=device, dtype=dtype),
        torch.tensor([27.0], device=device, dtype=dtype),
        torch.tensor([21.0], device=device, dtype=dtype),
        thermal_optics=optics,
        nlayers=4,
    )

    transfer = model.layered_transport.build(
        torch.full((1, 1), optics.rho_thermal, device=device, dtype=dtype),
        torch.full((1, 1), optics.tau_thermal, device=device, dtype=dtype),
        torch.full((1, 1), optics.rs_thermal, device=device, dtype=dtype),
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        hotspot=torch.full((1,), model.reflectance_model.default_hotspot, device=device, dtype=dtype),
        lidf=model.reflectance_model.lidf,
        nlayers=4,
    )
    emissivity_soil = 1.0 - optics.rs_thermal
    hssu = emissivity_soil * model._stefan_boltzmann(torch.tensor([27.0], device=device, dtype=dtype))
    hssh = emissivity_soil * model._stefan_boltzmann(torch.tensor([21.0], device=device, dtype=dtype))
    hs = hssu * transfer.Ps[:, -1] + hssh * (1.0 - transfer.Ps[:, -1])
    assert result.Emint.shape == (1, 5)
    assert result.Eplut.shape == (1, 5)
    assert result.Lote.shape == (1,)
    assert result.Eoutte.shape == (1,)
    assert torch.all(result.canopyemis > 0)
    assert torch.allclose(
        result.Eplut[:, -1],
        transfer.R_dd[:, -2, 0] * result.Emint[:, -2] + hs,
        atol=1e-12,
        rtol=1e-10,
    )
    assert torch.allclose(result.Rnust, emissivity_soil * (result.Emint[:, -1] - hssu), atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.Rnhst, emissivity_soil * (result.Emint[:, -1] - hssh), atol=1e-12, rtol=1e-10)


def test_canopy_thermal_integrated_balance_blackbody_emissivity_is_unity():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    result = model.integrated_balance(
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        torch.full((1, 3), 25.0, device=device, dtype=dtype),
        torch.full((1, 3), 23.0, device=device, dtype=dtype),
        torch.tensor([27.0], device=device, dtype=dtype),
        torch.tensor([21.0], device=device, dtype=dtype),
        thermal_optics=ThermalOptics(rho_thermal=0.0, tau_thermal=0.0, rs_thermal=0.0),
        nlayers=3,
    )

    assert torch.allclose(result.canopyemis, torch.ones_like(result.canopyemis), atol=1e-12, rtol=1e-10)


def test_canopy_thermal_integrated_balance_orientation_constant_collapse():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = scope_lidf(0.0, 0.0, device=device, dtype=dtype)
    model = CanopyThermalRadianceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    layer_mean = model.integrated_balance(
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
    oriented = model.integrated_balance(
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

    assert oriented.Rnuct.shape == (1, 3, 468)
    assert torch.allclose(oriented.Eoutte, layer_mean.Eoutte, atol=1e-12, rtol=1e-10)
    assert torch.allclose(oriented.Lote, layer_mean.Lote, atol=1e-12, rtol=1e-10)
    assert torch.allclose(oriented.Rnuct.mean(dim=-1), layer_mean.Rnuct, atol=1e-12, rtol=1e-10)
