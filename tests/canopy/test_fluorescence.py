import torch
from scipy.interpolate import CubicSpline

from scope_torch.biochem import LeafBiochemistryInputs
from scope_torch.canopy.fluorescence import CanopyFluorescenceModel
from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf, scope_lidf
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.spectral.fluspect import LeafBioBatch
from scope_torch.spectral.fluspect import FluspectModel
from scope_torch.spectral.loaders import load_soil_spectra
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
    expected_sigmaf = model._scope_sigmaf(
        result.LoF_,
        result.EoutFrc_,
        model.reflectance_model.fluspect.spectral.wlF,
    )
    assert torch.allclose(result.sigmaF, expected_sigmaf, atol=1e-12, rtol=1e-10)
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
    absorbed_cab = 1e3 * torch.trapz(model._e2phot(wlE, excitation * epsc_e * kchl_e), wlE, dim=-1)
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
    expected_sigmaf = model._scope_sigmaf(
        result.LoF_,
        result.EoutFrc_,
        model.reflectance_model.fluspect.spectral.wlF,
    )
    assert torch.allclose(result.sigmaF, expected_sigmaf, atol=1e-12, rtol=1e-10)


def test_canopy_fluorescence_profiles_match_layered_outputs():
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

    layered = model.layered(
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
    profiles = model.profiles(
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

    assert profiles.Ps.shape == (1, 5)
    assert torch.allclose(profiles.Fmin_, layered.Fmin_, atol=1e-12, rtol=1e-10)
    assert torch.allclose(profiles.Fplu_, layered.Fplu_, atol=1e-12, rtol=1e-10)
    expected = 0.001 * torch.trapz(layered.Fplu_[:, :-1, :], model.reflectance_model.fluspect.spectral.wlF, dim=-1)
    assert torch.allclose(profiles.layer_fluorescence, expected, atol=1e-12, rtol=1e-10)


def test_canopy_fluorescence_directional_matches_single_angle_solution():
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

    single = model.layered(
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
    directional = model.directional(
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

    assert directional.LoF_.shape == (1, 1, model.reflectance_model.fluspect.spectral.wlF.numel())
    assert torch.allclose(directional.LoF_[:, 0, :], single.LoF_, atol=1e-12, rtol=1e-10)


def test_canopy_fluorescence_layered_preserves_gradients_on_custom_output_grids():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    fluspect = FluspectModel.from_scope_assets(
        device=device,
        dtype=dtype,
        wlF=torch.arange(642.0, 849.0, 3.0, device=device, dtype=dtype),
        wlE=torch.arange(402.0, 751.0, 7.0, device=device, dtype=dtype),
    )
    reflectance = CanopyReflectanceModel(
        fluspect,
        FourSAILModel(lidf=lidf),
        lidf=lidf,
        soil_spectra=load_soil_spectra(device=device, dtype=dtype),
    )
    model = CanopyFluorescenceModel(reflectance)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([45.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.01], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        fqe=torch.tensor([0.01], device=device, dtype=dtype),
    )
    soil = reflectance.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))
    Esun_ = torch.full((1, fluspect.spectral.wlE.numel()), 1.0, device=device, dtype=dtype, requires_grad=True)
    Esky_ = torch.full((1, fluspect.spectral.wlE.numel()), 0.2, device=device, dtype=dtype)

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

    (result.LoF_.sum() + result.EoutF_.sum()).backward()

    assert Esun_.grad is not None
    assert torch.all(torch.isfinite(Esun_.grad))
    assert torch.count_nonzero(Esun_.grad) > 0


def test_canopy_fluorescence_matlab_spline_matches_not_a_knot_reference():
    device = torch.device("cpu")
    dtype = torch.float64
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    model = CanopyFluorescenceModel.from_scope_assets(lidf=lidf, device=device, dtype=dtype)

    source_x = torch.arange(640.0, 849.0, 4.0, device=device, dtype=dtype)
    target_x = torch.arange(640.0, 851.0, 1.0, device=device, dtype=dtype)
    source_y = torch.stack(
        [
            torch.sin(source_x / 35.0) + 0.1 * torch.cos(source_x / 21.0),
            torch.cos(source_x / 42.0) - 0.05 * torch.sin(source_x / 18.0),
        ],
        dim=0,
    )

    result = model._matlab_spline_interp1(source_x, source_y, target_x)
    reference = torch.as_tensor(
        CubicSpline(
            source_x.cpu().numpy(),
            source_y.cpu().numpy(),
            axis=-1,
            bc_type="not-a-knot",
        )(target_x.cpu().numpy()),
        device=device,
        dtype=dtype,
    )

    assert torch.allclose(result, reference, atol=1e-10, rtol=1e-10)


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


def test_canopy_fluorescence_layered_biochemical_matches_manual_eta_path():
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
    biochem = LeafBiochemistryInputs(
        Type="C3",
        Vcmax25=torch.tensor([60.0], device=device, dtype=dtype),
        BallBerrySlope=torch.tensor([8.0], device=device, dtype=dtype),
        BallBerry0=torch.tensor([0.01], device=device, dtype=dtype),
        RdPerVcmax25=torch.tensor([0.015], device=device, dtype=dtype),
        Kn0=torch.tensor([2.48], device=device, dtype=dtype),
        Knalpha=torch.tensor([2.83], device=device, dtype=dtype),
        Knbeta=torch.tensor([0.114], device=device, dtype=dtype),
        stressfactor=torch.tensor([1.0], device=device, dtype=dtype),
    )
    n_wle = model.reflectance_model.fluspect.spectral.wlE.numel()
    soil = model.reflectance_model.soil_reflectance(soil_spectrum=torch.tensor([1.0], device=device, dtype=dtype))
    Esun_ = torch.full((1, n_wle), 1.0, device=device, dtype=dtype)
    Esky_ = torch.full((1, n_wle), 0.2, device=device, dtype=dtype)

    coupled = model.layered_biochemical(
        leafbio,
        biochem,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun_,
        Esky_,
        Csu=torch.full((1, 4), 390.0, device=device, dtype=dtype),
        Csh=torch.full((1, 4), 390.0, device=device, dtype=dtype),
        ebu=torch.full((1, 4), 20.0, device=device, dtype=dtype),
        ebh=torch.full((1, 4), 20.0, device=device, dtype=dtype),
        Tcu=torch.full((1, 4), 25.0, device=device, dtype=dtype),
        Tch=torch.full((1, 4), 23.0, device=device, dtype=dtype),
        Oa=torch.tensor([209.0], device=device, dtype=dtype),
        p=torch.tensor([970.0], device=device, dtype=dtype),
        nlayers=4,
    )
    manual = model.layered(
        leafbio,
        soil,
        torch.tensor([3.0], device=device, dtype=dtype),
        torch.tensor([30.0], device=device, dtype=dtype),
        torch.tensor([20.0], device=device, dtype=dtype),
        torch.tensor([10.0], device=device, dtype=dtype),
        Esun_,
        Esky_,
        etau=coupled.sunlit.eta,
        etah=coupled.shaded.eta,
        Pnu_Cab=coupled.Pnu_Cab,
        Pnh_Cab=coupled.Pnh_Cab,
        nlayers=4,
    )

    assert coupled.Pnu_Cab.shape == coupled.sunlit.eta.shape
    assert coupled.Pnh_Cab.shape == coupled.shaded.eta.shape
    assert torch.all(coupled.Pnu_Cab >= 0)
    assert torch.all(coupled.Pnh_Cab >= 0)
    assert torch.allclose(coupled.sunlit.SIF, coupled.sunlit.fs * coupled.Pnu_Cab, atol=1e-12, rtol=1e-10)
    assert torch.allclose(coupled.shaded.SIF, coupled.shaded.fs * coupled.Pnh_Cab, atol=1e-12, rtol=1e-10)
    for field in manual.__dataclass_fields__:
        assert torch.allclose(getattr(coupled.fluorescence, field), getattr(manual, field), atol=1e-12, rtol=1e-10)
