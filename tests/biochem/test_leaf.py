import pytest
import torch

from scope.biochem import BiochemicalOptions, LeafBiochemistryInputs, LeafBiochemistryModel, LeafMeteo


def _make_c3_inputs(*, intercept: float = 0.01) -> tuple[LeafBiochemistryInputs, LeafMeteo]:
    leafbio = LeafBiochemistryInputs(
        Type="C3",
        Vcmax25=torch.tensor([60.0], dtype=torch.float64),
        BallBerrySlope=torch.tensor([8.0], dtype=torch.float64),
        BallBerry0=torch.tensor([intercept], dtype=torch.float64),
        RdPerVcmax25=torch.tensor([0.015], dtype=torch.float64),
        Kn0=torch.tensor([2.48], dtype=torch.float64),
        Knalpha=torch.tensor([2.83], dtype=torch.float64),
        Knbeta=torch.tensor([0.114], dtype=torch.float64),
        stressfactor=torch.tensor([1.0], dtype=torch.float64),
    )
    meteo = LeafMeteo(
        Q=torch.tensor([1200.0], dtype=torch.float64),
        Cs=torch.tensor([390.0], dtype=torch.float64),
        T=torch.tensor([25.0], dtype=torch.float64),
        eb=torch.tensor([20.0], dtype=torch.float64),
        Oa=torch.tensor([209.0], dtype=torch.float64),
        p=torch.tensor([970.0], dtype=torch.float64),
    )
    return leafbio, meteo


def test_leaf_biochemistry_c3_outputs_are_physical():
    model = LeafBiochemistryModel(dtype=torch.float64)
    leafbio, meteo = _make_c3_inputs()

    result = model(leafbio, meteo)

    assert result.A.shape == (1,)
    assert torch.all(result.A > 0)
    assert torch.all(result.gs > 0)
    assert torch.all(result.rcw > 0)
    assert torch.all(result.Ci > 0)
    assert torch.all(result.Ci < meteo.Cs)
    assert torch.all(result.RH >= 0)
    assert torch.all(result.RH <= 1)
    assert torch.all(result.eta >= 0)
    assert torch.all(result.Kn >= 0)
    assert torch.allclose(result.SIF, result.fs * meteo.Q, atol=1e-12, rtol=1e-10)


def test_leaf_biochemistry_zero_intercept_matches_closed_form_ci():
    model = LeafBiochemistryModel(dtype=torch.float64)
    leafbio, meteo = _make_c3_inputs(intercept=0.0)

    result = model(leafbio, meteo)

    temperature_k = meteo.T + 273.15
    rh = torch.clamp(meteo.eb / model._satvap(temperature_k - 273.15), max=1.0)
    expected_ci = torch.maximum(
        0.3 * meteo.Cs,
        meteo.Cs * (1.0 - 1.6 / (leafbio.BallBerrySlope * rh)),
    )
    assert torch.allclose(result.Ci, expected_ci, atol=1e-10, rtol=1e-10)
    assert result.fcount == 1


def test_leaf_biochemistry_iterative_intercept_solves_ball_berry_fixed_point():
    model = LeafBiochemistryModel(dtype=torch.float64)
    leafbio, meteo = _make_c3_inputs(intercept=0.01)

    result = model(leafbio, meteo, options=BiochemicalOptions(ci_tol=1e-10, max_iter=120))

    ppm2bar = 1e-6 * (meteo.p * 1e-3)
    ci_bar = result.Ci * ppm2bar
    cs_bar = meteo.Cs * ppm2bar
    ci_next = model._ball_berry(
        cs_bar,
        result.RH,
        result.A * ppm2bar,
        leafbio.BallBerrySlope,
        leafbio.BallBerry0,
        0.3,
    )
    assert torch.allclose(ci_bar, ci_next, atol=2e-10, rtol=1e-10)
    assert result.fcount > 1


def test_leaf_biochemistry_vectorised_ci_solver_matches_scalar_brent_bit_for_bit():
    """The vectorised Brent path must produce bit-exact Ci as the scalar path.

    The vectorised solver runs the same Brent-Dekker algorithm as the
    per-cell scalar fallback, with state vectors and torch.where-masked
    updates so all cells advance in lockstep. Same root selection, same
    iteration count per cell, same residuals — so the converged Ci must
    agree exactly (no atol/rtol slack). This is what lets the MATLAB
    benchmark parity tests pass under vectorised_ci_solver=True at the
    strict 1e-9 relative tolerance.
    """
    model = LeafBiochemistryModel(dtype=torch.float64)
    # A batch where every cell exercises the non-zero-intercept Brent path.
    torch.manual_seed(0)
    batch = 32
    leafbio = LeafBiochemistryInputs(
        Vcmax25=torch.empty(batch, dtype=torch.float64).uniform_(40.0, 120.0),
        BallBerrySlope=torch.empty(batch, dtype=torch.float64).uniform_(5.0, 12.0),
        BallBerry0=torch.full((batch,), 0.01, dtype=torch.float64),
        RdPerVcmax25=torch.full((batch,), 0.025, dtype=torch.float64),
        Kn0=torch.full((batch,), 2.48, dtype=torch.float64),
        Knalpha=torch.full((batch,), 2.83, dtype=torch.float64),
        Knbeta=torch.full((batch,), 0.114, dtype=torch.float64),
        stressfactor=torch.full((batch,), 1.0, dtype=torch.float64),
    )
    meteo = LeafMeteo(
        Q=torch.empty(batch, dtype=torch.float64).uniform_(200.0, 1500.0),
        Cs=torch.empty(batch, dtype=torch.float64).uniform_(300.0, 420.0),
        T=torch.empty(batch, dtype=torch.float64).uniform_(18.0, 32.0),
        eb=torch.empty(batch, dtype=torch.float64).uniform_(10.0, 22.0),
        Oa=torch.full((batch,), 209.0, dtype=torch.float64),
        p=torch.full((batch,), 1000.0, dtype=torch.float64),
    )

    scalar = model(leafbio, meteo, options=BiochemicalOptions(ci_tol=1e-7, max_iter=120, vectorised_ci_solver=False))
    vectorised = model(leafbio, meteo, options=BiochemicalOptions(ci_tol=1e-7, max_iter=120, vectorised_ci_solver=True))

    assert torch.equal(scalar.Ci, vectorised.Ci)
    assert torch.equal(scalar.A, vectorised.A)


def test_leaf_biochemistry_ball_berry_intercept_params_have_nonzero_gradient():
    """BallBerrySlope and BallBerry0 must carry gradient through the SIF output.

    Both parameters only enter the Ci solve via the Ball-Berry equation
    inside `_solve_ci`, which discards autograd (scalar Brent uses .item()
    calls; vectorised Brent runs under torch.no_grad). Before the
    implicit-gradient recovery in `__call__`, both gradients were exactly
    zero — making BallBerrySlope and BallBerry0 effectively non-
    differentiable parameters. That broke EXP-008 / EXP-009 / EXP-011-
    style inversions, which would report "BallBerrySlope pinned at prior"
    as a science finding when it was an autograd artifact.

    This test pins the gradient-recovery contract: gradients on both
    Ball-Berry intercept parameters and on Vcmax25 (which gets an implicit
    component through ∂Ci/∂Vcmax) must be finite and non-zero through SIF.
    The forward output stays bit-exact because the one-step Picard at the
    converged Ci uses a stop-gradient surrogate (see __call__).
    """
    model = LeafBiochemistryModel(dtype=torch.float64)
    bbs = torch.tensor([9.0], dtype=torch.float64, requires_grad=True)
    bb0 = torch.tensor([0.01], dtype=torch.float64, requires_grad=True)
    vcm = torch.tensor([80.0], dtype=torch.float64, requires_grad=True)
    leafbio = LeafBiochemistryInputs(
        Vcmax25=vcm,
        BallBerrySlope=bbs,
        BallBerry0=bb0,
        RdPerVcmax25=torch.tensor([0.025], dtype=torch.float64),
        Kn0=torch.tensor([2.48], dtype=torch.float64),
        Knalpha=torch.tensor([2.83], dtype=torch.float64),
        Knbeta=torch.tensor([0.114], dtype=torch.float64),
        stressfactor=torch.tensor([1.0], dtype=torch.float64),
    )
    meteo = LeafMeteo(
        Q=torch.tensor([1000.0], dtype=torch.float64),
        Cs=torch.tensor([400.0], dtype=torch.float64),
        T=torch.tensor([25.0], dtype=torch.float64),
        eb=torch.tensor([15.0], dtype=torch.float64),
        Oa=torch.tensor([209.0], dtype=torch.float64),
        p=torch.tensor([1000.0], dtype=torch.float64),
    )
    result = model(leafbio, meteo)
    sif = result.fs * meteo.Q
    sif.sum().backward()

    for name, grad in (("BallBerrySlope", bbs.grad), ("BallBerry0", bb0.grad), ("Vcmax25", vcm.grad)):
        assert grad is not None, f"{name} got no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite: {grad}"
        assert not torch.equal(grad, torch.zeros_like(grad)), f"{name} gradient is exactly zero"


def test_leaf_biochemistry_c4_path_returns_finite_fluxes():
    model = LeafBiochemistryModel(dtype=torch.float64)
    leafbio = LeafBiochemistryInputs(
        Type="C4",
        Vcmax25=torch.tensor([40.0], dtype=torch.float64),
        BallBerrySlope=torch.tensor([4.0], dtype=torch.float64),
        BallBerry0=torch.tensor([0.01], dtype=torch.float64),
        RdPerVcmax25=torch.tensor([0.025], dtype=torch.float64),
        Kn0=torch.tensor([2.48], dtype=torch.float64),
        Knalpha=torch.tensor([2.83], dtype=torch.float64),
        Knbeta=torch.tensor([0.114], dtype=torch.float64),
        stressfactor=torch.tensor([1.0], dtype=torch.float64),
    )
    meteo = LeafMeteo(
        Q=torch.tensor([1500.0], dtype=torch.float64),
        Cs=torch.tensor([390.0], dtype=torch.float64),
        T=torch.tensor([30.0], dtype=torch.float64),
        eb=torch.tensor([18.0], dtype=torch.float64),
        Oa=torch.tensor([209.0], dtype=torch.float64),
        p=torch.tensor([970.0], dtype=torch.float64),
    )

    result = model(leafbio, meteo)

    assert torch.all(torch.isfinite(result.A))
    assert torch.all(torch.isfinite(result.Ci))
    assert torch.all(torch.isfinite(result.eta))
    assert torch.all(result.gs >= 0)


def test_leaf_biochemistry_accepts_constant_tensor_type_codes():
    model = LeafBiochemistryModel(dtype=torch.float64)
    leafbio, meteo = _make_c3_inputs()
    leafbio.Type = torch.zeros(2, dtype=torch.float64)
    meteo = LeafMeteo(
        Q=torch.tensor([1200.0, 1100.0], dtype=torch.float64),
        Cs=torch.tensor([390.0, 395.0], dtype=torch.float64),
        T=torch.tensor([25.0, 24.0], dtype=torch.float64),
        eb=torch.tensor([20.0, 19.0], dtype=torch.float64),
        Oa=torch.tensor([209.0, 209.0], dtype=torch.float64),
        p=torch.tensor([970.0, 970.0], dtype=torch.float64),
    )
    leafbio.Vcmax25 = torch.tensor([60.0, 62.0], dtype=torch.float64)
    leafbio.BallBerrySlope = torch.tensor([8.0, 8.0], dtype=torch.float64)
    leafbio.BallBerry0 = torch.tensor([0.01, 0.01], dtype=torch.float64)
    leafbio.RdPerVcmax25 = torch.tensor([0.015, 0.015], dtype=torch.float64)
    leafbio.Kn0 = torch.tensor([2.48, 2.48], dtype=torch.float64)
    leafbio.Knalpha = torch.tensor([2.83, 2.83], dtype=torch.float64)
    leafbio.Knbeta = torch.tensor([0.114, 0.114], dtype=torch.float64)
    leafbio.stressfactor = torch.tensor([1.0, 1.0], dtype=torch.float64)

    result = model(leafbio, meteo)

    assert result.A.shape == (2,)
    assert torch.all(result.A > 0)


def test_leaf_biochemistry_rejects_mixed_tensor_type_codes():
    model = LeafBiochemistryModel(dtype=torch.float64)
    leafbio, meteo = _make_c3_inputs()
    leafbio.Type = torch.tensor([0.0, 1.0], dtype=torch.float64)
    meteo = LeafMeteo(
        Q=torch.tensor([1200.0, 1200.0], dtype=torch.float64),
        Cs=torch.tensor([390.0, 390.0], dtype=torch.float64),
        T=torch.tensor([25.0, 25.0], dtype=torch.float64),
        eb=torch.tensor([20.0, 20.0], dtype=torch.float64),
        Oa=torch.tensor([209.0, 209.0], dtype=torch.float64),
        p=torch.tensor([970.0, 970.0], dtype=torch.float64),
    )
    leafbio.Vcmax25 = torch.tensor([60.0, 60.0], dtype=torch.float64)
    leafbio.BallBerrySlope = torch.tensor([8.0, 8.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="mixed C3/C4 batches"):
        model(leafbio, meteo)
