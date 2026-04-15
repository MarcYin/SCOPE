import math

import numpy as np
import pytest
import torch
from prosail.FourSAIL import foursail as foursail_np

from scope.canopy.foursail import FourSAILModel, campbell_lidf


def _hotspot_terms_scalar(hotspot, dso, ks, ko, lai):
    if hotspot <= 0 or dso == 0:
        ts = math.exp(-ks * lai)
        return ts, (1.0 - ts) / (ks * lai)
    alf = (dso / hotspot) * 2.0 / (ks + ko)
    if alf == 0:
        ts = math.exp(-ks * lai)
        return ts, (1.0 - ts) / (ks * lai)
    fhot = lai * math.sqrt(ko * ks)
    x1 = 0.0
    y1 = 0.0
    f1 = 1.0
    fint = (1.0 - math.exp(-alf)) * 0.05
    acc = 0.0
    for istep in range(1, 21):
        if istep < 20:
            x2 = -math.log(1.0 - istep * fint) / alf
        else:
            x2 = 1.0
        y2 = -(ko + ks) * lai * x2 + fhot * (1.0 - math.exp(-alf * x2)) / alf
        f2 = math.exp(y2)
        if abs(y2 - y1) > 1e-9:
            acc += (f2 - f1) * (x2 - x1) / (y2 - y1)
        x1, y1, f1 = x2, y2, f2
    return f1, 0.0 if math.isnan(acc) else acc


def test_foursail_matches_prosail():
    device = torch.device("cpu")
    dtype = torch.float64
    nwl = 64
    torch.manual_seed(0)
    rho = torch.rand(nwl, device=device, dtype=dtype) * 0.2 + 0.05
    tau = torch.rand(nwl, device=device, dtype=dtype) * 0.2 + 0.03
    soil = torch.rand(nwl, device=device, dtype=dtype) * 0.3 + 0.1
    lai = torch.tensor(3.2, device=device, dtype=dtype)
    hotspot = torch.tensor(0.2, device=device, dtype=dtype)
    tts = torch.tensor(35.0, device=device, dtype=dtype)
    tto = torch.tensor(20.0, device=device, dtype=dtype)
    psi = torch.tensor(10.0, device=device, dtype=dtype)
    lidfa = 57.0

    lidf = campbell_lidf(lidfa, device=device, dtype=dtype)
    model = FourSAILModel(lidf=lidf)
    torch_out = model(rho, tau, soil, lai, hotspot, tts, tto, psi)

    np_result = foursail_np(
        rho.cpu().numpy(),
        tau.cpu().numpy(),
        lidfa,
        0.0,
        2,
        float(lai.item()),
        float(hotspot.item()),
        float(tts.item()),
        float(tto.item()),
        float(psi.item()),
        soil.cpu().numpy(),
    )
    # unpack numpy outputs
    keys = [
        "tss",
        "too",
        "tsstoo",
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
        "gammasdf",
        "gammasdb",
        "gammaso",
    ]
    numpy_out = dict(zip(keys, np_result))

    assert np.allclose(torch_out.rdd.cpu().numpy(), numpy_out["rdd"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rsd.cpu().numpy(), numpy_out["rsd"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rdo.cpu().numpy(), numpy_out["rdo"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rso.cpu().numpy(), numpy_out["rso"], atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.rsot.cpu().numpy(), numpy_out["rsot"], atol=1e-8, rtol=1e-6)


def test_hotspot_terms_match_scalar_reference():
    dtype = torch.float64
    model = FourSAILModel()
    hotspot = torch.tensor([0.2, 0.0, 0.5, 0.3], dtype=dtype)
    dso = torch.tensor([0.4, 0.7, 0.0, 0.2], dtype=dtype)
    ks = torch.tensor([0.6, 0.7, 0.8, 0.9], dtype=dtype)
    ko = torch.tensor([0.5, 0.4, 0.3, 0.2], dtype=dtype)
    lai = torch.tensor([3.0, 2.5, 1.8, 4.0], dtype=dtype)

    tsstoo, sumint = model._hotspot_terms(hotspot, dso, ks, ko, lai)

    expected = np.array(
        [
            _hotspot_terms_scalar(*vals)
            for vals in zip(hotspot.tolist(), dso.tolist(), ks.tolist(), ko.tolist(), lai.tolist())
        ]
    )
    assert np.allclose(tsstoo.numpy(), expected[:, 0], atol=1e-12, rtol=1e-10)
    assert np.allclose(sumint.numpy(), expected[:, 1], atol=1e-12, rtol=1e-10)


# ── vectorized campbell_lidf tests ──────────────────────────────────────


class TestCampbellLidfVectorized:
    """Tests for the batched/vectorized campbell_lidf path."""

    def test_scalar_vs_batch_equivalence(self):
        """Each row of a batched call must match the corresponding scalar call."""
        alphas = [10.0, 30.0, 57.0, 70.0, 85.0]
        batch = campbell_lidf(torch.tensor(alphas, dtype=torch.float64))
        for i, a in enumerate(alphas):
            scalar = campbell_lidf(a, dtype=torch.float64)
            torch.testing.assert_close(batch[i], scalar, atol=1e-12, rtol=1e-10)

    def test_batch_shape(self):
        result = campbell_lidf(torch.tensor([10.0, 20.0, 30.0]))
        assert result.shape == (3, 18)

    def test_scalar_shape(self):
        result = campbell_lidf(57.0)
        assert result.shape == (18,)

    def test_single_element_tensor_returns_1d(self):
        result = campbell_lidf(torch.tensor(57.0))
        assert result.shape == (18,)

    def test_custom_n_elements(self):
        result_scalar = campbell_lidf(57.0, n_elements=13)
        assert result_scalar.shape == (13,)
        result_batch = campbell_lidf(torch.tensor([30.0, 57.0]), n_elements=13)
        assert result_batch.shape == (2, 13)

    @pytest.mark.parametrize("alpha", [0.1, 10.0, 30.0, 57.0, 70.0, 89.9])
    def test_valid_distribution(self, alpha):
        """Output must be a valid probability distribution (sums to 1, all >= 0)."""
        lidf = campbell_lidf(alpha, dtype=torch.float64)
        assert (lidf >= 0).all()
        torch.testing.assert_close(lidf.sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_batch_valid_distribution(self):
        alphas = torch.tensor([0.1, 30.0, 57.0, 89.9], dtype=torch.float64)
        lidf = campbell_lidf(alphas)
        assert (lidf >= 0).all()
        sums = lidf.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4, dtype=torch.float64), atol=1e-10, rtol=0)

    def test_planophile_weight_in_low_angles(self):
        """alpha near 0 should concentrate weight in the first (low-angle) bins."""
        lidf = campbell_lidf(5.0, dtype=torch.float64)
        assert lidf[:3].sum() > lidf[-3:].sum()

    def test_erectophile_weight_in_high_angles(self):
        """alpha near 90 should concentrate weight in the last (high-angle) bins."""
        lidf = campbell_lidf(85.0, dtype=torch.float64)
        assert lidf[-3:].sum() > lidf[:3].sum()

    @pytest.mark.parametrize("dt", [torch.float32, torch.float64])
    def test_dtype_preserved(self, dt):
        lidf = campbell_lidf(57.0, dtype=dt)
        assert lidf.dtype == dt
        lidf_b = campbell_lidf(torch.tensor([57.0], dtype=dt))
        assert lidf_b.dtype == dt
