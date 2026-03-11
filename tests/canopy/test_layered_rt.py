import math

import numpy as np
import torch
from scipy.integrate import quad

from scope_torch.canopy.foursail import FourSAILModel, scope_lidf, scope_litab
from scope_torch.canopy.layered_rt import LayeredCanopyTransportModel


def _pso_function(y: float, K: float, k: float, lai: float, hotspot: float, dso: float) -> float:
    if abs(dso) <= 1e-12 or hotspot <= 0.0:
        return math.exp((K + k) * lai * y - math.sqrt(max(K * k, 0.0)) * lai * y)
    alf = (dso / hotspot) * 2.0 / (k + K)
    return math.exp((K + k) * lai * y + math.sqrt(max(K * k, 0.0)) * lai / alf * (1.0 - math.exp(y * alf)))


def test_scope_lidf_uses_scope_angular_grid():
    dtype = torch.float64
    lidf = scope_lidf(0.0, 0.0, dtype=dtype)
    model = FourSAILModel(lidf=lidf)

    assert lidf.shape == (13,)
    assert torch.all(lidf >= 0)
    assert torch.isclose(lidf.sum(), torch.tensor(1.0, dtype=dtype))
    assert torch.allclose(model._litab, scope_litab(dtype=dtype))


def test_bidirectional_gap_profile_matches_quad_reference():
    dtype = torch.float64
    transport = LayeredCanopyTransportModel(FourSAILModel())
    ko = torch.tensor([0.55, 1.2, 0.2, 1.5], dtype=dtype)
    ks = torch.tensor([0.73, 1.0, 0.9, 1.3], dtype=dtype)
    lai = torch.tensor([3.0, 6.0, 1.0, 7.0], dtype=dtype)
    hotspot = torch.tensor([0.2, 0.05, 0.5, 0.01], dtype=dtype)
    dso = torch.tensor([0.4, 0.9, 0.2, 1.5], dtype=dtype)

    nlayers = 6
    dx = torch.tensor(1.0 / nlayers, dtype=dtype)
    xl = torch.cat([torch.zeros(1, dtype=dtype), -torch.arange(1, nlayers + 1, dtype=dtype) * dx], dim=0)

    actual = transport._bidirectional_gap_profile(ko, ks, lai, hotspot, dso, xl, dx).numpy()
    expected = np.zeros_like(actual)
    for batch_idx in range(ko.numel()):
        for layer_idx, upper in enumerate(xl.tolist()):
            lower = upper - float(dx)
            integral, _ = quad(
                _pso_function,
                lower,
                upper,
                args=(
                    float(ko[batch_idx]),
                    float(ks[batch_idx]),
                    float(lai[batch_idx]),
                    float(hotspot[batch_idx]),
                    float(dso[batch_idx]),
                ),
                epsabs=1e-12,
                epsrel=1e-12,
                limit=200,
            )
            expected[batch_idx, layer_idx] = integral / float(dx)

    assert np.allclose(actual, expected, atol=1e-11, rtol=1e-10)
