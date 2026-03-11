import math

import numpy as np
import torch

from prosail.FourSAIL import foursail as foursail_np

from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf


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

    expected = np.array([_hotspot_terms_scalar(*vals) for vals in zip(hotspot.tolist(), dso.tolist(), ks.tolist(), ko.tolist(), lai.tolist())])
    assert np.allclose(tsstoo.numpy(), expected[:, 0], atol=1e-12, rtol=1e-10)
    assert np.allclose(sumint.numpy(), expected[:, 1], atol=1e-12, rtol=1e-10)
