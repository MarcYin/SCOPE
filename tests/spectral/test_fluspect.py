import numpy as np
import pytest
import torch
from scipy.special import exp1

from scope_torch.spectral.fluspect import (
    FluspectModel,
    LeafBioBatch,
    OptiPar,
    SpectralGrids,
)


def _make_spectral(device, dtype):
    wlP = torch.linspace(400.0, 700.0, 64, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 700.0, 16, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 700.0, 16, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


def _make_optipar(spectral: SpectralGrids) -> OptiPar:
    wl = spectral.wlP
    base = torch.linspace(0, 1, wl.numel(), dtype=wl.dtype, device=wl.device)
    nr = 1.4 + 0.05 * torch.sin(base)
    Kab = 0.01 + 0.005 * torch.cos(base)
    Kca = 0.008 + 0.003 * torch.sin(base * 2)
    KcaV = Kca * 0.9
    KcaZ = Kca * 1.1
    Kdm = 0.005 + 0.002 * torch.cos(base * 3)
    Kw = 0.002 + 0.001 * torch.sin(base * 4)
    Ks = 0.001 + 0.0005 * torch.cos(base * 5)
    Kant = 0.0002 + 0.0001 * torch.sin(base * 6)
    phi = torch.full_like(wl, 0.5)
    return OptiPar(
        nr=nr,
        Kab=Kab,
        Kca=Kca,
        KcaV=KcaV,
        KcaZ=KcaZ,
        Kdm=Kdm,
        Kw=Kw,
        Ks=Ks,
        Kant=Kant,
        phi=phi,
    )


def _make_leafbio(batch: int, device, dtype) -> LeafBioBatch:
    Cab = torch.full((batch,), 40.0, device=device, dtype=dtype)
    return LeafBioBatch(
        Cab=Cab,
        Cca=Cab * 0.2,
        V2Z=torch.full((batch,), -999.0, device=device, dtype=dtype),
        Cw=torch.full((batch,), 0.01, device=device, dtype=dtype),
        Cdm=torch.full((batch,), 0.012, device=device, dtype=dtype),
        Cs=torch.zeros(batch, device=device, dtype=dtype),
        Cant=torch.ones(batch, device=device, dtype=dtype),
        Cp=torch.zeros(batch, device=device, dtype=dtype),
        Cbc=torch.zeros(batch, device=device, dtype=dtype),
        N=torch.full((batch,), 1.5, device=device, dtype=dtype),
        fqe=torch.zeros(batch, device=device, dtype=dtype),
    )


def _calctav_np(alfa, nr):
    rd = np.pi / 180.0
    sa = np.sin(alfa * rd)
    n2 = nr**2
    np_ = n2 + 1
    nm = n2 - 1
    a = (nr + 1) ** 2 / 2
    k = -((n2 - 1) ** 2) / 4
    b1 = np.sqrt(np.maximum((sa**2 - np_ / 2) ** 2 + k, 0.0))
    b2 = sa**2 - np_ / 2
    b = b1 - b2
    ts = ((k**2) / (6 * b**3) + k / b - b / 2) - ((k**2) / (6 * a**3) + k / a - a / 2)
    tp1 = -2 * n2 * (b - a) / (np_**2)
    tp2 = -2 * n2 * np_ * np.log(b / a) / (nm**2)
    tp3 = n2 * (1 / b - 1 / a) / 2
    tp4 = 16 * n2**2 * (n2**2 + 1) * np.log((2 * np_ * b - nm**2) / (2 * np_ * a - nm**2)) / (np_**3 * nm**2)
    tp5 = 16 * n2**3 * (1 / (2 * np_ * b - nm**2) - 1 / (2 * np_ * a - nm**2)) / (np_**3)
    tp = tp1 + tp2 + tp3 + tp4 + tp5
    return (ts + tp) / (2 * sa**2)


def _stack_layers_np(r, t, N):
    D = np.sqrt(np.maximum((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), 0.0))
    rq = r**2
    tq = t**2
    a = (1 + rq - tq + D) / (2 * np.maximum(r, 1e-9))
    b = (1 - rq + tq + D) / (2 * np.maximum(t, 1e-9))
    bNm1 = b ** (N - 1)
    bN2 = bNm1**2
    a2 = a**2
    denom = a2 * bN2 - 1
    Rsub = a * (bN2 - 1) / denom
    Tsub = bNm1 * (a2 - 1) / denom
    idx = (r + t) >= 1
    if np.any(idx):
        denom_zero = t[idx] + (1 - t[idx]) * (N[idx] - 1)
        Tsub[idx] = t[idx] / denom_zero
        Rsub[idx] = 1 - Tsub[idx]
    return Rsub, Tsub


def _fluspect_numpy(optipar: OptiPar, leafbio: LeafBioBatch, spectral: SpectralGrids):
    wl = spectral.wlP.cpu().numpy()
    nr = optipar.nr.cpu().numpy()
    Kab = optipar.Kab.cpu().numpy()
    Kca = optipar.Kca.cpu().numpy()
    Kdm = optipar.Kdm.cpu().numpy()
    Kw = optipar.Kw.cpu().numpy()
    Ks = optipar.Ks.cpu().numpy()
    Kant = optipar.Kant.cpu().numpy()

    Cab = leafbio.Cab.cpu().numpy()
    Cca = (leafbio.Cca if leafbio.Cca is not None else leafbio.Cab * 0.25).cpu().numpy()
    Cw = leafbio.Cw.cpu().numpy()
    Cdm = leafbio.Cdm.cpu().numpy()
    Cs = leafbio.Cs.cpu().numpy()
    Cant = leafbio.Cant.cpu().numpy()
    N = leafbio.N.cpu().numpy()

    numerator = Cab[:, None] * Kab
    numerator += Cca[:, None] * Kca
    numerator += Cdm[:, None] * Kdm
    numerator += Cw[:, None] * Kw
    numerator += Cs[:, None] * Ks
    numerator += Cant[:, None] * Kant
    Kall = numerator / N[:, None]

    t1 = (1 - Kall) * np.exp(-Kall)
    t2 = (Kall**2) * exp1(np.maximum(Kall, 1e-9))
    tau = np.where(Kall > 0, t1 + t2, 1.0)

    talf = _calctav_np(59.0, nr)[None, :]
    ralf = 1 - talf
    t12 = _calctav_np(90.0, nr)[None, :]
    r12 = 1 - t12
    t21 = t12 / (nr[None, :] ** 2)
    r21 = 1 - t21

    denom = 1 - r21 * r21 * tau**2
    Ta = talf * tau * t21 / denom
    Ra = ralf + r21 * tau * Ta
    t = t12 * tau * t21 / denom
    r = r12 + r21 * tau * t

    Rsub, Tsub = _stack_layers_np(r, t, N[:, None])
    denom2 = 1 - Rsub * r
    tran = Ta * Tsub / denom2
    refl = Ra + Ta * Rsub * t / denom2
    return refl, tran


def test_leafopt_shapes():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _make_spectral(device, dtype)
    optipar = _make_optipar(spectral)
    model = FluspectModel(spectral, optipar, dtype=dtype)
    leafbio = _make_leafbio(batch=3, device=device, dtype=dtype)
    outputs = model(leafbio)
    assert outputs.refl.shape == (3, spectral.wlP.numel())
    assert outputs.tran.shape == (3, spectral.wlP.numel())
    assert outputs.kChlrel.shape == (3, spectral.wlP.numel())


def test_numpy_parity():
    device = torch.device("cpu")
    dtype = torch.float64
    spectral = _make_spectral(device, dtype)
    optipar = _make_optipar(spectral)
    model = FluspectModel(spectral, optipar, dtype=dtype)
    leafbio = _make_leafbio(batch=2, device=device, dtype=dtype)
    torch_out = model(leafbio)
    refl_np, tran_np = _fluspect_numpy(optipar, leafbio, spectral)
    assert np.allclose(torch_out.refl.cpu().numpy(), refl_np, atol=1e-8, rtol=1e-6)
    assert np.allclose(torch_out.tran.cpu().numpy(), tran_np, atol=1e-8, rtol=1e-6)
