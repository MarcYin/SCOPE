from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import torch
from scipy import special as sp_special


@dataclass(slots=True)
class SpectralGrids:
    wlP: torch.Tensor
    wlF: Optional[torch.Tensor] = None
    wlE: Optional[torch.Tensor] = None

    @staticmethod
    def default(device: torch.device, dtype: torch.dtype) -> "SpectralGrids":
        wlP = torch.arange(400.0, 2501.0, 1.0, device=device, dtype=dtype)
        wlF = torch.arange(640.0, 851.0, 4.0, device=device, dtype=dtype)
        wlE = torch.arange(400.0, 751.0, 5.0, device=device, dtype=dtype)
        return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


@dataclass(slots=True)
class OptiPar:
    nr: torch.Tensor
    Kab: torch.Tensor
    Kca: torch.Tensor
    KcaV: torch.Tensor
    KcaZ: torch.Tensor
    Kdm: torch.Tensor
    Kw: torch.Tensor
    Ks: torch.Tensor
    Kant: torch.Tensor
    phi: torch.Tensor
    Kp: Optional[torch.Tensor] = None
    Kcbc: Optional[torch.Tensor] = None

    def to(self, device: torch.device, dtype: torch.dtype) -> "OptiPar":
        data = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                data[f.name] = None
                continue
            data[f.name] = torch.as_tensor(value, device=device, dtype=dtype)
        return OptiPar(**data)


@dataclass(slots=True)
class LeafBioBatch:
    Cab: torch.Tensor | float
    Cca: Optional[torch.Tensor | float] = None
    V2Z: torch.Tensor | float = 0.0
    Cw: torch.Tensor | float = 0.009
    Cdm: torch.Tensor | float = 0.012
    Cs: torch.Tensor | float = 0.0
    Cant: torch.Tensor | float = 1.0
    Cbc: torch.Tensor | float = 0.0
    Cp: torch.Tensor | float = 0.0
    N: torch.Tensor | float = 1.5
    fqe: torch.Tensor | float = 0.01


@dataclass(slots=True)
class LeafOptics:
    refl: torch.Tensor
    tran: torch.Tensor
    kChlrel: torch.Tensor
    Mb: Optional[torch.Tensor] = None
    Mf: Optional[torch.Tensor] = None


class FluspectModel:
    def __init__(
        self,
        spectral: SpectralGrids,
        optipar: OptiPar,
        ndub: int = 15,
        doublings_step: int = 5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device or spectral.wlP.device
        self.dtype = dtype
        self.spectral = self._coerce_spectral(spectral)
        self.optipar = optipar.to(self.device, self.dtype)
        self.ndub = ndub
        self.step = doublings_step

    def __call__(self, leafbio: LeafBioBatch) -> LeafOptics:
        batch_size, tensors = self._prepare_leafbio(leafbio)
        spectral = self.spectral
        optipar = self.optipar

        wlP = spectral.wlP
        nr = optipar.nr.unsqueeze(0)
        Kab = optipar.Kab.unsqueeze(0)
        Kdm = optipar.Kdm.unsqueeze(0)
        Kw = optipar.Kw.unsqueeze(0)
        Ks = optipar.Ks.unsqueeze(0)
        Kant = optipar.Kant.unsqueeze(0)
        Kp = self._optional_tensor(optipar.Kp, batch_size, wlP.numel())
        Kcbc = self._optional_tensor(optipar.Kcbc, batch_size, wlP.numel())

        Cab = tensors["Cab"]
        Cca = tensors["Cca"]
        V2Z = tensors["V2Z"]
        Cw = tensors["Cw"]
        Cdm = tensors["Cdm"]
        Cs = tensors["Cs"]
        Cant = tensors["Cant"]
        Cp = tensors["Cp"]
        Cbc = tensors["Cbc"]
        N = tensors["N"]
        fqe = tensors["fqe"]

        Kca = torch.where(
            (V2Z == -999).unsqueeze(-1),
            optipar.Kca.unsqueeze(0).expand(batch_size, -1),
            (1 - V2Z).unsqueeze(-1) * optipar.KcaV.unsqueeze(0) + V2Z.unsqueeze(-1) * optipar.KcaZ.unsqueeze(0),
        )

        numerator = Cab.unsqueeze(-1) * Kab
        numerator += Cca.unsqueeze(-1) * Kca
        numerator += Cdm.unsqueeze(-1) * Kdm
        numerator += Cw.unsqueeze(-1) * Kw
        numerator += Cs.unsqueeze(-1) * Ks
        numerator += Cant.unsqueeze(-1) * Kant
        if Cp is not None:
            numerator += Cp.unsqueeze(-1) * (Kp if Kp is not None else torch.zeros_like(Kab))
        if Cbc is not None:
            numerator += Cbc.unsqueeze(-1) * (Kcbc if Kcbc is not None else torch.zeros_like(Kab))
        Kall = numerator / N.unsqueeze(-1)

        tau, kChlrel = self._prospect_layer(Kall, Cab, Kab, N)
        kCarrel = torch.where(Kall > 0, Cca.unsqueeze(-1) * Kca / (Kall * N.unsqueeze(-1)), torch.zeros_like(Kall))

        (
            refl,
            tran,
            rho_core,
            tau_core,
            r21,
            talf,
            Rsub,
            Tsub,
            r,
            t,
        ) = self._combine_interfaces(tau, nr, N)

        leafopt = LeafOptics(refl=refl, tran=tran, kChlrel=kChlrel)

        if (fqe > 0).any():
            Mb, Mf = self._fluorescence(
                refl,
                tran,
                tau,
                talf,
                r21,
                kChlrel,
                kCarrel,
                k=N,
                k_full=None,
                Cab=Cab,
                Kall=Kall,
                r12=r12,
                t12=t12,
                t21=t21,
                r=r,
                leafbio=tensors,
                fqe=fqe,
                phi=optipar.phi,
                ralf=ralf,
            )
            leafopt.Mb = Mb
            leafopt.Mf = Mf

        return leafopt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_spectral(self, spectral: SpectralGrids) -> SpectralGrids:
        wlP = torch.as_tensor(spectral.wlP, device=self.device, dtype=self.dtype)
        wlF = spectral.wlF
        wlE = spectral.wlE
        if wlF is None or wlE is None:
            default = SpectralGrids.default(self.device, self.dtype)
            wlF = default.wlF if wlF is None else torch.as_tensor(wlF, device=self.device, dtype=self.dtype)
            wlE = default.wlE if wlE is None else torch.as_tensor(wlE, device=self.device, dtype=self.dtype)
        else:
            wlF = torch.as_tensor(wlF, device=self.device, dtype=self.dtype)
            wlE = torch.as_tensor(wlE, device=self.device, dtype=self.dtype)
        return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)

    def _prepare_leafbio(self, leafbio: LeafBioBatch) -> Tuple[int, dict[str, torch.Tensor]]:
        tensors: dict[str, torch.Tensor] = {}
        batch_size = None
        for f in fields(LeafBioBatch):
            value = getattr(leafbio, f.name)
            if value is None and f.name == "Cca":
                continue
            tensor = torch.as_tensor(value if value is not None else 0.0, device=self.device, dtype=self.dtype)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                if tensor.shape[0] == 1:
                    tensor = tensor.expand(batch_size)
                else:
                    raise ValueError("Leaf bio parameters must broadcast to a common batch size")
            tensors[f.name] = tensor
        if "Cca" not in tensors or (tensors["Cca"] == 0).all():
            tensors["Cca"] = 0.25 * tensors["Cab"]
        if batch_size is None:
            batch_size = 1
        return batch_size, {k: v for k, v in tensors.items()}

    def _optional_tensor(self, tensor: Optional[torch.Tensor], batch: int, size: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.unsqueeze(0).expand(batch, -1).to(self.device, self.dtype)

    def _prospect_layer(self, Kall: torch.Tensor, Cab: torch.Tensor, Kab: torch.Tensor, N: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t1 = (1 - Kall) * torch.exp(-Kall)
        exp_term = self._expint(Kall)
        t2 = (Kall**2) * exp_term
        tau = torch.where(Kall > 0, t1 + t2, torch.ones_like(Kall))
        kChlrel = torch.where(
            Kall > 0,
            Cab.unsqueeze(-1) * Kab / (Kall * N.unsqueeze(-1)),
            torch.zeros_like(Kall),
        )
        return tau, kChlrel

    def _expint(self, x: torch.Tensor) -> torch.Tensor:
        x_clamped = torch.clamp(x, min=1e-9)
        values = torch.from_numpy(sp_special.exp1(x_clamped.detach().cpu().numpy()))
        return values.to(x_clamped.device, x_clamped.dtype)

    def _calctav(self, alfa: float, nr: torch.Tensor) -> torch.Tensor:
        rd = math.pi / 180
        alfa_tensor = torch.as_tensor(alfa * rd, device=nr.device, dtype=nr.dtype)
        sa = torch.sin(alfa_tensor)
        n2 = nr**2
        np_ = n2 + 1
        nm = n2 - 1
        a = (nr + 1) ** 2 / 2
        k = -((n2 - 1) ** 2) / 4
        b1 = torch.sqrt(torch.clamp((sa**2 - np_ / 2) ** 2 + k, min=0.0))
        b2 = sa**2 - np_ / 2
        b = b1 - b2
        ts = ((k**2) / (6 * b**3) + k / b - b / 2) - ((k**2) / (6 * a**3) + k / a - a / 2)
        tp1 = -2 * n2 * (b - a) / (np_**2)
        tp2 = -2 * n2 * np_ * torch.log(b / a) / (nm**2)
        tp3 = n2 * (1 / b - 1 / a) / 2
        tp4 = 16 * n2**2 * (n2**2 + 1) * torch.log((2 * np_ * b - nm**2) / (2 * np_ * a - nm**2)) / (np_**3 * nm**2)
        tp5 = 16 * n2**3 * (1 / (2 * np_ * b - nm**2) - 1 / (2 * np_ * a - nm**2)) / (np_**3)
        tp = tp1 + tp2 + tp3 + tp4 + tp5
        tav = (ts + tp) / (2 * sa**2)
        return tav

    def _stacked_layers(self, r: torch.Tensor, t: torch.Tensor, N: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = r.shape[0]
        D = torch.sqrt(torch.clamp((1 + r + t) * (1 + r - t) * (1 - r + t) * (1 - r - t), min=0.0))
        rq = r**2
        tq = t**2
        a = (1 + rq - tq + D) / (2 * r.clamp(min=1e-9))
        b = (1 - rq + tq + D) / (2 * t.clamp(min=1e-9))
        bNm1 = b ** (N.unsqueeze(-1) - 1)
        bN2 = bNm1**2
        a2 = a**2
        denom = a2 * bN2 - 1
        Rsub = a * (bN2 - 1) / denom
        Tsub = bNm1 * (a2 - 1) / denom
        zero_abs = (r + t) >= 1
        if zero_abs.any():
            idx = zero_abs
            denom_zero = t[idx] + (1 - t[idx]) * (N.unsqueeze(-1)[idx] - 1)
            Tsub[idx] = t[idx] / denom_zero
            Rsub[idx] = 1 - Tsub[idx]
        return Rsub, Tsub

    def _combine_interfaces(
        self,
        tau: torch.Tensor,
        nr: torch.Tensor,
        N: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        talf = self._calctav(59.0, nr)
        ralf = 1 - talf
        t12 = self._calctav(90.0, nr)
        r12 = 1 - t12
        t21 = t12 / (nr**2)
        r21 = 1 - t21

        denom = 1 - r21 * r21 * tau**2
        Ta = talf * tau * t21 / denom
        Ra = ralf + r21 * tau * Ta
        t = t12 * tau * t21 / denom
        r = r12 + r21 * tau * t

        Rsub, Tsub = self._stacked_layers(r, t, N)
        denom2 = 1 - Rsub * r
        tran = Ta * Tsub / denom2
        refl = Ra + Ta * Rsub * t / denom2

        Rb = (refl - ralf) / (talf * t21 + (refl - ralf) * r21)
        Z = tran * (1 - Rb * r21) / (talf * t21)
        rho_core = (Rb - r21 * Z**2) / (1 - (r21 * Z) ** 2)
        tau_core = (1 - Rb * r21) / (1 - (r21 * Z) ** 2) * Z

        return refl, tran, rho_core, tau_core, r21, talf, Rsub, Tsub, r, t

    def _fluorescence(
        self,
        refl: torch.Tensor,
        tran: torch.Tensor,
        tau: torch.Tensor,
        talf: torch.Tensor,
        r21: torch.Tensor,
        kChlrel: torch.Tensor,
        kCarrel: torch.Tensor,
        **_: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Placeholder for the full fluorescence implementation.
        raise NotImplementedError("Fluorescence transport is not implemented yet")
