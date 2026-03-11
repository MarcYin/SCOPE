from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import torch


_EULER_GAMMA = 0.5772156649015328606


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
    kCarrel: torch.Tensor
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

    @classmethod
    def from_scope_assets(
        cls,
        path: str | None = None,
        *,
        scope_root_path: str | None = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        wlF: Optional[torch.Tensor] = None,
        wlE: Optional[torch.Tensor] = None,
    ) -> "FluspectModel":
        from .loaders import load_fluspect_resources

        resources = load_fluspect_resources(
            path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
            wlF=wlF,
            wlE=wlE,
        )
        return cls(
            resources.spectral,
            resources.optipar,
            ndub=ndub,
            doublings_step=doublings_step,
            device=device,
            dtype=dtype,
        )

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
            ralf,
            t21,
            r12,
            t12,
        ) = self._combine_interfaces(tau, nr, N)

        leafopt = LeafOptics(refl=refl, tran=tran, kChlrel=kChlrel, kCarrel=kCarrel)

        if (fqe > 0).any():
            Mb, Mf = self._fluorescence(
                rho_core=rho_core,
                tau_core=tau_core,
                refl=refl,
                tran=tran,
                talf=talf,
                r21=r21,
                t21=t21,
                kChlrel=kChlrel,
                fqe=fqe,
                phi=optipar.phi,
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

    def _interp1d(self, source_x: torch.Tensor, source_y: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        source_x = source_x.to(self.device, self.dtype)
        target_x = target_x.to(self.device, self.dtype)
        idx = torch.bucketize(target_x, source_x) - 1
        idx = idx.clamp(0, source_x.numel() - 2)
        x0 = source_x[idx]
        x1 = source_x[idx + 1]
        denom = (x1 - x0).clamp(min=1e-9)
        frac = (target_x - x0) / denom
        batch = source_y.shape[0]
        gather0 = source_y.gather(1, idx.unsqueeze(0).expand(batch, -1))
        gather1 = source_y.gather(1, (idx + 1).unsqueeze(0).expand(batch, -1))
        return gather0 + (gather1 - gather0) * frac

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
        result = torch.empty_like(x_clamped)

        small = x_clamped <= 1.0
        if small.any():
            xs = x_clamped[small]
            series = torch.zeros_like(xs)
            term = torch.ones_like(xs)
            for n in range(1, 21):
                term = term * (-xs / n)
                series = series - term / n
            result[small] = -torch.log(xs) - _EULER_GAMMA + series

        large = ~small
        if large.any():
            xl = x_clamped[large]
            b = xl + 1.0
            c = torch.full_like(xl, 1e30)
            d = 1.0 / b
            h = d.clone()
            for i in range(1, 101):
                a = -(i * i)
                b = b + 2.0
                d = 1.0 / (a * d + b)
                c = b + a / c
                delta = c * d
                h = h * delta
            result[large] = h * torch.exp(-xl)

        return result

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

        return refl, tran, rho_core, tau_core, r21, talf, ralf, t21, r12, t12

    def _fluorescence(
        self,
        *,
        rho_core: torch.Tensor,
        tau_core: torch.Tensor,
        refl: torch.Tensor,
        tran: torch.Tensor,
        talf: torch.Tensor,
        r21: torch.Tensor,
        t21: torch.Tensor,
        kChlrel: torch.Tensor,
        fqe: torch.Tensor,
        phi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        dtype = self.dtype
        spectral = self.spectral
        wlP = spectral.wlP
        wlE = spectral.wlE
        wlF = spectral.wlF
        batch = rho_core.shape[0]

        if wlE is None or wlF is None:
            raise ValueError("Spectral grids must define excitation and fluorescence wavelengths")

        rho = torch.clamp(rho_core, min=0.0)
        tau = tau_core

        sum_rt = rho + tau
        I_rt = sum_rt < 1 - 1e-9

        D = torch.zeros_like(rho)
        temp = (1 + rho + tau) * (1 + rho - tau) * (1 - rho + tau) * (1 - rho - tau)
        D[I_rt] = torch.sqrt(torch.clamp(temp[I_rt], min=0.0))
        a = torch.ones_like(rho)
        b = torch.ones_like(rho)
        safe_r = rho.clamp(min=1e-9)
        safe_t = tau.clamp(min=1e-9)
        a[I_rt] = (1 + rho[I_rt] ** 2 - tau[I_rt] ** 2 + D[I_rt]) / (2 * safe_r[I_rt])
        b[I_rt] = (1 - rho[I_rt] ** 2 + tau[I_rt] ** 2 + D[I_rt]) / (2 * safe_t[I_rt])

        s = torch.where(tau.abs() > 0, rho / tau, torch.zeros_like(rho))
        logb = torch.log(torch.clamp(b, min=1e-12))
        I_a = (a > 1) & torch.isfinite(a)
        s[I_a] = 2 * a[I_a] / (a[I_a] ** 2 - 1) * logb[I_a]

        k = logb.clone()
        k[I_a] = (a[I_a] - 1) / (a[I_a] + 1) * logb[I_a]
        kChl = kChlrel * k

        r21_batch = r21.expand(batch, -1)
        talf_batch = talf.expand(batch, -1)
        t21_batch = t21.expand(batch, -1)

        k_ex = self._interp1d(wlP, k, wlE)
        s_ex = self._interp1d(wlP, s, wlE)
        kChl_ex = self._interp1d(wlP, kChl, wlE)
        r21_ex = self._interp1d(wlP, r21_batch, wlE)
        rho_ex = self._interp1d(wlP, rho, wlE)
        tau_ex = self._interp1d(wlP, tau, wlE)
        talf_ex = self._interp1d(wlP, talf_batch, wlE)

        k_em = self._interp1d(wlP, k, wlF)
        s_em = self._interp1d(wlP, s, wlF)
        r21_em = self._interp1d(wlP, r21_batch, wlF)
        rho_em = self._interp1d(wlP, rho, wlF)
        tau_em = self._interp1d(wlP, tau, wlF)
        t21_em = self._interp1d(wlP, t21_batch, wlF)

        eps = 2.0 ** (-self.ndub)
        te = 1 - (k_ex + s_ex) * eps
        tf = 1 - (k_em + s_em) * eps
        re = s_ex * eps
        rf = s_em * eps

        sigmoid = 1.0 / (
            1.0
            + torch.exp(-wlF.view(-1, 1).to(device, dtype) / 10.0)
            * torch.exp(wlE.view(1, -1).to(device, dtype) / 10.0)
        )
        sigmoid = sigmoid.unsqueeze(0)

        phi_em = self._interp1d(wlP, phi.unsqueeze(0).expand(batch, -1), wlF)
        coeff = self.step * eps * 0.5 * fqe.unsqueeze(-1).unsqueeze(-1)
        Mf = coeff * phi_em.unsqueeze(-1) * kChl_ex.unsqueeze(1) * sigmoid
        Mb = Mf.clone()

        for _ in range(self.ndub):
            xe = te / (1 - re * re)
            xf = tf / (1 - rf * rf)
            ten = te * xe
            tfn = tf * xf
            ren = re * (1 + ten)
            rfn = rf * (1 + tfn)

            prod = xf.unsqueeze(-1) * xe.unsqueeze(1)
            A11 = xf.unsqueeze(-1) + xe.unsqueeze(1)
            A12 = prod * (rf.unsqueeze(-1) + re.unsqueeze(1))
            A21 = 1 + prod * (1 + rf.unsqueeze(-1) * re.unsqueeze(1))
            A22 = (xf * rf).unsqueeze(-1) + (xe * re).unsqueeze(1)

            Mf_new = Mf * A11 + Mb * A12
            Mb_new = Mb * A21 + Mf * A22

            te, tf, re, rf = ten, tfn, ren, rfn
            Mf, Mb = Mf_new, Mb_new

        Rb = rho + tau**2 * r21_batch / (1 - rho * r21_batch)
        Rb_ex = self._interp1d(wlP, Rb, wlE)
        Rb_em = self._interp1d(wlP, Rb, wlF)

        Xe = talf_ex / (1 - r21_ex * Rb_ex)
        Ye = tau_ex * r21_ex / (1 - rho_ex * r21_ex)
        Xf = t21_em / (1 - r21_em * Rb_em)
        Yf = tau_em * r21_em / (1 - rho_em * r21_em)

        Xe_mat = Xe.unsqueeze(1)
        Ye_mat = Ye.unsqueeze(1)
        Xf_mat = Xf.unsqueeze(-1)
        Yf_mat = Yf.unsqueeze(-1)

        A = Xe_mat * (1 + Ye_mat * Yf_mat) * Xf_mat
        B = Xe_mat * (Ye_mat + Yf_mat) * Xf_mat

        g = Mb
        f = Mf
        Mb_final = A * g + B * f
        Mf_final = A * f + B * g

        return Mb_final, Mf_final
