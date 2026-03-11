from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import torch


def scope_litab(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64
    coarse = torch.arange(5.0, 76.0, 10.0, device=device, dtype=dtype)
    fine = torch.arange(81.0, 90.0, 2.0, device=device, dtype=dtype)
    return torch.cat([coarse, fine], dim=0)


def scope_lazitab(*, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    device = device or torch.device("cpu")
    dtype = dtype or torch.float64
    return torch.arange(5.0, 356.0, 10.0, device=device, dtype=dtype)


def scope_lidf(
    lidfa: float,
    lidfb: float = 0.0,
    *,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """SCOPE/SAIL leaf-angle distribution using the upstream `leafangles.m` discretization."""

    device = device or torch.device("cpu")
    dtype = dtype or torch.float64
    a = torch.as_tensor(lidfa, device=device, dtype=dtype)
    b = torch.as_tensor(lidfb, device=device, dtype=dtype)
    if a.ndim != 0 or b.ndim != 0:
        raise ValueError("scope_lidf expects scalar lidfa/lidfb values")

    def dcum(theta_deg: float) -> torch.Tensor:
        theta = torch.as_tensor(theta_deg, device=device, dtype=dtype)
        rd = torch.pi / 180.0
        if bool((a > 1).item()):
            return 1.0 - torch.cos(theta * rd)

        eps = torch.as_tensor(1e-8, device=device, dtype=dtype)
        x = 2.0 * rd * theta
        theta2 = x.clone()
        delx = torch.as_tensor(1.0, device=device, dtype=dtype)
        y = torch.zeros_like(x)
        while bool((delx > eps).item()):
            y = a * torch.sin(x) + 0.5 * b * torch.sin(2.0 * x)
            dx = 0.5 * (y - x + theta2)
            x = x + dx
            delx = torch.abs(dx)
        return (2.0 * y + theta2) / torch.pi

    F = torch.zeros(13, device=device, dtype=dtype)
    for idx in range(8):
        F[idx] = dcum((idx + 1) * 10.0)
    for idx in range(8, 12):
        F[idx] = dcum(80.0 + (idx - 7) * 2.0)
    F[-1] = 1.0

    lidf = torch.zeros_like(F)
    lidf[0] = F[0]
    lidf[1:] = F[1:] - F[:-1]
    lidf = torch.clamp(lidf, min=0.0)
    return lidf / lidf.sum().clamp(min=1e-12)


def campbell_lidf(alpha: float, n_elements: int = 18, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Compute an ellipsoidal leaf inclination distribution (Campbell, 1990).

    Parameters
    ----------
    alpha:
        Mean leaf angle in degrees (57 deg approximates a spherical distribution).
    n_elements:
        Number of discrete inclination bins spanning 0-90 degrees.
    device, dtype:
        Optional torch placement for the returned tensor.
    """

    device = device or torch.device("cpu")
    dtype = dtype or torch.float64
    alpha_tensor = torch.as_tensor(alpha, dtype=dtype, device=device)
    excent = torch.exp(
        -1.6184e-5 * alpha_tensor**3 + 2.1145e-3 * alpha_tensor**2 - 1.2390e-1 * alpha_tensor + 3.2491
    )
    step = 90.0 / n_elements
    freq = torch.zeros(n_elements, dtype=dtype, device=device)
    for idx in range(n_elements):
        tl1 = torch.deg2rad(torch.tensor(idx * step, dtype=dtype, device=device))
        tl2 = torch.deg2rad(torch.tensor((idx + 1.0) * step, dtype=dtype, device=device))
        x1 = excent / torch.sqrt(1.0 + excent**2 * torch.tan(tl1) ** 2)
        x2 = excent / torch.sqrt(1.0 + excent**2 * torch.tan(tl2) ** 2)
        if torch.isclose(excent, torch.tensor(1.0, dtype=dtype, device=device), atol=1e-12):
            freq[idx] = torch.abs(torch.cos(tl1) - torch.cos(tl2))
        else:
            alph = excent / torch.sqrt(torch.abs(1.0 - excent**2))
            alph2 = alph**2
            x1_sq = x1**2
            x2_sq = x2**2
            if excent > 1:
                alpx1 = torch.sqrt(alph2 + x1_sq)
                alpx2 = torch.sqrt(alph2 + x2_sq)
                term1 = x1 * alpx1 + alph2 * torch.log(x1 + alpx1)
                term2 = x2 * alpx2 + alph2 * torch.log(x2 + alpx2)
                freq[idx] = torch.abs(term1 - term2)
            else:
                almx1 = torch.sqrt(alph2 - x1_sq)
                almx2 = torch.sqrt(alph2 - x2_sq)
                term1 = x1 * almx1 + alph2 * torch.arcsin(x1 / alph)
                term2 = x2 * almx2 + alph2 * torch.arcsin(x2 / alph)
                freq[idx] = torch.abs(term1 - term2)
    freq_sum = freq.sum()
    lidf = torch.where(freq_sum > 0, freq / freq_sum, torch.full_like(freq, 1.0 / n_elements))
    return lidf


@dataclass(slots=True)
class FourSAILResult:
    rdd: torch.Tensor
    tdd: torch.Tensor
    rsd: torch.Tensor
    tsd: torch.Tensor
    rdo: torch.Tensor
    tdo: torch.Tensor
    rso: torch.Tensor
    rsos: torch.Tensor
    rsod: torch.Tensor
    rddt: torch.Tensor
    rsdt: torch.Tensor
    rdot: torch.Tensor
    rsodt: torch.Tensor
    rsost: torch.Tensor
    rsot: torch.Tensor
    tss: torch.Tensor
    too: torch.Tensor
    tsstoo: torch.Tensor
    gammasdf: torch.Tensor
    gammasdb: torch.Tensor
    gammaso: torch.Tensor


class FourSAILModel:
    """PyTorch translation of the 4-stream canopy model (SAILh)."""

    def __init__(self, lidf: Optional[torch.Tensor] = None, n_angles: int = 18, *, litab: Optional[torch.Tensor] = None) -> None:
        self.default_lidf = lidf
        if litab is not None:
            self._litab = torch.as_tensor(litab, dtype=torch.float64)
            self.n_angles = int(self._litab.numel())
            if lidf is not None and int(lidf.shape[-1]) != self.n_angles:
                raise ValueError("litab and lidf must have matching angle counts")
            return

        self.n_angles = int(lidf.shape[-1]) if lidf is not None else n_angles
        if self.n_angles == 13:
            self._litab = scope_litab(dtype=torch.float64)
        else:
            step = 90.0 / self.n_angles
            self._litab = torch.linspace(step / 2.0, 90.0 - step / 2.0, self.n_angles, dtype=torch.float64)

    def __call__(
        self,
        rho: torch.Tensor,
        tau: torch.Tensor,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        hotspot: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        lidf: Optional[torch.Tensor] = None,
    ) -> FourSAILResult:
        rho = self._ensure_2d(rho)
        tau = self._ensure_2d(tau)
        soil = self._ensure_2d(soil_refl, target_shape=rho.shape)
        batch, nwl = rho.shape
        device = rho.device
        dtype = rho.dtype

        lai = self._expand_param(lai, batch, device, dtype)
        hotspot = self._expand_param(hotspot, batch, device, dtype)
        tts = self._expand_param(tts, batch, device, dtype)
        tto = self._expand_param(tto, batch, device, dtype)
        psi = self._expand_param(psi, batch, device, dtype)

        lidf_tensor = lidf if lidf is not None else self.default_lidf
        if lidf_tensor is None:
            raise ValueError("A leaf inclination distribution (lidf) must be provided")
        lidf_tensor = torch.as_tensor(lidf_tensor, device=device, dtype=dtype)
        if lidf_tensor.ndim == 1:
            lidf_tensor = lidf_tensor.unsqueeze(0).expand(batch, -1)
        elif lidf_tensor.shape[0] != batch:
            raise ValueError("lidf must broadcast to the batch dimension")
        lidf_tensor = lidf_tensor / lidf_tensor.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        n_angles = lidf_tensor.shape[1]
        if n_angles != self.n_angles:
            raise ValueError(f"lidf has {n_angles} elements but model expects {self.n_angles}")
        litab = self._litab.to(device=device, dtype=dtype)

        geometry = self._geometry_constants(tts, tto, psi)
        ks, ko, bf, sob, sof = self._weighted_sum_over_lidf(tts, tto, psi, lidf_tensor, litab)

        sdb = 0.5 * (ks + bf)
        sdf = 0.5 * (ks - bf)
        dob = 0.5 * (ko + bf)
        dof = 0.5 * (ko - bf)
        ddb = 0.5 * (1.0 + bf)
        ddf = 0.5 * (1.0 - bf)

        sigb = ddb.unsqueeze(-1) * rho + ddf.unsqueeze(-1) * tau
        sigf = ddf.unsqueeze(-1) * rho + ddb.unsqueeze(-1) * tau
        sigb = torch.clamp(sigb, min=1e-36)
        sigf = torch.clamp(sigf, min=1e-36)
        att = 1.0 - sigf
        m = torch.sqrt(torch.clamp(att**2 - sigb**2, min=1e-36))
        sb = sdb.unsqueeze(-1) * rho + sdf.unsqueeze(-1) * tau
        sf = sdf.unsqueeze(-1) * rho + sdb.unsqueeze(-1) * tau
        vb = dob.unsqueeze(-1) * rho + dof.unsqueeze(-1) * tau
        vf = dof.unsqueeze(-1) * rho + dob.unsqueeze(-1) * tau
        w = sob.unsqueeze(-1) * rho + sof.unsqueeze(-1) * tau

        outputs = self._init_outputs(batch, nwl, device, dtype)

        positive = lai > 0
        if positive.any():
            ks_pos = ks[positive]
            ko_pos = ko[positive]
            bf_pos = bf[positive]
            lai_pos = lai[positive]
            hotspot_pos = hotspot[positive]
            m_pos = m[positive]
            att_pos = att[positive]
            sigb_pos = sigb[positive]
            sb_pos = sb[positive]
            sf_pos = sf[positive]
            vb_pos = vb[positive]
            vf_pos = vf[positive]
            w_pos = w[positive]
            rho_pos = rho[positive]
            tau_pos = tau[positive]
            soil_pos = soil[positive]
            geometry_pos = tuple(g[positive] for g in geometry)

            pos_outputs = self._solve_four_stream(
                ks_pos,
                ko_pos,
                bf_pos,
                lai_pos,
                hotspot_pos,
                m_pos,
                att_pos,
                sigb_pos,
                sb_pos,
                sf_pos,
                vb_pos,
                vf_pos,
                w_pos,
                rho_pos,
                tau_pos,
                soil_pos,
                geometry_pos,
            )
            for field in fields(FourSAILResult):
                name = field.name
                getattr(outputs, name)[positive] = getattr(pos_outputs, name)

        return outputs

    def _ensure_2d(self, tensor: torch.Tensor, target_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        tensor = torch.as_tensor(tensor)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError("Spectral inputs must be 1D or 2D tensors")
        if target_shape is not None and tensor.shape != target_shape:
            raise ValueError("Spectral tensors must have matching shapes")
        return tensor

    def _expand_param(
        self, value: torch.Tensor | float, batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.repeat(batch)
        elif tensor.shape[0] != batch:
            raise ValueError("Scalar parameters must broadcast to the batch dimension")
        return tensor

    def _init_outputs(self, batch: int, nwl: int, device: torch.device, dtype: torch.dtype) -> FourSAILResult:
        zeros = lambda: torch.zeros((batch, nwl), device=device, dtype=dtype)
        zeros_scalar = lambda: torch.zeros(batch, device=device, dtype=dtype)
        ones_scalar = lambda: torch.ones(batch, device=device, dtype=dtype)
        ones_spectral = lambda: torch.ones((batch, nwl), device=device, dtype=dtype)
        return FourSAILResult(
            rdd=zeros(),
            tdd=ones_spectral(),
            rsd=zeros(),
            tsd=zeros(),
            rdo=zeros(),
            tdo=zeros(),
            rso=zeros(),
            rsos=zeros(),
            rsod=zeros(),
            rddt=zeros(),
            rsdt=zeros(),
            rdot=zeros(),
            rsodt=zeros(),
            rsost=zeros(),
            rsot=zeros(),
            tss=ones_scalar(),
            too=ones_scalar(),
            tsstoo=ones_scalar(),
            gammasdf=zeros(),
            gammasdb=zeros(),
            gammaso=zeros(),
        )

    def _geometry_constants(
        self, tts: torch.Tensor, tto: torch.Tensor, psi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        deg2rad = torch.pi / 180.0
        tts_rad = torch.deg2rad(tts)
        tto_rad = torch.deg2rad(tto)
        cts = torch.cos(tts_rad)
        cto = torch.cos(tto_rad)
        tants = torch.tan(tts_rad)
        tanto = torch.tan(tto_rad)
        cospsi = torch.cos(torch.deg2rad(psi))
        dso = torch.sqrt(torch.clamp(tants**2 + tanto**2 - 2 * tants * tanto * cospsi, min=0.0))
        ctscto = cts * cto
        return cts, cto, ctscto, tants, tanto, cospsi, dso

    def _weighted_sum_over_lidf(
        self,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        lidf: torch.Tensor,
        litab: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = tts.shape[0]
        n_angles = litab.shape[0]
        device = tts.device
        dtype = tts.dtype
        chi_s = torch.zeros((batch, n_angles), device=device, dtype=dtype)
        chi_o = torch.zeros_like(chi_s)
        frho = torch.zeros_like(chi_s)
        ftau = torch.zeros_like(chi_s)
        for idx in range(n_angles):
            angle = litab[idx]
            chi_s[:, idx], chi_o[:, idx], frho[:, idx], ftau[:, idx] = self._volscatt(tts, tto, psi, angle)
        cts = torch.cos(torch.deg2rad(tts)).unsqueeze(-1).clamp(min=1e-6)
        cto = torch.cos(torch.deg2rad(tto)).unsqueeze(-1).clamp(min=1e-6)
        cos_tts = cts
        cos_tto = cto
        sobli = frho * torch.pi / (cos_tts * cos_tto)
        sofli = ftau * torch.pi / (cos_tts * cos_tto)
        bfli = torch.cos(torch.deg2rad(litab)).to(device=device, dtype=dtype) ** 2
        bfli = bfli.unsqueeze(0).expand(batch, -1)
        ksli = chi_s / cos_tts
        koli = chi_o / cos_tto
        ks = torch.sum(ksli * lidf, dim=-1)
        ko = torch.sum(koli * lidf, dim=-1)
        bf = torch.sum(bfli * lidf, dim=-1)
        sob = torch.sum(sobli * lidf, dim=-1)
        sof = torch.sum(sofli * lidf, dim=-1)
        return ks, ko, bf, sob, sof

    def _volscatt(
        self, tts: torch.Tensor, tto: torch.Tensor, psi: torch.Tensor, ttl: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        deg2rad = torch.pi / 180.0
        tts_rad = torch.deg2rad(tts)
        tto_rad = torch.deg2rad(tto)
        psi_rad = torch.deg2rad(psi)
        ttl_rad = torch.deg2rad(torch.as_tensor(ttl, device=tts.device, dtype=tts.dtype))
        cos_psi = torch.cos(psi_rad)
        cos_ttli = torch.cos(ttl_rad)
        sin_ttli = torch.sin(ttl_rad)
        cos_tts = torch.cos(tts_rad)
        sin_tts = torch.sin(tts_rad)
        cos_tto = torch.cos(tto_rad)
        sin_tto = torch.sin(tto_rad)
        Cs = cos_ttli * cos_tts
        Ss = sin_ttli * sin_tts
        Co = cos_ttli * cos_tto
        So = sin_ttli * sin_tto
        As = torch.maximum(Ss, Cs)
        Ao = torch.maximum(So, Co)
        eps = 1e-9
        cosbts = torch.where(torch.abs(Ss) > eps, -Cs / torch.clamp(Ss, min=eps), torch.full_like(Ss, 5.0))
        mask_bts = torch.abs(cosbts) < 1.0
        bts = torch.where(mask_bts, torch.acos(cosbts.clamp(-1 + 1e-9, 1 - 1e-9)), torch.full_like(Ss, torch.pi))
        ds = torch.where(mask_bts, Ss, Cs)
        cosbto = torch.where(torch.abs(So) > eps, -Co / torch.clamp(So, min=eps), torch.full_like(So, 5.0))
        mask_bto = torch.abs(cosbto) < 1.0
        bto = torch.zeros_like(So)
        do_term = torch.zeros_like(So)
        bto = torch.where(mask_bto, torch.acos(cosbto.clamp(-1 + 1e-9, 1 - 1e-9)), bto)
        do_term = torch.where(mask_bto, So, do_term)
        mask_high = (~mask_bto) & (tto < 90)
        bto = torch.where(mask_high, torch.full_like(So, torch.pi), bto)
        do_term = torch.where(mask_high, Co, do_term)
        mask_low = (~mask_bto) & (~mask_high)
        bto = torch.where(mask_low, torch.zeros_like(So), bto)
        do_term = torch.where(mask_low, -Co, do_term)
        chi_s = 2.0 / torch.pi * ((bts - torch.pi * 0.5) * Cs + torch.sin(bts) * Ss)
        chi_o = 2.0 / torch.pi * ((bto - torch.pi * 0.5) * Co + torch.sin(bto) * So)
        delta1 = torch.abs(bts - bto)
        delta2 = torch.pi - torch.abs(bts + bto - torch.pi)
        Tot = psi_rad + delta1 + delta2
        bt1 = torch.minimum(psi_rad, delta1)
        bt3 = torch.maximum(psi_rad, delta2)
        bt2 = Tot - bt1 - bt3
        T1 = 2.0 * Cs * Co + Ss * So * cos_psi
        T2 = torch.sin(bt2) * (2 * ds * do_term + Ss * So * torch.cos(bt1) * torch.cos(bt3))
        denom = 2.0 * (torch.pi**2)
        frho = torch.clamp(((torch.pi - bt2) * T1 + T2) / denom, min=0.0)
        ftau = torch.clamp((-(bt2) * T1 + T2) / denom, min=0.0)
        return chi_s, chi_o, frho, ftau

    def _jfunc1(self, k: torch.Tensor, l: torch.Tensor, lai: torch.Tensor) -> torch.Tensor:
        k_exp = k.unsqueeze(-1)
        t = lai.unsqueeze(-1)
        denom = k_exp - l
        delta = denom * t
        numerator = torch.exp(-l * t) - torch.exp(-k_exp * t)
        result = numerator / denom
        mask = torch.abs(denom) <= 1e-6
        avg = 0.5 * t * (torch.exp(-k_exp * t) + torch.exp(-l * t))
        correction = 1.0 - (delta**2) / 12.0
        result = torch.where(mask, avg * correction, result)
        return result

    def _jfunc2(self, k: torch.Tensor, l: torch.Tensor, lai: torch.Tensor) -> torch.Tensor:
        k_exp = k.unsqueeze(-1)
        denom = (k_exp + l).clamp(min=1e-9)
        return (1.0 - torch.exp(-denom * lai.unsqueeze(-1))) / denom

    def _jfunc2_scalar(self, k: torch.Tensor, l: torch.Tensor, lai: torch.Tensor) -> torch.Tensor:
        denom = (k + l).clamp(min=1e-9)
        return (1.0 - torch.exp(-denom * lai)) / denom

    def _hotspot_terms(
        self, hotspot: torch.Tensor, dso: torch.Tensor, ks: torch.Tensor, ko: torch.Tensor, lai: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tsstoo = torch.exp(-ks * lai)
        sumint = (1.0 - tsstoo) / (ks * lai)

        active = (hotspot > 0) & (dso != 0)
        if not active.any():
            return tsstoo, sumint

        alf = torch.zeros_like(hotspot)
        alf[active] = (dso[active] / hotspot[active]) * 2.0 / (ks[active] + ko[active])
        active = active & (alf != 0)
        if not active.any():
            return tsstoo, sumint

        lai_a = lai[active]
        ks_a = ks[active]
        ko_a = ko[active]
        alf_a = alf[active]

        fhot = lai_a * torch.sqrt(ko_a * ks_a)
        fint = (1.0 - torch.exp(-alf_a)) * 0.05

        x1 = torch.zeros_like(alf_a)
        y1 = torch.zeros_like(alf_a)
        f1 = torch.ones_like(alf_a)
        acc = torch.zeros_like(alf_a)

        for istep in range(1, 21):
            if istep < 20:
                x2 = -torch.log1p(-istep * fint) / alf_a
            else:
                x2 = torch.ones_like(alf_a)
            y2 = -(ko_a + ks_a) * lai_a * x2 + fhot * (1.0 - torch.exp(-alf_a * x2)) / alf_a
            f2 = torch.exp(y2)
            delta = y2 - y1
            valid = torch.abs(delta) > 1e-9
            acc = acc + torch.where(valid, (f2 - f1) * (x2 - x1) / delta, torch.zeros_like(acc))
            x1, y1, f1 = x2, y2, f2

        tsstoo[active] = f1
        sumint[active] = torch.nan_to_num(acc, nan=0.0)
        return tsstoo, sumint

    def _solve_four_stream(
        self,
        ks: torch.Tensor,
        ko: torch.Tensor,
        bf: torch.Tensor,
        lai: torch.Tensor,
        hotspot: torch.Tensor,
        m: torch.Tensor,
        att: torch.Tensor,
        sigb: torch.Tensor,
        sb: torch.Tensor,
        sf: torch.Tensor,
        vb: torch.Tensor,
        vf: torch.Tensor,
        w: torch.Tensor,
        rho: torch.Tensor,
        tau: torch.Tensor,
        soil: torch.Tensor,
        geometry: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> FourSAILResult:
        batch, nwl = rho.shape
        device = rho.device
        dtype = rho.dtype
        cts, cto, ctscto, tants, tanto, cospsi, dso = geometry

        e1 = torch.exp(-m * lai.unsqueeze(-1))
        e2 = e1**2
        rinf = (att - m) / sigb
        rinf2 = rinf**2
        re = rinf * e1
        denom = 1.0 - rinf2 * e2
        denom = denom.clamp(min=1e-9)
        J1ks = self._jfunc1(ks, m, lai)
        J2ks = self._jfunc2(ks, m, lai)
        J1ko = self._jfunc1(ko, m, lai)
        J2ko = self._jfunc2(ko, m, lai)
        Pss = (sf + sb * rinf) * J1ks
        Qss = (sf * rinf + sb) * J2ks
        Pv = (vf + vb * rinf) * J1ko
        Qv = (vf * rinf + vb) * J2ko
        tdd = (1.0 - rinf2) * e1 / denom
        rdd = rinf * (1.0 - e2) / denom
        tsd = (Pss - re * Qss) / denom
        rsd = (Qss - re * Pss) / denom
        tdo = (Pv - re * Qv) / denom
        rdo = (Qv - re * Pv) / denom
        gammasdf = (1.0 + rinf) * (J1ks - re * J2ks) / denom
        gammasdb = (1.0 + rinf) * (-re * J1ks + J2ks) / denom
        tss = torch.exp(-ks * lai)
        too = torch.exp(-ko * lai)
        z = self._jfunc2_scalar(ks, ko, lai)
        z_term = z.unsqueeze(-1)
        g1 = (z_term - (J1ks * too.unsqueeze(-1))) / (ko.unsqueeze(-1) + m)
        g2 = (z_term - (J1ko * tss.unsqueeze(-1))) / (ks.unsqueeze(-1) + m)
        Tv1 = (vf * rinf + vb) * g1
        Tv2 = (vf + vb * rinf) * g2
        T1 = Tv1 * (sf + sb * rinf)
        T2 = Tv2 * (sf * rinf + sb)
        T3 = (rdo * Qss + tdo * Pss) * rinf
        rsod = (T1 + T2 - T3) / (1.0 - rinf2)
        T4 = Tv1 * (1.0 + rinf)
        T5 = Tv2 * (1.0 + rinf)
        T6 = (rdo * J2ks + tdo * J1ks) * (1.0 + rinf) * rinf
        gammasod = (T4 + T5 - T6) / (1.0 - rinf2)
        alf = torch.zeros_like(hotspot)
        mask = hotspot > 0
        alf[mask] = (dso[mask] / hotspot[mask]) * 2.0 / ((ks + ko)[mask])
        tsstoo, sumint = self._hotspot_terms(hotspot, dso, ks, ko, lai)
        rsos = w * lai.unsqueeze(-1) * sumint.unsqueeze(-1)
        gammasos = ko * lai * sumint
        rso = rsos + rsod
        gammaso = gammasos.unsqueeze(-1) + gammasod
        dn = (1.0 - soil * rdd).clamp(min=1e-9)
        rddt = rdd + tdd * soil * tdd / dn
        rsdt = rsd + (tsd + tss.unsqueeze(-1)) * soil * tdd / dn
        rdot = rdo + tdd * soil * (tdo + too.unsqueeze(-1)) / dn
        rsodt = ((tss.unsqueeze(-1) + tsd) * tdo + (tsd + tss.unsqueeze(-1) * soil * rdd) * too.unsqueeze(-1)) * soil / dn
        rsost = rso + tsstoo.unsqueeze(-1) * soil
        rsot = rsost + rsodt
        return FourSAILResult(
            rdd=rdd,
            tdd=tdd,
            rsd=rsd,
            tsd=tsd,
            rdo=rdo,
            tdo=tdo,
            rso=rso,
            rsos=rsos,
            rsod=rsod,
            rddt=rddt,
            rsdt=rsdt,
            rdot=rdot,
            rsodt=rsodt,
            rsost=rsost,
            rsot=rsot,
            tss=tss,
            too=too,
            tsstoo=tsstoo,
            gammasdf=gammasdf,
            gammasdb=gammasdb,
            gammaso=gammaso,
        )
