from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from .foursail import FourSAILModel, scope_lazitab


@dataclass(slots=True)
class LayerFluxProfiles:
    Es_: torch.Tensor
    Emin_: torch.Tensor
    Eplu_: torch.Tensor


@dataclass(slots=True)
class LayeredCanopyTransfer:
    nlayers: int
    litab: torch.Tensor
    lazitab: torch.Tensor
    lidf: torch.Tensor
    xl: torch.Tensor
    dx: torch.Tensor
    iLAI: torch.Tensor
    ks: torch.Tensor
    ko: torch.Tensor
    dso: torch.Tensor
    Ps: torch.Tensor
    Po: torch.Tensor
    Pso: torch.Tensor
    rho_dd: torch.Tensor
    tau_dd: torch.Tensor
    R_sd: torch.Tensor
    R_dd: torch.Tensor
    Xss: torch.Tensor
    Xsd: torch.Tensor
    Xdd: torch.Tensor
    vb: torch.Tensor
    vf: torch.Tensor
    w: torch.Tensor
    fs: torch.Tensor
    absfs: torch.Tensor
    absfo: torch.Tensor
    fsfo: torch.Tensor
    foctl: torch.Tensor
    fsctl: torch.Tensor
    ctl2: torch.Tensor
    lidf_azimuth: torch.Tensor


class LayeredCanopyTransportModel:
    """Reusable layered transport helper derived from SCOPE's RTMo recursions."""

    def __init__(self, sail: FourSAILModel, *, lazitab: Optional[torch.Tensor] = None) -> None:
        self.sail = sail
        self.default_lazitab = lazitab

    def build(
        self,
        rho: torch.Tensor,
        tau: torch.Tensor,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        *,
        hotspot: torch.Tensor,
        lidf: Optional[torch.Tensor] = None,
        nlayers: int = 60,
    ) -> LayeredCanopyTransfer:
        rho = self.sail._ensure_2d(rho)
        tau = self.sail._ensure_2d(tau, target_shape=rho.shape)
        soil = self.sail._ensure_2d(soil_refl, target_shape=rho.shape)

        batch, nwl = rho.shape
        device = rho.device
        dtype = rho.dtype

        lai = self.sail._expand_param(lai, batch, device, dtype)
        tts = self.sail._expand_param(tts, batch, device, dtype)
        tto = self.sail._expand_param(tto, batch, device, dtype)
        psi = self.sail._expand_param(psi, batch, device, dtype)
        hotspot = self.sail._expand_param(hotspot, batch, device, dtype)

        lidf_tensor = lidf if lidf is not None else self.sail.default_lidf
        if lidf_tensor is None:
            raise ValueError("A leaf inclination distribution (lidf) must be provided")
        lidf_tensor = torch.as_tensor(lidf_tensor, device=device, dtype=dtype)
        if lidf_tensor.ndim == 1:
            lidf_tensor = lidf_tensor.unsqueeze(0).expand(batch, -1)
        elif lidf_tensor.shape[0] != batch:
            raise ValueError("lidf must broadcast to the batch dimension")
        lidf_tensor = lidf_tensor / lidf_tensor.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        litab = self.sail._litab.to(device=device, dtype=dtype)
        lazitab = self._lazitab(device=device, dtype=dtype)

        geometry = self.sail._geometry_constants(tts, tto, psi)
        _, _, _, _, _, _, dso = geometry
        ks, ko, bf, sob, sof = self.sail._weighted_sum_over_lidf(tts, tto, psi, lidf_tensor, litab)

        sdb = 0.5 * (ks + bf)
        sdf = 0.5 * (ks - bf)
        dob = 0.5 * (ko + bf)
        dof = 0.5 * (ko - bf)
        ddb = 0.5 * (1.0 + bf)
        ddf = 0.5 * (1.0 - bf)

        sigb = ddb.unsqueeze(-1) * rho + ddf.unsqueeze(-1) * tau
        sigf = ddf.unsqueeze(-1) * rho + ddb.unsqueeze(-1) * tau
        sb = sdb.unsqueeze(-1) * rho + sdf.unsqueeze(-1) * tau
        sf = sdf.unsqueeze(-1) * rho + sdb.unsqueeze(-1) * tau
        vb = dob.unsqueeze(-1) * rho + dof.unsqueeze(-1) * tau
        vf = dof.unsqueeze(-1) * rho + dob.unsqueeze(-1) * tau
        w = sob.unsqueeze(-1) * rho + sof.unsqueeze(-1) * tau
        a = 1.0 - sigf

        dx = torch.as_tensor(1.0 / nlayers, device=device, dtype=dtype)
        iLAI = lai * dx

        tau_ss = (1.0 - ks * iLAI).unsqueeze(-1).expand(-1, nlayers)
        tau_dd = (1.0 - a * iLAI.unsqueeze(-1)).unsqueeze(1).expand(-1, nlayers, -1)
        tau_sd = (sf * iLAI.unsqueeze(-1)).unsqueeze(1).expand(-1, nlayers, -1)
        rho_sd = (sb * iLAI.unsqueeze(-1)).unsqueeze(1).expand(-1, nlayers, -1)
        rho_dd = (sigb * iLAI.unsqueeze(-1)).unsqueeze(1).expand(-1, nlayers, -1)

        R_sd = torch.zeros((batch, nlayers + 1, nwl), device=device, dtype=dtype)
        R_dd = torch.zeros_like(R_sd)
        Xsd = torch.zeros((batch, nlayers, nwl), device=device, dtype=dtype)
        Xdd = torch.zeros_like(Xsd)
        R_sd[:, -1, :] = soil
        R_dd[:, -1, :] = soil
        for layer in range(nlayers - 1, -1, -1):
            dnorm = (1.0 - rho_dd[:, layer, :] * R_dd[:, layer + 1, :]).clamp(min=1e-9)
            Xsd[:, layer, :] = (tau_sd[:, layer, :] + tau_ss[:, layer].unsqueeze(-1) * R_sd[:, layer + 1, :] * rho_dd[:, layer, :]) / dnorm
            Xdd[:, layer, :] = tau_dd[:, layer, :] / dnorm
            R_sd[:, layer, :] = rho_sd[:, layer, :] + tau_dd[:, layer, :] * (
                R_sd[:, layer + 1, :] * tau_ss[:, layer].unsqueeze(-1) + R_dd[:, layer + 1, :] * Xsd[:, layer, :]
            )
            R_dd[:, layer, :] = rho_dd[:, layer, :] + tau_dd[:, layer, :] * R_dd[:, layer + 1, :] * Xdd[:, layer, :]

        xl = torch.cat([torch.zeros(1, device=device, dtype=dtype), -torch.arange(1, nlayers + 1, device=device, dtype=dtype) * dx], dim=0)
        Ps = torch.exp(ks.unsqueeze(-1) * xl.unsqueeze(0) * lai.unsqueeze(-1))
        Po = torch.exp(ko.unsqueeze(-1) * xl.unsqueeze(0) * lai.unsqueeze(-1))
        Ps[:, :nlayers] = Ps[:, :nlayers] * self._finite_layer_average(ks, lai, dx).unsqueeze(-1)
        Po[:, :nlayers] = Po[:, :nlayers] * self._finite_layer_average(ko, lai, dx).unsqueeze(-1)

        Pso = self._bidirectional_gap_profile(ko, ks, lai, hotspot, dso, xl, dx)
        Pso = torch.minimum(Pso, Po)
        Pso = torch.minimum(Pso, Ps)

        fs, absfs, absfo, fsfo, foctl, fsctl, ctl2, lidf_azimuth = self._orientation_factors(
            tts=tts,
            tto=tto,
            psi=psi,
            litab=litab,
            lazitab=lazitab,
            lidf=lidf_tensor,
        )

        return LayeredCanopyTransfer(
            nlayers=nlayers,
            litab=litab,
            lazitab=lazitab,
            lidf=lidf_tensor,
            xl=xl,
            dx=dx,
            iLAI=iLAI,
            ks=ks,
            ko=ko,
            dso=dso,
            Ps=Ps,
            Po=Po,
            Pso=Pso,
            rho_dd=rho_dd,
            tau_dd=tau_dd,
            R_sd=R_sd,
            R_dd=R_dd,
            Xss=tau_ss,
            Xsd=Xsd,
            Xdd=Xdd,
            vb=vb,
            vf=vf,
            w=w,
            fs=fs,
            absfs=absfs,
            absfo=absfo,
            fsfo=fsfo,
            foctl=foctl,
            fsctl=fsctl,
            ctl2=ctl2,
            lidf_azimuth=lidf_azimuth,
        )

    def flux_profiles(self, transfer: LayeredCanopyTransfer, Esun_: torch.Tensor, Esky_: torch.Tensor) -> LayerFluxProfiles:
        Esun = self.sail._ensure_2d(Esun_)
        Esky = self.sail._ensure_2d(Esky_, target_shape=Esun.shape)
        batch, nwl = Esun.shape

        Es = torch.zeros((batch, transfer.nlayers + 1, nwl), device=Esun.device, dtype=Esun.dtype)
        Emin = torch.zeros_like(Es)
        Eplu = torch.zeros_like(Es)
        Es[:, 0, :] = Esun
        Emin[:, 0, :] = Esky

        for layer in range(transfer.nlayers):
            Es[:, layer + 1, :] = transfer.Xss[:, layer].unsqueeze(-1) * Es[:, layer, :]
            Emin[:, layer + 1, :] = transfer.Xsd[:, layer, :] * Es[:, layer, :] + transfer.Xdd[:, layer, :] * Emin[:, layer, :]
            Eplu[:, layer, :] = transfer.R_sd[:, layer, :] * Es[:, layer, :] + transfer.R_dd[:, layer, :] * Emin[:, layer, :]
        Eplu[:, -1, :] = transfer.R_dd[:, -1, :] * (Es[:, -1, :] + Emin[:, -1, :])
        return LayerFluxProfiles(Es_=Es, Emin_=Emin, Eplu_=Eplu)

    def _finite_layer_average(self, extinction: torch.Tensor, lai: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        term = extinction * lai * dx
        result = torch.ones_like(term)
        mask = torch.abs(term) > 1e-9
        result[mask] = (1.0 - torch.exp(-term[mask])) / term[mask]
        return result

    def _bidirectional_gap_profile(
        self,
        ko: torch.Tensor,
        ks: torch.Tensor,
        lai: torch.Tensor,
        hotspot: torch.Tensor,
        dso: torch.Tensor,
        xl: torch.Tensor,
        dx: torch.Tensor,
        quadrature_order: int = 64,
    ) -> torch.Tensor:
        device = ko.device
        dtype = ko.dtype
        lower = (xl - dx).view(1, -1, 1)
        upper = xl.view(1, -1, 1)

        ko_b = ko.unsqueeze(-1).unsqueeze(-1)
        ks_b = ks.unsqueeze(-1).unsqueeze(-1)
        lai_b = lai.unsqueeze(-1).unsqueeze(-1)
        hotspot_b = hotspot.unsqueeze(-1).unsqueeze(-1)
        dso_b = dso.unsqueeze(-1).unsqueeze(-1)

        same_path = (torch.abs(dso_b) <= 1e-12) | (hotspot_b <= 1e-12)
        alf = torch.zeros_like(dso_b)
        active = ~same_path
        alf[active] = (dso_b[active] / hotspot_b[active]) * 2.0 / (ks_b[active] + ko_b[active]).clamp(min=1e-9)

        pso = torch.empty((ko.shape[0], xl.numel()), device=device, dtype=dtype)
        sqrt_kk = torch.sqrt((ko_b * ks_b).clamp(min=0.0))
        if same_path.any():
            rate = ((ko_b + ks_b) - sqrt_kk) * lai_b
            pso_same = self._exp_interval_average(rate, lower, upper).squeeze(-1)
            pso = torch.where(same_path.expand(-1, xl.numel(), 1).squeeze(-1), pso_same, pso)
        if active.any():
            nodes, weights = self._gauss_legendre(order=quadrature_order, device=device, dtype=dtype)
            midpoint = 0.5 * (upper + lower)
            half_width = 0.5 * (upper - lower)
            y = midpoint + half_width * nodes.view(1, 1, -1)
            exp_term = torch.exp(y * alf)
            exponent = (ko_b + ks_b) * lai_b * y + sqrt_kk * lai_b / alf * (1.0 - exp_term)
            pso_active = 0.5 * torch.sum(weights.view(1, 1, -1) * torch.exp(exponent), dim=-1)
            pso = torch.where(active.expand(-1, xl.numel(), 1).squeeze(-1), pso_active, pso)
        return pso

    def _orientation_factors(
        self,
        *,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        litab: torch.Tensor,
        lazitab: torch.Tensor,
        lidf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        device = tts.device
        dtype = tts.dtype
        batch = tts.shape[0]
        ninc = litab.numel()
        nazi = lazitab.numel()

        cos_tts = torch.cos(torch.deg2rad(tts)).view(batch, 1, 1)
        sin_tts = torch.sin(torch.deg2rad(tts)).view(batch, 1, 1)
        cos_tto = torch.cos(torch.deg2rad(tto)).view(batch, 1, 1)
        sin_tto = torch.sin(torch.deg2rad(tto)).view(batch, 1, 1)

        cos_ttli = torch.cos(torch.deg2rad(litab)).view(1, ninc, 1)
        sin_ttli = torch.sin(torch.deg2rad(litab)).view(1, ninc, 1)
        cos_phils = torch.cos(torch.deg2rad(lazitab)).view(1, 1, nazi)
        cos_philo = torch.cos(torch.deg2rad(lazitab.view(1, 1, nazi) - psi.view(batch, 1, 1)))

        cds = cos_ttli * cos_tts + sin_ttli * sin_tts * cos_phils
        cdo = cos_ttli * cos_tto + sin_ttli * sin_tto * cos_philo
        fs = cds / cos_tts.clamp(min=1e-9)
        fo = cdo / cos_tto.clamp(min=1e-9)
        fs_flat = fs.reshape(batch, ninc * nazi)
        absfs = fs.abs().reshape(batch, ninc * nazi)
        absfo = fo.abs().reshape(batch, ninc * nazi)
        fsfo = (fs * fo).reshape(batch, ninc * nazi)
        foctl = (fo * cos_ttli).reshape(batch, ninc * nazi)
        fsctl = (fs * cos_ttli).reshape(batch, ninc * nazi)
        ctl2 = (cos_ttli**2).expand(batch, -1, nazi).reshape(batch, ninc * nazi)
        lidf_azimuth = (lidf / float(nazi)).unsqueeze(-1).expand(-1, -1, nazi).reshape(batch, ninc * nazi)
        return fs_flat, absfs, absfo, fsfo, foctl, fsctl, ctl2, lidf_azimuth

    def _lazitab(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.default_lazitab is not None:
            return torch.as_tensor(self.default_lazitab, device=device, dtype=dtype)
        return scope_lazitab(device=device, dtype=dtype)

    def _exp_interval_average(self, rate: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        span = upper - lower
        scaled = rate * span
        midpoint = 0.5 * (lower + upper)
        denom = torch.where(torch.abs(scaled) <= 1e-12, torch.ones_like(scaled), scaled)
        exact = torch.exp(rate * lower) * torch.expm1(scaled) / denom
        limit = torch.exp(rate * midpoint)
        return torch.where(torch.abs(scaled) <= 1e-9, limit, exact)

    @staticmethod
    @lru_cache(maxsize=None)
    def _gauss_legendre_numpy(order: int) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        return nodes.astype(np.float64), weights.astype(np.float64)

    def _gauss_legendre(self, *, order: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        nodes_np, weights_np = self._gauss_legendre_numpy(order)
        nodes = torch.as_tensor(nodes_np, device=device, dtype=dtype)
        weights = torch.as_tensor(weights_np, device=device, dtype=dtype)
        return nodes, weights
