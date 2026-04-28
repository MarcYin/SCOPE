from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from .loaders import FluspectResources, load_fluspect_resources
from .optics import fresnel_tav


@dataclass(slots=True)
class SoilEmpiricalParams:
    """Empirical constants used by SCOPE's BSM wet-soil transport."""

    SMC: torch.Tensor | float = 25.0
    film: torch.Tensor | float = 0.015


@dataclass(slots=True)
class BSMSoilParameters:
    """SCOPE-compatible inputs for the BSM soil reflectance model."""

    BSMBrightness: torch.Tensor | float
    BSMlat: torch.Tensor | float
    BSMlon: torch.Tensor | float
    SMC: torch.Tensor | float


class SoilBSMModel:
    """Tensor-native implementation of SCOPE's BSM soil model."""

    def __init__(
        self,
        GSV: torch.Tensor,
        Kw: torch.Tensor,
        nw: torch.Tensor,
        *,
        empirical: SoilEmpiricalParams | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        device_obj = torch.device(device) if device is not None else torch.as_tensor(Kw).device
        self.device = device_obj
        self.dtype = dtype
        self.GSV = torch.as_tensor(GSV, device=device_obj, dtype=dtype)
        self.Kw = torch.as_tensor(Kw, device=device_obj, dtype=dtype)
        self.nw = torch.as_tensor(nw, device=device_obj, dtype=dtype)
        self.empirical = self._coerce_empirical(empirical or SoilEmpiricalParams())

        if self.GSV.ndim != 2 or self.GSV.shape[1] != 3:
            raise ValueError(f"GSV must have shape (n_wavelength, 3), got {tuple(self.GSV.shape)}")
        if self.Kw.ndim != 1:
            raise ValueError(f"Kw must be 1D, got shape {tuple(self.Kw.shape)}")
        if self.nw.ndim != 1:
            raise ValueError(f"nw must be 1D, got shape {tuple(self.nw.shape)}")
        n_wavelength = self.GSV.shape[0]
        if self.Kw.shape[0] != n_wavelength or self.nw.shape[0] != n_wavelength:
            raise ValueError("GSV, Kw, and nw must share the same wavelength axis")

    @classmethod
    def from_resources(
        cls,
        resources: FluspectResources,
        *,
        empirical: SoilEmpiricalParams | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> SoilBSMModel:
        try:
            GSV = resources.extras["GSV"]
            nw = resources.extras["nw"]
        except KeyError as exc:
            raise ValueError(
                "FLUSPECT resources must include 'GSV' and 'nw' extras to build the BSM soil model"
            ) from exc

        return cls(
            GSV,
            resources.optipar.Kw,
            nw,
            empirical=empirical,
            device=device if device is not None else resources.optipar.Kw.device,
            dtype=dtype,
        )

    @classmethod
    def from_scope_assets(
        cls,
        path: str | None = None,
        *,
        scope_root_path: str | None = None,
        empirical: SoilEmpiricalParams | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> SoilBSMModel:
        resources = load_fluspect_resources(
            path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
        )
        return cls.from_resources(resources, empirical=empirical, device=device, dtype=dtype)

    def __call__(self, soil: BSMSoilParameters) -> torch.Tensor:
        _, params = self._prepare_soil(soil)
        smc = self._normalize_soil_moisture(params["SMC"])

        lat = params["BSMlat"] * (math.pi / 180.0)
        lon = params["BSMlon"] * (math.pi / 180.0)
        brightness = params["BSMBrightness"]

        coefficients = torch.stack(
            [
                brightness * torch.sin(lat),
                brightness * torch.cos(lat) * torch.sin(lon),
                brightness * torch.cos(lat) * torch.cos(lon),
            ],
            dim=-1,
        )
        rdry = torch.matmul(coefficients, self.GSV.transpose(0, 1))
        return self._soilwat(rdry, smc * 1e2)

    def _coerce_empirical(self, empirical: SoilEmpiricalParams) -> SoilEmpiricalParams:
        SMC = torch.as_tensor(empirical.SMC, device=self.device, dtype=self.dtype)
        film = torch.as_tensor(empirical.film, device=self.device, dtype=self.dtype)
        if SMC.ndim != 0 or film.ndim != 0:
            raise ValueError("Soil empirical parameters must be scalar values")
        if float(SMC) <= 0:
            raise ValueError("Soil empirical SMC capacity must be positive")
        if float(film) <= 0:
            raise ValueError("Soil empirical film thickness must be positive")
        return SoilEmpiricalParams(SMC=SMC, film=film)

    def _prepare_soil(self, soil: BSMSoilParameters) -> tuple[int, dict[str, torch.Tensor]]:
        tensors: dict[str, torch.Tensor] = {}
        batch_size = None
        for field in fields(BSMSoilParameters):
            tensor = torch.as_tensor(getattr(soil, field.name), device=self.device, dtype=self.dtype)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                if tensor.shape[0] == 1:
                    tensor = tensor.expand(batch_size)
                else:
                    raise ValueError("BSM soil parameters must broadcast to a common batch size")
            tensors[field.name] = tensor
        if batch_size is None:
            batch_size = 1
        return batch_size, tensors

    def _normalize_soil_moisture(self, smc: torch.Tensor) -> torch.Tensor:
        if smc.numel() and float(smc.mean()) > 1.0:
            return smc / 100.0
        return smc

    def _soilwat(self, rdry: torch.Tensor, SMp: torch.Tensor) -> torch.Tensor:
        mu = (SMp - 5.0) / self.empirical.SMC
        wet_mask = mu > 0
        if not wet_mask.any():
            return rdry

        two = torch.as_tensor(2.0, device=self.device, dtype=self.dtype)
        tav90_2 = fresnel_tav(90.0, two)
        tav90_2_over_nw = fresnel_tav(90.0, two / self.nw)
        p = 1 - fresnel_tav(90.0, self.nw) / (self.nw**2)
        Rw = 1 - fresnel_tav(40.0, self.nw)
        rbac = 1 - (1 - rdry) * (rdry * tav90_2_over_nw.unsqueeze(0) / tav90_2 + 1 - rdry)

        k = torch.arange(0.0, 7.0, device=self.device, dtype=self.dtype)
        mu_safe = mu.clamp(min=1e-12).unsqueeze(-1)
        poisson = torch.exp(k.unsqueeze(0) * torch.log(mu_safe) - mu_safe - torch.lgamma(k + 1.0).unsqueeze(0))
        tw = torch.exp((-2.0 * self.Kw.unsqueeze(-1) * self.empirical.film) * k)

        fraction = (
            (1 - Rw).unsqueeze(0).unsqueeze(-1)
            * (1 - p).unsqueeze(0).unsqueeze(-1)
            * tw.unsqueeze(0)
            * rbac.unsqueeze(-1)
        )
        fraction = fraction / (1 - p.unsqueeze(0).unsqueeze(-1) * tw.unsqueeze(0) * rbac.unsqueeze(-1))
        Rwet_k = Rw.unsqueeze(0).unsqueeze(-1) + fraction

        rwet = rdry * poisson[:, :1]
        rwet = rwet + torch.einsum("bnk,bk->bn", Rwet_k[:, :, 1:], poisson[:, 1:])
        return torch.where(wet_mask.unsqueeze(-1), rwet, rdry)
