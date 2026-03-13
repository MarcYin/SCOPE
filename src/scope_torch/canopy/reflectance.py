from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch

from .foursail import FourSAILModel, FourSAILResult
from .layered_rt import LayeredCanopyTransportModel
from ..spectral.fluspect import FluspectModel, LeafBioBatch, LeafOptics
from ..spectral.loaders import SoilSpectraLibrary, load_fluspect_resources, load_soil_spectra
from ..spectral.soil import BSMSoilParameters, SoilBSMModel, SoilEmpiricalParams


@dataclass(slots=True)
class CanopyReflectanceResult:
    leaf_refl: torch.Tensor
    leaf_tran: torch.Tensor
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

    @classmethod
    def from_components(cls, leafopt: LeafOptics, sail: FourSAILResult) -> "CanopyReflectanceResult":
        return cls(
            leaf_refl=leafopt.refl,
            leaf_tran=leafopt.tran,
            **{field.name: getattr(sail, field.name) for field in fields(FourSAILResult)},
        )


@dataclass(slots=True)
class CanopyRadiationProfileResult:
    Ps: torch.Tensor
    Po: torch.Tensor
    Pso: torch.Tensor
    Es_direct_: torch.Tensor
    Emin_direct_: torch.Tensor
    Eplu_direct_: torch.Tensor
    Es_diffuse_: torch.Tensor
    Emin_diffuse_: torch.Tensor
    Eplu_diffuse_: torch.Tensor
    Es_: torch.Tensor
    Emin_: torch.Tensor
    Eplu_: torch.Tensor


@dataclass(slots=True)
class CanopyDirectionalReflectanceResult:
    tto: torch.Tensor
    psi: torch.Tensor
    refl_: torch.Tensor
    rso_: torch.Tensor


class CanopyReflectanceModel:
    """Stable SCOPE-facing reflectance wrapper around leaf optics + 4SAIL."""

    def __init__(
        self,
        fluspect: FluspectModel,
        sail: FourSAILModel,
        *,
        lidf: torch.Tensor,
        default_hotspot: float = 0.2,
        soil_spectra: Optional[SoilSpectraLibrary] = None,
        soil_bsm: Optional[SoilBSMModel] = None,
        soil_index_base: int = 1,
    ) -> None:
        self.fluspect = fluspect
        self.sail = sail
        self.layered_transport = LayeredCanopyTransportModel(sail)
        self.lidf = lidf
        self.default_hotspot = default_hotspot
        self.soil_spectra = soil_spectra
        self.soil_bsm = soil_bsm
        self.soil_index_base = soil_index_base

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: Optional[FourSAILModel] = None,
        path: Optional[str] = None,
        soil_path: Optional[str] = None,
        scope_root_path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
        soil_index_base: int = 1,
        soil_empirical: SoilEmpiricalParams | None = None,
    ) -> "CanopyReflectanceModel":
        resources = load_fluspect_resources(
            path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
        )
        fluspect = FluspectModel(
            resources.spectral,
            resources.optipar,
            ndub=ndub,
            doublings_step=doublings_step,
            device=torch.device(device) if device is not None else resources.optipar.Kw.device,
            dtype=dtype,
        )
        soil_spectra = load_soil_spectra(
            soil_path,
            scope_root_path=scope_root_path,
            device=fluspect.device,
            dtype=fluspect.dtype,
        )
        soil_bsm = SoilBSMModel.from_resources(resources, empirical=soil_empirical, device=fluspect.device, dtype=fluspect.dtype)
        sail_model = sail if sail is not None else FourSAILModel(lidf=lidf)
        return cls(
            fluspect,
            sail_model,
            lidf=lidf,
            default_hotspot=default_hotspot,
            soil_spectra=soil_spectra,
            soil_bsm=soil_bsm,
            soil_index_base=soil_index_base,
        )

    def soil_reflectance(
        self,
        *,
        soil_refl: Optional[torch.Tensor] = None,
        soil_spectrum: Optional[torch.Tensor] = None,
        bsm: BSMSoilParameters | None = None,
        BSMBrightness: Optional[torch.Tensor] = None,
        BSMlat: Optional[torch.Tensor] = None,
        BSMlon: Optional[torch.Tensor] = None,
        SMC: Optional[torch.Tensor] = None,
        soil_index_base: Optional[int] = None,
    ) -> torch.Tensor:
        if soil_refl is not None:
            return torch.as_tensor(soil_refl, device=self.fluspect.device, dtype=self.fluspect.dtype)

        if soil_spectrum is not None:
            if self.soil_spectra is None:
                raise ValueError("soil_spectrum was provided but no soil spectra library is configured")
            return self.soil_spectra.batch(
                soil_spectrum,
                index_base=self.soil_index_base if soil_index_base is None else soil_index_base,
            )

        if bsm is None and all(value is not None for value in (BSMBrightness, BSMlat, BSMlon, SMC)):
            bsm = BSMSoilParameters(
                BSMBrightness=BSMBrightness,
                BSMlat=BSMlat,
                BSMlon=BSMlon,
                SMC=SMC,
            )
        if bsm is not None:
            if self.soil_bsm is None:
                raise ValueError("BSM soil inputs were provided but no BSM soil model is configured")
            return self.soil_bsm(bsm)

        raise KeyError("Provide either soil_refl, soil_spectrum, or BSM soil parameters")

    def __call__(
        self,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        *,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int | torch.Tensor] = None,
    ) -> CanopyReflectanceResult:
        leafopt = self.fluspect(leafbio)
        hotspot_value = hotspot if hotspot is not None else torch.full_like(torch.as_tensor(lai), self.default_hotspot)
        sail_out = self.sail(
            leafopt.refl,
            leafopt.tran,
            soil_refl,
            lai,
            hotspot_value,
            tts,
            tto,
            psi,
            lidf=self.lidf if lidf is None else lidf,
        )
        rtmo_core = self._rtmo_optical_reflectance(
            leafopt=leafopt,
            soil_refl=soil_refl,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            hotspot=hotspot_value,
            lidf=self.lidf if lidf is None else lidf,
            nlayers=nlayers,
        )
        sail_out.rdd = rtmo_core["rdd"]
        sail_out.rsd = rtmo_core["rsd"]
        sail_out.rdo = rtmo_core["rdo"]
        sail_out.rso = rtmo_core["rso"]
        sail_out.rsos = rtmo_core["rsos"]
        sail_out.rsod = rtmo_core["rsod"]
        # The layered RTMo terms already include the soil boundary condition, so
        # the 4SAIL "total" aliases should track those values directly rather
        # than retaining stale pre-override four-stream totals.
        sail_out.rddt = rtmo_core["rdd"]
        sail_out.rsdt = rtmo_core["rsd"]
        sail_out.rdot = rtmo_core["rdo"]
        sail_out.rsost = rtmo_core["rsos"]
        sail_out.rsodt = rtmo_core["rsod"]
        sail_out.rsot = rtmo_core["rso"]
        return CanopyReflectanceResult.from_components(leafopt, sail_out)

    def profiles(
        self,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_: torch.Tensor,
        Esky_: torch.Tensor,
        *,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
    ) -> CanopyRadiationProfileResult:
        leafopt = self.fluspect(leafbio)
        batch, nwl = leafopt.refl.shape
        soil = self.sail._ensure_2d(soil_refl, target_shape=leafopt.refl.shape)
        hotspot_value = hotspot if hotspot is not None else torch.full_like(torch.as_tensor(lai), self.default_hotspot)
        nl = self._resolve_uniform_nlayers(
            nlayers=nlayers,
            lai=lai,
            batch=batch,
            device=leafopt.refl.device,
            dtype=leafopt.refl.dtype,
        )
        transfer = self.layered_transport.build(
            leafopt.refl,
            leafopt.tran,
            soil,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.lidf if lidf is None else lidf,
            nlayers=nl,
        )
        Esun = self._prepare_spectrum(Esun_, batch, nwl, device=leafopt.refl.device, dtype=leafopt.refl.dtype)
        Esky = self._prepare_spectrum(Esky_, batch, nwl, device=leafopt.refl.device, dtype=leafopt.refl.dtype)
        direct = self.layered_transport.flux_profiles(transfer, Esun, torch.zeros_like(Esky))
        diffuse = self.layered_transport.flux_profiles(transfer, torch.zeros_like(Esun), Esky)
        return CanopyRadiationProfileResult(
            Ps=transfer.Ps,
            Po=transfer.Po,
            Pso=transfer.Pso,
            Es_direct_=direct.Es_,
            Emin_direct_=direct.Emin_,
            Eplu_direct_=direct.Eplu_,
            Es_diffuse_=diffuse.Es_,
            Emin_diffuse_=diffuse.Emin_,
            Eplu_diffuse_=diffuse.Eplu_,
            Es_=direct.Es_ + diffuse.Es_,
            Emin_=direct.Emin_ + diffuse.Emin_,
            Eplu_=direct.Eplu_ + diffuse.Eplu_,
        )

    def directional(
        self,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        directional_tto: torch.Tensor,
        directional_psi: torch.Tensor,
        Esun_: torch.Tensor,
        Esky_: torch.Tensor,
        *,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
    ) -> CanopyDirectionalReflectanceResult:
        device = self.fluspect.device
        dtype = self.fluspect.dtype
        tto_angles = torch.as_tensor(directional_tto, device=device, dtype=dtype).reshape(-1)
        psi_angles = torch.as_tensor(directional_psi, device=device, dtype=dtype).reshape(-1)
        if tto_angles.shape != psi_angles.shape:
            raise ValueError("directional_tto and directional_psi must have the same shape")

        soil = self.sail._ensure_2d(soil_refl)
        batch = soil.shape[0]
        lai_tensor = self.sail._expand_param(lai, batch, device, dtype)
        tts_tensor = self.sail._expand_param(tts, batch, device, dtype)
        hotspot_value = hotspot if hotspot is not None else torch.full_like(lai_tensor, self.default_hotspot)
        Esun = self._prepare_spectrum(Esun_, batch, self.fluspect.spectral.wlP.numel(), device=device, dtype=dtype)
        Esky = self._prepare_spectrum(Esky_, batch, self.fluspect.spectral.wlP.numel(), device=device, dtype=dtype)

        refl = []
        rso = []
        for idx in range(tto_angles.numel()):
            result = self(
                leafbio,
                soil_refl,
                lai_tensor,
                tts_tensor,
                tto_angles[idx].expand(batch),
                psi_angles[idx].expand(batch),
                hotspot=hotspot_value,
                lidf=lidf,
                nlayers=nlayers,
            )
            refl.append(self._directional_reflectance(result.rso, result.rdo, Esun, Esky).squeeze(1))
            rso.append(result.rso.squeeze(1))

        return CanopyDirectionalReflectanceResult(
            tto=tto_angles,
            psi=psi_angles,
            refl_=torch.stack(refl, dim=1),
            rso_=torch.stack(rso, dim=1),
        )

    def _rtmo_optical_reflectance(
        self,
        *,
        leafopt: LeafOptics,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        hotspot: torch.Tensor,
        lidf: torch.Tensor,
        nlayers: Optional[int | torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        soil = self.sail._ensure_2d(soil_refl, target_shape=leafopt.refl.shape)
        batch, nwl = leafopt.refl.shape
        outputs = {"rdd": [], "rsd": [], "rdo": [], "rso": [], "rsos": [], "rsod": []}
        batch_nlayers = self._resolve_nlayers(nlayers, lai, batch, device=leafopt.refl.device, dtype=leafopt.refl.dtype)
        for idx in range(batch):
            nlayers_i = int(batch_nlayers[idx].item())
            transfer = self.layered_transport.build(
                leafopt.refl[idx : idx + 1],
                leafopt.tran[idx : idx + 1],
                soil[idx : idx + 1],
                lai[idx : idx + 1],
                tts[idx : idx + 1],
                tto[idx : idx + 1],
                psi[idx : idx + 1],
                hotspot=hotspot[idx : idx + 1],
                lidf=lidf[idx : idx + 1] if lidf.ndim == 2 else lidf,
                nlayers=nlayers_i,
            )
            unit = torch.ones((1, nwl), device=leafopt.refl.device, dtype=leafopt.refl.dtype)
            zeros = torch.zeros_like(unit)
            direct = self.layered_transport.flux_profiles(transfer, unit, zeros)
            diffuse = self.layered_transport.flux_profiles(transfer, zeros, unit)

            Po = transfer.Po[:, : transfer.nlayers].unsqueeze(-1)
            Pso = transfer.Pso[:, : transfer.nlayers].unsqueeze(-1)
            iLAI = transfer.iLAI.view(1, 1, 1)

            piLocd = iLAI * torch.sum(
                (transfer.vb.unsqueeze(1) * diffuse.Emin_[:, : transfer.nlayers, :] + transfer.vf.unsqueeze(1) * diffuse.Eplu_[:, : transfer.nlayers, :]) * Po,
                dim=1,
            )
            piLosd = soil[idx : idx + 1] * diffuse.Emin_[:, -1, :] * transfer.Po[:, -1].unsqueeze(-1)
            piLocu_diffuse = iLAI * torch.sum(
                transfer.vb.unsqueeze(1) * direct.Emin_[:, : transfer.nlayers, :] * Po
                + transfer.vf.unsqueeze(1) * direct.Eplu_[:, : transfer.nlayers, :] * Po,
                dim=1,
            )
            piLocu_hotspot = iLAI * torch.sum(transfer.w.unsqueeze(1) * Pso, dim=1)
            piLosu_diffuse = soil[idx : idx + 1] * direct.Emin_[:, -1, :] * transfer.Po[:, -1].unsqueeze(-1)
            piLosu_hotspot = soil[idx : idx + 1] * transfer.Pso[:, -1].unsqueeze(-1)
            rsod = piLocu_diffuse + piLosu_diffuse
            rsos = piLocu_hotspot + piLosu_hotspot

            outputs["rdd"].append(transfer.R_dd[:, 0, :])
            outputs["rsd"].append(transfer.R_sd[:, 0, :])
            outputs["rdo"].append(piLocd + piLosd)
            outputs["rso"].append(rsod + rsos)
            outputs["rsos"].append(rsos)
            outputs["rsod"].append(rsod)

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def _prepare_spectrum(
        self,
        value: torch.Tensor,
        batch: int,
        n_wavelength: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Spectra must be 1D or 2D, got shape {tuple(tensor.shape)}")
        if tensor.shape[-1] != n_wavelength:
            raise ValueError(f"Spectra must have length {n_wavelength}, got {tensor.shape[-1]}")
        if tensor.shape[0] == 1 and batch != 1:
            tensor = tensor.expand(batch, -1)
        elif tensor.shape[0] != batch:
            raise ValueError("Spectra must broadcast to the batch dimension")
        return tensor

    def _directional_reflectance(
        self,
        rso: torch.Tensor,
        rdo: torch.Tensor,
        Esun: torch.Tensor,
        Esky: torch.Tensor,
    ) -> torch.Tensor:
        irradiance = Esun + Esky
        refl = (rso * Esun + rdo * Esky) / irradiance.clamp(min=1e-12)
        threshold = 2e-4 * torch.max(Esky, dim=-1, keepdim=True).values
        output = torch.where(Esky < threshold, rso, refl)
        if output.ndim == 3 and output.shape[1] == 1:
            return output.squeeze(1)
        return output

    def _resolve_uniform_nlayers(
        self,
        *,
        nlayers: Optional[int],
        lai: torch.Tensor,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> int:
        resolved = self._resolve_nlayers(nlayers, lai, batch, device=device, dtype=dtype)
        unique = torch.unique(resolved)
        if unique.numel() != 1:
            raise ValueError("Profile outputs require a uniform nlayers value across the batch")
        return int(unique.item())

    def _resolve_nlayers(
        self,
        nlayers: Optional[int | torch.Tensor],
        lai: torch.Tensor,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if nlayers is None:
            lai_tensor = torch.as_tensor(lai, device=device, dtype=dtype).reshape(-1)
            if lai_tensor.shape[0] == 1 and batch != 1:
                lai_tensor = lai_tensor.expand(batch)
            return torch.clamp(torch.ceil(lai_tensor * 10.0), min=2.0).to(torch.int64)

        if isinstance(nlayers, int):
            return torch.full((batch,), max(2, nlayers), device=device, dtype=torch.int64)

        nlayers_tensor = torch.as_tensor(nlayers, device=device)
        if nlayers_tensor.ndim == 0:
            return torch.full((batch,), max(2, int(nlayers_tensor.item())), device=device, dtype=torch.int64)
        nlayers_tensor = nlayers_tensor.reshape(-1)
        if nlayers_tensor.shape[0] == 1 and batch != 1:
            nlayers_tensor = nlayers_tensor.expand(batch)
        if nlayers_tensor.shape[0] != batch:
            raise ValueError("nlayers must be scalar or match the batch dimension")
        return torch.clamp(torch.round(nlayers_tensor), min=2).to(torch.int64)
