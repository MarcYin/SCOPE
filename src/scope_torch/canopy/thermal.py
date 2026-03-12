from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .foursail import FourSAILModel
from .layered_rt import LayeredCanopyTransportModel
from .reflectance import CanopyReflectanceModel
from ..spectral.soil import SoilEmpiricalParams


@dataclass(slots=True)
class ThermalOptics:
    rho_thermal: torch.Tensor | float = 0.01
    tau_thermal: torch.Tensor | float = 0.01
    rs_thermal: torch.Tensor | float = 0.06


@dataclass(slots=True)
class CanopyThermalRadianceResult:
    Lot_: torch.Tensor
    Eoutte_: torch.Tensor
    Emint_: torch.Tensor
    Eplut_: torch.Tensor
    LotBB_: torch.Tensor
    Loutt: torch.Tensor
    Eoutt: torch.Tensor


@dataclass(slots=True)
class CanopyThermalBalanceResult:
    Lote: torch.Tensor
    Eoutte: torch.Tensor
    Emint: torch.Tensor
    Eplut: torch.Tensor
    Rnuct: torch.Tensor
    Rnhct: torch.Tensor
    Rnust: torch.Tensor
    Rnhst: torch.Tensor
    canopyemis: torch.Tensor


def default_thermal_wavelengths(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    reg2 = torch.arange(2500.0, 15001.0, 100.0, device=device, dtype=dtype)
    reg3 = torch.arange(16000.0, 50001.0, 1000.0, device=device, dtype=dtype)
    return torch.cat([reg2, reg3], dim=0)


class CanopyThermalRadianceModel:
    """Layered canopy thermal radiance using SCOPE-style thermal defaults."""

    def __init__(self, reflectance_model: CanopyReflectanceModel) -> None:
        self.reflectance_model = reflectance_model
        self.layered_transport = LayeredCanopyTransportModel(reflectance_model.sail)

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
    ) -> "CanopyThermalRadianceModel":
        reflectance = CanopyReflectanceModel.from_scope_assets(
            lidf=lidf,
            sail=sail,
            path=path,
            soil_path=soil_path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
            ndub=ndub,
            doublings_step=doublings_step,
            default_hotspot=default_hotspot,
            soil_index_base=soil_index_base,
            soil_empirical=soil_empirical,
        )
        return cls(reflectance)

    def __call__(
        self,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Tcu: torch.Tensor,
        Tch: torch.Tensor,
        Tsu: torch.Tensor,
        Tsh: torch.Tensor,
        *,
        thermal_optics: ThermalOptics | None = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
        wlT: Optional[torch.Tensor] = None,
    ) -> CanopyThermalRadianceResult:
        thermal = thermal_optics or ThermalOptics()
        device = self.reflectance_model.fluspect.device
        dtype = self.reflectance_model.fluspect.dtype
        wlT = default_thermal_wavelengths(device=device, dtype=dtype) if wlT is None else torch.as_tensor(wlT, device=device, dtype=dtype)

        lai_tensor = torch.as_tensor(lai, device=device, dtype=dtype)
        if lai_tensor.ndim == 0:
            lai_tensor = lai_tensor.unsqueeze(0)
        batch = lai_tensor.shape[0]
        hotspot_value = hotspot if hotspot is not None else torch.full_like(lai_tensor, self.reflectance_model.default_hotspot)

        nl = self._resolve_nlayers(nlayers, Tcu=Tcu, Tch=Tch)
        rho = self._broadcast_scalar_spectrum(thermal.rho_thermal, batch, wlT)
        tau = self._broadcast_scalar_spectrum(thermal.tau_thermal, batch, wlT)
        soil = self._broadcast_scalar_spectrum(thermal.rs_thermal, batch, wlT)
        transfer = self.layered_transport.build(
            rho,
            tau,
            soil,
            lai_tensor,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nl,
        )

        Tcu_tensor = self._prepare_layer_temperature(Tcu, batch, nl, transfer, device=device, dtype=dtype)
        Tch_tensor = self._prepare_layer_temperature(Tch, batch, nl, transfer, device=device, dtype=dtype)
        Tsu_tensor = self._expand_batch(Tsu, batch, device=device, dtype=dtype)
        Tsh_tensor = self._expand_batch(Tsh, batch, device=device, dtype=dtype)

        epsc = 1.0 - rho - tau
        epss = 1.0 - soil
        Hcsu = self._canopy_emission(Tcu_tensor, epsc, transfer, wlT)
        Hcsh = self._canopy_emission(Tch_tensor, epsc, transfer, wlT)
        Hssu = torch.pi * self._planck(wlT, Tsu_tensor.unsqueeze(-1) + 273.15, epss)
        Hssh = torch.pi * self._planck(wlT, Tsh_tensor.unsqueeze(-1) + 273.15, epss)

        Hc = Hcsu * transfer.Ps[:, :nl].unsqueeze(-1) + Hcsh * (1.0 - transfer.Ps[:, :nl]).unsqueeze(-1)
        Hs = Hssu * transfer.Ps[:, -1].unsqueeze(-1) + Hssh * (1.0 - transfer.Ps[:, -1]).unsqueeze(-1)

        Emin = torch.zeros((batch, nl + 1, wlT.numel()), device=device, dtype=dtype)
        Eplu = torch.zeros_like(Emin)
        U = torch.zeros_like(Emin)
        Y = torch.zeros((batch, nl, wlT.numel()), device=device, dtype=dtype)
        U[:, -1, :] = Hs
        for layer in range(nl - 1, -1, -1):
            denom = (1.0 - transfer.rho_dd[:, layer, :] * transfer.R_dd[:, layer + 1, :]).clamp(min=1e-9)
            source = Hc[:, layer, :] * transfer.iLAI.unsqueeze(-1)
            Y[:, layer, :] = (transfer.rho_dd[:, layer, :] * U[:, layer + 1, :] + source) / denom
            U[:, layer, :] = transfer.tau_dd[:, layer, :] * (transfer.R_dd[:, layer + 1, :] * Y[:, layer, :] + U[:, layer + 1, :]) + source
        for layer in range(nl):
            Emin[:, layer + 1, :] = transfer.Xdd[:, layer, :] * Emin[:, layer, :] + Y[:, layer, :]
            Eplu[:, layer, :] = transfer.R_dd[:, layer, :] * Emin[:, layer, :] + U[:, layer, :]
        Eplu[:, -1, :] = transfer.R_dd[:, -1, :] * Emin[:, -1, :] + Hs

        Po = transfer.Po[:, :nl].unsqueeze(-1)
        Pso = transfer.Pso[:, :nl].unsqueeze(-1)
        piLov = transfer.iLAI.unsqueeze(-1) * (
            transfer.ko.unsqueeze(-1).unsqueeze(-1) * Hcsh * (Po - Pso)
            + transfer.ko.unsqueeze(-1).unsqueeze(-1) * Hcsu * Pso
            + (transfer.vb.unsqueeze(1) * Emin[:, :nl, :] + transfer.vf.unsqueeze(1) * Eplu[:, :nl, :]) * Po
        ).sum(dim=1)
        piLos = Hssh * (transfer.Po[:, -1].unsqueeze(-1) - transfer.Pso[:, -1].unsqueeze(-1)) + Hssu * transfer.Pso[:, -1].unsqueeze(-1)
        piLot = piLov + piLos
        Lot = piLot / torch.pi
        Eoutte = Eplu[:, 0, :]
        Loutt = 0.001 * torch.trapz(Lot, wlT, dim=-1)
        Eoutt = 0.001 * torch.trapz(Eoutte, wlT, dim=-1)
        sigma_sb = torch.as_tensor(5.67e-8, device=device, dtype=dtype)
        Tbr = torch.clamp(torch.pi * Loutt / sigma_sb, min=0.0) ** 0.25
        LotBB = self._planck(wlT, Tbr.unsqueeze(-1), torch.ones((batch, wlT.numel()), device=device, dtype=dtype))

        return CanopyThermalRadianceResult(
            Lot_=Lot,
            Eoutte_=Eoutte,
            Emint_=Emin,
            Eplut_=Eplu,
            LotBB_=LotBB,
            Loutt=Loutt,
            Eoutt=Eoutt,
        )

    def integrated_balance(
        self,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Tcu: torch.Tensor,
        Tch: torch.Tensor,
        Tsu: torch.Tensor,
        Tsh: torch.Tensor,
        *,
        thermal_optics: ThermalOptics | None = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
    ) -> CanopyThermalBalanceResult:
        thermal = thermal_optics or ThermalOptics()
        device = self.reflectance_model.fluspect.device
        dtype = self.reflectance_model.fluspect.dtype

        lai_tensor = torch.as_tensor(lai, device=device, dtype=dtype)
        if lai_tensor.ndim == 0:
            lai_tensor = lai_tensor.unsqueeze(0)
        batch = lai_tensor.shape[0]
        hotspot_value = hotspot if hotspot is not None else torch.full_like(lai_tensor, self.reflectance_model.default_hotspot)

        nl = self._resolve_nlayers(nlayers, Tcu=Tcu, Tch=Tch)
        rho = self._broadcast_scalar_value(thermal.rho_thermal, batch, device=device, dtype=dtype)
        tau = self._broadcast_scalar_value(thermal.tau_thermal, batch, device=device, dtype=dtype)
        soil = self._broadcast_scalar_value(thermal.rs_thermal, batch, device=device, dtype=dtype)
        transfer = self.layered_transport.build(
            rho.unsqueeze(-1),
            tau.unsqueeze(-1),
            soil.unsqueeze(-1),
            lai_tensor,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nl,
        )

        Tcu_tensor = self._prepare_layer_temperature(Tcu, batch, nl, transfer, device=device, dtype=dtype)
        Tch_tensor = self._prepare_layer_temperature(Tch, batch, nl, transfer, device=device, dtype=dtype)
        Tsu_tensor = self._expand_batch(Tsu, batch, device=device, dtype=dtype)
        Tsh_tensor = self._expand_batch(Tsh, batch, device=device, dtype=dtype)

        epsc = 1.0 - rho - tau
        epss = 1.0 - soil
        result = self._integrated_balance_core(
            transfer=transfer,
            epsc=epsc,
            epss=epss,
            Tcu=Tcu_tensor,
            Tch=Tch_tensor,
            Tsu=Tsu_tensor,
            Tsh=Tsh_tensor,
        )

        # MATLAB ebal.m reuses the same RTMo thermal transport coefficients for
        # the blackbody normalization and only swaps the leaf/soil emissivities
        # to one. Rebuilding a separate zero-reflectance transfer changes the
        # scattering path and biases canopy emissivity.
        black = self._integrated_balance_core(
            transfer=transfer,
            epsc=torch.ones_like(epsc),
            epss=torch.ones_like(epss),
            Tcu=Tcu_tensor,
            Tch=Tch_tensor,
            Tsu=Tsu_tensor,
            Tsh=Tsh_tensor,
        )
        canopyemis = result["Eoutte"] / black["Eoutte"].clamp(min=1e-12)

        return CanopyThermalBalanceResult(
            Lote=result["Lote"],
            Eoutte=result["Eoutte"],
            Emint=result["Emint"],
            Eplut=result["Eplut"],
            Rnuct=result["Rnuct"],
            Rnhct=result["Rnhct"],
            Rnust=result["Rnust"],
            Rnhst=result["Rnhst"],
            canopyemis=canopyemis,
        )

    def _planck(self, wavelength_nm: torch.Tensor, temperature_k: torch.Tensor, emissivity: torch.Tensor) -> torch.Tensor:
        c1 = torch.as_tensor(1.191066e-22, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        c2 = torch.as_tensor(14388.33, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        wavelength_term = (wavelength_nm * 1e-9) ** -5
        exponent = c2 / ((wavelength_nm * 1e-3) * temperature_k)
        return emissivity * c1 * wavelength_term / torch.expm1(exponent)

    def _stefan_boltzmann(self, temperature_c: torch.Tensor) -> torch.Tensor:
        sigma_sb = torch.as_tensor(5.67e-8, device=temperature_c.device, dtype=temperature_c.dtype)
        kelvin = torch.clamp(temperature_c + 273.15, min=0.0)
        return sigma_sb * kelvin**4

    def _broadcast_scalar_spectrum(self, value: torch.Tensor | float, batch: int, wavelength: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=wavelength.device, dtype=wavelength.dtype)
        if tensor.ndim == 0:
            tensor = tensor.view(1, 1).expand(batch, wavelength.numel())
        elif tensor.ndim == 1:
            if tensor.shape[0] == batch:
                tensor = tensor.view(batch, 1).expand(batch, wavelength.numel())
            elif tensor.shape[0] == wavelength.numel():
                tensor = tensor.view(1, -1).expand(batch, wavelength.numel())
            else:
                raise ValueError("Thermal optics must broadcast to batch or wavelength dimensions")
        elif tensor.ndim == 2:
            if tensor.shape[0] == 1 and tensor.shape[1] == wavelength.numel():
                tensor = tensor.expand(batch, wavelength.numel())
            elif tensor.shape[0] != batch or tensor.shape[1] != wavelength.numel():
                raise ValueError("Thermal optics must match (batch, n_wavelength)")
        else:
            raise ValueError("Thermal optics must be scalar, 1D, or 2D")
        return tensor

    def _broadcast_scalar_value(self, value: torch.Tensor | float, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.repeat(batch)
        if tensor.ndim == 1 and tensor.shape[0] == batch:
            return tensor
        if tensor.ndim == 1 and tensor.shape[0] == 1 and batch != 1:
            return tensor.expand(batch)
        raise ValueError("Integrated thermal optics must broadcast to the batch dimension")

    def _prepare_layer_temperature(
        self,
        value: torch.Tensor | float,
        batch: int,
        nlayers: int,
        transfer,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        ninc = transfer.litab.numel()
        nazi = transfer.lazitab.numel()
        nori = transfer.lidf_azimuth.shape[1]
        if tensor.ndim == 0:
            return tensor.view(1, 1).expand(batch, nlayers)
        if tensor.ndim == 1:
            if tensor.shape[0] == batch:
                return tensor.view(batch, 1).expand(batch, nlayers)
            if tensor.shape[0] == nlayers:
                return tensor.view(1, nlayers).expand(batch, nlayers)
        if tensor.ndim == 2:
            if tensor.shape == (batch, nlayers):
                return tensor
            if tensor.shape[0] == 1 and tensor.shape[1] == nlayers:
                return tensor.expand(batch, nlayers)
        if tensor.ndim == 3:
            if tensor.shape == (batch, nlayers, nori):
                return tensor
            if tensor.shape == (1, nlayers, nori):
                return tensor.expand(batch, nlayers, nori)
            if tensor.shape == (nlayers, ninc, nazi):
                return tensor.reshape(1, nlayers, nori).expand(batch, nlayers, nori)
        if tensor.ndim == 4 and tensor.shape[-2:] == (ninc, nazi):
            if tensor.shape[0] == batch and tensor.shape[1] == nlayers:
                return tensor.reshape(batch, nlayers, nori)
            if tensor.shape[0] == 1 and tensor.shape[1] == nlayers:
                return tensor.reshape(1, nlayers, nori).expand(batch, nlayers, nori)
        raise ValueError(f"Layer temperatures must broadcast to shape ({batch}, {nlayers}) or ({batch}, {nlayers}, {nori})")

    def _expand_batch(self, value: torch.Tensor | float, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.repeat(batch)
        if tensor.ndim == 1 and tensor.shape[0] == batch:
            return tensor
        if tensor.ndim == 1 and tensor.shape[0] == 1 and batch != 1:
            return tensor.expand(batch)
        raise ValueError("Scalar thermal parameters must broadcast to the batch dimension")

    def _resolve_nlayers(self, nlayers: Optional[int], *, Tcu: torch.Tensor, Tch: torch.Tensor) -> int:
        for value in (Tcu, Tch):
            tensor = torch.as_tensor(value)
            if tensor.ndim >= 2:
                return int(tensor.shape[1])
        return int(nlayers) if nlayers is not None else 60

    def _canopy_emission(self, temperature_c: torch.Tensor, emissivity: torch.Tensor, transfer, wavelength_nm: torch.Tensor) -> torch.Tensor:
        thermal = torch.pi * self._planck(wavelength_nm, temperature_c.unsqueeze(-1) + 273.15, emissivity.unsqueeze(1))
        if thermal.ndim == 4:
            weights = transfer.lidf_azimuth.unsqueeze(1).unsqueeze(-1)
            return torch.sum(thermal * weights, dim=2)
        return thermal

    def _integrated_canopy_emission(
        self,
        temperature_c: torch.Tensor,
        emissivity: torch.Tensor,
        transfer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stefan = self._stefan_boltzmann(temperature_c)
        if stefan.ndim == 3:
            thermal = emissivity.view(-1, 1, 1) * stefan
        else:
            thermal = emissivity.view(-1, 1) * stefan
        if thermal.ndim == 3:
            weights = transfer.lidf_azimuth.unsqueeze(1)
            mean_thermal = torch.sum(thermal * weights, dim=2)
            return thermal, mean_thermal
        return thermal, thermal

    def _integrated_balance_core(
        self,
        *,
        transfer,
        epsc: torch.Tensor,
        epss: torch.Tensor,
        Tcu: torch.Tensor,
        Tch: torch.Tensor,
        Tsu: torch.Tensor,
        Tsh: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = transfer.Ps.shape[0]
        nl = transfer.nlayers
        device = transfer.Ps.device
        dtype = transfer.Ps.dtype

        Hcsu3, Hcsu = self._integrated_canopy_emission(Tcu, epsc, transfer)
        _, Hcsh = self._integrated_canopy_emission(Tch, epsc, transfer)
        Hssu = epss * self._stefan_boltzmann(Tsu)
        Hssh = epss * self._stefan_boltzmann(Tsh)

        Hc = Hcsu * transfer.Ps[:, :nl] + Hcsh * (1.0 - transfer.Ps[:, :nl])
        Hs = Hssu * transfer.Ps[:, -1] + Hssh * (1.0 - transfer.Ps[:, -1])

        rho_dd = transfer.rho_dd[..., 0]
        tau_dd = transfer.tau_dd[..., 0]
        R_dd = transfer.R_dd[..., 0]
        Emin = torch.zeros((batch, nl + 1), device=device, dtype=dtype)
        Eplu = torch.zeros_like(Emin)
        U = torch.zeros_like(Emin)
        Y = torch.zeros((batch, nl), device=device, dtype=dtype)
        U[:, -1] = Hs
        for layer in range(nl - 1, -1, -1):
            denom = (1.0 - rho_dd[:, layer] * R_dd[:, layer + 1]).clamp(min=1e-9)
            Y[:, layer] = (rho_dd[:, layer] * U[:, layer + 1] + Hc[:, layer] * transfer.iLAI) / denom
            U[:, layer] = tau_dd[:, layer] * (R_dd[:, layer + 1] * Y[:, layer] + U[:, layer + 1]) + Hc[:, layer] * transfer.iLAI
        for layer in range(nl):
            Emin[:, layer + 1] = transfer.Xdd[:, layer, 0] * Emin[:, layer] + Y[:, layer]
            Eplu[:, layer] = transfer.R_dd[:, layer, 0] * Emin[:, layer] + U[:, layer]
        # Match RTMt_sb.m: the soil-surface upward flux uses the canopy-bottom
        # coefficients (index nl in MATLAB, nl-1 in 0-based indexing), not
        # the background slot stored at the last R_dd entry.
        Eplu[:, -1] = transfer.R_dd[:, -2, 0] * Emin[:, -2] + Hs

        Po = transfer.Po[:, :nl]
        Pso = transfer.Pso[:, :nl]
        piLov = transfer.iLAI * (
            transfer.ko.unsqueeze(-1) * Hcsh * (Po - Pso)
            + transfer.ko.unsqueeze(-1) * Hcsu * Pso
            + (transfer.vb[:, 0].unsqueeze(-1) * Emin[:, :nl] + transfer.vf[:, 0].unsqueeze(-1) * Eplu[:, :nl]) * Po
        ).sum(dim=1)
        piLos = Hssh * (transfer.Po[:, -1] - transfer.Pso[:, -1]) + Hssu * transfer.Pso[:, -1]
        piLot = piLov + piLos
        Lote = piLot / torch.pi
        Eoutte = Eplu[:, 0]

        common_leaf = epsc.unsqueeze(-1) * (Emin[:, :-1] + Eplu[:, 1:])
        if Hcsu3.ndim == 3:
            Rnuct = common_leaf.unsqueeze(-1) - 2.0 * Hcsu3
        else:
            Rnuct = common_leaf - 2.0 * Hcsu
        Rnhct = common_leaf - 2.0 * Hcsh
        Rnust = epss * (Emin[:, -1] - Hssu)
        Rnhst = epss * (Emin[:, -1] - Hssh)
        return {
            "Lote": Lote,
            "Eoutte": Eoutte,
            "Emint": Emin,
            "Eplut": Eplu,
            "Rnuct": Rnuct,
            "Rnhct": Rnhct,
            "Rnust": Rnust,
            "Rnhst": Rnhst,
        }
