from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .foursail import FourSAILModel
from .reflectance import CanopyReflectanceModel
from ..spectral.fluspect import LeafBioBatch
from ..spectral.soil import SoilEmpiricalParams


@dataclass(slots=True)
class CanopyFluorescenceResult:
    """First-pass canopy fluorescence outputs on the SCOPE fluorescence grid."""

    leaf_fluor_back: torch.Tensor
    leaf_fluor_forw: torch.Tensor
    Femleaves_: torch.Tensor
    EoutFrc_: torch.Tensor
    EoutF_: torch.Tensor
    LoF_: torch.Tensor
    sigmaF: torch.Tensor
    gammasdf: torch.Tensor
    gammasdb: torch.Tensor
    gammaso: torch.Tensor
    F685: torch.Tensor
    wl685: torch.Tensor
    F740: torch.Tensor
    wl740: torch.Tensor
    F684: torch.Tensor
    F761: torch.Tensor
    LoutF: torch.Tensor
    EoutF: torch.Tensor


class CanopyFluorescenceModel:
    """Canopy fluorescence wrapper built on leaf Mb/Mf and canopy gamma transport.

    This is intentionally narrower than upstream ``RTMf``. It uses the existing
    leaf excitation-emission matrices and 4SAIL gamma terms to provide a stable
    fluorescence API before the full layered radiance solver is ported.
    """

    def __init__(self, reflectance_model: CanopyReflectanceModel) -> None:
        self.reflectance_model = reflectance_model

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
    ) -> "CanopyFluorescenceModel":
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
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        excitation: torch.Tensor,
        *,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
    ) -> CanopyFluorescenceResult:
        fluspect = self.reflectance_model.fluspect
        sail_model = self.reflectance_model.sail

        leafopt = fluspect(leafbio)
        if leafopt.Mb is None or leafopt.Mf is None:
            raise ValueError("Leaf fluorescence matrices are unavailable. Provide leafbio with fqe > 0.")

        hotspot_value = hotspot if hotspot is not None else torch.full_like(torch.as_tensor(lai), self.reflectance_model.default_hotspot)
        sail = sail_model(
            leafopt.refl,
            leafopt.tran,
            soil_refl,
            lai,
            hotspot_value,
            tts,
            tto,
            psi,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
        )

        excitation_tensor = self._prepare_excitation(excitation, leafopt.Mb.shape[0], fluspect.spectral.wlE.numel())
        leaf_fluor_back = torch.einsum("bfe,be->bf", leafopt.Mb, excitation_tensor)
        leaf_fluor_forw = torch.einsum("bfe,be->bf", leafopt.Mf, excitation_tensor)

        lai_tensor = self._expand_batch(lai, leaf_fluor_back.shape[0], device=leaf_fluor_back.device, dtype=leaf_fluor_back.dtype)
        Femleaves_ = lai_tensor.unsqueeze(-1) * (leaf_fluor_back + leaf_fluor_forw)
        EoutFrc_ = Femleaves_.clone()

        wlP = fluspect.spectral.wlP
        wlF = fluspect.spectral.wlF
        if wlF is None:
            raise ValueError("Spectral grids must define fluorescence wavelengths")

        gammasdf = self._interp1d(wlP, sail.gammasdf, wlF)
        gammasdb = self._interp1d(wlP, sail.gammasdb, wlF)
        gammaso = self._interp1d(wlP, sail.gammaso, wlF)

        transport_total = (gammasdf + gammasdb).clamp(min=1e-9)
        # `gammasdb` is interpreted here as the upward hemispherical branch.
        upward_escape = torch.clamp(gammasdb / transport_total, min=0.0, max=1.0)
        sigmaF = torch.clamp(gammaso / transport_total, min=0.0, max=1.0)

        EoutF_ = upward_escape * EoutFrc_
        LoF_ = sigmaF * EoutFrc_ / torch.pi

        F685, wl685 = self._peak_in_window(LoF_, wlF, max_wavelength=700.0)
        F740, wl740 = self._peak_in_window(LoF_, wlF, min_wavelength=700.0)
        F684 = self._sample_nearest(LoF_, wlF, 684.0)
        F761 = self._sample_nearest(LoF_, wlF, 761.0)
        LoutF = 0.001 * torch.trapz(LoF_, wlF, dim=-1)
        EoutF = 0.001 * torch.trapz(EoutF_, wlF, dim=-1)

        return CanopyFluorescenceResult(
            leaf_fluor_back=leaf_fluor_back,
            leaf_fluor_forw=leaf_fluor_forw,
            Femleaves_=Femleaves_,
            EoutFrc_=EoutFrc_,
            EoutF_=EoutF_,
            LoF_=LoF_,
            sigmaF=sigmaF,
            gammasdf=gammasdf,
            gammasdb=gammasdb,
            gammaso=gammaso,
            F685=F685,
            wl685=wl685,
            F740=F740,
            wl740=wl740,
            F684=F684,
            F761=F761,
            LoutF=LoutF,
            EoutF=EoutF,
        )

    def _prepare_excitation(self, excitation: torch.Tensor, batch: int, n_wle: int) -> torch.Tensor:
        tensor = torch.as_tensor(excitation, device=self.reflectance_model.fluspect.device, dtype=self.reflectance_model.fluspect.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Excitation spectra must be 1D or 2D, got shape {tuple(tensor.shape)}")
        if tensor.shape[-1] != n_wle:
            raise ValueError(f"Excitation spectra must have length {n_wle} to match wlE, got {tensor.shape[-1]}")
        if tensor.shape[0] == 1 and batch != 1:
            tensor = tensor.expand(batch, -1)
        elif tensor.shape[0] != batch:
            raise ValueError("Excitation spectra must broadcast to the batch dimension")
        return tensor

    def _expand_batch(self, value: torch.Tensor | float, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.repeat(batch)
        elif tensor.shape[0] == 1 and batch != 1:
            tensor = tensor.expand(batch)
        elif tensor.shape[0] != batch:
            raise ValueError("Scalar canopy parameters must broadcast to the batch dimension")
        return tensor

    def _interp1d(self, source_x: torch.Tensor, source_y: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        source_x = source_x.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype)
        source_y = source_y.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype)
        target_x = target_x.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype)
        idx = torch.bucketize(target_x, source_x) - 1
        idx = idx.clamp(0, source_x.numel() - 2)
        x0 = source_x[idx]
        x1 = source_x[idx + 1]
        denom = (x1 - x0).clamp(min=1e-9)
        frac = (target_x - x0) / denom
        y0 = source_y.gather(1, idx.unsqueeze(0).expand(source_y.shape[0], -1))
        y1 = source_y.gather(1, (idx + 1).unsqueeze(0).expand(source_y.shape[0], -1))
        return y0 + (y1 - y0) * frac

    def _peak_in_window(
        self,
        spectrum: torch.Tensor,
        wavelength: torch.Tensor,
        *,
        min_wavelength: float | None = None,
        max_wavelength: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones_like(wavelength, dtype=torch.bool)
        if min_wavelength is not None:
            mask &= wavelength >= min_wavelength
        if max_wavelength is not None:
            mask &= wavelength <= max_wavelength
        if not mask.any():
            raise ValueError("Requested peak window does not overlap the fluorescence grid")

        window_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        window = spectrum.index_select(1, window_idx)
        peak_values, peak_offsets = window.max(dim=-1)
        peak_indices = window_idx.index_select(0, peak_offsets)
        peak_wavelengths = wavelength.index_select(0, peak_indices)
        return peak_values, peak_wavelengths

    def _sample_nearest(self, spectrum: torch.Tensor, wavelength: torch.Tensor, target: float) -> torch.Tensor:
        index = torch.argmin(torch.abs(wavelength - target))
        return spectrum[:, index]
