from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..biochem import BiochemicalOptions, LeafBiochemistryInputs, LeafBiochemistryModel, LeafBiochemistryResult, LeafMeteo
from .foursail import FourSAILModel
from .layered_rt import LayeredCanopyTransfer, LayeredCanopyTransportModel
from .reflectance import CanopyReflectanceModel
from ..spectral.fluspect import FluspectModel, LeafBioBatch, SpectralGrids
from ..spectral.soil import SoilEmpiricalParams


@dataclass(slots=True)
class CanopyFluorescenceResult:
    leaf_fluor_back: torch.Tensor
    leaf_fluor_forw: torch.Tensor
    LoF_sunlit: torch.Tensor
    LoF_shaded: torch.Tensor
    LoF_scattered: torch.Tensor
    LoF_soil: torch.Tensor
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
    Fmin_: torch.Tensor
    Fplu_: torch.Tensor


@dataclass(slots=True)
class CanopyBiochemicalFluorescenceResult:
    fluorescence: CanopyFluorescenceResult
    sunlit: LeafBiochemistryResult
    shaded: LeafBiochemistryResult
    Pnu_Cab: torch.Tensor
    Pnh_Cab: torch.Tensor


@dataclass(slots=True)
class _LayeredFluorescenceDiagnostics:
    leaf_fluor_back: torch.Tensor
    leaf_fluor_forw: torch.Tensor
    poutfrc: torch.Tensor
    MpluEsun: torch.Tensor
    MminEsun: torch.Tensor
    piLs: torch.Tensor
    piLd: torch.Tensor
    Femmin: torch.Tensor
    Femplu: torch.Tensor
    Fmin_: torch.Tensor
    Fplu_: torch.Tensor
    piLo1: torch.Tensor
    piLo2: torch.Tensor
    piLo3: torch.Tensor
    piLo4: torch.Tensor
    Femleaves_: torch.Tensor
    EoutFrc_: torch.Tensor
    EoutF_: torch.Tensor
    LoF_: torch.Tensor


class CanopyFluorescenceModel:
    """Canopy fluorescence transport with both simple and layered modes."""

    def __init__(self, reflectance_model: CanopyReflectanceModel) -> None:
        self.reflectance_model = reflectance_model
        self.layered_transport = LayeredCanopyTransportModel(reflectance_model.sail)
        self.leaf_biochemistry = LeafBiochemistryModel(
            device=reflectance_model.fluspect.device,
            dtype=reflectance_model.fluspect.dtype,
        )
        default_spectral = SpectralGrids.default(reflectance_model.fluspect.device, reflectance_model.fluspect.dtype)
        self._rtmf_fluspect = FluspectModel(
            SpectralGrids(
                wlP=reflectance_model.fluspect.spectral.wlP,
                wlF=default_spectral.wlF,
                wlE=default_spectral.wlE,
            ),
            reflectance_model.fluspect.optipar,
            ndub=reflectance_model.fluspect.ndub,
            doublings_step=reflectance_model.fluspect.step,
            device=reflectance_model.fluspect.device,
            dtype=reflectance_model.fluspect.dtype,
        )

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
        return self.one_pass(
            leafbio,
            soil_refl,
            lai,
            tts,
            tto,
            psi,
            excitation,
            hotspot=hotspot,
            lidf=lidf,
        )

    def one_pass(
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

        wlE = fluspect.spectral.wlE
        wlF = fluspect.spectral.wlF
        wlP = fluspect.spectral.wlP
        if wlE is None or wlF is None:
            raise ValueError("Spectral grids must define excitation and fluorescence wavelengths")

        excitation_tensor = self._prepare_spectrum(excitation, leafopt.Mb.shape[0], wlE.numel())
        leaf_fluor_back = torch.einsum("bfe,be->bf", leafopt.Mb, excitation_tensor)
        leaf_fluor_forw = torch.einsum("bfe,be->bf", leafopt.Mf, excitation_tensor)

        lai_tensor = self._expand_batch(lai, leaf_fluor_back.shape[0], device=leaf_fluor_back.device, dtype=leaf_fluor_back.dtype)
        Femleaves_ = lai_tensor.unsqueeze(-1) * (leaf_fluor_back + leaf_fluor_forw)
        EoutFrc_ = self._simple_reabsorption_corrected_source(
            leafopt=leafopt,
            leafbio=leafbio,
            excitation=excitation_tensor,
            lai=lai_tensor,
            wlP=wlP,
            wlE=wlE,
            wlF=wlF,
        )

        gammasdf = self._interp1d(wlP, sail.gammasdf, wlF)
        gammasdb = self._interp1d(wlP, sail.gammasdb, wlF)
        gammaso = self._interp1d(wlP, sail.gammaso, wlF)

        transport_total = (gammasdf + gammasdb).clamp(min=1e-9)
        upward_escape = torch.clamp(gammasdb / transport_total, min=0.0, max=1.0)
        sigmaF = torch.clamp(gammaso / transport_total, min=0.0, max=1.0)

        EoutF_ = upward_escape * EoutFrc_
        LoF_ = sigmaF * EoutFrc_ / torch.pi
        sigmaF = self._scope_sigmaf(LoF_, EoutFrc_, wlF)

        F685, wl685 = self._peak_in_window(LoF_, wlF, max_wavelength=700.0)
        F740, wl740 = self._peak_in_window(LoF_, wlF, min_wavelength=700.0)
        F684 = self._sample_nearest(LoF_, wlF, 684.0)
        F761 = self._sample_nearest(LoF_, wlF, 761.0)
        LoutF = 0.001 * torch.trapz(LoF_, wlF, dim=-1)
        EoutF = 0.001 * torch.trapz(EoutF_, wlF, dim=-1)

        zeros = torch.zeros_like(LoF_)
        empty_profile = torch.zeros((LoF_.shape[0], 1, LoF_.shape[1]), device=LoF_.device, dtype=LoF_.dtype)
        return CanopyFluorescenceResult(
            leaf_fluor_back=leaf_fluor_back,
            leaf_fluor_forw=leaf_fluor_forw,
            LoF_sunlit=LoF_,
            LoF_shaded=zeros,
            LoF_scattered=zeros,
            LoF_soil=zeros,
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
            Fmin_=empty_profile,
            Fplu_=empty_profile,
        )

    def layered(
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
        etau: Optional[torch.Tensor] = None,
        etah: Optional[torch.Tensor] = None,
        Pnu_Cab: Optional[torch.Tensor] = None,
        Pnh_Cab: Optional[torch.Tensor] = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
    ) -> CanopyFluorescenceResult:
        fluspect = self.reflectance_model.fluspect
        leafopt = self._rtmf_fluspect(leafbio)
        if leafopt.Mb is None or leafopt.Mf is None:
            raise ValueError("Leaf fluorescence matrices are unavailable. Provide leafbio with fqe > 0.")

        wlP = self._rtmf_fluspect.spectral.wlP
        wlE = self._rtmf_fluspect.spectral.wlE
        wlF = self._rtmf_fluspect.spectral.wlF
        if wlE is None or wlF is None:
            raise ValueError("Spectral grids must define excitation and fluorescence wavelengths")
        nl = self._resolve_nlayers(nlayers, etau=etau, etah=etah)

        source_wlE = fluspect.spectral.wlE
        if source_wlE is None:
            raise ValueError("Spectral grids must define excitation wavelengths")
        Esun_source = self._prepare_spectrum(Esun_, leafopt.refl.shape[0], source_wlE.numel())
        Esky_source = self._prepare_spectrum(Esky_, leafopt.refl.shape[0], source_wlE.numel())
        Esun_working = self._sample_spectrum(Esun_source, source_wlE, wlE)
        Esky_working = self._sample_spectrum(Esky_source, source_wlE, wlE)

        diagnostics = self._layered_diagnostics(
            leafopt=leafopt,
            leafbio=leafbio,
            soil_refl=soil_refl,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            Esun_=Esun_working,
            Esky_=Esky_working,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nl,
            wlP=wlP,
            wlE=wlE,
            wlF=wlF,
            etau=etau,
            etah=etah,
        )

        reflectance = self.reflectance_model(
            leafbio,
            soil_refl,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot,
            lidf=lidf,
        )
        target_wlF = fluspect.spectral.wlF
        if target_wlF is None:
            raise ValueError("Spectral grids must define fluorescence wavelengths")

        gammasdf = self._sample_spectrum(reflectance.gammasdf, wlP, target_wlF)
        gammasdb = self._sample_spectrum(reflectance.gammasdb, wlP, target_wlF)
        gammaso = self._sample_spectrum(reflectance.gammaso, wlP, target_wlF)

        LoF_interp = self._matlab_spline_interp1(wlF, diagnostics.LoF_, target_wlF)
        EoutF_interp = self._matlab_spline_interp1(wlF, diagnostics.EoutF_, target_wlF)
        LoF_sunlit = self._matlab_spline_interp1(wlF, diagnostics.piLo1 / torch.pi, target_wlF)
        LoF_shaded = self._matlab_spline_interp1(wlF, diagnostics.piLo2 / torch.pi, target_wlF)
        LoF_scattered = self._matlab_spline_interp1(wlF, diagnostics.piLo3 / torch.pi, target_wlF)
        LoF_soil = self._matlab_spline_interp1(wlF, diagnostics.piLo4 / torch.pi, target_wlF)
        Femleaves_interp = self._matlab_spline_interp1(wlF, diagnostics.Femleaves_, target_wlF)

        leafopt_source = fluspect(leafbio)
        poutfrc_target = self._layered_source_poutfrc(
            leafopt=leafopt_source,
            leafbio=leafbio,
            soil_refl=soil_refl,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            Esun=Esun_source,
            Esky=Esky_source,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nl,
            etau=etau,
            etah=etah,
            Pnu_Cab=Pnu_Cab,
            Pnh_Cab=Pnh_Cab,
            wlP=fluspect.spectral.wlP,
            wlE=source_wlE,
        )
        EoutFrc_target = self._source_spectrum_from_poutfrc(poutfrc_target, wlP=fluspect.spectral.wlP, wlF=target_wlF)
        sigmaF = self._scope_sigmaf(LoF_interp, EoutFrc_target, target_wlF)

        F685, wl685 = self._peak_in_window(LoF_interp, target_wlF, max_wavelength=700.0)
        F740, wl740 = self._peak_in_window(LoF_interp, target_wlF, min_wavelength=700.0)
        F684 = self._sample_nearest(LoF_interp, target_wlF, 684.0)
        F761 = self._sample_nearest(LoF_interp, target_wlF, 761.0)
        LoutF = 0.001 * torch.trapz(diagnostics.LoF_, wlF, dim=-1)
        EoutF = 0.001 * torch.trapz(diagnostics.EoutF_, wlF, dim=-1)

        return CanopyFluorescenceResult(
            leaf_fluor_back=diagnostics.leaf_fluor_back,
            leaf_fluor_forw=diagnostics.leaf_fluor_forw,
            LoF_sunlit=LoF_sunlit,
            LoF_shaded=LoF_shaded,
            LoF_scattered=LoF_scattered,
            LoF_soil=LoF_soil,
            Femleaves_=Femleaves_interp,
            EoutFrc_=EoutFrc_target,
            EoutF_=EoutF_interp,
            LoF_=LoF_interp,
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
            Fmin_=diagnostics.Fmin_,
            Fplu_=diagnostics.Fplu_,
        )

    def _layered_diagnostics(
        self,
        *,
        leafopt,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_: torch.Tensor,
        Esky_: torch.Tensor,
        hotspot: Optional[torch.Tensor],
        lidf: Optional[torch.Tensor],
        nlayers: int,
        wlP: torch.Tensor,
        wlE: torch.Tensor,
        wlF: torch.Tensor,
        etau: Optional[torch.Tensor],
        etah: Optional[torch.Tensor],
    ) -> _LayeredFluorescenceDiagnostics:
        nl = nlayers
        hotspot_value = hotspot if hotspot is not None else torch.full_like(
            torch.as_tensor(lai),
            self.reflectance_model.default_hotspot,
        )

        rho_e = self._sample_spectrum(leafopt.refl, wlP, wlE)
        tau_e = self._sample_spectrum(leafopt.tran, wlP, wlE)
        soil_e = self._sample_spectrum(soil_refl, wlP, wlE)
        rho_f = self._sample_spectrum(leafopt.refl, wlP, wlF)
        tau_f = self._sample_spectrum(leafopt.tran, wlP, wlF)
        soil_f = self._sample_spectrum(soil_refl, wlP, wlF)

        transport_e = self.layered_transport.build(
            rho_e,
            tau_e,
            soil_e,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nl,
        )
        transport_f = self.layered_transport.build(
            rho_f,
            tau_f,
            soil_f,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nl,
        )

        Esun = self._prepare_spectrum(Esun_, leafopt.Mb.shape[0], wlE.numel())
        Esky = self._prepare_spectrum(Esky_, leafopt.Mb.shape[0], wlE.numel())
        direct = self.layered_transport.flux_profiles(transport_e, Esun, torch.zeros_like(Esky))
        diffuse = self.layered_transport.flux_profiles(transport_e, torch.zeros_like(Esun), Esky)
        total_Emin = direct.Emin_ + diffuse.Emin_
        total_Eplu = direct.Eplu_ + diffuse.Eplu_

        leaf_fluor_back = torch.einsum("bfe,be->bf", leafopt.Mb, Esun)
        leaf_fluor_forw = torch.einsum("bfe,be->bf", leafopt.Mf, Esun)

        Mplu = 0.5 * (leafopt.Mb + leafopt.Mf)
        Mmin = 0.5 * (leafopt.Mb - leafopt.Mf)
        MpluEsun = self._matrix_energy_product(Mplu, Esun, wlE, wlF)
        MminEsun = self._matrix_energy_product(Mmin, Esun, wlE, wlF)
        MpluEmin = self._matrix_energy_product_layered(Mplu, total_Emin[:, :-1, :], wlE, wlF)
        MpluEplu = self._matrix_energy_product_layered(Mplu, total_Eplu[:, :-1, :], wlE, wlF)
        MminEmin = self._matrix_energy_product_layered(Mmin, total_Emin[:, :-1, :], wlE, wlF)
        MminEplu = self._matrix_energy_product_layered(Mmin, total_Eplu[:, :-1, :], wlE, wlF)

        etau_orient = self._prepare_etau(etau, transport_e)
        etah_orient = self._prepare_etah(etah, transport_e)
        etau_lidf = etau_orient * transport_e.lidf_azimuth.unsqueeze(1)
        etah_lidf = etah_orient * transport_e.lidf_azimuth.unsqueeze(1)

        absfsfo_tau = (etau_lidf * transport_e.absfs.unsqueeze(1) * transport_e.absfo.unsqueeze(1)).sum(dim=-1)
        fsfo_tau = (etau_lidf * transport_e.fsfo.unsqueeze(1)).sum(dim=-1)
        absfs_tau = (etau_lidf * transport_e.absfs.unsqueeze(1)).sum(dim=-1)
        fsctl_tau = (etau_lidf * transport_e.fsctl.unsqueeze(1)).sum(dim=-1)
        absfo_tau = (etau_lidf * transport_e.absfo.unsqueeze(1)).sum(dim=-1)
        foctl_tau = (etau_lidf * transport_e.foctl.unsqueeze(1)).sum(dim=-1)
        ctl2_tau = (etau_lidf * transport_e.ctl2.unsqueeze(1)).sum(dim=-1)

        absfo_h = (etah_lidf * transport_e.absfo.unsqueeze(1)).sum(dim=-1)
        foctl_h = (etah_lidf * transport_e.foctl.unsqueeze(1)).sum(dim=-1)
        ctl2_h = (etah_lidf * transport_e.ctl2.unsqueeze(1)).sum(dim=-1)
        sum_tau = etau_lidf.sum(dim=-1)
        sum_h = etah_lidf.sum(dim=-1)

        wfEs = absfsfo_tau.unsqueeze(-1) * MpluEsun.unsqueeze(1) + fsfo_tau.unsqueeze(-1) * MminEsun.unsqueeze(1)
        sfEs = absfs_tau.unsqueeze(-1) * MpluEsun.unsqueeze(1) - fsctl_tau.unsqueeze(-1) * MminEsun.unsqueeze(1)
        sbEs = absfs_tau.unsqueeze(-1) * MpluEsun.unsqueeze(1) + fsctl_tau.unsqueeze(-1) * MminEsun.unsqueeze(1)
        vfEplu_h = absfo_h.unsqueeze(-1) * MpluEplu - foctl_h.unsqueeze(-1) * MminEplu
        vfEplu_u = absfo_tau.unsqueeze(-1) * MpluEplu - foctl_tau.unsqueeze(-1) * MminEplu
        vbEmin_h = absfo_h.unsqueeze(-1) * MpluEmin + foctl_h.unsqueeze(-1) * MminEmin
        vbEmin_u = absfo_tau.unsqueeze(-1) * MpluEmin + foctl_tau.unsqueeze(-1) * MminEmin
        sigfEmin_h = sum_h.unsqueeze(-1) * MpluEmin - ctl2_h.unsqueeze(-1) * MminEmin
        sigfEmin_u = sum_tau.unsqueeze(-1) * MpluEmin - ctl2_tau.unsqueeze(-1) * MminEmin
        sigbEmin_h = sum_h.unsqueeze(-1) * MpluEmin + ctl2_h.unsqueeze(-1) * MminEmin
        sigbEmin_u = sum_tau.unsqueeze(-1) * MpluEmin + ctl2_tau.unsqueeze(-1) * MminEmin
        sigfEplu_h = sum_h.unsqueeze(-1) * MpluEplu - ctl2_h.unsqueeze(-1) * MminEplu
        sigfEplu_u = sum_tau.unsqueeze(-1) * MpluEplu - ctl2_tau.unsqueeze(-1) * MminEplu
        sigbEplu_h = sum_h.unsqueeze(-1) * MpluEplu + ctl2_h.unsqueeze(-1) * MminEplu
        sigbEplu_u = sum_tau.unsqueeze(-1) * MpluEplu + ctl2_tau.unsqueeze(-1) * MminEplu

        piLs = wfEs + vfEplu_u + vbEmin_u
        piLd = vbEmin_h + vfEplu_h
        Fsmin = sfEs + sigfEmin_u + sigbEplu_u
        Fsplu = sbEs + sigbEmin_u + sigfEplu_u
        Fdmin = sigfEmin_h + sigbEplu_h
        Fdplu = sigbEmin_h + sigfEplu_h

        Qs = transport_f.Ps[:, :nl]
        iLAI = transport_f.iLAI.unsqueeze(-1).unsqueeze(-1)
        Femmin = iLAI * (Qs.unsqueeze(-1) * Fsmin + (1.0 - Qs).unsqueeze(-1) * Fdmin)
        Femplu = iLAI * (Qs.unsqueeze(-1) * Fsplu + (1.0 - Qs).unsqueeze(-1) * Fdplu)

        batch = leafopt.Mb.shape[0]
        nf = wlF.numel()
        Fmin = torch.zeros((batch, nl + 1, nf), device=wlF.device, dtype=wlF.dtype)
        Fplu = torch.zeros_like(Fmin)
        U = torch.zeros_like(Fmin)
        Y = torch.zeros((batch, nl, nf), device=wlF.device, dtype=wlF.dtype)
        for layer in range(nl - 1, -1, -1):
            denom = (1.0 - transport_f.rho_dd[:, layer, :] * transport_f.R_dd[:, layer + 1, :]).clamp(min=1e-9)
            Y[:, layer, :] = (transport_f.rho_dd[:, layer, :] * U[:, layer + 1, :] + Femmin[:, layer, :]) / denom
            U[:, layer, :] = transport_f.tau_dd[:, layer, :] * (transport_f.R_dd[:, layer + 1, :] * Y[:, layer, :] + U[:, layer + 1, :]) + Femplu[:, layer, :]
        for layer in range(nl):
            Fmin[:, layer + 1, :] = transport_f.Xdd[:, layer, :] * Fmin[:, layer, :] + Y[:, layer, :]
            Fplu[:, layer, :] = transport_f.R_dd[:, layer, :] * Fmin[:, layer, :] + U[:, layer, :]

        Po = transport_f.Po[:, :nl].unsqueeze(-1)
        Pso = transport_f.Pso[:, :nl].unsqueeze(-1)
        piLo1 = transport_f.iLAI.unsqueeze(-1) * (piLs * Pso).sum(dim=1)
        piLo2 = transport_f.iLAI.unsqueeze(-1) * (piLd * (Po - Pso)).sum(dim=1)
        piLo3 = transport_f.iLAI.unsqueeze(-1) * (
            (transport_f.vb.unsqueeze(1) * Fmin[:, :nl, :] + transport_f.vf.unsqueeze(1) * Fplu[:, :nl, :]) * Po
        ).sum(dim=1)
        piLo4 = soil_f * Fmin[:, -1, :] * transport_f.Po[:, -1].unsqueeze(-1)
        piLtot = piLo1 + piLo2 + piLo3 + piLo4

        LoF_ = piLtot / torch.pi
        EoutF_ = Fplu[:, 0, :]
        Femleaves_ = (Femmin + Femplu).sum(dim=1)
        poutfrc = self._layered_reabsorption_corrected_poutfrc(
            leafopt=leafopt,
            leafbio=leafbio,
            Esun=Esun,
            total_Emin=total_Emin,
            total_Eplu=total_Eplu,
            transport=transport_e,
            etau=etau_orient,
            etah=etah_orient,
            wlP=wlP,
            wlE=wlE,
        )
        EoutFrc_ = self._source_spectrum_from_poutfrc(poutfrc, wlP=wlP, wlF=wlF)
        return _LayeredFluorescenceDiagnostics(
            leaf_fluor_back=leaf_fluor_back,
            leaf_fluor_forw=leaf_fluor_forw,
            poutfrc=poutfrc,
            MpluEsun=MpluEsun,
            MminEsun=MminEsun,
            piLs=piLs,
            piLd=piLd,
            Femmin=Femmin,
            Femplu=Femplu,
            Fmin_=Fmin,
            Fplu_=Fplu,
            piLo1=piLo1,
            piLo2=piLo2,
            piLo3=piLo3,
            piLo4=piLo4,
            Femleaves_=Femleaves_,
            EoutFrc_=EoutFrc_,
            EoutF_=EoutF_,
            LoF_=LoF_,
        )

    def layered_biochemical(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_: torch.Tensor,
        Esky_: torch.Tensor,
        *,
        Csu: torch.Tensor,
        Csh: torch.Tensor,
        ebu: torch.Tensor,
        ebh: torch.Tensor,
        Tcu: torch.Tensor,
        Tch: torch.Tensor,
        Oa: torch.Tensor,
        p: torch.Tensor,
        fV: torch.Tensor | float = 1.0,
        biochem_options: Optional[BiochemicalOptions] = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
    ) -> CanopyBiochemicalFluorescenceResult:
        fluspect = self.reflectance_model.fluspect
        leafopt = fluspect(leafbio)
        if leafopt.Mb is None or leafopt.Mf is None:
            raise ValueError("Leaf fluorescence matrices are unavailable. Provide leafbio with fqe > 0.")

        wlP = fluspect.spectral.wlP
        wlE = fluspect.spectral.wlE
        if wlE is None:
            raise ValueError("Spectral grids must define excitation wavelengths")

        hotspot_value = hotspot if hotspot is not None else torch.full_like(torch.as_tensor(lai), self.reflectance_model.default_hotspot)
        nl = self._resolve_nlayers(nlayers, etau=Tcu, etah=Tch)
        rho_e = self._sample_spectrum(leafopt.refl, wlP, wlE)
        tau_e = self._sample_spectrum(leafopt.tran, wlP, wlE)
        soil_e = self._sample_spectrum(soil_refl, wlP, wlE)
        transport_e = self.layered_transport.build(
            rho_e,
            tau_e,
            soil_e,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nl,
        )
        Pnu_Cab, Pnh_Cab = self._absorbed_cab_profiles(
            leafopt=leafopt,
            transport=transport_e,
            Esun_=Esun_,
            Esky_=Esky_,
            wlP=wlP,
            wlE=wlE,
        )

        sunlit = self._run_leaf_biochemistry(
            biochemistry=biochemistry,
            Q=Pnu_Cab,
            Cs=self._prepare_etau(Csu, transport_e),
            T=self._prepare_etau(Tcu, transport_e),
            eb=self._prepare_etau(ebu, transport_e),
            Oa=self._prepare_etau(Oa, transport_e),
            p=self._prepare_etau(p, transport_e),
            fV=self._prepare_etau(fV, transport_e),
            options=biochem_options,
            target_shape=Pnu_Cab.shape,
        )
        shaded = self._run_leaf_biochemistry(
            biochemistry=biochemistry,
            Q=Pnh_Cab,
            Cs=self._prepare_layer_profile(Csh, transport_e),
            T=self._prepare_layer_profile(Tch, transport_e),
            eb=self._prepare_layer_profile(ebh, transport_e),
            Oa=self._prepare_layer_profile(Oa, transport_e),
            p=self._prepare_layer_profile(p, transport_e),
            fV=self._prepare_layer_profile(fV, transport_e),
            options=biochem_options,
            target_shape=Pnh_Cab.shape,
        )

        fluorescence = self.layered(
            leafbio,
            soil_refl,
            lai,
            tts,
            tto,
            psi,
            Esun_,
            Esky_,
            etau=sunlit.eta,
            etah=shaded.eta,
            Pnu_Cab=Pnu_Cab,
            Pnh_Cab=Pnh_Cab,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nl,
        )
        return CanopyBiochemicalFluorescenceResult(
            fluorescence=fluorescence,
            sunlit=sunlit,
            shaded=shaded,
            Pnu_Cab=Pnu_Cab,
            Pnh_Cab=Pnh_Cab,
        )

    def _prepare_spectrum(self, value: torch.Tensor, batch: int, n_wavelength: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.reflectance_model.fluspect.device, dtype=self.reflectance_model.fluspect.dtype)
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

    def _resolve_nlayers(self, nlayers: Optional[int], *, etau: Optional[torch.Tensor], etah: Optional[torch.Tensor]) -> int:
        for value in (etau, etah):
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if tensor.ndim >= 2:
                return int(tensor.shape[1])
        return int(nlayers) if nlayers is not None else 60

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
        source_x = source_x.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype).contiguous()
        source_y = source_y.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype).contiguous()
        target_x = target_x.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype).contiguous()
        idx = torch.bucketize(target_x, source_x) - 1
        idx = idx.clamp(0, source_x.numel() - 2)
        x0 = source_x[idx]
        x1 = source_x[idx + 1]
        denom = (x1 - x0).clamp(min=1e-9)
        frac = (target_x - x0) / denom
        y0 = source_y.gather(1, idx.unsqueeze(0).expand(source_y.shape[0], -1))
        y1 = source_y.gather(1, (idx + 1).unsqueeze(0).expand(source_y.shape[0], -1))
        return y0 + (y1 - y0) * frac

    def _sample_spectrum(self, spectrum: torch.Tensor, source_x: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        if spectrum.shape[-1] == target_x.numel() and source_x.numel() == target_x.numel() and torch.allclose(source_x, target_x):
            return spectrum
        return self._interp1d(source_x, spectrum, target_x)

    def _ephoton(self, wavelength_nm: torch.Tensor) -> torch.Tensor:
        h = torch.as_tensor(6.62607015e-34, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        c = torch.as_tensor(299792458.0, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        avogadro = torch.as_tensor(6.02214076e23, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        wavelength_m = wavelength_nm * 1e-9
        return avogadro * h * c / wavelength_m

    def _e2phot(self, wavelength_nm: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        return energy / self._ephoton(wavelength_nm).view(*([1] * (energy.ndim - 1)), -1)

    def _matrix_energy_product(self, matrix: torch.Tensor, excitation: torch.Tensor, wlE: torch.Tensor, wlF: torch.Tensor) -> torch.Tensor:
        photons = self._e2phot(wlE, excitation)
        emitted = torch.einsum("bfe,be->bf", matrix, photons)
        return self._ephoton(wlF).unsqueeze(0) * emitted

    def _matrix_energy_product_layered(self, matrix: torch.Tensor, excitation: torch.Tensor, wlE: torch.Tensor, wlF: torch.Tensor) -> torch.Tensor:
        photons = self._e2phot(wlE, excitation)
        emitted = torch.einsum("bfe,ble->blf", matrix, photons)
        return self._ephoton(wlF).view(1, 1, -1) * emitted

    def _absorbed_cab_profiles(
        self,
        *,
        leafopt,
        transport: LayeredCanopyTransfer,
        Esun_: torch.Tensor,
        Esky_: torch.Tensor,
        wlP: torch.Tensor,
        wlE: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = leafopt.refl.shape[0]
        Esun = self._prepare_spectrum(Esun_, batch, wlE.numel())
        Esky = self._prepare_spectrum(Esky_, batch, wlE.numel())
        direct = self.layered_transport.flux_profiles(transport, Esun, torch.zeros_like(Esky))
        diffuse = self.layered_transport.flux_profiles(transport, torch.zeros_like(Esun), Esky)
        total_Emin = direct.Emin_ + diffuse.Emin_
        total_Eplu = direct.Eplu_ + diffuse.Eplu_

        rho_e = self._sample_spectrum(leafopt.refl, wlP, wlE)
        tau_e = self._sample_spectrum(leafopt.tran, wlP, wlE)
        kchl_e = self._sample_spectrum(leafopt.kChlrel, wlP, wlE)
        epsc_e = (1.0 - rho_e - tau_e).clamp(min=0.0)

        pnsun_cab = 1e3 * torch.trapz(self._e2phot(wlE, Esun * epsc_e * kchl_e), wlE, dim=-1)
        E_layer = 0.5 * (total_Emin[:, :-1, :] + total_Emin[:, 1:, :] + total_Eplu[:, :-1, :] + total_Eplu[:, 1:, :])
        pndif_cab = 1e3 * torch.trapz(
            self._e2phot(wlE, E_layer * epsc_e.unsqueeze(1) * kchl_e.unsqueeze(1)),
            wlE,
            dim=-1,
        )
        pnuc_cab = transport.absfs.unsqueeze(1) * pnsun_cab.view(batch, 1, 1) + pndif_cab.unsqueeze(-1)
        return pnuc_cab, pndif_cab

    def _simple_reabsorption_corrected_source(
        self,
        *,
        leafopt,
        leafbio: LeafBioBatch,
        excitation: torch.Tensor,
        lai: torch.Tensor,
        wlP: torch.Tensor,
        wlE: torch.Tensor,
        wlF: torch.Tensor,
    ) -> torch.Tensor:
        batch = leafopt.refl.shape[0]
        rho_e = self._sample_spectrum(leafopt.refl, wlP, wlE)
        tau_e = self._sample_spectrum(leafopt.tran, wlP, wlE)
        kchl_e = self._sample_spectrum(leafopt.kChlrel, wlP, wlE)
        epsc_e = (1.0 - rho_e - tau_e).clamp(min=0.0)
        absorbed_cab = 1e3 * torch.trapz(self._e2phot(wlE, excitation * epsc_e * kchl_e), wlE, dim=-1)
        fqe = self._expand_batch(leafbio.fqe, batch, device=excitation.device, dtype=excitation.dtype)
        poutfrc = fqe * lai * absorbed_cab
        return self._source_spectrum_from_poutfrc(poutfrc, wlP=wlP, wlF=wlF)

    def _layered_reabsorption_corrected_poutfrc(
        self,
        *,
        leafopt,
        leafbio: LeafBioBatch,
        Esun: torch.Tensor,
        total_Emin: torch.Tensor,
        total_Eplu: torch.Tensor,
        transport: LayeredCanopyTransfer,
        etau: torch.Tensor,
        etah: torch.Tensor,
        wlP: torch.Tensor,
        wlE: torch.Tensor,
    ) -> torch.Tensor:
        batch = leafopt.refl.shape[0]
        rho_e = self._sample_spectrum(leafopt.refl, wlP, wlE)
        tau_e = self._sample_spectrum(leafopt.tran, wlP, wlE)
        kchl_e = self._sample_spectrum(leafopt.kChlrel, wlP, wlE)
        epsc_e = (1.0 - rho_e - tau_e).clamp(min=0.0)

        pnsun_cab = 1e3 * torch.trapz(self._e2phot(wlE, Esun * epsc_e * kchl_e), wlE, dim=-1)
        E_layer = 0.5 * (total_Emin[:, :-1, :] + total_Emin[:, 1:, :] + total_Eplu[:, :-1, :] + total_Eplu[:, 1:, :])
        pndif_cab = 1e3 * torch.trapz(
            self._e2phot(wlE, E_layer * epsc_e.unsqueeze(1) * kchl_e.unsqueeze(1)),
            wlE,
            dim=-1,
        )

        eta_weights = transport.lidf_azimuth.unsqueeze(1)
        pnuc_cab = transport.absfs.unsqueeze(1) * pnsun_cab.view(batch, 1, 1) + pndif_cab.unsqueeze(-1)
        sunlit_eta_abs = (etau * eta_weights * pnuc_cab).sum(dim=-1)
        shaded_eta_abs = (etah * eta_weights * pndif_cab.unsqueeze(-1)).sum(dim=-1)

        Ps = transport.Ps[:, : transport.nlayers]
        fqe = self._expand_batch(leafbio.fqe, batch, device=Esun.device, dtype=Esun.dtype)
        return fqe * transport.iLAI * torch.sum(Ps * sunlit_eta_abs + (1.0 - Ps) * shaded_eta_abs, dim=-1)

    def _layered_source_poutfrc(
        self,
        *,
        leafopt,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun: torch.Tensor,
        Esky: torch.Tensor,
        hotspot: Optional[torch.Tensor],
        lidf: Optional[torch.Tensor],
        nlayers: int,
        etau: Optional[torch.Tensor],
        etah: Optional[torch.Tensor],
        Pnu_Cab: Optional[torch.Tensor],
        Pnh_Cab: Optional[torch.Tensor],
        wlP: torch.Tensor,
        wlE: torch.Tensor,
    ) -> torch.Tensor:
        hotspot_value = hotspot if hotspot is not None else torch.full_like(
            torch.as_tensor(lai),
            self.reflectance_model.default_hotspot,
        )
        rho_e = self._sample_spectrum(leafopt.refl, wlP, wlE)
        tau_e = self._sample_spectrum(leafopt.tran, wlP, wlE)
        soil_e = self._sample_spectrum(soil_refl, wlP, wlE)
        transport = self.layered_transport.build(
            rho_e,
            tau_e,
            soil_e,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot_value,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nlayers,
        )
        etau_orient = self._prepare_etau(etau, transport)
        etah_orient = self._prepare_etah(etah, transport)
        if Pnu_Cab is not None and Pnh_Cab is not None:
            return self._poutfrc_from_absorbed_cab_profiles(
                leafbio=leafbio,
                transport=transport,
                etau=etau_orient,
                etah=etah_orient,
                Pnu_Cab=Pnu_Cab,
                Pnh_Cab=Pnh_Cab,
            )
        direct = self.layered_transport.flux_profiles(transport, Esun, torch.zeros_like(Esky))
        diffuse = self.layered_transport.flux_profiles(transport, torch.zeros_like(Esun), Esky)
        total_Emin = direct.Emin_ + diffuse.Emin_
        total_Eplu = direct.Eplu_ + diffuse.Eplu_
        return self._layered_reabsorption_corrected_poutfrc(
            leafopt=leafopt,
            leafbio=leafbio,
            Esun=Esun,
            total_Emin=total_Emin,
            total_Eplu=total_Eplu,
            transport=transport,
            etau=etau_orient,
            etah=etah_orient,
            wlP=wlP,
            wlE=wlE,
        )

    def _poutfrc_from_absorbed_cab_profiles(
        self,
        *,
        leafbio: LeafBioBatch,
        transport: LayeredCanopyTransfer,
        etau: torch.Tensor,
        etah: torch.Tensor,
        Pnu_Cab: torch.Tensor,
        Pnh_Cab: torch.Tensor,
    ) -> torch.Tensor:
        batch = transport.Ps.shape[0]
        device = transport.Ps.device
        dtype = transport.Ps.dtype
        eta_weights = transport.lidf_azimuth.unsqueeze(1)
        Ps = transport.Ps[:, : transport.nlayers]
        fqe = self._expand_batch(leafbio.fqe, batch, device=device, dtype=dtype)

        pnu_tensor = torch.as_tensor(Pnu_Cab, device=device, dtype=dtype)
        if pnu_tensor.ndim == 3 and pnu_tensor.shape[:2] == (batch, transport.nlayers):
            sunlit_term = (etau * eta_weights * pnu_tensor).sum(dim=-1)
        else:
            pnu = self._prepare_layer_profile(Pnu_Cab, transport)
            sunlit_eta = (etau * eta_weights).sum(dim=-1)
            sunlit_term = sunlit_eta * pnu

        pnh_tensor = torch.as_tensor(Pnh_Cab, device=device, dtype=dtype)
        if pnh_tensor.ndim == 3 and pnh_tensor.shape[:2] == (batch, transport.nlayers):
            shaded_term = (etah * eta_weights * pnh_tensor).sum(dim=-1)
        else:
            pnh = self._prepare_layer_profile(Pnh_Cab, transport)
            shaded_eta = (etah * eta_weights).sum(dim=-1)
            shaded_term = shaded_eta * pnh

        return fqe * transport.iLAI * torch.sum(Ps * sunlit_term + (1.0 - Ps) * shaded_term, dim=-1)

    def _source_spectrum_from_poutfrc(self, poutfrc: torch.Tensor, *, wlP: torch.Tensor, wlF: torch.Tensor) -> torch.Tensor:
        batch = poutfrc.shape[0]
        phi = self.reflectance_model.fluspect.optipar.phi.unsqueeze(0).expand(batch, -1)
        phi_em = self._sample_spectrum(phi, wlP, wlF)
        ep = self._ephoton(wlF).unsqueeze(0)
        return 1e-3 * ep * poutfrc.unsqueeze(-1) * phi_em

    def _matlab_spline_interp1(self, source_x: torch.Tensor, source_y: torch.Tensor, target_x: torch.Tensor) -> torch.Tensor:
        if source_x.numel() == target_x.numel() and torch.allclose(source_x, target_x):
            return source_y
        source_x = source_x.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype).contiguous()
        source_y = source_y.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype).contiguous()
        target_x = target_x.to(self.reflectance_model.fluspect.device, self.reflectance_model.fluspect.dtype).contiguous()
        if source_x.numel() < 4:
            return self._interp1d(source_x, source_y, target_x)

        h = source_x[1:] - source_x[:-1]
        n = source_x.numel()

        system = torch.zeros((n, n), device=source_x.device, dtype=source_x.dtype)
        rhs = torch.zeros((*source_y.shape[:-1], n), device=source_y.device, dtype=source_y.dtype)

        system[0, 0] = -h[1]
        system[0, 1] = h[0] + h[1]
        system[0, 2] = -h[0]
        system[-1, -3] = h[-1]
        system[-1, -2] = -(h[-2] + h[-1])
        system[-1, -1] = h[-2]

        slopes = (source_y[..., 1:] - source_y[..., :-1]) / h.view(*([1] * (source_y.ndim - 1)), -1)
        rhs[..., 1:-1] = 6.0 * (slopes[..., 1:] - slopes[..., :-1])

        for row in range(1, n - 1):
            system[row, row - 1] = h[row - 1]
            system[row, row] = 2.0 * (h[row - 1] + h[row])
            system[row, row + 1] = h[row]

        second = torch.linalg.solve(system, rhs.reshape(-1, n).transpose(0, 1)).transpose(0, 1).reshape(rhs.shape)

        idx = torch.bucketize(target_x, source_x) - 1
        idx = idx.clamp(0, n - 2)
        x0 = source_x[idx]
        x1 = source_x[idx + 1]
        h_eval = x1 - x0
        left = (x1 - target_x) / h_eval
        right = (target_x - x0) / h_eval

        expand_idx = idx.view(*([1] * (source_y.ndim - 1)), -1).expand(*source_y.shape[:-1], -1)
        y0 = source_y.gather(-1, expand_idx)
        y1 = source_y.gather(-1, expand_idx + 1)
        m0 = second.gather(-1, expand_idx)
        m1 = second.gather(-1, expand_idx + 1)
        h_batch = h_eval.view(*([1] * (source_y.ndim - 1)), -1)

        return (
            m0 * (left.view(*([1] * (source_y.ndim - 1)), -1) ** 3) * h_batch**2 / 6.0
            + m1 * (right.view(*([1] * (source_y.ndim - 1)), -1) ** 3) * h_batch**2 / 6.0
            + (y0 - m0 * h_batch**2 / 6.0) * left.view(*([1] * (source_y.ndim - 1)), -1)
            + (y1 - m1 * h_batch**2 / 6.0) * right.view(*([1] * (source_y.ndim - 1)), -1)
        )

    def _scope_sigmaf(self, LoF: torch.Tensor, EoutFrc: torch.Tensor, wlF: torch.Tensor) -> torch.Tensor:
        raw = (LoF * torch.pi) / EoutFrc.clamp(min=1e-12)
        coarse_wlF = wlF[::4]
        coarse_raw = raw[..., ::4]
        return self._sample_spectrum(coarse_raw, coarse_wlF, wlF)

    def _prepare_layer_profile(self, value: torch.Tensor | float, transfer: LayeredCanopyTransfer) -> torch.Tensor:
        batch = transfer.Ps.shape[0]
        nl = transfer.nlayers
        device = transfer.Ps.device
        dtype = transfer.Ps.dtype
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.view(1, 1).expand(batch, nl)
        if tensor.ndim == 1:
            if tensor.shape[0] == batch:
                return tensor.view(batch, 1).expand(batch, nl)
            if tensor.shape[0] == nl:
                return tensor.view(1, nl).expand(batch, nl)
        if tensor.ndim == 2:
            if tensor.shape == (batch, nl):
                return tensor
            if tensor.shape == (1, nl):
                return tensor.expand(batch, nl)
            if tensor.shape == (batch, 1):
                return tensor.expand(batch, nl)
        raise ValueError(f"Layer profiles must broadcast to shape ({batch}, {nl})")

    def _run_leaf_biochemistry(
        self,
        *,
        biochemistry: LeafBiochemistryInputs,
        Q: torch.Tensor,
        Cs: torch.Tensor,
        T: torch.Tensor,
        eb: torch.Tensor,
        Oa: torch.Tensor,
        p: torch.Tensor,
        fV: torch.Tensor,
        options: Optional[BiochemicalOptions],
        target_shape: torch.Size,
    ) -> LeafBiochemistryResult:
        flat_inputs = self._flatten_biochemistry_inputs(biochemistry, target_shape)
        result = self.leaf_biochemistry(
            flat_inputs,
            LeafMeteo(
                Q=Q.reshape(-1),
                Cs=Cs.reshape(-1),
                T=T.reshape(-1),
                eb=eb.reshape(-1),
                Oa=Oa.reshape(-1),
                p=p.reshape(-1),
            ),
            options=options,
            fV=fV.reshape(-1),
        )
        return self._reshape_biochemistry_result(result, target_shape)

    def _flatten_biochemistry_inputs(
        self,
        biochemistry: LeafBiochemistryInputs,
        target_shape: torch.Size,
    ) -> LeafBiochemistryInputs:
        device = self.reflectance_model.fluspect.device
        dtype = self.reflectance_model.fluspect.dtype
        numel = int(torch.tensor(target_shape).prod().item())

        def _flatten(value: torch.Tensor | float | None) -> torch.Tensor | None:
            if value is None:
                return None
            tensor = torch.as_tensor(value, device=device, dtype=dtype)
            if tensor.ndim == 0:
                return tensor.repeat(numel)
            if tensor.shape == target_shape:
                return tensor.reshape(-1)
            if len(target_shape) == 2:
                batch, nl = target_shape
                if tensor.ndim == 1:
                    if tensor.shape[0] == batch:
                        return tensor.view(batch, 1).expand(batch, nl).reshape(-1)
                    if tensor.shape[0] == nl:
                        return tensor.view(1, nl).expand(batch, nl).reshape(-1)
                if tensor.ndim == 2:
                    if tensor.shape == (1, nl):
                        return tensor.expand(batch, nl).reshape(-1)
                    if tensor.shape == (batch, 1):
                        return tensor.expand(batch, nl).reshape(-1)
            if len(target_shape) == 3:
                batch, nl, nori = target_shape
                if tensor.ndim == 1:
                    if tensor.shape[0] == batch:
                        return tensor.view(batch, 1, 1).expand(batch, nl, nori).reshape(-1)
                    if tensor.shape[0] == nl:
                        return tensor.view(1, nl, 1).expand(batch, nl, nori).reshape(-1)
                    if tensor.shape[0] == nori:
                        return tensor.view(1, 1, nori).expand(batch, nl, nori).reshape(-1)
                if tensor.ndim == 2:
                    if tensor.shape == (batch, nl):
                        return tensor.unsqueeze(-1).expand(batch, nl, nori).reshape(-1)
                    if tensor.shape == (1, nl):
                        return tensor.view(1, nl, 1).expand(batch, nl, nori).reshape(-1)
                    if tensor.shape == (batch, 1):
                        return tensor.view(batch, 1, 1).expand(batch, nl, nori).reshape(-1)
                    if tensor.shape == (batch, nori):
                        return tensor.unsqueeze(1).expand(batch, nl, nori).reshape(-1)
                if tensor.ndim == 3:
                    if tensor.shape == (batch, nl, nori):
                        return tensor.reshape(-1)
                    if tensor.shape == (1, nl, nori):
                        return tensor.expand(batch, nl, nori).reshape(-1)
            raise ValueError(f"Leaf biochemistry inputs must broadcast to shape {tuple(target_shape)}")

        return LeafBiochemistryInputs(
            Vcmax25=_flatten(biochemistry.Vcmax25),
            BallBerrySlope=_flatten(biochemistry.BallBerrySlope),
            Type=biochemistry.Type,
            BallBerry0=_flatten(biochemistry.BallBerry0),
            RdPerVcmax25=_flatten(biochemistry.RdPerVcmax25),
            Kn0=_flatten(biochemistry.Kn0),
            Knalpha=_flatten(biochemistry.Knalpha),
            Knbeta=_flatten(biochemistry.Knbeta),
            stressfactor=_flatten(biochemistry.stressfactor),
            g_m=_flatten(biochemistry.g_m),
            TDP=biochemistry.TDP,
        )

    def _reshape_biochemistry_result(
        self,
        result: LeafBiochemistryResult,
        target_shape: torch.Size,
    ) -> LeafBiochemistryResult:
        data = {}
        for field in LeafBiochemistryResult.__dataclass_fields__:
            value = getattr(result, field)
            if isinstance(value, torch.Tensor):
                data[field] = value.reshape(target_shape)
            else:
                data[field] = value
        return LeafBiochemistryResult(**data)

    def _prepare_etau(self, etau: Optional[torch.Tensor], transfer: LayeredCanopyTransfer) -> torch.Tensor:
        batch = transfer.Ps.shape[0]
        nl = transfer.nlayers
        ninc = transfer.litab.numel()
        nazi = transfer.lazitab.numel()
        nori = transfer.lidf_azimuth.shape[1]
        device = transfer.Ps.device
        dtype = transfer.Ps.dtype
        if etau is None:
            return torch.ones((batch, nl, nori), device=device, dtype=dtype)
        tensor = torch.as_tensor(etau, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.view(1, 1, 1).expand(batch, nl, nori)
        if tensor.ndim == 1:
            if tensor.shape[0] == batch:
                return tensor.view(batch, 1, 1).expand(batch, nl, nori)
            if tensor.shape[0] == nl:
                return tensor.view(1, nl, 1).expand(batch, nl, nori)
        if tensor.ndim == 2:
            if tensor.shape == (batch, nl):
                return tensor.unsqueeze(-1).expand(batch, nl, nori)
            if tensor.shape[0] == 1 and tensor.shape[1] == nl:
                return tensor.unsqueeze(-1).expand(batch, nl, nori)
        if tensor.ndim == 3 and tensor.shape[-1] == nori:
            if tensor.shape[0] == 1 and tensor.shape[1] == nl:
                return tensor.expand(batch, nl, nori)
            if tensor.shape[0] == batch and tensor.shape[1] == nl:
                return tensor
        if tensor.ndim == 3 and tensor.shape == (nl, ninc, nazi):
            return tensor.reshape(1, nl, nori).expand(batch, nl, nori)
        if tensor.ndim == 4 and tensor.shape[-2:] == (ninc, nazi):
            if tensor.shape[0] == 1 and tensor.shape[1] == nl:
                return tensor.reshape(1, nl, nori).expand(batch, nl, nori)
            if tensor.shape[0] == batch and tensor.shape[1] == nl:
                return tensor.reshape(batch, nl, nori)
        raise ValueError(f"etau must broadcast to shape ({batch}, {nl}, {nori}), got {tuple(tensor.shape)}")

    def _prepare_etah(self, etah: Optional[torch.Tensor], transfer: LayeredCanopyTransfer) -> torch.Tensor:
        batch = transfer.Ps.shape[0]
        nl = transfer.nlayers
        ninc = transfer.litab.numel()
        nazi = transfer.lazitab.numel()
        nori = transfer.lidf_azimuth.shape[1]
        device = transfer.Ps.device
        dtype = transfer.Ps.dtype
        if etah is None:
            return torch.ones((batch, nl, nori), device=device, dtype=dtype)
        tensor = torch.as_tensor(etah, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.view(1, 1, 1).expand(batch, nl, nori)
        if tensor.ndim == 1:
            if tensor.shape[0] == batch:
                return tensor.view(batch, 1, 1).expand(batch, nl, nori)
            if tensor.shape[0] == nl:
                return tensor.view(1, nl, 1).expand(batch, nl, nori)
        if tensor.ndim == 2:
            if tensor.shape == (batch, nl):
                return tensor.unsqueeze(-1).expand(batch, nl, nori)
            if tensor.shape[0] == 1 and tensor.shape[1] == nl:
                return tensor.unsqueeze(-1).expand(batch, nl, nori)
        if tensor.ndim == 3 and tensor.shape[-1] == nori:
            if tensor.shape[0] == 1 and tensor.shape[1] == nl:
                return tensor.expand(batch, nl, nori)
            if tensor.shape[0] == batch and tensor.shape[1] == nl:
                return tensor
        if tensor.ndim == 3 and tensor.shape == (nl, ninc, nazi):
            return tensor.reshape(1, nl, nori).expand(batch, nl, nori)
        if tensor.ndim == 4 and tensor.shape[-2:] == (ninc, nazi):
            if tensor.shape[0] == 1 and tensor.shape[1] == nl:
                return tensor.reshape(1, nl, nori).expand(batch, nl, nori)
            if tensor.shape[0] == batch and tensor.shape[1] == nl:
                return tensor.reshape(batch, nl, nori)
        raise ValueError(f"etah must broadcast to shape ({batch}, {nl}, {nori}), got {tuple(tensor.shape)}")

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
