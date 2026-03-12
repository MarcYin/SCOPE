from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from ..biochem import BiochemicalOptions, LeafBiochemistryInputs, LeafBiochemistryResult
from ..canopy.fluorescence import CanopyFluorescenceModel, CanopyFluorescenceResult
from ..canopy.foursail import FourSAILModel
from ..canopy.layered_rt import LayeredCanopyTransfer
from ..canopy.reflectance import CanopyReflectanceModel
from ..canopy.thermal import CanopyThermalRadianceModel, CanopyThermalRadianceResult, ThermalOptics, default_thermal_wavelengths
from ..spectral.fluspect import LeafBioBatch
from ..spectral.soil import SoilEmpiricalParams
from .fluxes import HeatFluxInputs, ResistanceInputs, aerodynamic_resistances, heat_fluxes


@dataclass(slots=True)
class EnergyBalanceMeteo:
    Ta: torch.Tensor | float
    ea: torch.Tensor | float
    Ca: torch.Tensor | float
    Oa: torch.Tensor | float
    p: torch.Tensor | float
    z: torch.Tensor | float
    u: torch.Tensor | float
    L: torch.Tensor | float = -1e6


@dataclass(slots=True)
class EnergyBalanceCanopy:
    Cd: torch.Tensor | float
    rwc: torch.Tensor | float
    z0m: torch.Tensor | float
    d: torch.Tensor | float
    h: torch.Tensor | float
    kV: torch.Tensor | float = 0.0
    fV: Optional[torch.Tensor | float] = None


@dataclass(slots=True)
class EnergyBalanceSoil:
    rss: torch.Tensor | float
    rbs: torch.Tensor | float
    thermal_optics: ThermalOptics = field(default_factory=ThermalOptics)
    soil_heat_method: int = 2
    GAM: torch.Tensor | float = 0.0
    Tsold: Optional[torch.Tensor] = None
    dt_seconds: Optional[torch.Tensor | float] = None


@dataclass(slots=True)
class EnergyBalanceOptions:
    max_iter: int = 100
    max_energy_error: float = 1.0
    monin_obukhov: bool = True
    initial_shaded_leaf_offset: float = 0.1
    initial_sunlit_leaf_offset: float = 0.3
    initial_soil_offset: float = 3.0


@dataclass(slots=True)
class ShortwaveRadiationResult:
    transfer: LayeredCanopyTransfer
    Rnuc: torch.Tensor
    Rnhc: torch.Tensor
    Rnus: torch.Tensor
    Rnhs: torch.Tensor
    Pnu_Cab: torch.Tensor
    Pnh_Cab: torch.Tensor


@dataclass(slots=True)
class IncidentThermalRadiationResult:
    Rnuc: torch.Tensor
    Rnhc: torch.Tensor
    Rnus: torch.Tensor
    Rnhs: torch.Tensor


@dataclass(slots=True)
class CanopyEnergyBalanceResult:
    sunlit: LeafBiochemistryResult
    shaded: LeafBiochemistryResult
    sunlit_Cs_input: torch.Tensor
    shaded_Cs_input: torch.Tensor
    sunlit_eb_input: torch.Tensor
    shaded_eb_input: torch.Tensor
    sunlit_T_input: torch.Tensor
    shaded_T_input: torch.Tensor
    Pnu_Cab: torch.Tensor
    Pnh_Cab: torch.Tensor
    Rnuc_sw: torch.Tensor
    Rnhc_sw: torch.Tensor
    Rnus_sw: torch.Tensor
    Rnhs_sw: torch.Tensor
    Rnuct: torch.Tensor
    Rnhct: torch.Tensor
    Rnust: torch.Tensor
    Rnhst: torch.Tensor
    canopyemis: torch.Tensor
    Rnuc: torch.Tensor
    Rnhc: torch.Tensor
    Rnus: torch.Tensor
    Rnhs: torch.Tensor
    lEcu: torch.Tensor
    lEch: torch.Tensor
    lEsu: torch.Tensor
    lEsh: torch.Tensor
    Hcu: torch.Tensor
    Hch: torch.Tensor
    Hsu: torch.Tensor
    Hsh: torch.Tensor
    Gsu: torch.Tensor
    Gsh: torch.Tensor
    Csu: torch.Tensor
    Csh: torch.Tensor
    ebu: torch.Tensor
    ebh: torch.Tensor
    Tcu: torch.Tensor
    Tch: torch.Tensor
    Tsu: torch.Tensor
    Tsh: torch.Tensor
    L: torch.Tensor
    ustar: torch.Tensor
    raa: torch.Tensor
    rawc: torch.Tensor
    raws: torch.Tensor
    rac: torch.Tensor
    ras: torch.Tensor
    Rnctot: torch.Tensor
    lEctot: torch.Tensor
    Hctot: torch.Tensor
    Actot: torch.Tensor
    Tcave: torch.Tensor
    Rnstot: torch.Tensor
    lEstot: torch.Tensor
    Hstot: torch.Tensor
    Gtot: torch.Tensor
    Tsave: torch.Tensor
    Rntot: torch.Tensor
    lEtot: torch.Tensor
    Htot: torch.Tensor
    max_error: torch.Tensor
    converged: torch.Tensor
    counter: torch.Tensor
    Tsold: Optional[torch.Tensor] = None


@dataclass(slots=True)
class CanopyEnergyBalanceFluorescenceResult:
    energy: CanopyEnergyBalanceResult
    fluorescence: CanopyFluorescenceResult


@dataclass(slots=True)
class CanopyEnergyBalanceThermalResult:
    energy: CanopyEnergyBalanceResult
    thermal: CanopyThermalRadianceResult


class CanopyEnergyBalanceModel:
    """Tensor-native SCOPE-style energy balance closure for layered canopies."""

    def __init__(self, reflectance_model: CanopyReflectanceModel) -> None:
        self.reflectance_model = reflectance_model
        self.fluorescence_model = CanopyFluorescenceModel(reflectance_model)
        self.thermal_model = CanopyThermalRadianceModel(reflectance_model)
        self.rhoa = 1.2047
        self.cp = 1004.0
        self.kappa = 0.4
        self.g = 9.81
        self.sigma_sb = 5.67e-8
        self.MH2O = 18.0
        self.Mair = 28.96

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
    ) -> "CanopyEnergyBalanceModel":
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

    def solve(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        *,
        Esun_lw: Optional[torch.Tensor] = None,
        Esky_lw: Optional[torch.Tensor] = None,
        wlT: Optional[torch.Tensor] = None,
        meteo: EnergyBalanceMeteo,
        canopy: EnergyBalanceCanopy,
        soil: EnergyBalanceSoil,
        options: Optional[EnergyBalanceOptions] = None,
        biochem_options: Optional[BiochemicalOptions] = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
        Tcu0: Optional[torch.Tensor] = None,
        Tch0: Optional[torch.Tensor] = None,
        Tsu0: Optional[torch.Tensor] = None,
        Tsh0: Optional[torch.Tensor] = None,
    ) -> CanopyEnergyBalanceResult:
        opts = options or EnergyBalanceOptions()
        device = self.reflectance_model.fluspect.device
        dtype = self.reflectance_model.fluspect.dtype

        leafopt = self.reflectance_model.fluspect(leafbio)
        batch = leafopt.refl.shape[0]
        lai_tensor = self._expand_batch(lai, batch, device=device, dtype=dtype)
        ta = self._expand_batch(meteo.Ta, batch, device=device, dtype=dtype)
        ea = self._expand_batch(meteo.ea, batch, device=device, dtype=dtype)
        ca = self._expand_batch(meteo.Ca, batch, device=device, dtype=dtype)
        oa = self._expand_batch(meteo.Oa, batch, device=device, dtype=dtype)
        pressure = self._expand_batch(meteo.p, batch, device=device, dtype=dtype)
        L = self._expand_batch(meteo.L, batch, device=device, dtype=dtype)
        hotspot_value = hotspot if hotspot is not None else torch.full_like(lai_tensor, self.reflectance_model.default_hotspot)

        nl = self._resolve_nlayers(nlayers, canopy.fV, Tcu0, Tch0)
        shortwave = self._shortwave_radiation(
            leafopt=leafopt,
            soil_refl=soil_refl,
            lai=lai_tensor,
            tts=tts,
            tto=tto,
            psi=psi,
            Esun_sw=Esun_sw,
            Esky_sw=Esky_sw,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            thermal_optics=soil.thermal_optics,
            hotspot=hotspot_value,
            lidf=lidf,
            nlayers=nl,
            wlT=wlT,
        )

        fV = self._fV_profile(canopy, shortwave.transfer, batch)
        e_to_q = (self.MH2O / self.Mair) / pressure

        Tch = self._initial_layer_state(Tch0, ta + opts.initial_shaded_leaf_offset, batch, nl, device=device, dtype=dtype)
        Tcu = self._initial_layer_state(Tcu0, ta + opts.initial_sunlit_leaf_offset, batch, nl, device=device, dtype=dtype)
        Tsh = self._expand_batch(Tsh0 if Tsh0 is not None else ta + opts.initial_soil_offset, batch, device=device, dtype=dtype)
        Tsu = self._expand_batch(Tsu0 if Tsu0 is not None else ta + opts.initial_soil_offset, batch, device=device, dtype=dtype)
        Cch = ca.view(batch, 1).expand(batch, nl)
        Ccu = ca.view(batch, 1).expand(batch, nl)
        ech = ea.view(batch, 1).expand(batch, nl)
        ecu = ea.view(batch, 1).expand(batch, nl)

        Wc = 1.0
        active = torch.ones(batch, device=device, dtype=torch.bool)
        sunlit = None
        shaded = None
        thermal = None
        resistances = None
        lEch = Hch = lEcu = Hcu = None
        lEs = Hs = G = None
        max_error = torch.full((batch,), torch.inf, device=device, dtype=dtype)
        sunlit_Cs_input = shaded_Cs_input = None
        sunlit_eb_input = shaded_eb_input = None
        sunlit_T_input = shaded_T_input = None

        for iteration in range(1, opts.max_iter + 1):
            iterating = active.clone()
            sunlit_Cs_input = Ccu.clone()
            shaded_Cs_input = Cch.clone()
            sunlit_eb_input = ecu.clone()
            shaded_eb_input = ech.clone()
            sunlit_T_input = Tcu.clone()
            shaded_T_input = Tch.clone()
            sunlit = self.fluorescence_model._run_leaf_biochemistry(
                biochemistry=biochemistry,
                Q=shortwave.Pnu_Cab,
                Cs=Ccu,
                T=Tcu,
                eb=ecu,
                Oa=oa.view(batch, 1).expand(batch, nl),
                p=pressure.view(batch, 1).expand(batch, nl),
                fV=fV,
                options=biochem_options,
                target_shape=shortwave.Pnu_Cab.shape,
            )
            shaded = self.fluorescence_model._run_leaf_biochemistry(
                biochemistry=biochemistry,
                Q=shortwave.Pnh_Cab,
                Cs=Cch,
                T=Tch,
                eb=ech,
                Oa=oa.view(batch, 1).expand(batch, nl),
                p=pressure.view(batch, 1).expand(batch, nl),
                fV=fV,
                options=biochem_options,
                target_shape=shortwave.Pnh_Cab.shape,
            )

            resistances = aerodynamic_resistances(
                ResistanceInputs(
                    LAI=lai_tensor,
                    Cd=canopy.Cd,
                    rwc=canopy.rwc,
                    z0m=canopy.z0m,
                    d=canopy.d,
                    h=canopy.h,
                    z=meteo.z,
                    u=meteo.u,
                    L=L,
                    rbs=soil.rbs,
                ),
                kappa=self.kappa,
            )
            rac = (lai_tensor + 1.0) * (resistances.raa + resistances.rawc)
            ras = (lai_tensor + 1.0) * (resistances.raa + resistances.raws)

            thermal = self.thermal_model.integrated_balance(
                lai=lai_tensor,
                tts=tts,
                tto=tto,
                psi=psi,
                Tcu=Tcu,
                Tch=Tch,
                Tsu=Tsu,
                Tsh=Tsh,
                thermal_optics=soil.thermal_optics,
                hotspot=hotspot_value,
                lidf=lidf,
                nlayers=nl,
            )
            Rnhc = shortwave.Rnhc + thermal.Rnhct
            Rnuc = shortwave.Rnuc + thermal.Rnuct
            Rnhs = shortwave.Rnhs + thermal.Rnhst
            Rnus = shortwave.Rnus + thermal.Rnust

            shaded_flux = heat_fluxes(
                HeatFluxInputs(
                    ra=rac.view(batch, 1),
                    rs=shaded.rcw,
                    Tc=Tch,
                    ea=ea.view(batch, 1),
                    Ta=ta.view(batch, 1),
                    e_to_q=e_to_q.view(batch, 1),
                    Ca=ca.view(batch, 1),
                    Ci=shaded.Ci,
                ),
                rhoa=self.rhoa,
                cp=self.cp,
            )
            sunlit_flux = heat_fluxes(
                HeatFluxInputs(
                    ra=rac.view(batch, 1),
                    rs=sunlit.rcw,
                    Tc=Tcu,
                    ea=ea.view(batch, 1),
                    Ta=ta.view(batch, 1),
                    e_to_q=e_to_q.view(batch, 1),
                    Ca=ca.view(batch, 1),
                    Ci=sunlit.Ci,
                ),
                rhoa=self.rhoa,
                cp=self.cp,
            )
            soil_t = torch.stack([Tsh, Tsu], dim=1)
            soil_flux = heat_fluxes(
                HeatFluxInputs(
                    ra=ras.view(batch, 1),
                    rs=self._prepare_soil_profile(soil.rss, batch, device=device, dtype=dtype),
                    Tc=soil_t,
                    ea=ea.view(batch, 1),
                    Ta=ta.view(batch, 1),
                    e_to_q=e_to_q.view(batch, 1),
                    Ca=ca.view(batch, 1),
                    Ci=ca.view(batch, 1),
                ),
                rhoa=self.rhoa,
                cp=self.cp,
            )

            lEch = shaded_flux.latent_heat
            Hch = shaded_flux.sensible_heat
            lEcu = sunlit_flux.latent_heat
            Hcu = sunlit_flux.sensible_heat
            lEs = soil_flux.latent_heat
            Hs = soil_flux.sensible_heat

            ps_bottom = shortwave.transfer.Ps[:, -1]
            Hctot = self._aggregate_canopy(shortwave.transfer, Hcu, Hch, lai_tensor)
            Hstot = (1.0 - ps_bottom) * Hs[:, 0] + ps_bottom * Hs[:, 1]
            Htot = Hctot + Hstot
            if opts.monin_obukhov:
                L_new = self._monin_obukhov_length(ta, resistances.ustar, Htot)
            else:
                L_new = L

            G, dG = self._soil_heat_flux(
                Rnhs=Rnhs,
                Rnus=Rnus,
                Tsh=Tsh,
                Tsu=Tsu,
                soil=soil,
                device=device,
                dtype=dtype,
            )

            EBerch = Rnhc - lEch - Hch
            EBercu = Rnuc - lEcu - Hcu
            EBers = torch.stack([Rnhs, Rnus], dim=1) - lEs - Hs - G

            max_error = torch.stack(
                [
                    torch.amax(torch.abs(EBercu), dim=1),
                    torch.amax(torch.abs(EBerch), dim=1),
                    torch.amax(torch.abs(EBers), dim=1),
                ],
                dim=1,
            ).amax(dim=1)

            iterating_2d = iterating.view(batch, 1)
            Cch = torch.where(iterating_2d, shaded_flux.Cc, Cch)
            Ccu = torch.where(iterating_2d, sunlit_flux.Cc, Ccu)
            ech = torch.where(iterating_2d, shaded_flux.ec, ech)
            ecu = torch.where(iterating_2d, sunlit_flux.ec, ecu)
            L = torch.where(iterating, L_new, L)

            active = max_error > opts.max_energy_error
            if not bool(active.any().item()) or iteration == opts.max_iter:
                break

            if iteration == 10:
                Wc = 0.8
            elif iteration == 20:
                Wc = 0.6

            rac_term = rac.view(batch, 1).clamp(min=1e-12)
            ras_term = ras.view(batch, 1).clamp(min=1e-12)
            epsc = self._leaf_thermal_emissivity(soil.thermal_optics, batch, device=device, dtype=dtype).view(batch, 1)
            epss = self._soil_thermal_emissivity(soil.thermal_optics, batch, device=device, dtype=dtype).view(batch, 1)

            Tch_new = Tch + Wc * EBerch / (
                (self.rhoa * self.cp) / rac_term
                + self.rhoa * shaded_flux.lambda_evap * e_to_q.view(batch, 1) * shaded_flux.slope_satvap / (rac_term + shaded.rcw)
                + 4.0 * epsc * self.sigma_sb * (Tch + 273.15) ** 3
            )
            Tcu_new = Tcu + Wc * EBercu / (
                (self.rhoa * self.cp) / rac_term
                + self.rhoa * sunlit_flux.lambda_evap * e_to_q.view(batch, 1) * sunlit_flux.slope_satvap / (rac_term + sunlit.rcw)
                + 4.0 * epsc * self.sigma_sb * (Tcu + 273.15) ** 3
            )
            soil_rss = self._prepare_soil_profile(soil.rss, batch, device=device, dtype=dtype)
            Tsoil_new = soil_t + Wc * EBers / (
                (self.rhoa * self.cp) / ras_term
                + self.rhoa * soil_flux.lambda_evap * e_to_q.view(batch, 1) * soil_flux.slope_satvap / (ras_term + soil_rss)
                + 4.0 * epss * self.sigma_sb * (soil_t + 273.15) ** 3
                + dG
            )
            active_2d = active.view(batch, 1)
            Tch = torch.where(active_2d, Tch_new, Tch)
            Tcu = torch.where(active_2d, Tcu_new, Tcu)
            soil_t = torch.where(active_2d, Tsoil_new, soil_t)
            Tch = torch.where(torch.abs(Tch) > 100.0, ta.view(batch, 1), Tch)
            Tcu = torch.where(torch.abs(Tcu) > 100.0, ta.view(batch, 1), Tcu)
            soil_t = torch.where(torch.abs(soil_t) > 100.0, ta.view(batch, 1), soil_t)
            Tsh = soil_t[:, 0]
            Tsu = soil_t[:, 1]

        if (
            sunlit is None
            or shaded is None
            or thermal is None
            or resistances is None
            or lEch is None
            or Hch is None
            or lEcu is None
            or Hcu is None
            or lEs is None
            or Hs is None
            or G is None
            or sunlit_Cs_input is None
            or shaded_Cs_input is None
            or sunlit_eb_input is None
            or shaded_eb_input is None
            or sunlit_T_input is None
            or shaded_T_input is None
        ):
            raise RuntimeError("Energy balance did not execute")

        ps_bottom = shortwave.transfer.Ps[:, -1]
        Rnhc = shortwave.Rnhc + thermal.Rnhct
        Rnuc = shortwave.Rnuc + thermal.Rnuct
        Rnhs = shortwave.Rnhs + thermal.Rnhst
        Rnus = shortwave.Rnus + thermal.Rnust

        Rnctot = self._aggregate_canopy(shortwave.transfer, Rnuc, Rnhc, lai_tensor)
        lEctot = self._aggregate_canopy(shortwave.transfer, lEcu, lEch, lai_tensor)
        Hctot = self._aggregate_canopy(shortwave.transfer, Hcu, Hch, lai_tensor)
        Actot = self._aggregate_canopy(shortwave.transfer, sunlit.A, shaded.A, lai_tensor)
        Tcave = self._aggregate_canopy(shortwave.transfer, Tcu, Tch, torch.ones_like(lai_tensor))
        Rnstot = (1.0 - ps_bottom) * Rnhs + ps_bottom * Rnus
        lEstot = (1.0 - ps_bottom) * lEs[:, 0] + ps_bottom * lEs[:, 1]
        Hstot = (1.0 - ps_bottom) * Hs[:, 0] + ps_bottom * Hs[:, 1]
        Gtot = (1.0 - ps_bottom) * G[:, 0] + ps_bottom * G[:, 1]
        Tsave = (1.0 - ps_bottom) * Tsh + ps_bottom * Tsu
        counter = torch.full((batch,), iteration, device=device, dtype=torch.int64)

        return CanopyEnergyBalanceResult(
            sunlit=sunlit,
            shaded=shaded,
            sunlit_Cs_input=sunlit_Cs_input,
            shaded_Cs_input=shaded_Cs_input,
            sunlit_eb_input=sunlit_eb_input,
            shaded_eb_input=shaded_eb_input,
            sunlit_T_input=sunlit_T_input,
            shaded_T_input=shaded_T_input,
            Pnu_Cab=shortwave.Pnu_Cab,
            Pnh_Cab=shortwave.Pnh_Cab,
            Rnuc_sw=shortwave.Rnuc,
            Rnhc_sw=shortwave.Rnhc,
            Rnus_sw=shortwave.Rnus,
            Rnhs_sw=shortwave.Rnhs,
            Rnuct=thermal.Rnuct,
            Rnhct=thermal.Rnhct,
            Rnust=thermal.Rnust,
            Rnhst=thermal.Rnhst,
            canopyemis=thermal.canopyemis,
            Rnuc=Rnuc,
            Rnhc=Rnhc,
            Rnus=Rnus,
            Rnhs=Rnhs,
            lEcu=lEcu,
            lEch=lEch,
            lEsu=lEs[:, 1],
            lEsh=lEs[:, 0],
            Hcu=Hcu,
            Hch=Hch,
            Hsu=Hs[:, 1],
            Hsh=Hs[:, 0],
            Gsu=G[:, 1],
            Gsh=G[:, 0],
            Csu=Ccu,
            Csh=Cch,
            ebu=ecu,
            ebh=ech,
            Tcu=Tcu,
            Tch=Tch,
            Tsu=Tsu,
            Tsh=Tsh,
            L=L,
            ustar=resistances.ustar,
            raa=resistances.raa,
            rawc=resistances.rawc,
            raws=resistances.raws,
            rac=(lai_tensor + 1.0) * (resistances.raa + resistances.rawc),
            ras=(lai_tensor + 1.0) * (resistances.raa + resistances.raws),
            Rnctot=Rnctot,
            lEctot=lEctot,
            Hctot=Hctot,
            Actot=Actot,
            Tcave=Tcave,
            Rnstot=Rnstot,
            lEstot=lEstot,
            Hstot=Hstot,
            Gtot=Gtot,
            Tsave=Tsave,
            Rntot=Rnctot + Rnstot,
            lEtot=lEctot + lEstot,
            Htot=Hctot + Hstot,
            max_error=max_error,
            converged=max_error <= opts.max_energy_error,
            counter=counter,
            Tsold=self._update_tsold(soil, Tsh=Tsh, Tsu=Tsu, device=device, dtype=dtype),
        )

    def solve_fluorescence(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        *,
        Esun_lw: Optional[torch.Tensor] = None,
        Esky_lw: Optional[torch.Tensor] = None,
        wlT: Optional[torch.Tensor] = None,
        meteo: EnergyBalanceMeteo,
        canopy: EnergyBalanceCanopy,
        soil: EnergyBalanceSoil,
        options: Optional[EnergyBalanceOptions] = None,
        biochem_options: Optional[BiochemicalOptions] = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
        Tcu0: Optional[torch.Tensor] = None,
        Tch0: Optional[torch.Tensor] = None,
        Tsu0: Optional[torch.Tensor] = None,
        Tsh0: Optional[torch.Tensor] = None,
    ) -> CanopyEnergyBalanceFluorescenceResult:
        energy = self.solve(
            leafbio,
            biochemistry,
            soil_refl,
            lai,
            tts,
            tto,
            psi,
            Esun_sw,
            Esky_sw,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            wlT=wlT,
            meteo=meteo,
            canopy=canopy,
            soil=soil,
            options=options,
            biochem_options=biochem_options,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
            Tcu0=Tcu0,
            Tch0=Tch0,
            Tsu0=Tsu0,
            Tsh0=Tsh0,
        )
        fluorescence = self._fluorescence_from_energy(
            energy=energy,
            leafbio=leafbio,
            soil_refl=soil_refl,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            Esun_sw=Esun_sw,
            Esky_sw=Esky_sw,
            hotspot=hotspot,
            lidf=lidf,
        )
        return CanopyEnergyBalanceFluorescenceResult(energy=energy, fluorescence=fluorescence)

    def solve_thermal(
        self,
        leafbio: LeafBioBatch,
        biochemistry: LeafBiochemistryInputs,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        *,
        Esun_lw: Optional[torch.Tensor] = None,
        Esky_lw: Optional[torch.Tensor] = None,
        wlT_forcing: Optional[torch.Tensor] = None,
        meteo: EnergyBalanceMeteo,
        canopy: EnergyBalanceCanopy,
        soil: EnergyBalanceSoil,
        options: Optional[EnergyBalanceOptions] = None,
        biochem_options: Optional[BiochemicalOptions] = None,
        hotspot: Optional[torch.Tensor] = None,
        lidf: Optional[torch.Tensor] = None,
        nlayers: Optional[int] = None,
        Tcu0: Optional[torch.Tensor] = None,
        Tch0: Optional[torch.Tensor] = None,
        Tsu0: Optional[torch.Tensor] = None,
        Tsh0: Optional[torch.Tensor] = None,
        wlT: Optional[torch.Tensor] = None,
    ) -> CanopyEnergyBalanceThermalResult:
        energy = self.solve(
            leafbio,
            biochemistry,
            soil_refl,
            lai,
            tts,
            tto,
            psi,
            Esun_sw,
            Esky_sw,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            wlT=wlT_forcing,
            meteo=meteo,
            canopy=canopy,
            soil=soil,
            options=options,
            biochem_options=biochem_options,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=nlayers,
            Tcu0=Tcu0,
            Tch0=Tch0,
            Tsu0=Tsu0,
            Tsh0=Tsh0,
        )
        thermal = self._thermal_from_energy(
            energy=energy,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            thermal_optics=soil.thermal_optics,
            hotspot=hotspot,
            lidf=lidf,
            wlT=wlT,
        )
        return CanopyEnergyBalanceThermalResult(energy=energy, thermal=thermal)

    def _fluorescence_from_energy(
        self,
        *,
        energy: CanopyEnergyBalanceResult,
        leafbio: LeafBioBatch,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        hotspot: Optional[torch.Tensor],
        lidf: Optional[torch.Tensor],
    ) -> CanopyFluorescenceResult:
        wlP = self.reflectance_model.fluspect.spectral.wlP
        wlE = self.reflectance_model.fluspect.spectral.wlE
        if wlE is None:
            raise ValueError("Spectral grids must define excitation wavelengths")
        batch = energy.Tsu.shape[0]
        Esun = self.fluorescence_model._prepare_spectrum(Esun_sw, batch, wlP.numel())
        Esky = self.fluorescence_model._prepare_spectrum(Esky_sw, batch, wlP.numel())
        Esun_e = self.fluorescence_model._sample_spectrum(Esun, wlP, wlE)
        Esky_e = self.fluorescence_model._sample_spectrum(Esky, wlP, wlE)
        fluorescence = self.fluorescence_model.layered(
            leafbio,
            soil_refl,
            lai,
            tts,
            tto,
            psi,
            Esun_e,
            Esky_e,
            etau=energy.sunlit.eta,
            etah=energy.shaded.eta,
            Pnu_Cab=energy.Pnu_Cab,
            Pnh_Cab=energy.Pnh_Cab,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=energy.Tcu.shape[1],
        )
        return fluorescence

    def _thermal_from_energy(
        self,
        *,
        energy: CanopyEnergyBalanceResult,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        thermal_optics: ThermalOptics,
        hotspot: Optional[torch.Tensor],
        lidf: Optional[torch.Tensor],
        wlT: Optional[torch.Tensor],
    ) -> CanopyThermalRadianceResult:
        return self.thermal_model(
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            Tcu=energy.Tcu,
            Tch=energy.Tch,
            Tsu=energy.Tsu,
            Tsh=energy.Tsh,
            thermal_optics=thermal_optics,
            hotspot=hotspot,
            lidf=lidf,
            nlayers=energy.Tcu.shape[1],
            wlT=wlT,
        )

    def _shortwave_radiation(
        self,
        *,
        leafopt,
        soil_refl: torch.Tensor,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_sw: torch.Tensor,
        Esky_sw: torch.Tensor,
        Esun_lw: Optional[torch.Tensor] = None,
        Esky_lw: Optional[torch.Tensor] = None,
        thermal_optics: Optional[ThermalOptics] = None,
        hotspot: torch.Tensor,
        lidf: Optional[torch.Tensor],
        nlayers: int,
        wlT: Optional[torch.Tensor] = None,
    ) -> ShortwaveRadiationResult:
        wlP = self.reflectance_model.fluspect.spectral.wlP
        device = leafopt.refl.device
        dtype = leafopt.refl.dtype
        batch = leafopt.refl.shape[0]
        Esun_optical = self.fluorescence_model._prepare_spectrum(Esun_sw, batch, wlP.numel())
        Esky_optical = self.fluorescence_model._prepare_spectrum(Esky_sw, batch, wlP.numel())
        soil_optical = self.reflectance_model.sail._ensure_2d(soil_refl, target_shape=leafopt.refl.shape)

        if thermal_optics is None:
            thermal_optics = ThermalOptics()
        wlT_tensor = default_thermal_wavelengths(device=device, dtype=dtype) if wlT is None else torch.as_tensor(wlT, device=device, dtype=dtype)
        rho_thermal = self.thermal_model._broadcast_scalar_spectrum(thermal_optics.rho_thermal, batch, wlT_tensor)
        tau_thermal = self.thermal_model._broadcast_scalar_spectrum(thermal_optics.tau_thermal, batch, wlT_tensor)
        soil_thermal = self.thermal_model._broadcast_scalar_spectrum(thermal_optics.rs_thermal, batch, wlT_tensor)
        Esun_thermal = (
            torch.zeros((batch, wlT_tensor.numel()), device=device, dtype=dtype)
            if Esun_lw is None
            else self.fluorescence_model._prepare_spectrum(Esun_lw, batch, wlT_tensor.numel())
        )
        Esky_thermal = (
            torch.zeros((batch, wlT_tensor.numel()), device=device, dtype=dtype)
            if Esky_lw is None
            else self.fluorescence_model._prepare_spectrum(Esky_lw, batch, wlT_tensor.numel())
        )

        Esun = torch.cat([Esun_optical, Esun_thermal], dim=-1)
        Esky = torch.cat([Esky_optical, Esky_thermal], dim=-1)
        rho = torch.cat([leafopt.refl, rho_thermal], dim=-1)
        tau = torch.cat([leafopt.tran, tau_thermal], dim=-1)
        soil = torch.cat([soil_optical, soil_thermal], dim=-1)
        transfer = self.fluorescence_model.layered_transport.build(
            rho,
            tau,
            soil,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nlayers,
        )
        direct = self.fluorescence_model.layered_transport.flux_profiles(transfer, Esun, torch.zeros_like(Esky))
        diffuse = self.fluorescence_model.layered_transport.flux_profiles(transfer, torch.zeros_like(Esun), Esky)
        total_Emin = direct.Emin_ + diffuse.Emin_
        total_Eplu = direct.Eplu_ + diffuse.Eplu_

        epsc = (1.0 - rho - tau).clamp(min=0.0)
        epss = (1.0 - soil).clamp(min=0.0)
        E_layer = 0.5 * (total_Emin[:, :-1, :] + total_Emin[:, 1:, :] + total_Eplu[:, :-1, :] + total_Eplu[:, 1:, :])

        n_optical = wlP.numel()
        E_layer_optical = E_layer[:, :, :n_optical]
        epsc_optical = epsc[:, :n_optical]
        epss_optical = epss[:, :n_optical]
        E_layer_thermal = E_layer[:, :, n_optical:]
        epsc_thermal = epsc[:, n_optical:]
        epss_thermal = epss[:, n_optical:]

        asun = 0.001 * self._integrate_spectral_blocks(
            optical=Esun_optical * epsc_optical,
            thermal=Esun_thermal * epsc_thermal,
            wl_optical=wlP,
            wl_thermal=wlT_tensor,
        )
        rndif = 0.001 * self._integrate_spectral_blocks(
            optical=E_layer_optical * epsc_optical.unsqueeze(1),
            thermal=E_layer_thermal * epsc_thermal.unsqueeze(1),
            wl_optical=wlP,
            wl_thermal=wlT_tensor,
        )
        sunlit_factor = (transfer.absfs * transfer.lidf_azimuth).sum(dim=-1)
        rndir = sunlit_factor.view(batch, 1) * asun.view(batch, 1)

        # RTMo stores absorbed PAR for biochemistry in umol m-2 s-1 while the
        # spectral forcing is in mW m-2 nm-1. Multiplying by 1e3 applies the
        # mW->W and mol->umol conversions together.
        pnsun_cab = 1e3 * torch.trapz(self._e2phot(wlP, Esun_optical * epsc_optical * leafopt.kChlrel), wlP, dim=-1)
        pndif_cab = 1e3 * torch.trapz(
            self._e2phot(wlP, E_layer_optical * epsc_optical.unsqueeze(1) * leafopt.kChlrel.unsqueeze(1)),
            wlP,
            dim=-1,
        )
        pnudir_cab = sunlit_factor.view(batch, 1) * pnsun_cab.view(batch, 1)

        rndir_soil = 0.001 * self._integrate_spectral_blocks(
            optical=Esun_optical * epss_optical,
            thermal=Esun_thermal * epss_thermal,
            wl_optical=wlP,
            wl_thermal=wlT_tensor,
        )
        rndif_soil = 0.001 * self._integrate_spectral_blocks(
            optical=total_Emin[:, -1, :n_optical] * epss_optical,
            thermal=total_Emin[:, -1, n_optical:] * epss_thermal,
            wl_optical=wlP,
            wl_thermal=wlT_tensor,
        )
        return ShortwaveRadiationResult(
            transfer=transfer,
            Rnuc=rndir + rndif,
            Rnhc=rndif,
            Rnus=rndir_soil + rndif_soil,
            Rnhs=rndif_soil,
            Pnu_Cab=pnudir_cab + pndif_cab,
            Pnh_Cab=pndif_cab,
        )

    def _incoming_thermal_radiation(
        self,
        *,
        lai: torch.Tensor,
        tts: torch.Tensor,
        tto: torch.Tensor,
        psi: torch.Tensor,
        Esun_lw: torch.Tensor,
        Esky_lw: torch.Tensor,
        thermal_optics: ThermalOptics,
        hotspot: torch.Tensor,
        lidf: Optional[torch.Tensor],
        nlayers: int,
        wlT: Optional[torch.Tensor],
    ) -> IncidentThermalRadiationResult:
        device = self.reflectance_model.fluspect.device
        dtype = self.reflectance_model.fluspect.dtype
        batch = hotspot.shape[0]
        wlT_tensor = default_thermal_wavelengths(device=device, dtype=dtype) if wlT is None else torch.as_tensor(wlT, device=device, dtype=dtype)
        Esun = self.fluorescence_model._prepare_spectrum(Esun_lw, batch, wlT_tensor.numel())
        Esky = self.fluorescence_model._prepare_spectrum(Esky_lw, batch, wlT_tensor.numel())

        rho = self.thermal_model._broadcast_scalar_spectrum(thermal_optics.rho_thermal, batch, wlT_tensor)
        tau = self.thermal_model._broadcast_scalar_spectrum(thermal_optics.tau_thermal, batch, wlT_tensor)
        soil = self.thermal_model._broadcast_scalar_spectrum(thermal_optics.rs_thermal, batch, wlT_tensor)
        transfer = self.fluorescence_model.layered_transport.build(
            rho,
            tau,
            soil,
            lai,
            tts,
            tto,
            psi,
            hotspot=hotspot,
            lidf=self.reflectance_model.lidf if lidf is None else lidf,
            nlayers=nlayers,
        )
        direct = self.fluorescence_model.layered_transport.flux_profiles(transfer, Esun, torch.zeros_like(Esky))
        diffuse = self.fluorescence_model.layered_transport.flux_profiles(transfer, torch.zeros_like(Esun), Esky)
        total_Emin = direct.Emin_ + diffuse.Emin_
        total_Eplu = direct.Eplu_ + diffuse.Eplu_

        epsc = (1.0 - rho - tau).clamp(min=0.0)
        epss = (1.0 - soil).clamp(min=0.0)
        E_layer = 0.5 * (total_Emin[:, :-1, :] + total_Emin[:, 1:, :] + total_Eplu[:, :-1, :] + total_Eplu[:, 1:, :])
        asun = 0.001 * torch.trapz(Esun * epsc, wlT_tensor, dim=-1)
        rndif = 0.001 * torch.trapz(E_layer * epsc.unsqueeze(1), wlT_tensor, dim=-1)
        sunlit_factor = (transfer.absfs * transfer.lidf_azimuth).sum(dim=-1)
        rndir = sunlit_factor.view(batch, 1) * asun.view(batch, 1)
        rndir_soil = 0.001 * torch.trapz(Esun * epss, wlT_tensor, dim=-1)
        rndif_soil = 0.001 * torch.trapz(total_Emin[:, -1, :] * epss, wlT_tensor, dim=-1)
        return IncidentThermalRadiationResult(
            Rnuc=rndir + rndif,
            Rnhc=rndif,
            Rnus=rndir_soil + rndif_soil,
            Rnhs=rndif_soil,
        )

    def _aggregate_canopy(
        self,
        transfer: LayeredCanopyTransfer,
        sunlit_flux: torch.Tensor,
        shaded_flux: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        ps = transfer.Ps[:, : transfer.nlayers]
        return transfer.dx * scale * torch.sum(ps * sunlit_flux + (1.0 - ps) * shaded_flux, dim=1)

    def _monin_obukhov_length(self, Ta: torch.Tensor, ustar: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        numerator = -self.rhoa * self.cp * ustar**3 * (Ta + 273.15)
        denom = self.kappa * self.g * H
        L = torch.where(torch.abs(denom) <= 1e-12, torch.full_like(denom, -1e6), numerator / denom)
        return torch.where(torch.isfinite(L), L, torch.full_like(L, -1e6))

    def _integrate_spectral_blocks(
        self,
        *,
        optical: torch.Tensor,
        thermal: torch.Tensor,
        wl_optical: torch.Tensor,
        wl_thermal: torch.Tensor,
    ) -> torch.Tensor:
        total = torch.trapz(optical, wl_optical, dim=-1)
        if thermal.shape[-1] == 0 or wl_thermal.numel() == 0:
            return total

        total = total + torch.trapz(thermal, wl_thermal, dim=-1)
        if wl_optical.numel() == 0:
            return total

        if wl_optical.numel() >= 2:
            optical_step = torch.abs(wl_optical[-1] - wl_optical[-2])
        else:
            optical_step = torch.tensor(0.0, device=wl_optical.device, dtype=wl_optical.dtype)
        if wl_thermal.numel() >= 2:
            thermal_step = torch.abs(wl_thermal[1] - wl_thermal[0])
        else:
            thermal_step = torch.tensor(0.0, device=wl_thermal.device, dtype=wl_thermal.dtype)
        seam_gap = torch.abs(wl_thermal[0] - wl_optical[-1])
        seam_limit = 5.0 * torch.maximum(optical_step, thermal_step).clamp(min=1.0)
        if bool((seam_gap <= seam_limit).item()):
            total = total + 0.5 * seam_gap * (optical[..., -1] + thermal[..., 0])
        return total

    def _soil_heat_flux(
        self,
        *,
        Rnhs: torch.Tensor,
        Rnus: torch.Tensor,
        Tsh: torch.Tensor,
        Tsu: torch.Tensor,
        soil: EnergyBalanceSoil,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        epss = self._soil_thermal_emissivity(soil.thermal_optics, Rnhs.shape[0], device=device, dtype=dtype).view(-1, 1)
        Rns = torch.stack([Rnhs, Rnus], dim=1)
        Ts = torch.stack([Tsh, Tsu], dim=1)
        if soil.soil_heat_method == 3:
            zeros = torch.zeros_like(Rns)
            return zeros, zeros
        if soil.soil_heat_method == 1:
            if soil.Tsold is None or soil.dt_seconds is None:
                raise ValueError("soil_heat_method=1 requires Tsold and dt_seconds")
            dt = self._expand_batch(soil.dt_seconds, Ts.shape[0], device=device, dtype=dtype).view(-1, 1, 1)
            history = torch.as_tensor(soil.Tsold, device=device, dtype=dtype)
            if history.ndim == 2:
                history = history.unsqueeze(0)
            if history.ndim != 3 or history.shape[0] not in (1, Ts.shape[0]) or history.shape[-1] != 2:
                raise ValueError("Tsold must have shape (history, 2) or (batch, history, 2)")
            if history.shape[0] == 1 and Ts.shape[0] != 1:
                history = history.expand(Ts.shape[0], -1, -1)
            steps = history.shape[1]
            x = torch.arange(1, steps + 1, device=device, dtype=dtype).view(1, steps, 1) * dt
            previous = torch.cat([Ts.unsqueeze(1), history[:, :-1, :]], dim=1)
            coeff = (torch.sqrt(x) - torch.sqrt((x - dt).clamp(min=0.0))) / dt
            pi_sqrt = torch.sqrt(torch.tensor(torch.pi, device=device, dtype=dtype))
            GAM = self._expand_batch(soil.GAM, Ts.shape[0], device=device, dtype=dtype).view(-1, 1)
            scale = GAM / pi_sqrt * 2.0
            G = scale * torch.sum((previous - history) * coeff, dim=1)
            dG = scale * coeff[:, :1, 0]
            return G, dG.expand_as(G)
        G = 0.35 * Rns
        dG = 4.0 * epss * self.sigma_sb * (Ts + 273.15) ** 3 * 0.35
        return G, dG

    def _initial_layer_state(
        self,
        explicit: Optional[torch.Tensor],
        default: torch.Tensor,
        batch: int,
        nlayers: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if explicit is None:
            return default.view(batch, 1).expand(batch, nlayers)
        return self.fluorescence_model._prepare_layer_profile(explicit, self._dummy_transfer(batch, nlayers, device=device, dtype=dtype))

    def _resolve_nlayers(
        self,
        nlayers: Optional[int],
        fV: Optional[torch.Tensor | float],
        *profiles: Optional[torch.Tensor],
    ) -> int:
        for profile in profiles:
            if profile is None:
                continue
            tensor = torch.as_tensor(profile)
            if tensor.ndim >= 2:
                return int(tensor.shape[1])
        if fV is not None:
            tensor = torch.as_tensor(fV)
            if tensor.ndim >= 2:
                return int(tensor.shape[1])
        return int(nlayers) if nlayers is not None else 60

    def _fV_profile(self, canopy: EnergyBalanceCanopy, transfer: LayeredCanopyTransfer, batch: int) -> torch.Tensor:
        device = transfer.Ps.device
        dtype = transfer.Ps.dtype
        if canopy.fV is not None:
            return self.fluorescence_model._prepare_layer_profile(canopy.fV, transfer)
        kV = self._expand_batch(canopy.kV, batch, device=device, dtype=dtype)
        # Match upstream SCOPE ebal.m: fV is evaluated on xl(1:end-1),
        # i.e. it includes the canopy top (0) and excludes the bottom edge.
        return torch.exp(kV.view(batch, 1) * transfer.xl[:-1].view(1, -1))

    def _prepare_soil_profile(self, value: torch.Tensor | float, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.view(1, 1).expand(batch, 2)
        if tensor.ndim == 1:
            if tensor.shape[0] == batch:
                return tensor.view(batch, 1).expand(batch, 2)
            if tensor.shape[0] == 2:
                return tensor.view(1, 2).expand(batch, 2)
        if tensor.ndim == 2:
            if tensor.shape == (batch, 2):
                return tensor
            if tensor.shape == (1, 2):
                return tensor.expand(batch, 2)
        raise ValueError(f"Soil resistances must broadcast to shape ({batch}, 2)")

    def _leaf_thermal_emissivity(self, optics: ThermalOptics, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        rho = self._expand_batch(optics.rho_thermal, batch, device=device, dtype=dtype)
        tau = self._expand_batch(optics.tau_thermal, batch, device=device, dtype=dtype)
        return 1.0 - rho - tau

    def _soil_thermal_emissivity(self, optics: ThermalOptics, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        soil = self._expand_batch(optics.rs_thermal, batch, device=device, dtype=dtype)
        return 1.0 - soil

    def _update_tsold(
        self,
        soil: EnergyBalanceSoil,
        *,
        Tsh: torch.Tensor,
        Tsu: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if soil.soil_heat_method != 1 or soil.Tsold is None:
            return None
        history = torch.as_tensor(soil.Tsold, device=device, dtype=dtype)
        if history.ndim == 2:
            history = history.unsqueeze(0)
        if history.ndim != 3:
            raise ValueError("Tsold must have shape (history, 2) or (batch, history, 2)")
        if history.shape[0] == 1 and Tsh.shape[0] != 1:
            history = history.expand(Tsh.shape[0], -1, -1).clone()
        updated = history.clone()
        updated[:, 1:, :] = history[:, :-1, :]
        updated[:, 0, 0] = Tsh
        updated[:, 0, 1] = Tsu
        return updated

    def _expand_batch(self, value: torch.Tensor | float, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=device, dtype=dtype)
        if tensor.ndim == 0:
            return tensor.repeat(batch)
        if tensor.ndim == 1 and tensor.shape[0] == batch:
            return tensor
        if tensor.ndim == 1 and tensor.shape[0] == 1 and batch != 1:
            return tensor.expand(batch)
        raise ValueError("Values must broadcast to the batch dimension")

    def _ephoton(self, wavelength_nm: torch.Tensor) -> torch.Tensor:
        h = torch.as_tensor(6.62607015e-34, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        c = torch.as_tensor(299792458.0, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        avogadro = torch.as_tensor(6.02214076e23, device=wavelength_nm.device, dtype=wavelength_nm.dtype)
        wavelength_m = wavelength_nm * 1e-9
        return avogadro * h * c / wavelength_m

    def _e2phot(self, wavelength_nm: torch.Tensor, energy: torch.Tensor) -> torch.Tensor:
        return energy / self._ephoton(wavelength_nm).view(*([1] * (energy.ndim - 1)), -1)

    def _dummy_transfer(self, batch: int, nlayers: int, *, device: torch.device, dtype: torch.dtype) -> LayeredCanopyTransfer:
        dummy = torch.zeros((batch, nlayers + 1), device=device, dtype=dtype)
        return LayeredCanopyTransfer(
            nlayers=nlayers,
            litab=torch.zeros(1, device=device, dtype=dtype),
            lazitab=torch.zeros(1, device=device, dtype=dtype),
            lidf=torch.ones((batch, 1), device=device, dtype=dtype),
            xl=torch.zeros(nlayers + 1, device=device, dtype=dtype),
            dx=torch.as_tensor(1.0 / nlayers, device=device, dtype=dtype),
            iLAI=torch.zeros(batch, device=device, dtype=dtype),
            ks=torch.zeros(batch, device=device, dtype=dtype),
            ko=torch.zeros(batch, device=device, dtype=dtype),
            dso=torch.zeros(batch, device=device, dtype=dtype),
            Ps=dummy,
            Po=dummy,
            Pso=dummy,
            rho_dd=torch.zeros((batch, nlayers, 1), device=device, dtype=dtype),
            tau_dd=torch.zeros((batch, nlayers, 1), device=device, dtype=dtype),
            R_sd=torch.zeros((batch, nlayers + 1, 1), device=device, dtype=dtype),
            R_dd=torch.zeros((batch, nlayers + 1, 1), device=device, dtype=dtype),
            Xss=torch.zeros((batch, nlayers), device=device, dtype=dtype),
            Xsd=torch.zeros((batch, nlayers, 1), device=device, dtype=dtype),
            Xdd=torch.zeros((batch, nlayers, 1), device=device, dtype=dtype),
            vb=torch.zeros((batch, 1), device=device, dtype=dtype),
            vf=torch.zeros((batch, 1), device=device, dtype=dtype),
            w=torch.zeros((batch, 1), device=device, dtype=dtype),
            fs=torch.zeros((batch, 1), device=device, dtype=dtype),
            absfs=torch.zeros((batch, 1), device=device, dtype=dtype),
            absfo=torch.zeros((batch, 1), device=device, dtype=dtype),
            fsfo=torch.zeros((batch, 1), device=device, dtype=dtype),
            foctl=torch.zeros((batch, 1), device=device, dtype=dtype),
            fsctl=torch.zeros((batch, 1), device=device, dtype=dtype),
            ctl2=torch.zeros((batch, 1), device=device, dtype=dtype),
            lidf_azimuth=torch.ones((batch, 1), device=device, dtype=dtype),
        )
