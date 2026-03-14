from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class VariableDefinition:
    name: str
    kind: str
    category: str
    units: str
    meaning: str
    workflows: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    notes: str = ""

    def search_blob(self) -> str:
        parts = [
            self.name,
            self.kind,
            self.category,
            self.units,
            self.meaning,
            self.notes,
            *self.workflows,
            *self.aliases,
        ]
        return " ".join(part.lower() for part in parts if part)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _v(
    name: str,
    *,
    kind: str,
    category: str,
    units: str,
    meaning: str,
    workflows: Sequence[str] = (),
    aliases: Sequence[str] = (),
    notes: str = "",
) -> VariableDefinition:
    return VariableDefinition(
        name=name,
        kind=kind,
        category=category,
        units=units,
        meaning=meaning,
        workflows=tuple(workflows),
        aliases=tuple(aliases),
        notes=notes,
    )


VARIABLES: tuple[VariableDefinition, ...] = (
    _v("y", kind="dimension", category="grid", units="-", meaning="Spatial row dimension of the ROI/grid."),
    _v("x", kind="dimension", category="grid", units="-", meaning="Spatial column dimension of the ROI/grid."),
    _v("time", kind="dimension", category="grid", units="datetime", meaning="Time axis for scenes or time-series runs."),
    _v("wavelength", kind="dimension", category="spectral", units="nm", meaning="Optical wavelength axis used by reflectance and shortwave forcing."),
    _v("excitation_wavelength", kind="dimension", category="spectral", units="nm", meaning="Excitation wavelength axis used to drive fluorescence source terms."),
    _v("fluorescence_wavelength", kind="dimension", category="spectral", units="nm", meaning="Fluorescence emission wavelength axis."),
    _v("thermal_wavelength", kind="dimension", category="spectral", units="um", meaning="Thermal wavelength axis used by longwave radiance outputs."),
    _v("layer", kind="dimension", category="canopy", units="-", meaning="Within-canopy layer axis for layered transport and physiology fields."),
    _v("layer_interface", kind="dimension", category="canopy", units="-", meaning="Layer-interface axis for cumulative transport profiles."),
    _v("direction", kind="dimension", category="geometry", units="-", meaning="Index over directional viewing geometries."),
    _v("directional_tto", kind="dimension", category="geometry", units="deg", meaning="Viewing zenith angles for directional outputs.", aliases=("tto grid",)),
    _v("directional_psi", kind="dimension", category="geometry", units="deg", meaning="Relative azimuth angles for directional outputs.", aliases=("psi grid",)),
    _v("calc_fluor", kind="option", category="workflow", units="0/1", meaning="Enables fluorescence workflows in high-level runner dispatch.", aliases=("options.calc_fluor",)),
    _v("calc_planck", kind="option", category="workflow", units="0/1", meaning="Enables thermal/Planck workflows in high-level runner dispatch."),
    _v("calc_directional", kind="option", category="workflow", units="0/1", meaning="Requests directional products for the selected workflows."),
    _v("calc_vert_profiles", kind="option", category="workflow", units="0/1", meaning="Requests vertical-profile products for the selected workflows."),
    _v("soil_heat_method", kind="option", category="workflow", units="index", meaning="Selects the soil heat-flux treatment in coupled energy-balance workflows."),
    _v("mSCOPE", kind="option", category="workflow", units="0/1", meaning="Upstream vertically heterogeneous leaf-optics mode. Present as metadata only; not implemented in this Python stack."),
    _v("Cab", kind="input", category="leaf biochemistry", units="ug cm-2", meaning="Leaf chlorophyll a+b content.", aliases=("chlorophyll",), workflows=("reflectance", "fluorescence", "thermal", "energy balance")),
    _v("Cca", kind="input", category="leaf biochemistry", units="ug cm-2", meaning="Leaf carotenoid content.", aliases=("carotenoids",)),
    _v("Cw", kind="input", category="leaf biochemistry", units="g cm-2", meaning="Equivalent leaf water thickness.", aliases=("leaf water",)),
    _v("Cdm", kind="input", category="leaf biochemistry", units="g cm-2", meaning="Leaf dry matter content per area.", aliases=("Cm", "dry matter")),
    _v("Cs", kind="input", category="leaf biochemistry", units="-", meaning="Senescent or brown pigment content used by FLUSPECT.", aliases=("cbrown",)),
    _v("Cant", kind="input", category="leaf biochemistry", units="ug cm-2", meaning="Anthocyanin content."),
    _v("Cbc", kind="input", category="leaf biochemistry", units="ug cm-2", meaning="Brown carbon or additional biochemical absorber term when provided."),
    _v("Cp", kind="input", category="leaf biochemistry", units="g cm-2", meaning="Protein content parameter for extended leaf optics."),
    _v("N", kind="input", category="leaf biochemistry", units="-", meaning="Leaf mesophyll structure parameter in PROSPECT/FLUSPECT."),
    _v("fqe", kind="input", category="leaf biochemistry", units="-", meaning="Leaf fluorescence quantum efficiency scaling used to build fluorescence source terms.", aliases=("fluorescence efficiency",)),
    _v("ala", kind="input", category="leaf structure", units="deg", meaning="Leaf inclination angle or Campbell mean leaf angle input used to derive the LIDF.", aliases=("LIDFa",)),
    _v("LAI", kind="input", category="canopy structure", units="m2 m-2", meaning="Leaf area index."),
    _v("tts", kind="input", category="geometry", units="deg", meaning="Solar zenith angle.", aliases=("sun zenith",)),
    _v("tto", kind="input", category="geometry", units="deg", meaning="Viewing zenith angle.", aliases=("observer zenith",)),
    _v("psi", kind="input", category="geometry", units="deg", meaning="Relative azimuth angle between sun and sensor.", aliases=("relative azimuth",)),
    _v("soil_refl", kind="input", category="soil", units="-", meaning="Explicit soil reflectance spectrum on the model optical wavelength grid."),
    _v("soil_spectrum", kind="input", category="soil", units="index", meaning="Index into the upstream SCOPE soil reflectance library.", aliases=("soilspectrum",)),
    _v("BSMBrightness", kind="input", category="soil", units="-", meaning="Brightness parameter for the SCOPE BSM soil model."),
    _v("BSMlat", kind="input", category="soil", units="deg", meaning="Latitude-like mineral composition parameter used by the SCOPE BSM soil model."),
    _v("BSMlon", kind="input", category="soil", units="deg", meaning="Longitude-like mineral composition parameter used by the SCOPE BSM soil model."),
    _v("SMC", kind="input", category="soil", units="fraction", meaning="Soil moisture content used by the BSM soil model."),
    _v("Esun_", kind="input", category="spectral forcing", units="W m-2 um-1", meaning="Direct excitation irradiance spectrum for fluorescence."),
    _v("Esky_", kind="input", category="spectral forcing", units="W m-2 um-1", meaning="Diffuse excitation irradiance spectrum for fluorescence."),
    _v("Esun_sw", kind="input", category="spectral forcing", units="W m-2 um-1", meaning="Direct shortwave irradiance spectrum used by reflectance and energy-balance workflows."),
    _v("Esky_sw", kind="input", category="spectral forcing", units="W m-2 um-1", meaning="Diffuse shortwave irradiance spectrum used by reflectance and energy-balance workflows."),
    _v("etau", kind="input", category="fluorescence transport", units="-", meaning="Forward fluorescence efficiency or source scaling per canopy layer.", notes="Used in layered and directional fluorescence paths."),
    _v("etah", kind="input", category="fluorescence transport", units="-", meaning="Backward fluorescence efficiency or source scaling per canopy layer.", notes="Used in layered and directional fluorescence paths."),
    _v("Ta", kind="input", category="meteorology", units="degC", meaning="Air temperature."),
    _v("ea", kind="input", category="meteorology", units="hPa", meaning="Ambient vapor pressure."),
    _v("Ca", kind="input", category="meteorology", units="ppm", meaning="Ambient CO2 concentration."),
    _v("Oa", kind="input", category="meteorology", units="permil or mmol mol-1", meaning="Ambient oxygen concentration used by leaf biochemistry."),
    _v("p", kind="input", category="meteorology", units="hPa", meaning="Air pressure."),
    _v("z", kind="input", category="meteorology", units="m", meaning="Reference meteorological height."),
    _v("u", kind="input", category="meteorology", units="m s-1", meaning="Wind speed at the reference height."),
    _v("L", kind="input", category="meteorology", units="m", meaning="Monin-Obukhov length when supplied as an external stability input."),
    _v("Cd", kind="input", category="canopy aerodynamics", units="-", meaning="Drag coefficient for canopy aerodynamic exchange."),
    _v("rwc", kind="input", category="canopy aerodynamics", units="-", meaning="Relative water content or canopy resistance scaling parameter used by the energy-balance closure."),
    _v("z0m", kind="input", category="canopy aerodynamics", units="m", meaning="Momentum roughness length."),
    _v("d", kind="input", category="canopy aerodynamics", units="m", meaning="Zero-plane displacement height."),
    _v("h", kind="input", category="canopy aerodynamics", units="m", meaning="Canopy height."),
    _v("kV", kind="input", category="canopy aerodynamics", units="-", meaning="Vertical extinction or partitioning parameter for layered canopy coupling."),
    _v("fV", kind="input", category="canopy aerodynamics", units="-", meaning="Vertical partitioning profile used by layered canopy physiology and energy balance."),
    _v("rss", kind="input", category="soil", units="s m-1", meaning="Soil resistance to evaporation.", aliases=("soil surface resistance",)),
    _v("rbs", kind="input", category="soil", units="s m-1", meaning="Boundary resistance near the soil surface."),
    _v("rho_thermal", kind="input", category="thermal optics", units="-", meaning="Thermal reflectance of the soil surface."),
    _v("tau_thermal", kind="input", category="thermal optics", units="-", meaning="Thermal transmittance of the canopy in the simplified thermal optics parameterization."),
    _v("rs_thermal", kind="input", category="thermal optics", units="-", meaning="Thermal soil reflectance/emissivity control in the simplified thermal optics parameterization."),
    _v("Tcu", kind="input", category="thermal state", units="degC", meaning="Sunlit canopy temperature."),
    _v("Tch", kind="input", category="thermal state", units="degC", meaning="Shaded canopy temperature."),
    _v("Tsu", kind="input", category="thermal state", units="degC", meaning="Sunlit soil temperature."),
    _v("Tsh", kind="input", category="thermal state", units="degC", meaning="Shaded soil temperature."),
    _v("Csu", kind="input", category="boundary state", units="ppm", meaning="CO2 concentration at the sunlit leaf surface."),
    _v("Csh", kind="input", category="boundary state", units="ppm", meaning="CO2 concentration at the shaded leaf surface."),
    _v("ebu", kind="input", category="boundary state", units="hPa", meaning="Vapor pressure at the sunlit leaf surface."),
    _v("ebh", kind="input", category="boundary state", units="hPa", meaning="Vapor pressure at the shaded leaf surface."),
    _v("GAM", kind="input", category="soil", units="W m-2", meaning="Optional soil heat flux forcing or initialization term."),
    _v("Tsold", kind="input", category="soil", units="degC", meaning="Previous-step soil temperature state for transient soil heat treatment."),
    _v("dt_seconds", kind="input", category="soil", units="s", meaning="Time-step length used by transient soil heat treatment."),
    _v("leaf_refl", kind="output", category="reflectance", units="-", meaning="Leaf hemispherical reflectance from FLUSPECT."),
    _v("leaf_tran", kind="output", category="reflectance", units="-", meaning="Leaf hemispherical transmittance from FLUSPECT."),
    _v("rsot", kind="output", category="reflectance", units="-", meaning="Total top-of-canopy reflectance factor in the observation direction.", aliases=("apparent reflectance", "reflectance.csv")),
    _v("rso", kind="output", category="reflectance", units="-", meaning="Bidirectional reflectance factor.", aliases=("BRF",), notes="Matches the original SCOPE rso definition."),
    _v("rsd", kind="output", category="reflectance", units="-", meaning="Directional-hemispherical reflectance factor.", aliases=("DHRF",)),
    _v("rdd", kind="output", category="reflectance", units="-", meaning="Bi-hemispherical reflectance factor.", aliases=("BHRF",)),
    _v("rdo", kind="output", category="reflectance", units="-", meaning="Hemispherical-directional reflectance factor.", aliases=("HDRF",)),
    _v("refl_", kind="output", category="reflectance", units="-", meaning="Directional apparent reflectance for a requested angle grid."),
    _v("Ps", kind="output", category="profiles", units="-", meaning="Cumulative direct-solar interception profile along canopy depth."),
    _v("Po", kind="output", category="profiles", units="-", meaning="Cumulative observation-path attenuation profile along canopy depth."),
    _v("Pso", kind="output", category="profiles", units="-", meaning="Joint sun-observer hotspot weighting profile across the canopy."),
    _v("Es_", kind="output", category="profiles", units="W m-2 um-1", meaning="Downward direct-plus-diffuse shortwave flux profile at canopy interfaces."),
    _v("Emin_", kind="output", category="profiles", units="W m-2 um-1", meaning="Downward diffuse shortwave flux profile at canopy interfaces."),
    _v("Eplu_", kind="output", category="profiles", units="W m-2 um-1", meaning="Upward diffuse shortwave flux profile at canopy interfaces."),
    _v("LoF_", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Fluorescence spectrum in the observation direction.", aliases=("fluorescence.csv",)),
    _v("LoF_sunlit", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Observation-direction fluorescence contribution from sunlit leaves."),
    _v("LoF_shaded", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Observation-direction fluorescence contribution from shaded leaves."),
    _v("LoF_scattered", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Observation-direction fluorescence contribution from multiply scattered canopy radiation."),
    _v("LoF_soil", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Observation-direction fluorescence contribution associated with the soil boundary term."),
    _v("EoutF_", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Top-of-canopy hemispherically integrated fluorescence.", aliases=("fluorescence_hemis.csv",)),
    _v("EoutFrc_", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Reabsorption-corrected hemispherical fluorescence spectrum.", aliases=("fluorescence_ReabsCorr.csv",)),
    _v("Femleaves_", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Fluorescence emitted by all leaves before canopy escape losses.", aliases=("fluorescence_AllLeaves.csv",)),
    _v("sigmaF", kind="output", category="fluorescence", units="-", meaning="Fluorescence escape probability.", aliases=("sigmaF.csv",)),
    _v("F685", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="First fluorescence peak radiance.", aliases=("F_1stpeak",)),
    _v("wl685", kind="output", category="fluorescence", units="nm", meaning="Wavelength of the first fluorescence peak.", aliases=("wl_1stpeak",)),
    _v("F740", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Second fluorescence peak radiance.", aliases=("F_2ndpeak",)),
    _v("wl740", kind="output", category="fluorescence", units="nm", meaning="Wavelength of the second fluorescence peak.", aliases=("wl_2ndpeak",)),
    _v("F684", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Fluorescence radiance sampled near 684-687 nm.", aliases=("F687",)),
    _v("F761", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Fluorescence radiance sampled near 760-761 nm.", aliases=("F760",)),
    _v("LoutF", kind="output", category="fluorescence", units="W m-2 um-1 sr-1", meaning="Integrated observation-direction fluorescence scalar.", aliases=("LFtot",)),
    _v("EoutF", kind="output", category="fluorescence", units="W m-2", meaning="Integrated hemispherical fluorescence scalar.", aliases=("EFtot",)),
    _v("Fmin_", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Downward or internal fluorescence transport profile term at canopy interfaces."),
    _v("Fplu_", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Upward fluorescence transport profile term at canopy interfaces."),
    _v("Lot_", kind="output", category="thermal", units="W m-2 um-1 sr-1", meaning="Thermal radiance spectrum in the observation direction."),
    _v("Eoutte_", kind="output", category="thermal", units="W m-2 um-1", meaning="Emitted longwave flux spectrum leaving the canopy-soil system."),
    _v("Emint_", kind="output", category="thermal", units="W m-2 um-1", meaning="Downward internal thermal flux profile."),
    _v("Eplut_", kind="output", category="thermal", units="W m-2 um-1", meaning="Upward internal thermal flux profile."),
    _v("LotBB_", kind="output", category="thermal", units="W m-2 um-1 sr-1", meaning="Blackbody-equivalent observation-direction thermal radiance."),
    _v("Loutt", kind="output", category="thermal", units="W m-2 um-1 sr-1", meaning="Integrated observation-direction thermal radiance scalar."),
    _v("Eoutt", kind="output", category="thermal", units="W m-2", meaning="Integrated hemispherical thermal flux scalar."),
    _v("BrightnessT", kind="output", category="thermal", units="K", meaning="Brightness temperature inferred from the directional thermal radiance."),
    _v("layer_thermal_upward", kind="output", category="profiles", units="W m-2", meaning="Integrated upward thermal transport per canopy layer."),
    _v("Pnu_Cab", kind="output", category="physiology", units="umol m-2 s-1", meaning="Sunlit absorbed PAR by chlorophyll a+b used to drive leaf biochemistry."),
    _v("Pnh_Cab", kind="output", category="physiology", units="umol m-2 s-1", meaning="Shaded absorbed PAR by chlorophyll a+b used to drive leaf biochemistry."),
    _v("sunlit_*", kind="output", category="physiology", units="varies", meaning="Prefix used for leaf-biochemistry outputs of sunlit leaves.", notes="Search for base names such as A, Ci, rcw, eta, or Ja."),
    _v("shaded_*", kind="output", category="physiology", units="varies", meaning="Prefix used for leaf-biochemistry outputs of shaded leaves.", notes="Search for base names such as A, Ci, rcw, eta, or Ja."),
    _v("sunlit_A", kind="output", category="physiology", units="umol m-2 s-1", meaning="Net assimilation rate of sunlit leaves."),
    _v("shaded_A", kind="output", category="physiology", units="umol m-2 s-1", meaning="Net assimilation rate of shaded leaves."),
    _v("sunlit_Ci", kind="output", category="physiology", units="ppm", meaning="Intercellular CO2 concentration of sunlit leaves."),
    _v("shaded_Ci", kind="output", category="physiology", units="ppm", meaning="Intercellular CO2 concentration of shaded leaves."),
    _v("sunlit_rcw", kind="output", category="physiology", units="s m-1", meaning="Canopy water-vapor resistance of sunlit leaves."),
    _v("shaded_rcw", kind="output", category="physiology", units="s m-1", meaning="Canopy water-vapor resistance of shaded leaves."),
    _v("sunlit_eta", kind="output", category="physiology", units="-", meaning="Fluorescence efficiency term returned by the sunlit leaf biochemistry solve."),
    _v("shaded_eta", kind="output", category="physiology", units="-", meaning="Fluorescence efficiency term returned by the shaded leaf biochemistry solve."),
    _v("Rnctot", kind="output", category="energy balance", units="W m-2", meaning="Net radiation of the canopy.", aliases=("fluxes.csv",)),
    _v("lEctot", kind="output", category="energy balance", units="W m-2", meaning="Latent heat flux of the canopy (transpiration)."),
    _v("Hctot", kind="output", category="energy balance", units="W m-2", meaning="Sensible heat flux of the canopy."),
    _v("Actot", kind="output", category="energy balance", units="umol m-2 s-1", meaning="Net photosynthesis of the canopy.", aliases=("Photosynthesis",)),
    _v("Tcave", kind="output", category="energy balance", units="degC", meaning="Average canopy temperature."),
    _v("Rnstot", kind="output", category="energy balance", units="W m-2", meaning="Net radiation of the soil."),
    _v("lEstot", kind="output", category="energy balance", units="W m-2", meaning="Latent heat flux of the soil (evaporation)."),
    _v("Hstot", kind="output", category="energy balance", units="W m-2", meaning="Sensible heat flux of the soil."),
    _v("Gtot", kind="output", category="energy balance", units="W m-2", meaning="Soil heat flux."),
    _v("Tsave", kind="output", category="energy balance", units="degC", meaning="Average soil temperature."),
    _v("Rntot", kind="output", category="energy balance", units="W m-2", meaning="Total net radiation."),
    _v("lEtot", kind="output", category="energy balance", units="W m-2", meaning="Total latent heat flux."),
    _v("Htot", kind="output", category="energy balance", units="W m-2", meaning="Total sensible heat flux."),
    _v("raa", kind="output", category="resistance", units="s m-1", meaning="Aerodynamic resistance above the canopy."),
    _v("raws", kind="output", category="resistance", units="s m-1", meaning="Aerodynamic resistance within the soil/canopy lower boundary layer.", aliases=("within-soil aerodynamic resistance",)),
    _v("rawc", kind="output", category="resistance", units="s m-1", meaning="Aerodynamic resistance within the canopy air space."),
    _v("rac", kind="output", category="resistance", units="s m-1", meaning="Boundary-layer resistance for canopy heat and vapor exchange."),
    _v("ras", kind="output", category="resistance", units="s m-1", meaning="Boundary-layer resistance for soil heat and vapor exchange."),
    _v("ustar", kind="output", category="resistance", units="m s-1", meaning="Friction velocity."),
    _v("canopyemis", kind="output", category="thermal", units="-", meaning="Effective canopy emissivity."),
    _v("Csu", kind="output", category="boundary state", units="ppm", meaning="Solved CO2 concentration at the sunlit leaf surface."),
    _v("Csh", kind="output", category="boundary state", units="ppm", meaning="Solved CO2 concentration at the shaded leaf surface."),
    _v("ebu", kind="output", category="boundary state", units="hPa", meaning="Solved vapor pressure at the sunlit leaf surface."),
    _v("ebh", kind="output", category="boundary state", units="hPa", meaning="Solved vapor pressure at the shaded leaf surface."),
    _v("Tcu", kind="output", category="thermal state", units="degC", meaning="Solved sunlit canopy temperature."),
    _v("Tch", kind="output", category="thermal state", units="degC", meaning="Solved shaded canopy temperature."),
    _v("Tsu", kind="output", category="thermal state", units="degC", meaning="Solved sunlit soil temperature."),
    _v("Tsh", kind="output", category="thermal state", units="degC", meaning="Solved shaded soil temperature."),
    _v("max_error", kind="output", category="solver", units="W m-2", meaning="Maximum residual or closure error in the final energy-balance iteration."),
    _v("converged", kind="output", category="solver", units="0/1", meaning="Whether the coupled energy-balance iteration converged."),
    _v("counter", kind="output", category="solver", units="-", meaning="Number of energy-balance iterations performed."),
    _v("scope_product", kind="output", category="dataset metadata", units="-", meaning="Dataset attribute naming the assembled product or workflow."),
    _v("scope_components", kind="output", category="dataset metadata", units="-", meaning="Dataset attribute listing merged workflow components."),
    _v("reflectance_directional_*", kind="output", category="namespaced outputs", units="varies", meaning="Prefix applied to directional reflectance variables in merged workflow datasets."),
    _v("reflectance_profile_*", kind="output", category="namespaced outputs", units="varies", meaning="Prefix applied to reflectance profile variables in merged workflow datasets."),
    _v("fluorescence_directional_*", kind="output", category="namespaced outputs", units="varies", meaning="Prefix applied to directional fluorescence variables in merged workflow datasets."),
    _v("fluorescence_profile_*", kind="output", category="namespaced outputs", units="varies", meaning="Prefix applied to fluorescence profile variables in merged workflow datasets."),
    _v("thermal_directional_*", kind="output", category="namespaced outputs", units="varies", meaning="Prefix applied to directional thermal variables in merged workflow datasets."),
    _v("thermal_profile_*", kind="output", category="namespaced outputs", units="varies", meaning="Prefix applied to thermal profile variables in merged workflow datasets."),
)

VARIABLES = VARIABLES + (
    _v("lite", kind="option", category="workflow", units="0/1", meaning="Upstream SCOPE lite-mode flag carried through prepared dataset attrs."),
    _v("calc_xanthophyllabs", kind="option", category="workflow", units="0/1", meaning="Upstream flag for xanthophyll absorption treatment. Present as metadata in the current Python stack."),
    _v("Fluorescence_model", kind="option", category="workflow", units="index", meaning="Upstream fluorescence-model selector. Present as metadata in the current Python stack."),
    _v("apply_T_corr", kind="option", category="workflow", units="0/1", meaning="Upstream temperature-correction flag carried with prepared dataset attrs."),
    _v("verify", kind="option", category="workflow", units="0/1", meaning="Upstream verification-mode flag carried with prepared dataset attrs."),
    _v("calc_rss_rbs", kind="option", category="workflow", units="0/1", meaning="Upstream flag controlling calculation of soil and boundary resistances."),
    _v("MoninObukhov", kind="option", category="workflow", units="0/1", meaning="Flag controlling Monin-Obukhov stability correction usage."),
    _v("save_spectral", kind="option", category="workflow", units="0/1", meaning="Upstream flag requesting spectral outputs to be saved."),
    _v("soilspectrum", kind="option", category="workflow", units="index", meaning="Upstream soil-spectrum mode flag or selector.", aliases=("soil_spectrum option",)),
    _v("tdd", kind="output", category="reflectance", units="-", meaning="Diffuse transmittance for diffuse incident radiation through the canopy."),
    _v("tsd", kind="output", category="reflectance", units="-", meaning="Direct-to-diffuse canopy transmittance for solar illumination."),
    _v("tdo", kind="output", category="reflectance", units="-", meaning="Diffuse transmittance from canopy layers toward the observation direction."),
    _v("rsos", kind="output", category="reflectance", units="-", meaning="Bidirectional reflectance contribution from the hotspot or single-scattering term."),
    _v("rsod", kind="output", category="reflectance", units="-", meaning="Bidirectional reflectance contribution from the diffuse multiple-scattering term."),
    _v("rddt", kind="output", category="reflectance", units="-", meaning="Bi-hemispherical reflectance including the soil boundary condition."),
    _v("rsdt", kind="output", category="reflectance", units="-", meaning="Directional-hemispherical reflectance including the soil boundary condition."),
    _v("rdot", kind="output", category="reflectance", units="-", meaning="Hemispherical-directional reflectance including the soil boundary condition."),
    _v("rsodt", kind="output", category="reflectance", units="-", meaning="Diffuse part of total bidirectional reflectance including the soil boundary condition."),
    _v("rsost", kind="output", category="reflectance", units="-", meaning="Hotspot or direct part of total bidirectional reflectance including the soil boundary condition."),
    _v("tss", kind="output", category="reflectance", units="-", meaning="Direct solar transmittance through the canopy.", aliases=("sun transmittance",)),
    _v("too", kind="output", category="reflectance", units="-", meaning="Direct transmittance along the observation path.", aliases=("observer transmittance",)),
    _v("tsstoo", kind="output", category="reflectance", units="-", meaning="Joint direct-transmittance term for the sun-observer hotspot path."),
    _v("rso_", kind="output", category="reflectance", units="-", meaning="Directional bidirectional reflectance factor on the requested angle grid.", aliases=("brdf_", "directional BRDF")),
    _v("gammasdf", kind="output", category="transport coefficients", units="-", meaning="Canopy transport coefficient for downward diffuse coupling used by fluorescence and reflectance transport."),
    _v("gammasdb", kind="output", category="transport coefficients", units="-", meaning="Canopy transport coefficient for upward diffuse coupling used by fluorescence and reflectance transport."),
    _v("gammaso", kind="output", category="transport coefficients", units="-", meaning="Canopy transport coefficient for escape toward the observation direction."),
    _v("Es_direct_", kind="output", category="profiles", units="W m-2 um-1", meaning="Direct shortwave irradiance profile at canopy interfaces."),
    _v("Emin_direct_", kind="output", category="profiles", units="W m-2 um-1", meaning="Downward shortwave flux profile originating from direct illumination."),
    _v("Eplu_direct_", kind="output", category="profiles", units="W m-2 um-1", meaning="Upward shortwave flux profile originating from direct illumination."),
    _v("Es_diffuse_", kind="output", category="profiles", units="W m-2 um-1", meaning="Diffuse shortwave irradiance profile at canopy interfaces."),
    _v("Emin_diffuse_", kind="output", category="profiles", units="W m-2 um-1", meaning="Downward diffuse shortwave flux profile at canopy interfaces."),
    _v("Eplu_diffuse_", kind="output", category="profiles", units="W m-2 um-1", meaning="Upward diffuse shortwave flux profile at canopy interfaces."),
    _v("leaf_fluor_back", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Backward leaf fluorescence source spectrum before canopy transport."),
    _v("leaf_fluor_forw", kind="output", category="fluorescence", units="W m-2 um-1", meaning="Forward leaf fluorescence source spectrum before canopy transport."),
    _v("layer_fluorescence", kind="output", category="profiles", units="W m-2", meaning="Upward fluorescence contribution aggregated per canopy layer.", aliases=("layer_fluorescence.dat",)),
    _v("sunlit", kind="output", category="structured outputs", units="LeafBiochemistryResult", meaning="Nested structured result holding sunlit leaf-biochemistry outputs."),
    _v("shaded", kind="output", category="structured outputs", units="LeafBiochemistryResult", meaning="Nested structured result holding shaded leaf-biochemistry outputs."),
    _v("sunlit_Cs_input", kind="output", category="solver diagnostics", units="ppm", meaning="CO2 boundary condition passed into the sunlit leaf solve during the final energy-balance iteration.", notes="Phase-lagged diagnostic; not the main same-state parity signal."),
    _v("shaded_Cs_input", kind="output", category="solver diagnostics", units="ppm", meaning="CO2 boundary condition passed into the shaded leaf solve during the final energy-balance iteration.", notes="Phase-lagged diagnostic; not the main same-state parity signal."),
    _v("sunlit_eb_input", kind="output", category="solver diagnostics", units="hPa", meaning="Leaf-surface vapor-pressure boundary condition passed into the sunlit leaf solve.", notes="Phase-lagged diagnostic; not the main same-state parity signal."),
    _v("shaded_eb_input", kind="output", category="solver diagnostics", units="hPa", meaning="Leaf-surface vapor-pressure boundary condition passed into the shaded leaf solve.", notes="Phase-lagged diagnostic; not the main same-state parity signal."),
    _v("sunlit_T_input", kind="output", category="solver diagnostics", units="degC", meaning="Leaf temperature passed into the sunlit leaf solve during the final iteration.", notes="Phase-lagged diagnostic; not the main same-state parity signal."),
    _v("shaded_T_input", kind="output", category="solver diagnostics", units="degC", meaning="Leaf temperature passed into the shaded leaf solve during the final iteration.", notes="Phase-lagged diagnostic; not the main same-state parity signal."),
    _v("Rnuc_sw", kind="output", category="energy balance", units="W m-2", meaning="Sunlit canopy net shortwave radiation."),
    _v("Rnhc_sw", kind="output", category="energy balance", units="W m-2", meaning="Shaded canopy net shortwave radiation."),
    _v("Rnus_sw", kind="output", category="energy balance", units="W m-2", meaning="Sunlit soil net shortwave radiation."),
    _v("Rnhs_sw", kind="output", category="energy balance", units="W m-2", meaning="Shaded soil net shortwave radiation."),
    _v("Rnuct", kind="output", category="energy balance", units="W m-2", meaning="Sunlit canopy net thermal radiation."),
    _v("Rnhct", kind="output", category="energy balance", units="W m-2", meaning="Shaded canopy net thermal radiation."),
    _v("Rnust", kind="output", category="energy balance", units="W m-2", meaning="Sunlit soil net thermal radiation."),
    _v("Rnhst", kind="output", category="energy balance", units="W m-2", meaning="Shaded soil net thermal radiation."),
    _v("Rnuc", kind="output", category="energy balance", units="W m-2", meaning="Total net radiation of the sunlit canopy fraction."),
    _v("Rnhc", kind="output", category="energy balance", units="W m-2", meaning="Total net radiation of the shaded canopy fraction."),
    _v("Rnus", kind="output", category="energy balance", units="W m-2", meaning="Total net radiation of the sunlit soil fraction."),
    _v("Rnhs", kind="output", category="energy balance", units="W m-2", meaning="Total net radiation of the shaded soil fraction."),
    _v("lEcu", kind="output", category="energy balance", units="W m-2", meaning="Latent heat flux of the sunlit canopy fraction."),
    _v("lEch", kind="output", category="energy balance", units="W m-2", meaning="Latent heat flux of the shaded canopy fraction."),
    _v("lEsu", kind="output", category="energy balance", units="W m-2", meaning="Latent heat flux of the sunlit soil fraction."),
    _v("lEsh", kind="output", category="energy balance", units="W m-2", meaning="Latent heat flux of the shaded soil fraction."),
    _v("Hcu", kind="output", category="energy balance", units="W m-2", meaning="Sensible heat flux of the sunlit canopy fraction."),
    _v("Hch", kind="output", category="energy balance", units="W m-2", meaning="Sensible heat flux of the shaded canopy fraction."),
    _v("Hsu", kind="output", category="energy balance", units="W m-2", meaning="Sensible heat flux of the sunlit soil fraction."),
    _v("Hsh", kind="output", category="energy balance", units="W m-2", meaning="Sensible heat flux of the shaded soil fraction."),
    _v("Gsu", kind="output", category="energy balance", units="W m-2", meaning="Soil heat flux associated with the sunlit soil fraction."),
    _v("Gsh", kind="output", category="energy balance", units="W m-2", meaning="Soil heat flux associated with the shaded soil fraction."),
)


def iter_variables() -> tuple[VariableDefinition, ...]:
    return VARIABLES


def search_variables(
    query: str | None = None,
    *,
    kind: str | None = None,
    category: str | None = None,
) -> list[VariableDefinition]:
    items = list(VARIABLES)
    if kind is not None:
        lowered = kind.lower()
        items = [item for item in items if item.kind.lower() == lowered]
    if category is not None:
        lowered = category.lower()
        items = [item for item in items if item.category.lower() == lowered]
    if not query:
        return items

    lowered_query = query.lower()
    matches = [item for item in items if lowered_query in item.search_blob()]
    if lowered_query.startswith("sunlit_") or lowered_query.startswith("shaded_"):
        prefix, _, base = lowered_query.partition("_")
        pattern = f"{prefix}_*"
        pattern_matches = [
            item
            for item in items
            if item.name.lower() == pattern or item.name.lower() == lowered_query or base in item.search_blob()
        ]
        combined = {item.name: item for item in matches}
        combined.update({item.name: item for item in pattern_matches})
        if combined:
            return sorted(combined.values(), key=lambda item: _match_rank(item, lowered_query))
    if matches:
        return sorted(matches, key=lambda item: _match_rank(item, lowered_query))
    return []


def _match_rank(item: VariableDefinition, lowered_query: str) -> tuple[int, int, str]:
    name = item.name.lower()
    aliases = tuple(alias.lower() for alias in item.aliases)
    if name == lowered_query:
        return (0, len(name), name)
    if lowered_query in aliases:
        return (1, len(name), name)
    if name.startswith(lowered_query):
        return (2, len(name), name)
    if any(alias.startswith(lowered_query) for alias in aliases):
        return (3, len(name), name)
    return (4, len(name), name)


def render_variable_markdown() -> str:
    lines: list[str] = [
        "# Variable Glossary",
        "",
        "This page is generated from the in-repo variable registry in `scope.variables`.",
        "",
        "Use it in two ways:",
        "",
        "- browser search through the MkDocs site search bar",
        "- terminal lookup with `scope vars <name>`",
        "",
        "The physical meanings are aligned to the current Python implementation and, where relevant, to the original SCOPE documentation at <https://scope-model.readthedocs.io/en/master/outfiles.html>.",
        "",
    ]

    kind_order = ("dimension", "option", "input", "output")
    for kind in kind_order:
        kind_items = [item for item in VARIABLES if item.kind == kind]
        if not kind_items:
            continue
        lines.append(f"## {kind.title()}s")
        lines.append("")
        category_names = []
        for item in kind_items:
            if item.category not in category_names:
                category_names.append(item.category)
        for category in category_names:
            lines.append(f"### {category.title()}")
            lines.append("")
            lines.extend(_render_table([item for item in kind_items if item.category == category]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_table(items: Iterable[VariableDefinition]) -> list[str]:
    lines = [
        "| Name | Units | Meaning | Workflows / aliases |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        workflow_text = ", ".join(item.workflows)
        alias_text = ", ".join(item.aliases)
        meta = "; ".join(part for part in (workflow_text, alias_text, item.notes) if part)
        lines.append(
            f"| `{item.name}` | {item.units} | {item.meaning} | {meta or '-'} |"
        )
    return lines
