# Variable Glossary

This page is generated from the in-repo variable registry in `scope.variables`.

Use it in two ways:

- browser search through the MkDocs site search bar
- terminal lookup with `scope vars <name>`

The physical meanings are aligned to the current Python implementation and, where relevant, to the original SCOPE documentation at <https://scope-model.readthedocs.io/en/master/outfiles.html>.

## Dimensions

### Grid

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `y` | - | Spatial row dimension of the ROI/grid. | - |
| `x` | - | Spatial column dimension of the ROI/grid. | - |
| `time` | datetime | Time axis for scenes or time-series runs. | - |

### Spectral

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `wavelength` | nm | Optical wavelength axis used by reflectance and shortwave forcing. | - |
| `excitation_wavelength` | nm | Excitation wavelength axis used to drive fluorescence source terms. | - |
| `fluorescence_wavelength` | nm | Fluorescence emission wavelength axis. | - |
| `thermal_wavelength` | um | Thermal wavelength axis used by longwave radiance outputs. | - |

### Canopy

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `layer` | - | Within-canopy layer axis for layered transport and physiology fields. | - |
| `layer_interface` | - | Layer-interface axis for cumulative transport profiles. | - |

### Geometry

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `direction` | - | Index over directional viewing geometries. | - |
| `directional_tto` | deg | Viewing zenith angles for directional outputs. | tto grid |
| `directional_psi` | deg | Relative azimuth angles for directional outputs. | psi grid |

## Options

### Workflow

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `calc_fluor` | 0/1 | Enables fluorescence workflows in high-level runner dispatch. | options.calc_fluor |
| `calc_planck` | 0/1 | Enables thermal/Planck workflows in high-level runner dispatch. | - |
| `calc_directional` | 0/1 | Requests directional products for the selected workflows. | - |
| `calc_vert_profiles` | 0/1 | Requests vertical-profile products for the selected workflows. | - |
| `soil_heat_method` | index | Selects the soil heat-flux treatment in coupled energy-balance workflows. | - |
| `mSCOPE` | 0/1 | Upstream vertically heterogeneous leaf-optics mode. Present as metadata only; not implemented in this Python stack. | - |
| `lite` | 0/1 | Upstream SCOPE lite-mode flag carried through prepared dataset attrs. | - |
| `calc_xanthophyllabs` | 0/1 | Upstream flag for xanthophyll absorption treatment. Present as metadata in the current Python stack. | - |
| `Fluorescence_model` | index | Upstream fluorescence-model selector. Present as metadata in the current Python stack. | - |
| `apply_T_corr` | 0/1 | Upstream temperature-correction flag carried with prepared dataset attrs. | - |
| `verify` | 0/1 | Upstream verification-mode flag carried with prepared dataset attrs. | - |
| `calc_rss_rbs` | 0/1 | Upstream flag controlling calculation of soil and boundary resistances. | - |
| `MoninObukhov` | 0/1 | Flag controlling Monin-Obukhov stability correction usage. | - |
| `save_spectral` | 0/1 | Upstream flag requesting spectral outputs to be saved. | - |
| `soilspectrum` | index | Upstream soil-spectrum mode flag or selector. | soil_spectrum option |

## Inputs

### Leaf Biochemistry

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Cab` | ug cm-2 | Leaf chlorophyll a+b content. | reflectance, fluorescence, thermal, energy balance; chlorophyll |
| `Cca` | ug cm-2 | Leaf carotenoid content. | carotenoids |
| `Cw` | g cm-2 | Equivalent leaf water thickness. | leaf water |
| `Cdm` | g cm-2 | Leaf dry matter content per area. | Cm, dry matter |
| `Cs` | - | Senescent or brown pigment content used by FLUSPECT. | cbrown |
| `Cant` | ug cm-2 | Anthocyanin content. | - |
| `Cbc` | ug cm-2 | Brown carbon or additional biochemical absorber term when provided. | - |
| `Cp` | g cm-2 | Protein content parameter for extended leaf optics. | - |
| `N` | - | Leaf mesophyll structure parameter in PROSPECT/FLUSPECT. | - |
| `fqe` | - | Leaf fluorescence quantum efficiency scaling used to build fluorescence source terms. | fluorescence efficiency |

### Leaf Structure

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `ala` | deg | Leaf inclination angle or Campbell mean leaf angle input used to derive the LIDF. | LIDFa |

### Canopy Structure

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `LAI` | m2 m-2 | Leaf area index. | - |

### Geometry

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `tts` | deg | Solar zenith angle. | sun zenith |
| `tto` | deg | Viewing zenith angle. | observer zenith |
| `psi` | deg | Relative azimuth angle between sun and sensor. | relative azimuth |

### Soil

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `soil_refl` | - | Explicit soil reflectance spectrum on the model optical wavelength grid. | - |
| `soil_spectrum` | index | Index into the upstream SCOPE soil reflectance library. | soilspectrum |
| `BSMBrightness` | - | Brightness parameter for the SCOPE BSM soil model. | - |
| `BSMlat` | deg | Latitude-like mineral composition parameter used by the SCOPE BSM soil model. | - |
| `BSMlon` | deg | Longitude-like mineral composition parameter used by the SCOPE BSM soil model. | - |
| `SMC` | fraction | Soil moisture content used by the BSM soil model. | - |
| `rss` | s m-1 | Soil resistance to evaporation. | soil surface resistance |
| `rbs` | s m-1 | Boundary resistance near the soil surface. | - |
| `GAM` | W m-2 | Optional soil heat flux forcing or initialization term. | - |
| `Tsold` | degC | Previous-step soil temperature state for transient soil heat treatment. | - |
| `dt_seconds` | s | Time-step length used by transient soil heat treatment. | - |

### Spectral Forcing

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Esun_` | W m-2 um-1 | Direct excitation irradiance spectrum for fluorescence. | - |
| `Esky_` | W m-2 um-1 | Diffuse excitation irradiance spectrum for fluorescence. | - |
| `Esun_sw` | W m-2 um-1 | Direct shortwave irradiance spectrum used by reflectance and energy-balance workflows. | - |
| `Esky_sw` | W m-2 um-1 | Diffuse shortwave irradiance spectrum used by reflectance and energy-balance workflows. | - |

### Fluorescence Transport

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `etau` | - | Forward fluorescence efficiency or source scaling per canopy layer. | Used in layered and directional fluorescence paths. |
| `etah` | - | Backward fluorescence efficiency or source scaling per canopy layer. | Used in layered and directional fluorescence paths. |

### Meteorology

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Ta` | degC | Air temperature. | - |
| `ea` | hPa | Ambient vapor pressure. | - |
| `Ca` | ppm | Ambient CO2 concentration. | - |
| `Oa` | permil or mmol mol-1 | Ambient oxygen concentration used by leaf biochemistry. | - |
| `p` | hPa | Air pressure. | - |
| `z` | m | Reference meteorological height. | - |
| `u` | m s-1 | Wind speed at the reference height. | - |
| `L` | m | Monin-Obukhov length when supplied as an external stability input. | - |

### Canopy Aerodynamics

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Cd` | - | Drag coefficient for canopy aerodynamic exchange. | - |
| `rwc` | - | Relative water content or canopy resistance scaling parameter used by the energy-balance closure. | - |
| `z0m` | m | Momentum roughness length. | - |
| `d` | m | Zero-plane displacement height. | - |
| `h` | m | Canopy height. | - |
| `kV` | - | Vertical extinction or partitioning parameter for layered canopy coupling. | - |
| `fV` | - | Vertical partitioning profile used by layered canopy physiology and energy balance. | - |

### Thermal Optics

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `rho_thermal` | - | Thermal reflectance of the soil surface. | - |
| `tau_thermal` | - | Thermal transmittance of the canopy in the simplified thermal optics parameterization. | - |
| `rs_thermal` | - | Thermal soil reflectance/emissivity control in the simplified thermal optics parameterization. | - |

### Thermal State

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Tcu` | degC | Sunlit canopy temperature. | - |
| `Tch` | degC | Shaded canopy temperature. | - |
| `Tsu` | degC | Sunlit soil temperature. | - |
| `Tsh` | degC | Shaded soil temperature. | - |

### Boundary State

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Csu` | ppm | CO2 concentration at the sunlit leaf surface. | - |
| `Csh` | ppm | CO2 concentration at the shaded leaf surface. | - |
| `ebu` | hPa | Vapor pressure at the sunlit leaf surface. | - |
| `ebh` | hPa | Vapor pressure at the shaded leaf surface. | - |

## Outputs

### Reflectance

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `leaf_refl` | - | Leaf hemispherical reflectance from FLUSPECT. | - |
| `leaf_tran` | - | Leaf hemispherical transmittance from FLUSPECT. | - |
| `rsot` | - | Total top-of-canopy reflectance factor in the observation direction. | apparent reflectance, reflectance.csv |
| `rso` | - | Bidirectional reflectance factor. | BRF; Matches the original SCOPE rso definition. |
| `rsd` | - | Directional-hemispherical reflectance factor. | DHRF |
| `rdd` | - | Bi-hemispherical reflectance factor. | BHRF |
| `rdo` | - | Hemispherical-directional reflectance factor. | HDRF |
| `refl_` | - | Directional apparent reflectance for a requested angle grid. | - |
| `tdd` | - | Diffuse transmittance for diffuse incident radiation through the canopy. | - |
| `tsd` | - | Direct-to-diffuse canopy transmittance for solar illumination. | - |
| `tdo` | - | Diffuse transmittance from canopy layers toward the observation direction. | - |
| `rsos` | - | Bidirectional reflectance contribution from the hotspot or single-scattering term. | - |
| `rsod` | - | Bidirectional reflectance contribution from the diffuse multiple-scattering term. | - |
| `rddt` | - | Bi-hemispherical reflectance including the soil boundary condition. | - |
| `rsdt` | - | Directional-hemispherical reflectance including the soil boundary condition. | - |
| `rdot` | - | Hemispherical-directional reflectance including the soil boundary condition. | - |
| `rsodt` | - | Diffuse part of total bidirectional reflectance including the soil boundary condition. | - |
| `rsost` | - | Hotspot or direct part of total bidirectional reflectance including the soil boundary condition. | - |
| `tss` | - | Direct solar transmittance through the canopy. | sun transmittance |
| `too` | - | Direct transmittance along the observation path. | observer transmittance |
| `tsstoo` | - | Joint direct-transmittance term for the sun-observer hotspot path. | - |
| `rso_` | - | Directional bidirectional reflectance factor on the requested angle grid. | brdf_, directional BRDF |

### Profiles

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Ps` | - | Cumulative direct-solar interception profile along canopy depth. | - |
| `Po` | - | Cumulative observation-path attenuation profile along canopy depth. | - |
| `Pso` | - | Joint sun-observer hotspot weighting profile across the canopy. | - |
| `Es_` | W m-2 um-1 | Downward direct-plus-diffuse shortwave flux profile at canopy interfaces. | - |
| `Emin_` | W m-2 um-1 | Downward diffuse shortwave flux profile at canopy interfaces. | - |
| `Eplu_` | W m-2 um-1 | Upward diffuse shortwave flux profile at canopy interfaces. | - |
| `layer_thermal_upward` | W m-2 | Integrated upward thermal transport per canopy layer. | - |
| `Es_direct_` | W m-2 um-1 | Direct shortwave irradiance profile at canopy interfaces. | - |
| `Emin_direct_` | W m-2 um-1 | Downward shortwave flux profile originating from direct illumination. | - |
| `Eplu_direct_` | W m-2 um-1 | Upward shortwave flux profile originating from direct illumination. | - |
| `Es_diffuse_` | W m-2 um-1 | Diffuse shortwave irradiance profile at canopy interfaces. | - |
| `Emin_diffuse_` | W m-2 um-1 | Downward diffuse shortwave flux profile at canopy interfaces. | - |
| `Eplu_diffuse_` | W m-2 um-1 | Upward diffuse shortwave flux profile at canopy interfaces. | - |
| `layer_fluorescence` | W m-2 | Upward fluorescence contribution aggregated per canopy layer. | layer_fluorescence.dat |

### Fluorescence

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `LoF_` | W m-2 um-1 sr-1 | Fluorescence spectrum in the observation direction. | fluorescence.csv |
| `LoF_sunlit` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution from sunlit leaves. | - |
| `LoF_shaded` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution from shaded leaves. | - |
| `LoF_scattered` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution from multiply scattered canopy radiation. | - |
| `LoF_soil` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution associated with the soil boundary term. | - |
| `EoutF_` | W m-2 um-1 | Top-of-canopy hemispherically integrated fluorescence. | fluorescence_hemis.csv |
| `EoutFrc_` | W m-2 um-1 | Reabsorption-corrected hemispherical fluorescence spectrum. | fluorescence_ReabsCorr.csv |
| `Femleaves_` | W m-2 um-1 | Fluorescence emitted by all leaves before canopy escape losses. | fluorescence_AllLeaves.csv |
| `sigmaF` | - | Fluorescence escape probability. | sigmaF.csv |
| `F685` | W m-2 um-1 sr-1 | First fluorescence peak radiance. | F_1stpeak |
| `wl685` | nm | Wavelength of the first fluorescence peak. | wl_1stpeak |
| `F740` | W m-2 um-1 sr-1 | Second fluorescence peak radiance. | F_2ndpeak |
| `wl740` | nm | Wavelength of the second fluorescence peak. | wl_2ndpeak |
| `F684` | W m-2 um-1 sr-1 | Fluorescence radiance sampled near 684-687 nm. | F687 |
| `F761` | W m-2 um-1 sr-1 | Fluorescence radiance sampled near 760-761 nm. | F760 |
| `LoutF` | W m-2 um-1 sr-1 | Integrated observation-direction fluorescence scalar. | LFtot |
| `EoutF` | W m-2 | Integrated hemispherical fluorescence scalar. | EFtot |
| `Fmin_` | W m-2 um-1 | Downward or internal fluorescence transport profile term at canopy interfaces. | - |
| `Fplu_` | W m-2 um-1 | Upward fluorescence transport profile term at canopy interfaces. | - |
| `leaf_fluor_back` | W m-2 um-1 | Backward leaf fluorescence source spectrum before canopy transport. | - |
| `leaf_fluor_forw` | W m-2 um-1 | Forward leaf fluorescence source spectrum before canopy transport. | - |

### Thermal

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Lot_` | W m-2 um-1 sr-1 | Thermal radiance spectrum in the observation direction. | - |
| `Eoutte_` | W m-2 um-1 | Emitted longwave flux spectrum leaving the canopy-soil system. | - |
| `Emint_` | W m-2 um-1 | Downward internal thermal flux profile. | - |
| `Eplut_` | W m-2 um-1 | Upward internal thermal flux profile. | - |
| `LotBB_` | W m-2 um-1 sr-1 | Blackbody-equivalent observation-direction thermal radiance. | - |
| `Loutt` | W m-2 um-1 sr-1 | Integrated observation-direction thermal radiance scalar. | - |
| `Eoutt` | W m-2 | Integrated hemispherical thermal flux scalar. | - |
| `BrightnessT` | K | Brightness temperature inferred from the directional thermal radiance. | - |
| `canopyemis` | - | Effective canopy emissivity. | - |

### Physiology

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Pnu_Cab` | umol m-2 s-1 | Sunlit absorbed PAR by chlorophyll a+b used to drive leaf biochemistry. | - |
| `Pnh_Cab` | umol m-2 s-1 | Shaded absorbed PAR by chlorophyll a+b used to drive leaf biochemistry. | - |
| `sunlit_*` | varies | Prefix used for leaf-biochemistry outputs of sunlit leaves. | Search for base names such as A, Ci, rcw, eta, or Ja. |
| `shaded_*` | varies | Prefix used for leaf-biochemistry outputs of shaded leaves. | Search for base names such as A, Ci, rcw, eta, or Ja. |
| `sunlit_A` | umol m-2 s-1 | Net assimilation rate of sunlit leaves. | - |
| `shaded_A` | umol m-2 s-1 | Net assimilation rate of shaded leaves. | - |
| `sunlit_Ci` | ppm | Intercellular CO2 concentration of sunlit leaves. | - |
| `shaded_Ci` | ppm | Intercellular CO2 concentration of shaded leaves. | - |
| `sunlit_rcw` | s m-1 | Canopy water-vapor resistance of sunlit leaves. | - |
| `shaded_rcw` | s m-1 | Canopy water-vapor resistance of shaded leaves. | - |
| `sunlit_eta` | - | Fluorescence efficiency term returned by the sunlit leaf biochemistry solve. | - |
| `shaded_eta` | - | Fluorescence efficiency term returned by the shaded leaf biochemistry solve. | - |

### Energy Balance

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Rnctot` | W m-2 | Net radiation of the canopy. | fluxes.csv |
| `lEctot` | W m-2 | Latent heat flux of the canopy (transpiration). | - |
| `Hctot` | W m-2 | Sensible heat flux of the canopy. | - |
| `Actot` | umol m-2 s-1 | Net photosynthesis of the canopy. | Photosynthesis |
| `Tcave` | degC | Average canopy temperature. | - |
| `Rnstot` | W m-2 | Net radiation of the soil. | - |
| `lEstot` | W m-2 | Latent heat flux of the soil (evaporation). | - |
| `Hstot` | W m-2 | Sensible heat flux of the soil. | - |
| `Gtot` | W m-2 | Soil heat flux. | - |
| `Tsave` | degC | Average soil temperature. | - |
| `Rntot` | W m-2 | Total net radiation. | - |
| `lEtot` | W m-2 | Total latent heat flux. | - |
| `Htot` | W m-2 | Total sensible heat flux. | - |
| `Rnuc_sw` | W m-2 | Sunlit canopy net shortwave radiation. | - |
| `Rnhc_sw` | W m-2 | Shaded canopy net shortwave radiation. | - |
| `Rnus_sw` | W m-2 | Sunlit soil net shortwave radiation. | - |
| `Rnhs_sw` | W m-2 | Shaded soil net shortwave radiation. | - |
| `Rnuct` | W m-2 | Sunlit canopy net thermal radiation. | - |
| `Rnhct` | W m-2 | Shaded canopy net thermal radiation. | - |
| `Rnust` | W m-2 | Sunlit soil net thermal radiation. | - |
| `Rnhst` | W m-2 | Shaded soil net thermal radiation. | - |
| `Rnuc` | W m-2 | Total net radiation of the sunlit canopy fraction. | - |
| `Rnhc` | W m-2 | Total net radiation of the shaded canopy fraction. | - |
| `Rnus` | W m-2 | Total net radiation of the sunlit soil fraction. | - |
| `Rnhs` | W m-2 | Total net radiation of the shaded soil fraction. | - |
| `lEcu` | W m-2 | Latent heat flux of the sunlit canopy fraction. | - |
| `lEch` | W m-2 | Latent heat flux of the shaded canopy fraction. | - |
| `lEsu` | W m-2 | Latent heat flux of the sunlit soil fraction. | - |
| `lEsh` | W m-2 | Latent heat flux of the shaded soil fraction. | - |
| `Hcu` | W m-2 | Sensible heat flux of the sunlit canopy fraction. | - |
| `Hch` | W m-2 | Sensible heat flux of the shaded canopy fraction. | - |
| `Hsu` | W m-2 | Sensible heat flux of the sunlit soil fraction. | - |
| `Hsh` | W m-2 | Sensible heat flux of the shaded soil fraction. | - |
| `Gsu` | W m-2 | Soil heat flux associated with the sunlit soil fraction. | - |
| `Gsh` | W m-2 | Soil heat flux associated with the shaded soil fraction. | - |

### Resistance

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `raa` | s m-1 | Aerodynamic resistance above the canopy. | - |
| `raws` | s m-1 | Aerodynamic resistance within the soil/canopy lower boundary layer. | within-soil aerodynamic resistance |
| `rawc` | s m-1 | Aerodynamic resistance within the canopy air space. | - |
| `rac` | s m-1 | Boundary-layer resistance for canopy heat and vapor exchange. | - |
| `ras` | s m-1 | Boundary-layer resistance for soil heat and vapor exchange. | - |
| `ustar` | m s-1 | Friction velocity. | - |

### Boundary State

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Csu` | ppm | Solved CO2 concentration at the sunlit leaf surface. | - |
| `Csh` | ppm | Solved CO2 concentration at the shaded leaf surface. | - |
| `ebu` | hPa | Solved vapor pressure at the sunlit leaf surface. | - |
| `ebh` | hPa | Solved vapor pressure at the shaded leaf surface. | - |

### Thermal State

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `Tcu` | degC | Solved sunlit canopy temperature. | - |
| `Tch` | degC | Solved shaded canopy temperature. | - |
| `Tsu` | degC | Solved sunlit soil temperature. | - |
| `Tsh` | degC | Solved shaded soil temperature. | - |

### Solver

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `max_error` | W m-2 | Maximum residual or closure error in the final energy-balance iteration. | - |
| `converged` | 0/1 | Whether the coupled energy-balance iteration converged. | - |
| `counter` | - | Number of energy-balance iterations performed. | - |

### Dataset Metadata

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `scope_product` | - | Dataset attribute naming the assembled product or workflow. | - |
| `scope_components` | - | Dataset attribute listing merged workflow components. | - |

### Namespaced Outputs

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `reflectance_directional_*` | varies | Prefix applied to directional reflectance variables in merged workflow datasets. | - |
| `reflectance_profile_*` | varies | Prefix applied to reflectance profile variables in merged workflow datasets. | - |
| `fluorescence_directional_*` | varies | Prefix applied to directional fluorescence variables in merged workflow datasets. | - |
| `fluorescence_profile_*` | varies | Prefix applied to fluorescence profile variables in merged workflow datasets. | - |
| `thermal_directional_*` | varies | Prefix applied to directional thermal variables in merged workflow datasets. | - |
| `thermal_profile_*` | varies | Prefix applied to thermal profile variables in merged workflow datasets. | - |

### Transport Coefficients

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `gammasdf` | - | Canopy transport coefficient for downward diffuse coupling used by fluorescence and reflectance transport. | - |
| `gammasdb` | - | Canopy transport coefficient for upward diffuse coupling used by fluorescence and reflectance transport. | - |
| `gammaso` | - | Canopy transport coefficient for escape toward the observation direction. | - |

### Structured Outputs

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `sunlit` | LeafBiochemistryResult | Nested structured result holding sunlit leaf-biochemistry outputs. | - |
| `shaded` | LeafBiochemistryResult | Nested structured result holding shaded leaf-biochemistry outputs. | - |

### Solver Diagnostics

| Name | Units | Meaning | Workflows / aliases |
| --- | --- | --- | --- |
| `sunlit_Cs_input` | ppm | CO2 boundary condition passed into the sunlit leaf solve during the final energy-balance iteration. | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_Cs_input` | ppm | CO2 boundary condition passed into the shaded leaf solve during the final energy-balance iteration. | Phase-lagged diagnostic; not the main same-state parity signal. |
| `sunlit_eb_input` | hPa | Leaf-surface vapor-pressure boundary condition passed into the sunlit leaf solve. | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_eb_input` | hPa | Leaf-surface vapor-pressure boundary condition passed into the shaded leaf solve. | Phase-lagged diagnostic; not the main same-state parity signal. |
| `sunlit_T_input` | degC | Leaf temperature passed into the sunlit leaf solve during the final iteration. | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_T_input` | degC | Leaf temperature passed into the shaded leaf solve during the final iteration. | Phase-lagged diagnostic; not the main same-state parity signal. |
