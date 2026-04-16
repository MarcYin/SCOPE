# Variable Glossary

This page is generated from the in-repo variable registry in `scope.variables`.

Use it in two ways:

- browser search through the MkDocs site search bar
- terminal lookup with `scope vars <name>`
- workflow filtering with `scope vars --workflow fluorescence`

The physical meanings are aligned to the current Python implementation and, where relevant, to the original SCOPE documentation at <https://scope-model.readthedocs.io/en/master/outfiles.html>.

## Dimensions

### Grid

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `y` | - | Spatial row dimension of the ROI/grid. | - | docs/input-output-reference.md | - |
| `x` | - | Spatial column dimension of the ROI/grid. | - | docs/input-output-reference.md | - |
| `time` | datetime | Time axis for scenes or time-series runs. | - | docs/input-output-reference.md | - |

### Spectral

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `wavelength` | nm | Optical wavelength axis used by reflectance and shortwave forcing. | - | docs/input-output-reference.md | - |
| `excitation_wavelength` | nm | Excitation wavelength axis used to drive fluorescence source terms. | - | docs/input-output-reference.md | - |
| `fluorescence_wavelength` | nm | Fluorescence emission wavelength axis. | - | docs/input-output-reference.md | - |
| `thermal_wavelength` | um | Thermal wavelength axis used by longwave radiance outputs. | - | docs/input-output-reference.md | - |

### Canopy

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `layer` | - | Within-canopy layer axis for layered transport and physiology fields. | - | docs/input-output-reference.md | - |
| `layer_interface` | - | Layer-interface axis for cumulative transport profiles. | - | docs/input-output-reference.md | - |

### Geometry

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `direction` | - | Index over directional viewing geometries. | - | docs/input-output-reference.md | - |
| `directional_tto` | deg | Viewing zenith angles for directional outputs. | - | docs/input-output-reference.md | tto grid |
| `directional_psi` | deg | Relative azimuth angles for directional outputs. | - | docs/input-output-reference.md | psi grid |

## Options

### Workflow

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `calc_ebal` | 0/1 | Requests coupled energy-balance closure in high-level runner dispatch. | - | docs/input-output-reference.md | options.calc_ebal |
| `calc_fluor` | 0/1 | Enables fluorescence workflows in high-level runner dispatch. When calc_ebal is true, this uses the coupled energy-balance fluorescence path. | - | docs/input-output-reference.md | options.calc_fluor |
| `calc_planck` | 0/1 | Enables thermal/Planck workflows in high-level runner dispatch. When calc_ebal is true, this uses the coupled energy-balance thermal path. | - | docs/input-output-reference.md | - |
| `calc_directional` | 0/1 | Requests directional products for the selected workflows. | - | docs/input-output-reference.md | - |
| `calc_vert_profiles` | 0/1 | Requests vertical-profile products for the selected workflows. | - | docs/input-output-reference.md | - |
| `soil_heat_method` | index | Selects the soil heat-flux treatment in coupled energy-balance workflows. | - | docs/input-output-reference.md | - |
| `mSCOPE` | 0/1 | Upstream vertically heterogeneous leaf-optics mode. Present as metadata only; not implemented in this Python stack. | - | docs/input-output-reference.md | - |
| `lite` | 0/1 | Upstream SCOPE lite-mode flag carried through prepared dataset attrs. | - | docs/input-output-reference.md | - |
| `calc_xanthophyllabs` | 0/1 | Upstream flag for xanthophyll absorption treatment. Present as metadata in the current Python stack. | - | docs/input-output-reference.md | - |
| `Fluorescence_model` | index | Upstream fluorescence-model selector. Present as metadata in the current Python stack. | - | docs/input-output-reference.md | - |
| `apply_T_corr` | 0/1 | Upstream temperature-correction flag carried with prepared dataset attrs. | - | docs/input-output-reference.md | - |
| `verify` | 0/1 | Upstream verification-mode flag carried with prepared dataset attrs. | - | docs/input-output-reference.md | - |
| `calc_rss_rbs` | 0/1 | Upstream flag controlling calculation of soil and boundary resistances. | - | docs/input-output-reference.md | - |
| `MoninObukhov` | 0/1 | Flag controlling Monin-Obukhov stability correction usage. | - | docs/input-output-reference.md | - |
| `save_spectral` | 0/1 | Upstream flag requesting spectral outputs to be saved. | - | docs/input-output-reference.md | - |
| `soilspectrum` | index | Upstream soil-spectrum mode flag or selector. | - | docs/input-output-reference.md | soil_spectrum option |

## Inputs

### Leaf Biochemistry

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Cab` | ug cm-2 | Leaf chlorophyll a+b content. | - | docs/input-output-reference.md | reflectance, fluorescence, thermal, energy balance; chlorophyll |
| `Cca` | ug cm-2 | Leaf carotenoid content. | - | docs/input-output-reference.md | carotenoids |
| `Cw` | g cm-2 | Equivalent leaf water thickness. | - | docs/input-output-reference.md | leaf water |
| `Cdm` | g cm-2 | Leaf dry matter content per area. | - | docs/input-output-reference.md | Cm, dry matter |
| `Cs` | - | Senescent or brown pigment content used by FLUSPECT. | - | docs/input-output-reference.md | cbrown |
| `Cant` | ug cm-2 | Anthocyanin content. | - | docs/input-output-reference.md | - |
| `Cbc` | ug cm-2 | Brown carbon or additional biochemical absorber term when provided. | - | docs/input-output-reference.md | - |
| `Cp` | g cm-2 | Protein content parameter for extended leaf optics. | - | docs/input-output-reference.md | - |
| `N` | - | Leaf mesophyll structure parameter in PROSPECT/FLUSPECT. | - | docs/input-output-reference.md | - |
| `fqe` | - | Leaf fluorescence quantum efficiency scaling used to build fluorescence source terms. | - | docs/input-output-reference.md | fluorescence efficiency |

### Canopy Structure

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `ala` | deg | Mean leaf angle parameter used to define the canopy leaf angle distribution function (LIDF). | Controls the overall canopy inclination distribution used by the LIDF. | docs/input-output-reference.md | LIDFa |
| `LAI` | m2 m-2 | Leaf area index. | - | docs/input-output-reference.md | - |

### Geometry

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `tts` | deg | Solar zenith angle. | - | docs/input-output-reference.md | sun zenith |
| `tto` | deg | Viewing zenith angle. | - | docs/input-output-reference.md | observer zenith |
| `psi` | deg | Relative azimuth angle between sun and sensor. | - | docs/input-output-reference.md | relative azimuth |

### Soil

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `soil_refl` | - | Explicit soil reflectance spectrum on the model optical wavelength grid. | - | docs/input-output-reference.md | - |
| `soil_spectrum` | index | Index into the upstream SCOPE soil reflectance library. | - | docs/input-output-reference.md | soilspectrum |
| `BSMBrightness` | - | Brightness parameter for the SCOPE BSM soil model. | - | docs/input-output-reference.md | - |
| `BSMlat` | deg | Latitude-like mineral composition parameter used by the SCOPE BSM soil model. | - | docs/input-output-reference.md | - |
| `BSMlon` | deg | Longitude-like mineral composition parameter used by the SCOPE BSM soil model. | - | docs/input-output-reference.md | - |
| `SMC` | fraction | Soil moisture content used by the BSM soil model. | - | docs/input-output-reference.md | - |
| `rss` | s m-1 | Soil resistance to evaporation. | - | docs/input-output-reference.md | soil surface resistance |
| `rbs` | s m-1 | Boundary resistance near the soil surface. | - | docs/input-output-reference.md | - |
| `GAM` | W m-2 | Optional soil heat flux forcing or initialization term. | - | docs/input-output-reference.md | - |
| `Tsold` | degC | Previous-step soil temperature state for transient soil heat treatment. | - | docs/input-output-reference.md | - |
| `dt_seconds` | s | Time-step length used by transient soil heat treatment. | - | docs/input-output-reference.md | - |

### Spectral Forcing

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Esun_` | W m-2 um-1 | Direct excitation irradiance spectrum for fluorescence. | - | docs/input-output-reference.md | - |
| `Esky_` | W m-2 um-1 | Diffuse excitation irradiance spectrum for fluorescence. | - | docs/input-output-reference.md | - |
| `Esun_sw` | W m-2 um-1 | Direct shortwave irradiance spectrum used by reflectance and energy-balance workflows. | - | docs/input-output-reference.md | - |
| `Esky_sw` | W m-2 um-1 | Diffuse shortwave irradiance spectrum used by reflectance and energy-balance workflows. | - | docs/input-output-reference.md | - |
| `Esun_lw` | W m-2 um-1 | Direct longwave irradiance spectrum used by coupled energy-balance workflows. | - | docs/input-output-reference.md | - |
| `Esky_lw` | W m-2 um-1 | Diffuse longwave irradiance spectrum used by coupled energy-balance workflows. | - | docs/input-output-reference.md | - |

### Fluorescence Transport

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `etau` | - | Forward fluorescence efficiency or source scaling per canopy layer. | - | docs/input-output-reference.md | Used in layered and directional fluorescence paths. |
| `etah` | - | Backward fluorescence efficiency or source scaling per canopy layer. | - | docs/input-output-reference.md | Used in layered and directional fluorescence paths. |

### Meteorology

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Ta` | degC | Air temperature. | - | docs/input-output-reference.md | - |
| `ea` | hPa | Ambient vapor pressure. | - | docs/input-output-reference.md | - |
| `Ca` | ppm | Ambient CO2 concentration. | - | docs/input-output-reference.md | - |
| `Oa` | permil or mmol mol-1 | Ambient oxygen concentration used by leaf biochemistry. | - | docs/input-output-reference.md | - |
| `p` | hPa | Air pressure. | - | docs/input-output-reference.md | - |
| `z` | m | Reference meteorological height. | - | docs/input-output-reference.md | - |
| `u` | m s-1 | Wind speed at the reference height. | - | docs/input-output-reference.md | - |
| `L` | m | Monin-Obukhov length when supplied as an external stability input. | - | docs/input-output-reference.md | - |

### Canopy Aerodynamics

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Cd` | - | Drag coefficient for canopy aerodynamic exchange. | - | docs/input-output-reference.md | - |
| `rwc` | - | Relative water content or canopy resistance scaling parameter used by the energy-balance closure. | - | docs/input-output-reference.md | - |
| `z0m` | m | Momentum roughness length. | - | docs/input-output-reference.md | - |
| `d` | m | Zero-plane displacement height. | - | docs/input-output-reference.md | - |
| `h` | m | Canopy height. | - | docs/input-output-reference.md | - |
| `kV` | - | Vertical extinction or partitioning parameter for layered canopy coupling. | - | docs/input-output-reference.md | - |
| `fV` | - | Vertical partitioning profile used by layered canopy physiology and energy balance. | - | docs/input-output-reference.md | - |

### Thermal Optics

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `rho_thermal` | - | Thermal reflectance of the soil surface. | - | docs/input-output-reference.md | - |
| `tau_thermal` | - | Thermal transmittance of the canopy in the simplified thermal optics parameterization. | - | docs/input-output-reference.md | - |
| `rs_thermal` | - | Thermal soil reflectance/emissivity control in the simplified thermal optics parameterization. | - | docs/input-output-reference.md | - |

### Thermal State

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Tcu` | degC | Sunlit canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tch` | degC | Shaded canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsu` | degC | Sunlit soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsh` | degC | Shaded soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Boundary State

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Csu` | ppm | CO2 concentration at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Csh` | ppm | CO2 concentration at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebu` | hPa | Vapor pressure at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebh` | hPa | Vapor pressure at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

## Outputs

### Reflectance

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `leaf_refl` | - | Leaf hemispherical reflectance from FLUSPECT. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `leaf_tran` | - | Leaf hemispherical transmittance from FLUSPECT. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsot` | - | Total top-of-canopy reflectance factor in the observation direction. | rsot = rsost + rsodt | https://scope-model.readthedocs.io/en/master/outfiles.html | apparent reflectance, reflectance.csv |
| `rso` | - | Bidirectional reflectance factor. | rso = rsos + rsod | https://scope-model.readthedocs.io/en/master/outfiles.html | BRF; Matches the original SCOPE rso definition. |
| `rsd` | - | Directional-hemispherical reflectance factor. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | DHRF |
| `rdd` | - | Bi-hemispherical reflectance factor. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | BHRF |
| `rdo` | - | Hemispherical-directional reflectance factor. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | HDRF |
| `refl_` | - | Directional apparent reflectance for a requested angle grid. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tdd` | - | Diffuse transmittance for diffuse incident radiation through the canopy. | Diffuse in -> diffuse out canopy transmittance term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tsd` | - | Direct-to-diffuse canopy transmittance for solar illumination. | Direct sun -> diffuse canopy transmittance term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tdo` | - | Diffuse transmittance from canopy layers toward the observation direction. | Diffuse canopy field -> observation-direction transmittance term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsos` | - | Bidirectional reflectance contribution from the hotspot or single-scattering term. | Direct/hotspot contribution to rso. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsod` | - | Bidirectional reflectance contribution from the diffuse multiple-scattering term. | Diffuse multiple-scattering contribution to rso. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rddt` | - | Bi-hemispherical reflectance including the soil boundary condition. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsdt` | - | Directional-hemispherical reflectance including the soil boundary condition. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rdot` | - | Hemispherical-directional reflectance including the soil boundary condition. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsodt` | - | Diffuse part of total bidirectional reflectance including the soil boundary condition. | Diffuse multiple-scattering part of rsot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rsost` | - | Hotspot or direct part of total bidirectional reflectance including the soil boundary condition. | Direct/hotspot part of rsot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `tss` | - | Direct solar transmittance through the canopy. | Beer-Lambert-like direct sun transmission term. | https://scope-model.readthedocs.io/en/master/outfiles.html | sun transmittance |
| `too` | - | Direct transmittance along the observation path. | Beer-Lambert-like observation-path transmission term. | https://scope-model.readthedocs.io/en/master/outfiles.html | observer transmittance |
| `tsstoo` | - | Joint direct-transmittance term for the sun-observer hotspot path. | Combined direct sun and observation-path transmission term. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rso_` | - | Directional bidirectional reflectance factor on the requested angle grid. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | brdf_, directional BRDF |

### Profiles

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Ps` | - | Cumulative direct-solar interception profile along canopy depth. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Po` | - | Cumulative observation-path attenuation profile along canopy depth. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Pso` | - | Joint sun-observer hotspot weighting profile across the canopy. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Es_` | W m-2 um-1 | Downward direct-plus-diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emin_` | W m-2 um-1 | Downward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplu_` | W m-2 um-1 | Upward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `layer_thermal_upward` | W m-2 | Integrated upward thermal transport per canopy layer. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Es_direct_` | W m-2 um-1 | Direct shortwave irradiance profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emin_direct_` | W m-2 um-1 | Downward shortwave flux profile originating from direct illumination. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplu_direct_` | W m-2 um-1 | Upward shortwave flux profile originating from direct illumination. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Es_diffuse_` | W m-2 um-1 | Diffuse shortwave irradiance profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emin_diffuse_` | W m-2 um-1 | Downward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplu_diffuse_` | W m-2 um-1 | Upward diffuse shortwave flux profile at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `layer_fluorescence` | W m-2 | Upward fluorescence contribution aggregated per canopy layer. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | layer_fluorescence.dat |

### Fluorescence

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `LoF_` | W m-2 um-1 sr-1 | Fluorescence spectrum in the observation direction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | fluorescence.csv |
| `LoF_sunlit` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution from sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `LoF_shaded` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution from shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `LoF_scattered` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution from multiply scattered canopy radiation. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `LoF_soil` | W m-2 um-1 sr-1 | Observation-direction fluorescence contribution associated with the soil boundary term. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `EoutF_` | W m-2 um-1 | Top-of-canopy hemispherically integrated fluorescence. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | fluorescence_hemis.csv |
| `EoutFrc_` | W m-2 um-1 | Reabsorption-corrected hemispherical fluorescence spectrum. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | fluorescence_ReabsCorr.csv |
| `Femleaves_` | W m-2 um-1 | Fluorescence emitted by all leaves before canopy escape losses. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | fluorescence_AllLeaves.csv |
| `sigmaF` | - | Fluorescence escape probability. | sigmaF = pi * LoF_ / EoutFrc_ | https://scope-model.readthedocs.io/en/master/outfiles.html | sigmaF.csv |
| `F685` | W m-2 um-1 sr-1 | First fluorescence peak radiance. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | F_1stpeak |
| `wl685` | nm | Wavelength of the first fluorescence peak. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | wl_1stpeak |
| `F740` | W m-2 um-1 sr-1 | Second fluorescence peak radiance. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | F_2ndpeak |
| `wl740` | nm | Wavelength of the second fluorescence peak. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | wl_2ndpeak |
| `F684` | W m-2 um-1 sr-1 | Fluorescence radiance sampled near 684-687 nm. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | F687 |
| `F761` | W m-2 um-1 sr-1 | Fluorescence radiance sampled near 760-761 nm. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | F760 |
| `LoutF` | W m-2 um-1 sr-1 | Integrated observation-direction fluorescence scalar. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | LFtot |
| `EoutF` | W m-2 | Integrated hemispherical fluorescence scalar. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | EFtot |
| `Fmin_` | W m-2 um-1 | Downward or internal fluorescence transport profile term at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Fplu_` | W m-2 um-1 | Upward fluorescence transport profile term at canopy interfaces. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `leaf_fluor_back` | W m-2 um-1 | Backward leaf fluorescence source spectrum before canopy transport. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `leaf_fluor_forw` | W m-2 um-1 | Forward leaf fluorescence source spectrum before canopy transport. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Thermal

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Lot_` | W m-2 um-1 sr-1 | Thermal radiance spectrum in the observation direction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eoutte_` | W m-2 um-1 | Emitted longwave flux spectrum leaving the canopy-soil system. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Emint_` | W m-2 um-1 | Downward internal thermal flux profile. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eplut_` | W m-2 um-1 | Upward internal thermal flux profile. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `LotBB_` | W m-2 um-1 sr-1 | Blackbody-equivalent observation-direction thermal radiance. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Loutt` | W m-2 um-1 sr-1 | Integrated observation-direction thermal radiance scalar. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Eoutt` | W m-2 | Integrated hemispherical thermal flux scalar. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `BrightnessT` | K | Brightness temperature inferred from the directional thermal radiance. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `canopyemis` | - | Effective canopy emissivity. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Physiology

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Pnu_Cab` | umol m-2 s-1 | Sunlit absorbed PAR by chlorophyll a+b used to drive leaf biochemistry. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Pnh_Cab` | umol m-2 s-1 | Shaded absorbed PAR by chlorophyll a+b used to drive leaf biochemistry. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_*` | varies | Prefix used for leaf-biochemistry outputs of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Search for base names such as A, Ci, rcw, eta, or Ja. |
| `shaded_*` | varies | Prefix used for leaf-biochemistry outputs of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Search for base names such as A, Ci, rcw, eta, or Ja. |
| `sunlit_A` | umol m-2 s-1 | Net assimilation rate of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_A` | umol m-2 s-1 | Net assimilation rate of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_Ci` | ppm | Intercellular CO2 concentration of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_Ci` | ppm | Intercellular CO2 concentration of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_rcw` | s m-1 | Canopy water-vapor resistance of sunlit leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_rcw` | s m-1 | Canopy water-vapor resistance of shaded leaves. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `sunlit_eta` | - | Fluorescence efficiency term returned by the sunlit leaf biochemistry solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `shaded_eta` | - | Fluorescence efficiency term returned by the shaded leaf biochemistry solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Energy Balance

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Rnctot` | W m-2 | Net radiation of the canopy. | Rnctot = Rnuc + Rnhc | https://scope-model.readthedocs.io/en/master/outfiles.html | fluxes.csv |
| `lEctot` | W m-2 | Latent heat flux of the canopy (transpiration). | lEctot = lEcu + lEch | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hctot` | W m-2 | Sensible heat flux of the canopy. | Hctot = Hcu + Hch | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Actot` | umol m-2 s-1 | Net photosynthesis of the canopy. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Photosynthesis |
| `Tcave` | degC | Average canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnstot` | W m-2 | Net radiation of the soil. | Rnstot = Rnus + Rnhs | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEstot` | W m-2 | Latent heat flux of the soil (evaporation). | lEstot = lEsu + lEsh | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hstot` | W m-2 | Sensible heat flux of the soil. | Hstot = Hsu + Hsh | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Gtot` | W m-2 | Soil heat flux. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsave` | degC | Average soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rntot` | W m-2 | Total net radiation. | Rntot = Rnctot + Rnstot | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEtot` | W m-2 | Total latent heat flux. | lEtot = lEctot + lEstot | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Htot` | W m-2 | Total sensible heat flux. | Htot = Hctot + Hstot | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnuc_sw` | W m-2 | Sunlit canopy net shortwave radiation. | Shortwave component of Rnuc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhc_sw` | W m-2 | Shaded canopy net shortwave radiation. | Shortwave component of Rnhc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnus_sw` | W m-2 | Sunlit soil net shortwave radiation. | Shortwave component of Rnus. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhs_sw` | W m-2 | Shaded soil net shortwave radiation. | Shortwave component of Rnhs. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnuct` | W m-2 | Sunlit canopy net thermal radiation. | Thermal component of Rnuc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhct` | W m-2 | Shaded canopy net thermal radiation. | Thermal component of Rnhc. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnust` | W m-2 | Sunlit soil net thermal radiation. | Thermal component of Rnus. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhst` | W m-2 | Shaded soil net thermal radiation. | Thermal component of Rnhs. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnuc` | W m-2 | Total net radiation of the sunlit canopy fraction. | Rnuc = Rnuc_sw + Rnuct | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhc` | W m-2 | Total net radiation of the shaded canopy fraction. | Rnhc = Rnhc_sw + Rnhct | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnus` | W m-2 | Total net radiation of the sunlit soil fraction. | Rnus = Rnus_sw + Rnust | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Rnhs` | W m-2 | Total net radiation of the shaded soil fraction. | Rnhs = Rnhs_sw + Rnhst | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEcu` | W m-2 | Latent heat flux of the sunlit canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEch` | W m-2 | Latent heat flux of the shaded canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEsu` | W m-2 | Latent heat flux of the sunlit soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `lEsh` | W m-2 | Latent heat flux of the shaded soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hcu` | W m-2 | Sensible heat flux of the sunlit canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hch` | W m-2 | Sensible heat flux of the shaded canopy fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hsu` | W m-2 | Sensible heat flux of the sunlit soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Hsh` | W m-2 | Sensible heat flux of the shaded soil fraction. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Gsu` | W m-2 | Soil heat flux associated with the sunlit soil fraction. | Sunlit-soil contribution to Gtot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Gsh` | W m-2 | Soil heat flux associated with the shaded soil fraction. | Shaded-soil contribution to Gtot. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Resistance

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `raa` | s m-1 | Aerodynamic resistance above the canopy. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `raws` | s m-1 | Aerodynamic resistance within the soil/canopy lower boundary layer. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | within-soil aerodynamic resistance |
| `rawc` | s m-1 | Aerodynamic resistance within the canopy air space. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `rac` | s m-1 | Boundary-layer resistance for canopy heat and vapor exchange. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ras` | s m-1 | Boundary-layer resistance for soil heat and vapor exchange. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ustar` | m s-1 | Friction velocity. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Boundary State

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Csu` | ppm | Solved CO2 concentration at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Csh` | ppm | Solved CO2 concentration at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebu` | hPa | Solved vapor pressure at the sunlit leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `ebh` | hPa | Solved vapor pressure at the shaded leaf surface. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Thermal State

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Tcu` | degC | Solved sunlit canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tch` | degC | Solved shaded canopy temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsu` | degC | Solved sunlit soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `Tsh` | degC | Solved shaded soil temperature. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Solver

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `max_error` | W m-2 | Maximum residual or closure error in the final energy-balance iteration. | - | docs/variable-glossary.md | - |
| `converged` | 0/1 | Whether the coupled energy-balance iteration converged. | - | docs/variable-glossary.md | - |
| `counter` | - | Number of energy-balance iterations performed. | - | docs/variable-glossary.md | - |

### Dataset Metadata

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `scope_product` | - | Dataset attribute naming the assembled product or workflow. | - | docs/variable-glossary.md | - |
| `scope_components` | - | Dataset attribute listing merged workflow components. | - | docs/variable-glossary.md | - |

### Namespaced Outputs

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `reflectance_directional_*` | varies | Prefix applied to directional reflectance variables in merged workflow datasets. | - | docs/variable-glossary.md | - |
| `reflectance_profile_*` | varies | Prefix applied to reflectance profile variables in merged workflow datasets. | - | docs/variable-glossary.md | - |
| `fluorescence_directional_*` | varies | Prefix applied to directional fluorescence variables in merged workflow datasets. | - | docs/variable-glossary.md | - |
| `fluorescence_profile_*` | varies | Prefix applied to fluorescence profile variables in merged workflow datasets. | - | docs/variable-glossary.md | - |
| `thermal_directional_*` | varies | Prefix applied to directional thermal variables in merged workflow datasets. | - | docs/variable-glossary.md | - |
| `thermal_profile_*` | varies | Prefix applied to thermal profile variables in merged workflow datasets. | - | docs/variable-glossary.md | - |

### Transport Coefficients

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `gammasdf` | - | Canopy transport coefficient for downward diffuse coupling used by fluorescence and reflectance transport. | Controls coupling from layer source terms into downward diffuse transport. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `gammasdb` | - | Canopy transport coefficient for upward diffuse coupling used by fluorescence and reflectance transport. | Controls coupling from layer source terms into upward diffuse transport. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `gammaso` | - | Canopy transport coefficient for escape toward the observation direction. | Controls coupling from layer source terms into observation-direction escape. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |

### Structured Outputs

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `sunlit` | LeafBiochemistryResult | Nested structured result holding sunlit leaf-biochemistry outputs. | - | docs/variable-glossary.md | - |
| `shaded` | LeafBiochemistryResult | Nested structured result holding shaded leaf-biochemistry outputs. | - | docs/variable-glossary.md | - |

### Solver Diagnostics

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `sunlit_Cs_input` | ppm | CO2 boundary condition passed into the sunlit leaf solve during the final energy-balance iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_Cs_input` | ppm | CO2 boundary condition passed into the shaded leaf solve during the final energy-balance iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `sunlit_eb_input` | hPa | Leaf-surface vapor-pressure boundary condition passed into the sunlit leaf solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_eb_input` | hPa | Leaf-surface vapor-pressure boundary condition passed into the shaded leaf solve. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `sunlit_T_input` | degC | Leaf temperature passed into the sunlit leaf solve during the final iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
| `shaded_T_input` | degC | Leaf temperature passed into the shaded leaf solve during the final iteration. | - | https://scope-model.readthedocs.io/en/master/outfiles.html | Phase-lagged diagnostic; not the main same-state parity signal. |
