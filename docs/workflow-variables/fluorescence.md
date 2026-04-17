# Fluorescence Variables

This page is generated from the in-repo variable registry for the `fluorescence` workflow family.

Use `scope vars --workflow fluorescence` for terminal lookup.

## Geometry

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `direction` | - | Index over directional viewing geometries. | - | docs/input-output-reference.md | - |
| `directional_tto` | deg | Viewing zenith angles for directional outputs. | - | docs/input-output-reference.md | tto grid |
| `directional_psi` | deg | Relative azimuth angles for directional outputs. | - | docs/input-output-reference.md | psi grid |
| `tts` | deg | Solar zenith angle. | - | docs/input-output-reference.md | sun zenith |
| `tto` | deg | Viewing zenith angle. | - | docs/input-output-reference.md | observer zenith |
| `psi` | deg | Relative azimuth angle between sun and sensor. | - | docs/input-output-reference.md | relative azimuth |

## Leaf Biochemistry

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

## Canopy Structure

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `ala` | deg | Mean leaf angle parameter used to define the canopy leaf angle distribution function (LIDF). | Controls the overall canopy inclination distribution used by the LIDF. | docs/input-output-reference.md | LIDFa |
| `LAI` | m2 m-2 | Leaf area index. | - | docs/input-output-reference.md | - |

## Soil

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

## Spectral Forcing

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `Esun_` | W m-2 um-1 | Direct excitation irradiance spectrum for fluorescence. | - | docs/input-output-reference.md | - |
| `Esky_` | W m-2 um-1 | Diffuse excitation irradiance spectrum for fluorescence. | - | docs/input-output-reference.md | - |
| `Esun_sw` | W m-2 um-1 | Direct shortwave irradiance spectrum used by reflectance and energy-balance workflows. | - | docs/input-output-reference.md | - |
| `Esky_sw` | W m-2 um-1 | Diffuse shortwave irradiance spectrum used by reflectance and energy-balance workflows. | - | docs/input-output-reference.md | - |
| `Esun_lw` | W m-2 um-1 | Direct longwave irradiance spectrum used by coupled energy-balance workflows. | - | docs/input-output-reference.md | - |
| `Esky_lw` | W m-2 um-1 | Diffuse longwave irradiance spectrum used by coupled energy-balance workflows. | - | docs/input-output-reference.md | - |

## Fluorescence Transport

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `etau` | - | Forward fluorescence efficiency or source scaling per canopy layer. | - | docs/input-output-reference.md | Used in layered and directional fluorescence paths. |
| `etah` | - | Backward fluorescence efficiency or source scaling per canopy layer. | - | docs/input-output-reference.md | Used in layered and directional fluorescence paths. |

## Profiles

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

## Fluorescence

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

## Physiology

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

## Transport Coefficients

| Name | Units | Meaning | Relationship / formula | Source | Workflows / aliases |
| --- | --- | --- | --- | --- | --- |
| `gammasdf` | - | Canopy transport coefficient for downward diffuse coupling used by fluorescence and reflectance transport. | Controls coupling from layer source terms into downward diffuse transport. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `gammasdb` | - | Canopy transport coefficient for upward diffuse coupling used by fluorescence and reflectance transport. | Controls coupling from layer source terms into upward diffuse transport. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
| `gammaso` | - | Canopy transport coefficient for escape toward the observation direction. | Controls coupling from layer source terms into observation-direction escape. | https://scope-model.readthedocs.io/en/master/outfiles.html | - |
