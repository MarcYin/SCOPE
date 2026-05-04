# Input / Output Reference

This page summarizes the practical input and output contracts for the current runner surface.

For name-by-name physical meanings, units, aliases, and search, use [Variable Glossary](variable-glossary.md) or the terminal command:

```bash
scope vars Rntot
scope vars --all --kind output
scope vars --workflow energy-balance
scope vars --related Rntot
```

## Preferred Entry Point

For most application code, prefer:

- `ScopeGridRunner.run_scope_dataset(...)`

That method dispatches the appropriate reflectance / fluorescence / thermal workflows from:

- runner-ready `xarray.Dataset` inputs
- a `varmap` that maps dataset variable names onto model names
- dataset attributes or explicit `scope_options`

Before running external datasets through the model, validate them with:

```python
from scope import validate_scope_dataset

validate_scope_dataset(dataset, workflow="scope")
```

For tensor-only production inference without `xarray`, prefer:

- `ScopeInferenceModel`

That surface accepts already-prepared tensor inputs and can return only selected fields such as:

- `("rsot", "rso")`
- `("energy.Rntot", "thermal.Lot_")`

## Common Dataset Dimensions

The current runner surface uses these dimensions:

- `y`, `x`, `time`
- `wavelength`
- `excitation_wavelength`
- `fluorescence_wavelength`
- `thermal_wavelength`
- `layer`
- `layer_interface`
- `direction`

## Common Input Groups

### Base reflectance workflow

Required:

- `Cab`
- `Cw`
- `Cdm`
- `LAI`
- `tts`
- `tto`
- `psi`
- one soil source:
  - `soil_refl`, or
  - `soil_spectrum`, or
  - `BSMBrightness`, `BSMlat`, `BSMlon`, `SMC`

Optional:

- additional leaf optical fields such as `Cca`, `Cs`, `Cant`, `Cbc`, `Cp`, `N`
- hotspot override

Outputs:

- `rsot`, `rdd`, `rso`, `rsd`, `rdo`, `leaf_refl`, `leaf_tran`, and related canopy terms

### Fluorescence extension

Additional required inputs:

- `fqe`
- `Esun_`
- `Esky_`

Optional:

- `etau`
- `etah`
- directional coordinates `directional_tto`, `directional_psi`

Outputs:

- `LoF_`
- `EoutF_`
- `EoutFrc_`
- `Femleaves_`
- `sigmaF`
- directional and profile fluorescence products when enabled

### Thermal workflow

For the standalone thermal path, provide:

- `LAI`
- `tts`
- `tto`
- `psi`
- `Tcu`
- `Tch`
- `Tsu`
- `Tsh`

Optional:

- `rho_thermal`
- `tau_thermal`
- `rs_thermal`

Outputs:

- `Lot_`
- `Eoutte_`
- `Emint_`
- `Eplut_`
- `LotBB_`

This path consumes explicit thermal state inputs. If you want the Python runner to reproduce the coupled `calc_ebal` solve instead, set `calc_ebal=1` on the high-level `scope` workflow or call one of the `energy-balance*` runners directly.

### Coupled energy-balance workflows

Additional meteorology / canopy / soil fields are required, typically including:

- `Ta`, `ea`, `Ca`, `Oa`, `p`, `z`, `u`
- `Cd`, `rwc`, `z0m`, `d`, `h`
- `rss`, `rbs`
- shortwave forcing on `wavelength`:
  - `Esun_sw`
  - `Esky_sw`
- optional longwave forcing on `thermal_wavelength`:
  - `Esun_lw`
  - `Esky_lw`

These workflows produce:

- temperatures
- resistances
- radiative terms
- fluxes
- coupled fluorescence or thermal outputs

## High-Level Workflow Options

`run_scope_dataset(...)` currently honors these dataset attrs or explicit `scope_options`:

- `calc_ebal`
- `calc_fluor`
- `calc_planck`
- `calc_directional`
- `calc_vert_profiles`

Recommended usage:

- keep the prepared dataset attrs as the default workflow intent
- use explicit `scope_options=...` only when you need to override them at runtime
- use `output_vars=(...)` to assemble only selected output variables; prefixed workflow names such as `fluorescence_directional_LoF_` match the final dataset variable names
- when `calc_ebal=1`, `calc_fluor` and `calc_planck` dispatch to the coupled energy-balance fluorescence / thermal paths rather than the standalone `Esun_/Esky_` or `Tcu/Tch/Tsu/Tsh` inputs

## Output Naming Conventions

Base workflow outputs are unprefixed:

- `rsot`
- `LoF_`
- `Lot_`

Directional and profile outputs are namespaced:

- `reflectance_directional_*`
- `reflectance_profile_*`
- `fluorescence_directional_*`
- `fluorescence_profile_*`
- `thermal_directional_*`
- `thermal_profile_*`

If multiple workflow components expose the same diagnostic name, the later component is also prefixed during merge so values are not silently overwritten.

## Output Metadata

Assembled workflow datasets preserve:

- original `xarray` coordinates where compatible
- workflow attrs such as `calc_fluor` and `calc_directional`
- `scope_product`
- `scope_components`
- per-variable metadata derived from the glossary registry:
  - `long_name`
  - `units`
  - `description`
  - `scope_category`
  - `scope_kind`
  - `scope_relationship` when applicable
  - `scope_source_doc`

NetCDF exports written with `write_netcdf_dataset(...)` additionally standardize:

- `Conventions = CF-1.10`
- dataset `title`
- dataset `source`
- upstream `references`
- append-only `history`
- coordinate axis metadata such as `time`, `x`, `y`, and `layer`

## Recommended Runtime Pattern

1. Prepare or assemble a runner-ready `xarray.Dataset`
2. Build a `SimulationConfig`
3. Build a `ScopeGridDataModule`
4. Instantiate `ScopeGridRunner.from_scope_assets(...)`
5. Call `run_scope_dataset(...)` or a more specific runner method
6. Persist outputs with `write_netcdf_dataset(...)` when needed
