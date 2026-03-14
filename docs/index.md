# SCOPE

`scope` is a PyTorch-first implementation of the SCOPE canopy radiative transfer model for:

- reflectance
- fluorescence
- thermal radiance
- coupled energy-balance workflows

It is designed for users who need:

- asset-backed SCOPE physics in Python
- batched ROI/time execution on `xarray` datasets
- differentiable model components in PyTorch
- reproducible MATLAB parity checks in local development and CI

## Attribution

`scope` is based on the original MATLAB SCOPE model by Christiaan van der Tol and collaborators:

- upstream repository: [Christiaanvandertol/SCOPE](https://github.com/Christiaanvandertol/SCOPE)
- upstream manual: [scope-model.readthedocs.io](https://scope-model.readthedocs.io/en/master/)
- foundational papers:
  - [Van der Tol et al. (2009)](https://doi.org/10.5194/bg-6-3109-2009)
  - [Yang et al. (2021)](https://doi.org/10.5194/gmd-14-4697-2021)

## Start Here

If this is your first time using the project:

1. Read [Installation](installation.md)
2. Run the examples in [Quickstart](quickstart.md)
3. Use [Input / Output Reference](input-output-reference.md) when building real workflows
4. Use [Variable Glossary](variable-glossary.md) when you need the physical meaning of a specific name

## Main User Entry Points

For most application code, prefer:

- `ScopeGridRunner.run_scope_dataset(...)`
- `prepare_scope_input_dataset(...)`
- `write_netcdf_dataset(...)`

These cover the common production path:

1. prepare a runner-ready `xarray.Dataset`
2. execute the requested SCOPE workflows
3. persist outputs to NetCDF when needed

## Current Feature Surface

The current implementation supports:

- FLUSPECT leaf optics
- 4SAIL-based canopy reflectance transport
- layered fluorescence and thermal radiative transfer
- leaf biochemistry and coupled energy balance
- directional and vertical-profile outputs on the homogeneous canopy path
- scene and time-series MATLAB parity checks

## What To Read Next

- [Quickstart](quickstart.md)
- [Model Mechanics](model-mechanics.md)
- [Variable Glossary](variable-glossary.md)
- [Examples](examples.md)
- [Production Notes](production-notes.md)
