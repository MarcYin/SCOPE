# SCOPE Torch

PyTorch-first port of the [SCOPE](https://github.com/Christiaanvandertol/SCOPE) canopy radiative transfer model with support for batched simulations, differentiable components, and GPU acceleration.

## Goals
- Maintain parity with the original MATLAB/Fortran implementations for all radiative, fluorescence, and energy balance components.
- Run many simulations (space *or* time grids) simultaneously so weather reanalyses, EO retrievals, and tower data can be fused efficiently.
- Enable gradient-based workflows (calibration, inversion, UQ) by keeping every module differentiable end-to-end.

## Repository Layout

```
src/
  scope_torch/
    config.py              # Simulation/IO dataclasses + device helpers
    data/grid.py           # ROI/time ingestion + batching (xarray -> torch)
    spectral/
      fluspect.py          # Leaf optics + fluorescence (PyTorch translation)
PLAN.md                    # Detailed implementation roadmap + physical equations
prepare_scope_input.py     # Legacy preprocessing prototype (ROI -> NetCDF)
scope_grid_netcdf_inmemory_refactored.m  # Legacy MATLAB grid runner reference
```

## Development Roadmap
See [PLAN.md](PLAN.md) for the physics summary, staged translation plan, and GPU-oriented design notes. Short version:
1. **Leaf optics (fluspect)** → done in `scope_torch.spectral.fluspect` with batched tensors and functional parity tests.
2. **Canopy RTMs (RTMo/RTMt/RTMf/RTMz)** → next major milestone after locking the leaf module API.
3. **Biochemistry + energy balance** → match Newton-style closures and photosynthesis (FvCB + Ball–Berry).
4. **Grid runners + IO** → reimplement `scope_grid_netcdf_inmemory_refactored.m` as a GPU-native runner backed by `ScopeGridDataModule`.

## Testing
Run the unit tests with

```bash
python -m pytest
```

`tests/spectral/test_fluspect.py` compares the PyTorch implementation to an analytically equivalent NumPy reference to guarantee physics-preserving behavior before we wire in full MATLAB parity datasets.
