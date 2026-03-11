# SCOPE Torch Implementation Plan

## 1. Current Repository Status

### Implemented and already useful

| Area | Current implementation | Evidence |
| --- | --- | --- |
| Leaf optics | `scope_torch.spectral.fluspect` ports the PROSPECT/fluspect-style leaf optics core, including fluorescence matrices `Mb`/`Mf` and batched inputs via `LeafBioBatch`. | [src/scope_torch/spectral/fluspect.py](src/scope_torch/spectral/fluspect.py), [tests/spectral/test_fluspect.py](tests/spectral/test_fluspect.py) |
| Canopy reflectance | `scope_torch.canopy.foursail` implements a batched 4SAIL canopy solver with LIDF support and soil coupling. | [src/scope_torch/canopy/foursail.py](src/scope_torch/canopy/foursail.py), [tests/canopy/test_foursail.py](tests/canopy/test_foursail.py) |
| Grid batching | `ScopeGridDataModule` stacks `xarray.Dataset` inputs into chunked torch batches. | [src/scope_torch/data/grid.py](src/scope_torch/data/grid.py), [tests/test_grid_data_module.py](tests/test_grid_data_module.py) |
| Minimal end-to-end runner | `ScopeGridRunner` already chains leaf optics and canopy reflectance over ROI/time batches. | [src/scope_torch/runners/grid.py](src/scope_torch/runners/grid.py), [tests/test_scope_grid_runner.py](tests/test_scope_grid_runner.py) |

### Present but still prototype-level

1. `prepare_scope_input.py` is still a host-specific script with hard-coded paths and no reusable library surface.
2. The grid runner currently produces only leaf reflectance/transmittance plus two canopy reflectance products (`rsot`, `rdd`).
3. The repository does not yet include the official MATLAB/SCOPE verification datasets, IO compatibility layers, or CI automation.

### Not implemented yet

1. `biochemical.m` equivalent for FvCB photosynthesis and stomatal conductance.
2. `ebal.m` equivalent for Newton-style energy balance closure.
3. SCOPE thermal and fluorescence canopy transport modules (`RTMf`, `RTMt_planck/sb`, `RTMz`).
4. Production output writers for SCOPE-compatible NetCDF/CSV/`.dat` products.
5. Reusable loaders for upstream spectral parameter files, soil spectra, and scenario definitions.

## 2. Main Gaps and Technical Risks

1. **Differentiability and GPU parity are not yet real guarantees.**
   `FluspectModel._expint()` calls SciPy on CPU after detaching tensors, and `FourSAILModel._hotspot_terms()` converts tensors to Python scalars inside a loop. Both break the stated autograd/GPU direction and should be treated as short-term hardening work.
2. **The canopy implementation is validated against PROSAIL/4SAIL, not full SCOPE canopy physics.**
   The current solver is a strong VIS/NIR reflectance base, but it is not yet a replacement for SCOPE's `RTMo`/`RTMf`/`RTMt`/`RTMz` stack.
3. **The grid path is functional but too narrow for the final product.**
   `ScopeGridDataModule` materializes full tensors before chunking, and `ScopeGridRunner` does not preserve metadata, emit all products, or expose SCOPE option parity.
4. **Verification coverage is still shallow.**
   Existing tests prove local mathematical correctness for implemented kernels, but there is no official MATLAB regression suite, no tolerance budget per product, and no CI to enforce it.

## 3. Revised Target Architecture

### Core packages

1. `scope_torch.spectral`
   Leaf optics, spectral grids, upstream parameter loaders, soil spectra, and wavelength interpolation utilities.
2. `scope_torch.canopy`
   Batched VIS/NIR canopy reflectance plus SCOPE-specific fluorescence and thermal transport.
3. `scope_torch.biochem`
   FvCB photosynthesis, stomatal conductance, and fluorescence-yield drivers.
4. `scope_torch.energy`
   Aerodynamic resistance, soil heat treatment, and Newton energy balance closure.
5. `scope_torch.runners`
   High-level simulation entry points for single scenes and ROI/time grids.
6. `scope_torch.io`
   Output assembly back to xarray/NetCDF plus SCOPE-compatible export tables where needed.

### Data model principles

1. Use batched tensors with spectral axes explicit, typically `[batch, wavelength]` for current kernels and `[batch, wavelength_out, wavelength_in]` for fluorescence transport matrices.
2. Keep the geometry and simulation options explicit in typed dataclasses so MATLAB parity cases are reproducible.
3. Preserve `xarray` metadata through the grid workflow so outputs can be reshaped back to `(y, x, time, wavelength)` without manual bookkeeping.

## 4. Improved Implementation Plan

### Phase 0: Harden what already exists

Goal: turn the current spectral + 4SAIL core into a stable base instead of building new modules on top of prototype assumptions.

1. Replace CPU-only and detached math paths in `fluspect` and `foursail` with tensor-native implementations.
2. Add device, dtype, and gradient tests for the implemented kernels.
3. Introduce loaders for spectral inputs (`OptiPar`, soil spectra, wavelength grids) from upstream SCOPE resources instead of synthetic test fixtures.
4. Add a reproducible local test workflow and CI entry point.

Exit criteria:
1. Leaf and canopy kernels run on CPU and GPU without host round-trips.
2. Existing tests pass in a fresh environment.
3. The repository can load real optical parameters from upstream files.

### Phase 1: Finish the spectral reflectance stack

Goal: move from "generic fluspect + 4SAIL" to a SCOPE-compatible reflectance core.

1. Wrap or refactor the current canopy solver behind a SCOPE-facing API that matches required `RTMo` inputs/outputs.
2. Add soil optics generation/loading rather than requiring precomputed soil reflectance cubes.
3. Expand canopy outputs beyond `rsot` and `rdd` to the full directional and hemispherical reflectance set already available from `FourSAILResult`.
4. Bring in curated MATLAB/PROSAIL reference cases under `tests/data/`.

Exit criteria:
1. Reflectance-only runs can be configured from real SCOPE inputs.
2. Output naming and shapes are stable enough for downstream coupling.
3. MATLAB/PROSAIL parity cases exist in-repo, not only synthetic tests.

### Phase 2: Add canopy fluorescence and thermal radiative transfer

Goal: complete the radiative-transfer side before coupling in physiology and energy balance.

1. Implement fluorescence transport (`RTMf`) using the existing leaf-level `Mb`/`Mf` outputs as inputs.
2. Implement thermal radiative transfer (`RTMt_planck` / `RTMt_sb`) and sun/shade separation outputs (`RTMz`) on the same geometry backbone.
3. Standardize canopy outputs into one result object covering reflectance, SIF, thermal radiance, and intermediate flux terms needed by energy balance.

Exit criteria:
1. The model can produce reflectance, fluorescence, and thermal products for prescribed leaf/soil temperatures and fluorescence yields.
2. Shared geometry factors are reused across VIS/NIR, fluorescence, and thermal calculations.

### Phase 3: Port biochemistry and energy balance

Goal: reproduce SCOPE's flux and temperature solution, not only its radiative outputs.

1. Implement `biochemical.m` equivalents for FvCB assimilation and Ball-Berry or equivalent stomatal conductance.
2. Implement `ebal.m` style Newton updates, damping, convergence checks, and flux bookkeeping.
3. Couple the biochemistry and energy modules to the radiative-transfer outputs with clear lite/full modes.

Exit criteria:
1. Net radiation, sensible heat, latent heat, and soil heat close within a defined tolerance.
2. Flux and temperature solutions match MATLAB references for benchmark scenarios.

### Phase 4: Make the grid workflow production-ready

Goal: convert the current prototype batch runner into the real ROI/time execution path.

1. Refactor `prepare_scope_input.py` into reusable library functions with configurable data sources and no machine-specific paths.
2. Make `ScopeGridDataModule` lazy/chunk-aware so it does not materialize the full dataset before batching.
3. Extend `ScopeGridRunner` to map the full option set, preserve metadata, and assemble outputs back into `xarray.Dataset`s.
4. Add NetCDF writers and, if needed, tabular exports for parity with existing workflows.

Exit criteria:
1. A prepared ROI/time dataset can be simulated end-to-end and written back with coordinates and metadata intact.
2. The grid pipeline supports chunked execution without rewriting model internals.

### Phase 5: Lock parity and regression coverage

Goal: prove equivalence and keep it stable.

1. Vendor official or curated SCOPE verification cases into `tests/data/`.
2. Add regression tests for single-case MATLAB parity, batched-vs-single consistency, and CPU-vs-GPU consistency.
3. Define explicit tolerances by product class: reflectance, fluorescence, thermal radiance, fluxes, and convergence metrics.
4. Wire the full suite into CI.

Exit criteria:
1. Each implemented module has reference-backed regression tests.
2. End-to-end cases fail fast when parity drifts.

## 5. Recommended Immediate Work Order

This is the sequence that reduces the most downstream rework.

1. **Kernel hardening first.**
   Remove the current CPU/detach hot spots in `fluspect` and `foursail` before adding more coupled physics.
2. **Real input loaders second.**
   Stop relying on synthetic `OptiPar` and soil spectra so later parity work is based on real SCOPE assets.
3. **SCOPE-facing canopy API third.**
   Stabilize the reflectance interface and output schema before building fluorescence, thermal, or grid products on top of it.
4. **Biochemistry and energy after the RT stack is stable.**
   These modules depend on the canopy outputs and are expensive to validate if the radiative interface keeps changing.
5. **Grid and IO last, but not forgotten.**
   The current runner is good enough for development, but production workflow work should wait until the core model contracts settle.
