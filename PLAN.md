# SCOPE Torch Translation Plan

## 1. Physical Basis & Key Equations

- **Canopy Net Radiation:** Shaded elements follow \(R_n(x) = (E^-(x) + E^+(x) - 2H_{cd}(x))(1 - \rho - \tau)\) (Eq. 18, van der Tol et al. 2009). Sunlit leaves add direct forcing \(f_s E_{\text{sun}}\) before applying the same absorptance term (Eq. 19). These equations bind the radiative transfer outputs \(E^\pm\) to the absorption/emission handled by Matlab `RTMo`/`RTMt`.
- **Energy Balance Closure:** Each canopy element obeys \(R_n - H - \lambda E - G = 0\) (Eq. 20) with sensible \(H = \rho_a c_p (T_s - T_a)/r_a\) (Eq. 21), latent \(\lambda E = \lambda (q_s - q_a)/(r_a + r_c)\) (Eq. 22), and soil heat \(G\). Matlab’s `ebal` routine enforces this closure; the PyTorch solver must mirror it exactly.
- **Iterative Temperature Updates:** SCOPE 2.0 introduces Newton updates \(T_{\text{new}} = T_{\text{old}} + W \cdot e_{\text{ebal}} / (\partial e_{\text{ebal}}/\partial T)\) with \(e_{\text{ebal}} = R_n - H - \lambda E - G\) (Equations 1, 6, 7; van der Tol et al. 2021). Analytic derivatives include sensible and latent contributions—retaining them avoids extra iterations even for batched GPU runs.
- **Photosynthesis & Fluorescence:** Leaf biochemistry follows Farquhar–von Caemmerer–Berry (FvCB) with \(w_c = \frac{V_{\max}(c_i - \Gamma_*)}{c_i + K_c(1 + O_i/K_o)}\) and \(w_j = \frac{J(c_i - \Gamma_*)}{4(c_i + 2\Gamma_*)}\); assimilation is the minimum of \(w_c, w_j, w_s\). `biochemical.m` combines these with Ball–Berry stomatal conductance, feeding latent flux and fluorescence submodels.
- **Fluorescence Transport:** Intrinsic leaf fluorescence splits into components (no canopy interaction vs. full canopy scattering) propagated by `RTMf`/`RTMo`. Tracking which tensors remain spectral versus band-integrated will keep the PyTorch implementation consistent with Matlab outputs.

## 2. MATLAB Module Responsibilities

| Module | Responsibility | PyTorch Translation Target |
| --- | --- | --- |
| `SCOPE.m` | Orchestrates IO, RTMs, fluxes, options | Entry script/module maintaining identical sequencing and IO schema |
| `fluspect_B_CX` | Leaf optical properties | `scope_torch.spectral.fluspect` returning batched tensors |
| `RTMo`, `RTMt_planck/sb`, `RTMf`, `RTMz` | Four-stream canopy radiative transfer (vis/NIR, thermal, fluorescence, sun–shade separation) | `scope_torch.canopy` subpackage handling spectral tensors `[batch, layer, angle, wavelength]` |
| `biochemical.m` | Photosynthesis + stomatal conductance | `scope_torch.biochem` with differentiable FvCB + Ball–Berry |
| `ebal.m` | Energy balance closure loop, flux bookkeeping | `scope_torch.energy` implementing Newton updates, convergence checks |
| Output helpers | CSV/`.dat` writers for fluxes, layer states, spectra | `scope_torch.io` writing identical files for regression diffs |
| `scope_grid_netcdf_inmemory_refactored.m` | Multi-pixel/time orchestration that ingests NetCDF cubes, configures per-pixel parameters, and accumulates outputs | `scope_torch.runners.grid` to expose batched PyTorch simulations driven by xarray/NetCDF tensors and dispatchable to GPU |
| `prepare_scope_input.py` | Region/time preprocessing: fuses ERA5 weather, Sentinel-2 biophysical retrievals, and TROPOSIF geometry into SCOPE-friendly NetCDF inputs | `scope_torch.data.prepare_scope_input` refactor that emits standardized datasets & metadata for batched PyTorch simulations |

Reference datasets (`output/verificationdata`) will be mirrored under `tests/data/` for automated parity checks.

## 3. PyTorch Architecture Plan

1. **Tensor Shapes:** Use `[batch, layer, angle, wavelength]` tensors; “batch” indexes independent simulations to leverage GPU parallelism.
2. **Subpackages:**
   - `spectral`: leaf/soil optics, LUT readers.
   - `canopy`: four-stream solver, gap probabilities, sunlit/shaded fractions.
   - `energy`: aerodynamic resistances, Newton closure loop.
   - `biochem`: FvCB + Ball–Berry, fluorescence yields.
   - `fluorescence`: transport from leaf to top-of-canopy.
3. **Device Management:** Central `SimulationConfig` places tensors on CPU/GPU; batched operations share precomputed geometric factors to reduce memory traffic.
4. **Differentiability:** Keep modules autograd-friendly to enable future inversion/gradient-based calibration.
5. **IO Compatibility:** Dataclasses enforce SCOPE’s existing CSV schema so downstream workflows remain unchanged.

## 4. Region & Time-Series Workflow (Multi-Simulation Design)

1. **ROI & Temporal Ingestion:** Reuse `prepare_scope_input.py` logic to accept (a) a region polygon/bounding box, (b) a time window, and (c) data sources (ERA5, TROPOSIF, Sentinel-2). Standardize outputs as xarray datasets with weather, geometry, biophysical, and soil layers.
2. **DataModule Abstraction:** Implement a `ScopeGridDataModule` that converts the prepared Dataset into PyTorch tensors shaped `[batch (=ny·nx·nt), variable]`, capturing metadata (lat/lon/time, site IDs) for back-references. Supports lazy chunking so extremely large regions stream through GPU memory.
3. **Grid Runner:** Translate `scope_grid_netcdf_inmemory_refactored.m` into a PyTorch runner that (a) vectorizes per-pixel configuration (meteorology, geometry, canopy, soil) and (b) forwards batches through the core model modules. Maintains feature parity (options struct, defaults, LIDF handling) while exposing asynchronous device execution.
4. **Spectral Products:** The runner must emit reflectance, SIF, thermal radiance, etc., with consistent spectral grids across the batch so that downstream tiling/NetCDF writers can reconstruct spatial rasters.
5. **Output Writer:** Provide converters back to NetCDF/GeoTIFF; metadata should include ROI/time tags, source dataset versions, and simulation options. This ensures compatibility with existing workflows that read the Matlab grids.

## 5. Translation & Verification Strategy

1. **Phase 1 – Leaf Optics:** Port `fluspect_B_CX` with PyTorch tensor math. Unit tests compare reflectance/transmittance spectra to Matlab outputs (RMSE per wavelength ≤ 1e-6).
2. **Phase 2 – Canopy Radiative Transfer:** Implement `RTMo`/`RTMt` single-layer first, then extend to full-layer stack. Integration tests ensure hemispherical fluxes and gap probabilities match Matlab within ≤0.5 % relative error.
3. **Phase 3 – Biochemistry & Fluorescence:** Translate `biochemical` and fluorescence pathways. Validate assimilation, transpiration, and fluorescence spectra against Matlab across multiple environmental scenarios.
4. **Phase 4 – Energy Balance:** Port `ebal` with analytic derivatives and damping. Tests check (a) closure tolerance <1 W m⁻² and (b) iteration counts comparable to Matlab.
5. **Phase 5 – End-to-End Regression:** Combine modules, run the official verification dataset plus randomized batched simulations; ensure batched GPU runs equal single CPU runs bitwise within tolerance.

Each phase culminates in pytest suites that load Matlab reference outputs. CI will run these suites on CPU immediately and add GPU runners when available.

## 6. Immediate Implementation Tasks

1. Scaffold Python package (`scope_torch/`) with the subpackages listed above plus configuration and IO layers.
2. Vendor Matlab verification CSVs into `tests/data/` and write loaders/utilities for comparing spectra/fluxes.
3. Begin translating `fluspect_B_CX` into PyTorch with targeted unit tests before tackling canopy RTMs.
4. Configure CI (GitHub Actions) for pytest + optional coverage, ensuring regression tests execute on every PR.
