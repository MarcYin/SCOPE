# SCOPE

PyTorch-first implementation of the SCOPE canopy radiative transfer model for reflectance, fluorescence, thermal radiance, and coupled energy-balance workflows.

## What It Is

`scope` is designed for users who need:

- asset-backed SCOPE physics in Python
- batched ROI/time execution on `xarray` datasets
- differentiable model components in PyTorch
- reproducible MATLAB parity checks in CI and local development

The current implementation supports:

- leaf optics through FLUSPECT
- canopy reflectance through 4SAIL-based transport
- layered fluorescence and thermal radiative transfer
- leaf biochemistry and coupled energy balance
- directional and vertical-profile outputs on the homogeneous canopy path
- ROI/time workflows with `xarray` input and output assembly

## Attribution

This package is a Python implementation of the original MATLAB SCOPE model:

- Soil Canopy Observation, Photochemistry and Energy fluxes (SCOPE)
- original repository: [Christiaanvandertol/SCOPE](https://github.com/Christiaanvandertol/SCOPE)
- upstream manual: [scope-model.readthedocs.io](https://scope-model.readthedocs.io/en/master/)

Please attribute the original SCOPE model and papers when using this package in research workflows:

- Van der Tol, C., Verhoef, W., Timmermans, J., Verhoef, A., and Su, Z. (2009), [Biogeosciences 6, 3109-3129](https://doi.org/10.5194/bg-6-3109-2009)
- Yang, P., Prikaziuk, E., Verhoef, W., and Van der Tol, C. (2021), [Geoscientific Model Development 14, 4697-4712](https://doi.org/10.5194/gmd-14-4697-2021)

## Install

Published package name:

```bash
python -m pip install SCOPE-RTM
```

Import name:

```python
import scope
```

Top-level CLI:

```bash
scope --help
scope fetch-upstream --help
scope prepare --help
scope run --help
scope vars Cab
scope vars --workflow fluorescence
scope vars --related Rntot
```

### 1. Clone the repository

```bash
git clone https://github.com/MarcYin/SCOPE scope
cd scope
```

### 2. Fetch the pinned upstream SCOPE assets

```bash
python scripts/fetch_upstream_scope.py
```

If you installed the package in an environment already, the same helper is available as:

```bash
scope-fetch-upstream
```

### 3. Create an environment and install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 4. Verify the install

```bash
PYTHONPATH=src python examples/basic_scene_reflectance.py
PYTHONPATH=src python -m pytest -q tests/test_scope_benchmark_parity.py tests/test_scope_timeseries_benchmark_parity.py
```

## 5-Minute Quickstart

### Minimal scene reflectance run

```bash
PYTHONPATH=src python examples/basic_scene_reflectance.py
```

Expected output:

```json
{
  "product": "reflectance",
  "dims": {"y": 1, "x": 1, "time": 1, "wavelength": 2001},
  "rsot_650nm": 0.047138178221010914,
  "rsot_865nm": 0.4100649627325952,
  "rsot_1600nm": 0.26994893328935227
}
```

### High-level workflow run

```bash
PYTHONPATH=src python examples/scope_workflow_demo.py
```

Expected output:

```json
{
  "product": "scope_workflow",
  "components": [
    "reflectance",
    "reflectance_directional",
    "reflectance_profile",
    "fluorescence",
    "fluorescence_directional",
    "fluorescence_profile"
  ],
  "rsot_650nm_t0": 0.04522854188089004,
  "LoF_peak_t0": 1.985767010834904e-05,
  "LoF_peak_wavelength_t0": 744.0
}
```

### Prepared-dataset CLI run

```bash
scope prepare \
  --weather weather.nc \
  --observation observation.nc \
  --bio-npz post_bio.npz \
  --year 2020 \
  --output scope_inputs.nc

scope run \
  --input scope_inputs.nc \
  --output scope_outputs.nc \
  --scope-root ./upstream/SCOPE \
  --workflow reflectance
```

Repeat `--output-var` to write only selected output variables for large runs.

## Main Entry Points

For most users, the preferred entry points are:

- [`ScopeGridRunner.run_scope_dataset(...)`](src/scope/runners/grid.py)
  High-level reflectance/fluorescence/thermal workflow dispatch from prepared `xarray` inputs.
- [`ScopeInferenceModel`](src/scope/inference.py)
  Lightweight tensor-only inference surface for production workloads that only need selected outputs.
- [`prepare_scope_input_dataset(...)`](src/scope/io/prepare.py)
  Build a runner-ready dataset from weather, observation, and Sentinel-2 bio inputs.
- [`validate_scope_dataset(...)`](src/scope/io/schema.py)
  Validate required variables, soil alternatives, and key dimensions before a workflow is executed.
- [`write_netcdf_dataset(...)`](src/scope/io/export.py)
  Persist prepared or simulated outputs to NetCDF with safe backend selection and compression handling.

For direct lower-level use:

- [`FluspectModel`](src/scope/spectral/fluspect.py)
- [`CanopyReflectanceModel`](src/scope/canopy/reflectance.py)
- [`CanopyFluorescenceModel`](src/scope/canopy/fluorescence.py)
- [`CanopyThermalRadianceModel`](src/scope/canopy/thermal.py)
- [`CanopyEnergyBalanceModel`](src/scope/energy/balance.py)

## Documentation Map

- [Installation Guide](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Model Mechanics](docs/model-mechanics.md)
- [Input / Output Reference](docs/input-output-reference.md)
- [Variable Glossary](docs/variable-glossary.md)
- [Workflow Variable Guides](docs/workflow-variables/reflectance.md)
- [Examples](docs/examples.md)
- [Production Notes](docs/production-notes.md)
- [Releasing](docs/releasing.md)
- [Benchmark Policy](docs/benchmark-policy.md)

Build the docs locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

## Production Notes

- Asset-backed constructors such as `from_scope_assets(...)` require an upstream SCOPE checkout. The recommended path is `scope-fetch-upstream`.
- The installed CLI now covers the common shell workflow: `scope fetch-upstream`, `scope prepare`, and `scope run`.
- Prepared inputs and assembled outputs now carry glossary-derived `xarray` metadata such as `long_name`, `units`, `description`, `scope_category`, and `scope_relationship`.
- NetCDF exports are now CF-enriched with dataset-level `Conventions`, `title`, `source`, `references`, `history`, and axis metadata on common coordinates.
- `scope run` validates workflow-specific inputs before execution, and the same validator is available directly as `validate_scope_dataset(...)`.
- The default CI suite runs parity tests in live-or-pregenerated mode. On machines without MATLAB, the tests compare against checked-in MATLAB fixtures.
- The self-hosted GPU and live-MATLAB lanes remain optional operational lanes; see [docs/benchmark-policy.md](docs/benchmark-policy.md).
- Documentation can be built locally with `mkdocs build --strict` and is deployed by the dedicated GitHub Pages workflow.
- Distribution artifacts can be built locally with `python -m build` and validated with `python -m twine check dist/*`.
- Release notes are drafted automatically on `main`, and tagged releases publish PyPI artifacts plus GitHub artifact attestations.

## Testing

Run the default suite with:

```bash
PYTHONPATH=src python -m pytest -q
```

The strongest automated checks currently include:

- kernel parity and execution-mode regression tests
- ROI/time runner consistency tests
- committed scene and time-series benchmark summary regression tests
- live-or-pregenerated MATLAB parity tests for the single-scene and time-series benchmark gates

## Performance Benchmarking

Use the committed kernel benchmark harness to compare eager and compiled execution on your own hardware:

```bash
PYTHONPATH=src python scripts/benchmark_kernels.py \
  --device cpu \
  --dtype float64 \
  --batch 32 \
  --fixture scope-assets \
  --mode compare
```

Current reference behavior on CPU with `torch 2.10.0`:

- `fluspect` and `reflectance` show strong steady-state speedups under `torch.compile`, but still require repeated same-shape calls to amortize compile cost.
- `thermal` speeds up in steady state, but the compile break-even is much higher.
- layered `fluorescence` currently fails under `torch.compile` on this environment.
- `leaf_biochemistry` currently becomes slower under `torch.compile` because of scalar-control-flow graph breaks and recompilation churn.

Because of that mix, the package does not enable compiled execution by default.

## Release Workflows

- `.github/workflows/release.yml`
  Verifies tag/version alignment, reruns the release-local CPU/docs gates, builds `sdist` and wheel artifacts for `SCOPE-RTM`, validates them with `twine check`, smoke-installs both artifact types through the documented `scope` CLI surface, and then publishes to PyPI on version tags. Manual dispatch still supports TestPyPI or PyPI.
- `.github/workflows/docs.yml`
  Builds the MkDocs site and deploys it to GitHub Pages.
