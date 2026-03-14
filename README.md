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

## Install

Published package name:

```bash
python -m pip install SCOPE-RTM
```

Import name:

```python
import scope
```

### 1. Clone the repository

```bash
git clone <your-repo-url> scope
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

## Main Entry Points

For most users, the preferred entry points are:

- [`ScopeGridRunner.run_scope_dataset(...)`](src/scope/runners/grid.py)
  High-level reflectance/fluorescence/thermal workflow dispatch from prepared `xarray` inputs.
- [`prepare_scope_input_dataset(...)`](src/scope/io/prepare.py)
  Build a runner-ready dataset from weather, observation, and Sentinel-2 bio inputs.
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
- [Examples](docs/examples.md)
- [Production Notes](docs/production-notes.md)
- [Benchmark Policy](docs/benchmark-policy.md)

Build the docs locally with:

```bash
python -m pip install -e ".[docs]"
mkdocs build --strict
```

## Production Notes

- Asset-backed constructors such as `from_scope_assets(...)` require an upstream SCOPE checkout. The recommended path is `scope-fetch-upstream`.
- The default CI suite runs parity tests in live-or-pregenerated mode. On machines without MATLAB, the tests compare against checked-in MATLAB fixtures.
- The self-hosted GPU and live-MATLAB lanes remain optional operational lanes; see [docs/benchmark-policy.md](docs/benchmark-policy.md).
- Documentation can be built locally with `mkdocs build --strict` and is deployed by the dedicated GitHub Pages workflow.
- Distribution artifacts can be built locally with `python -m build` and validated with `python -m twine check dist/*`.

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

## Release Workflows

- `.github/workflows/release.yml`
  Builds `sdist` and wheel artifacts for `SCOPE-RTM`, validates them with `twine check`, and auto-publishes to PyPI on version tags. Manual dispatch still supports TestPyPI or PyPI.
- `.github/workflows/docs.yml`
  Builds the MkDocs site and deploys it to GitHub Pages.
