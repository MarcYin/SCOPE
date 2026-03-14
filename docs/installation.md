# Installation

## Prerequisites

Required:

- Python `>=3.10`
- `git`
- the `scope` source tree

Optional:

- CUDA-capable PyTorch for GPU execution
- MATLAB for fresh live benchmark export instead of pregenerated-fixture fallback
- `netcdf4` or `h5netcdf` if you want HDF5-backed NetCDF output; otherwise the `scipy` backend is enough for basic NetCDF writing

## Recommended Source Installation

If you are installing from PyPI once a release has been published:

```bash
python -m pip install SCOPE-RTM
```

The public Python import remains:

```python
import scope
```

Clone the repo and fetch the pinned upstream SCOPE checkout:

```bash
git clone <your-repo-url> scope
cd scope
python scripts/fetch_upstream_scope.py
```

Then create an environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you also want to build the documentation site locally:

```bash
python -m pip install -e ".[docs]"
```

If you want the installed command-line helpers:

```bash
scope-fetch-upstream --help
scope-prepare --help
```

## Runtime Asset Requirement

The package itself does not bundle the full upstream MATLAB SCOPE tree inside `src/`. Asset-backed constructors such as:

- `FluspectModel.from_scope_assets(...)`
- `CanopyReflectanceModel.from_scope_assets(...)`
- `ScopeGridRunner.from_scope_assets(...)`

expect a usable upstream SCOPE checkout. The default working assumption is:

```text
./upstream/SCOPE
```

If your checkout lives elsewhere, pass `scope_root_path=...` explicitly.

## Installation Verification

### Minimal example

```bash
PYTHONPATH=src python examples/basic_scene_reflectance.py
```

### High-level workflow example

```bash
PYTHONPATH=src python examples/scope_workflow_demo.py
```

### Full test suite

```bash
PYTHONPATH=src python -m pytest -q
```

## GPU Notes

`scope` uses standard PyTorch device selection. For GPU usage:

1. Install a CUDA-enabled PyTorch build appropriate for your platform.
2. Pass `device="cuda"` in `SimulationConfig` or the relevant model constructor.
3. Re-run the example scripts or selected tests on CUDA before using GPU in production.

The current test suite includes optional CUDA checks but does not require CUDA to pass.

## MATLAB Notes

MATLAB is optional for normal model execution. It is only needed when you want:

- fresh MATLAB benchmark export
- live MATLAB parity runs instead of pregenerated-fixture fallback

If MATLAB is absent, the benchmark parity tests still run against checked-in MATLAB fixtures and summaries.
