# Quickstart

This page shows the shortest path from a fresh checkout to a successful run.

## 1. Install

```bash
git clone https://github.com/MarcYin/SCOPE scope
cd scope
python scripts/fetch_upstream_scope.py
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
scope --help
```

## 2. Run a Minimal Scene

```bash
PYTHONPATH=src python examples/basic_scene_reflectance.py
```

Typical output:

```json
{
  "product": "reflectance",
  "dims": {
    "y": 1,
    "x": 1,
    "time": 1,
    "wavelength": 2001
  },
  "rsot_650nm": 0.047138178221010914,
  "rsot_865nm": 0.4100649627325952,
  "rsot_1600nm": 0.26994893328935227,
  "rdd_mean": 0.22790320156253166,
  "rso_mean": 0.22750613414759552
}
```

## 3. Run the High-Level Workflow

```bash
PYTHONPATH=src python examples/scope_workflow_demo.py
```

Typical output:

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
  "dims": {
    "y": 1,
    "x": 1,
    "time": 2,
    "wavelength": 2001,
    "direction": 2,
    "layer_interface": 4,
    "fluorescence_wavelength": 53,
    "layer": 3
  },
  "rsot_650nm_t0": 0.04522854188089004,
  "LoF_peak_t0": 1.985767010834904e-05,
  "LoF_peak_wavelength_t0": 744.0
}
```

## 4. Prepare External Inputs

If you already have weather, observation, and Sentinel-2 bio inputs, build a runner-ready dataset with:

```bash
scope prepare \
  --weather weather.nc \
  --observation observation.nc \
  --bio-npz post_bio.npz \
  --year 2020 \
  --output scope_inputs.nc
```

Equivalent direct entry points are:

```bash
scope-prepare --help
python prepare_scope_input.py --help
```

If you need the meaning of a variable name while building inputs, use:

```bash
scope vars Cab
scope vars Rntot
scope vars --all --kind output
```

## 5. Run the Prepared Dataset

Once you have a prepared input dataset, run the installed workflow CLI:

```bash
scope run \
  --input scope_inputs.nc \
  --output scope_outputs.nc \
  --scope-root ./upstream/SCOPE \
  --workflow scope
```

For a reflectance-only run:

```bash
scope run \
  --input scope_inputs.nc \
  --output scope_reflectance.nc \
  --scope-root ./upstream/SCOPE \
  --workflow reflectance
```

Equivalent direct entry point:

```bash
scope-run --help
```

## 6. What to Read Next

- [Model Mechanics](model-mechanics.md)
- [Input / Output Reference](input-output-reference.md)
- [Examples](examples.md)
