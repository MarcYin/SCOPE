# Production Notes

## Recommended Deployment Assumptions

For the current codebase, the safest production assumptions are:

- run from a source checkout or controlled internal package build
- keep a pinned upstream SCOPE checkout available
- prefer the runner surface over direct low-level kernel composition unless you need custom research code
- persist prepared and simulated datasets through the shared NetCDF writer

## Asset Strategy

Asset-backed constructors expect upstream SCOPE resources such as:

- FLUSPECT parameter MAT files
- soil spectra
- atmospheric files

The intended operational path is:

1. fetch the pinned upstream checkout with `scope-fetch-upstream` or `scope fetch-upstream`
2. keep that checkout version-controlled or provisioned by deployment automation
3. pass `scope_root_path=...` explicitly if the checkout is not under `./upstream/SCOPE`

For scientific or operational deployments, keep the attribution trail visible to users: this package is based on the original MATLAB SCOPE repository at [Christiaanvandertol/SCOPE](https://github.com/Christiaanvandertol/SCOPE), and the upstream manual lives at [scope-model.readthedocs.io](https://scope-model.readthedocs.io/en/master/).

## Benchmark and CI Strategy

The current repository uses two parity modes:

- live MATLAB export when MATLAB is available
- pregenerated MATLAB fixture fallback when it is not

That makes hosted CI deterministic without requiring MATLAB on every runner, while still keeping a self-hosted live-MATLAB lane for stronger operational checks.

The documentation surface is also treated as a build artifact:

- example scripts are executed in the test suite
- the docs site can be built locally with `mkdocs build --strict`
- CI should keep docs build failures separate from physics regressions

## Inference API

For production code that does not need `xarray`, use:

- `ScopeInferenceModel`

This surface returns only the requested outputs and avoids the dataset assembly overhead of the runner path. It is the recommended deployment API for repeated same-shape inference workloads inside services or larger pipelines.

## Performance and Compilation

For kernel-level timing and eager-versus-compiled comparisons, use:

```bash
PYTHONPATH=src python scripts/benchmark_kernels.py --fixture scope-assets --mode compare
```

The current recommendation is:

- do not enable `torch.compile` by default in production workflows
- benchmark on the actual target hardware first
- only consider a compiled path for long-lived services with repeated same-shape calls

On the current reference CPU environment, `fluspect` and canopy `reflectance` benefit in steady state, `thermal` only pays off after a much larger number of calls, layered `fluorescence` currently fails under `torch.compile`, and leaf biochemistry currently regresses because scalar root-solving logic still causes graph breaks and recompilation churn.

## Autograd and Chunking

`chunk_size` slices SCOPE forward execution. For inference, export, or detached dataset assembly, that bounds the amount of model work done in one forward call.

Dataset assembly detaches each selected chunk and moves it to CPU before concatenating outputs, so non-autograd runs do not keep all completed chunks resident on the model device. Use `output_vars=(...)` on runner calls, or repeat `--output-var` in the CLI, when only a subset of variables is needed.

For gradient-based optimisation, slicing the forward pass is not enough by itself. If code collects all chunk outputs, concatenates them, computes one full-batch loss, and calls `loss.backward()` once, PyTorch must retain every chunk's intermediate graph until that backward pass finishes. In that pattern, `chunk_size` improves scheduling and per-forward temporary allocations, but it does not make activation memory scale with the chunk size.

The memory-oriented autograd pattern is to stream chunks and backpropagate each chunk loss before keeping the next chunk output:

```python
optim_tensor.grad = None
total_loss = 0.0

for outputs in runner.iter_scope_dataset_tensors(
    data_module,
    varmap=varmap,
    scope_options=scope_options,
):
    chunk_loss = loss_fn(outputs)
    chunk_loss.backward()
    total_loss += float(chunk_loss.detach())
```

For direct tensor inputs, use `runner.iter_scope_tensors(...)` the same way. If the objective is a global mean or sum, weight each chunk loss by its contribution to the full objective before calling `backward()`. Avoid storing chunk outputs that still have `grad_fn`; keeping those tensors alive also keeps their graphs alive.

When `calc_ebal=1` and the configured batch fits in a single chunk, the streaming tensor APIs use a single-chunk fast path. That path preserves the one-loss backward pattern for small optimisation slices while avoiding the extra outer chunk wrapper that is useful only for larger grids.

For multi-chunk `calc_ebal=1` runs where the configured chunk size does not cover the whole batch, set `truncate_backprop=True` on `EnergyBalanceOptions` (or `energy_truncate_backprop=1` / `truncate_backprop=1` in `scope_options` and dataset attrs). The Picard convergence sweep runs under `torch.no_grad()` — no autograd recording cost during the iteration — and a single final iteration is re-run with grad enabled on the converged state to produce the gradient-bearing outputs. Backward stays bounded to one iteration's biochem / thermal / heat-flux operations regardless of how many Picard iterations the forward took; forward stays at no_grad speed regardless of `max_iter`. This is the 1-step truncated BPTT ("phantom gradient") approximation that is standard for fixed-point iterations: at convergence the iterate is independent of the trajectory that produced it, so the gradient through earlier iterations is small and the truncation is a faithful approximation of the implicit gradient. The converged state and the final outputs are bitwise identical to the full-backprop solve — only the autograd recording cost and the graph backward walks differ.

## Release and Distribution

For maintainers, the repository now has separate operational paths for packaging and docs deployment:

- `.github/workflows/release.yml`
  Builds source and wheel distributions for `SCOPE-RTM`, runs `twine check`, smoke-installs the built wheel, auto-publishes to PyPI on version tags, and still supports manual TestPyPI or PyPI publishing through GitHub environments.
- `.github/workflows/docs.yml`
  Builds the MkDocs site and deploys it to GitHub Pages.

Local release verification uses:

```bash
python -m pip install -e ".[release]"
python -m build
python -m twine check dist/*
```

The packaged wheel is also smoke-installed in CI and must satisfy:

```bash
scope --help
scope-fetch-upstream --help
scope-prepare --help
scope-run --help
```

Release notes and provenance are also automated:

- `.github/workflows/release-drafter.yml`
  Maintains draft GitHub release notes from merged pull requests.
- `.github/workflows/release.yml`
  Creates the tagged GitHub release, uploads distribution artifacts, and emits GitHub artifact attestations for built packages.

## Current Operational Tradeoffs

The main operational tradeoffs are now:

- upstream SCOPE asset provisioning
- checked-in benchmark fixture footprint
- whether the self-hosted live-MATLAB lane should eventually become required CI

These are operational decisions, not open physics gaps.
