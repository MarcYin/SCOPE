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

1. fetch the pinned upstream checkout with `scope-fetch-upstream`
2. keep that checkout version-controlled or provisioned by deployment automation
3. pass `scope_root_path=...` explicitly if the checkout is not under `./upstream/SCOPE`

## Benchmark and CI Strategy

The current repository uses two parity modes:

- live MATLAB export when MATLAB is available
- pregenerated MATLAB fixture fallback when it is not

That makes hosted CI deterministic without requiring MATLAB on every runner, while still keeping a self-hosted live-MATLAB lane for stronger operational checks.

The documentation surface is also treated as a build artifact:

- example scripts are executed in the test suite
- the docs site can be built locally with `mkdocs build --strict`
- CI should keep docs build failures separate from physics regressions

## Release and Distribution

For maintainers, the repository now has separate operational paths for packaging and docs deployment:

- `.github/workflows/release.yml`
  Builds source and wheel distributions for `SCOPE-RTM`, runs `twine check`, auto-publishes to PyPI on version tags, and still supports manual TestPyPI or PyPI publishing through GitHub environments.
- `.github/workflows/docs.yml`
  Builds the MkDocs site and deploys it to GitHub Pages.

Local release verification uses:

```bash
python -m pip install -e ".[release]"
python -m build
python -m twine check dist/*
```

## Current Operational Tradeoffs

The main operational tradeoffs are now:

- upstream SCOPE asset provisioning
- checked-in benchmark fixture footprint
- whether the self-hosted live-MATLAB lane should eventually become required CI

These are operational decisions, not open physics gaps.
