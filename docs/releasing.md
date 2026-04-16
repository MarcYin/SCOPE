# Releasing

This page documents the current release path for the published package:

- PyPI package name: `SCOPE-RTM`
- Python import name: `scope`

## Release Trigger

The repository publishes automatically to PyPI from:

- `.github/workflows/release.yml`

The workflow behavior is:

- push a tag matching `v*` -> verify tag/version, rerun release-local validation, build, smoke-install, publish to PyPI
- manual dispatch with `publish_target=testpypi` -> publish to TestPyPI
- manual dispatch with `publish_target=pypi` -> publish to PyPI

Release notes are prepared continuously by:

- `.github/workflows/release-drafter.yml`

The tagged release workflow also:

- publishes an existing draft release for the tag if one already exists
- otherwise creates a new GitHub release with generated notes
- uploads built artifacts to that GitHub release
- emits GitHub artifact attestations for `dist/*`

## What the Release Workflow Verifies

Before publishing, the workflow:

1. verifies that the pushed tag matches `project.version`
2. reruns the release-local CPU test matrix and strict docs build
3. builds `sdist` and wheel artifacts
4. runs `twine check`
5. installs both the built wheel and the built `sdist` on clean runners
6. verifies:
   - `import scope`
   - `scope.__version__ == importlib.metadata.version("SCOPE-RTM")`
   - `scope --help`
   - `scope fetch-upstream --help`
   - `scope prepare --help`
   - `scope run --help`
   - a minimal installed-package `scope run` reflectance workflow using the checked-out upstream assets

## Required GitHub / PyPI Setup

Before the first real release, configure:

- GitHub environment `pypi`
- GitHub environment `testpypi`
- PyPI trusted publisher for project `SCOPE-RTM`
- TestPyPI trusted publisher for project `SCOPE-RTM`

The trusted publisher must point to:

- repository owner / repo
- workflow file: `.github/workflows/release.yml`
- environment: `pypi` or `testpypi`

## Maintainer Workflow

Recommended sequence:

1. Ensure `main` is green.
2. For releases that touch parity, workflows, or benchmark scripts, run the live MATLAB validation lane on the exact candidate commit and review the benchmark outputs:

```bash
PYTHONPATH=src python -m pytest -q tests/test_scope_benchmark_parity.py tests/test_scope_timeseries_benchmark_parity.py
PYTHONPATH=src python scripts/run_scope_benchmark_suite.py --matlab /Applications/MATLAB_R2025b.app/bin/matlab
```

3. Optionally run a manual TestPyPI publish.
4. Review the current GitHub draft release notes.
5. Bump version in `pyproject.toml`.
6. Create and push a matching tag like `v0.2.0`.
7. Confirm the GitHub release, artifact attestations, and PyPI install path:

```bash
python -m pip install SCOPE-RTM
python -c "import scope; print(scope.__version__)"
```

If your repository configuration enforces branch protections or required checks for tags, keep those settings documented alongside this workflow. The workflow itself now re-runs its own validation gates before publishing, but external protections still determine who can push release tags and when.
