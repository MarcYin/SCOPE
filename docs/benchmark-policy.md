# Benchmark Policy

This repo keeps three high-level MATLAB benchmark summary views under `tests/data/`:

1. `parity_worst_cases`
   This is the primary same-state parity contract. Use this summary for relative-error gating.
2. `absolute_policy_worst_cases`
   This is the fallback view for low-magnitude canopy thermal partition terms whose absolute errors stay negligible while relative errors can be unstable.
3. `stress_worst_cases`
   This contains upstream scenes or timesteps where MATLAB `ebal` hit `maxit`. These are tracked as stress diagnostics, not as parity-gating cases.

## Same-State vs Phase-Lagged Metrics

Do not read raw `energy_balance.sunlit_*` or `energy_balance.shaded_*` fields in isolation when judging leaf-physiology parity. Those are phase-lagged iterate diagnostics from the coupled energy loop.

For true same-state parity:

- Use `parity_worst_cases`.
- Prefer `leaf_iteration.*` metrics for leaf biochemistry quantities such as `A`, `Ci`, `eta`, and `rcw`.

The generated benchmark summaries now encode those replacements directly in their `parity_policy` section.

## CI Policy

The default required lane is the hosted CPU Python suite in [`tests.yml`](../.github/workflows/tests.yml).

The GPU and MATLAB parity lanes remain opt-in because they require self-hosted infrastructure:

- GPU needs a self-hosted CUDA runner.
- MATLAB parity needs a self-hosted MATLAB runner and a licensed MATLAB installation.

That keeps the required CI signal reproducible on standard GitHub-hosted infrastructure while still making the stronger parity checks available on demand.
