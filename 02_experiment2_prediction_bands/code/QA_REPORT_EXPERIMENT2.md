# Experiment 2 implementation QA report

## Assessment

A new standalone Experiment-2 production implementation has been built for the current Section-3 joint response-adaptive inference theory. It does not reuse the legacy fixed-realized-weight / Richardson / calibrated-inflation simultaneous-band branch and does not modify Experiment 1.

The included formal `R=500` results were generated under the frozen scientific protocol.

## Mathematical implementation audit

The code uses the final full-sample joint estimating-equation representation. With nuisance vector `gamma=(six predictive states, vech(W))`, it implements

`IF_full = -q' A_cc^{-1} {psi_c - A_c,gamma A_gamma,gamma^{-1} psi_gamma}`

and the deliberately naive comparator

`IF_naive = -q' A_cc^{-1} psi_c`.

The point estimator is identical under FULL and NAIVE; only covariance estimation differs.

Analytic derivatives include the complete chain

`z, W -> theta -> sample-standardized theta -> softmax alpha -> final local regression`.

The nuisance Jacobian is block-solved; the production path never constructs the full observation-by-27 score matrix. The explicit 27-dimensional score/sandwich is retained only as an independent QA parity path.

## Acceptance tests

`python test_experiment2_acceptance.py` returns:

`ALL_EXPERIMENT2_ACCEPTANCE_TESTS_PASS`

Passed gates:

1. strict one-sided history and no target-panel response leakage;
2. exact population bridge-balance identity;
3. prespecified bandwidth rule;
4. analytic softmax/discrepancy/state/W derivative vs central finite difference;
5. optimized scalar-influence covariance vs explicit 27-dimensional sandwich;
6. empirical roots of all state, W, and final-regression estimating equations;
7. nondegenerate `J_alpha_gamma`, `A_c,gamma`, and indirect influence term;
8. finite positive SE/covariance outputs;
9. deterministic same-seed reproducibility;
10. atomic checkpoint/resume without rewriting valid checkpoints.

No pseudoinverse, silent ridge, failed-replication substitution, target-response fallback, or coverage-driven tuning exists in the production code.

## Implementation pilot

Command:

```bash
python run_experiment2.py --R 30 --seed0 2026081601 --n-jobs 6 --N 200 500 1000 --outdir experiment2_pilot_R30
```

Status: `90/90` valid tasks, no failed replication and no nonfinite numeric output.

Numerical-stability maxima/minima across the 30-replication implementation pilot:

| N | max root error | max cond(W) | max state-Gram cond | max final-Gram cond | min ||A_c,gamma|| | min indirect-row norm |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3.16e-15 | 4.12 | 6.02 | 4.74 | 9.33e-4 | 1.79 |
| 500 | 4.55e-15 | 4.92 | 6.30 | 5.38 | 8.75e-4 | 2.74 |
| 1000 | 5.65e-16 | 5.75 | 6.83 | 6.06 | 8.66e-4 | 3.97 |

The smallest raw eigenvalues of the estimated grid correlation matrices are at numerical-zero scale (roughly `-3e-16` to `1e-14`), as expected for a highly correlated smooth 15-point grid. The code applies a `1e-12` eigenspectrum floor only when generating the Gaussian max-statistic correlation draws. It does not alter reported influence covariances or standard errors.

## Runtime and memory

Implementation-pilot wall time on the available container: `27.45 s` for 90 tasks with `n_jobs=6`; `/usr/bin/time -v` total wall time `28.35 s` and maximum RSS reported by `/usr/bin/time -v` for the command execution was approximately `141,464 KB`; this is not an aggregate sum of all worker RSS values.

Mean per-replication method runtime in that parallel pilot:

| N | mean seconds | max seconds | tracked large-array memory per task |
|---:|---:|---:|---:|
| 200 | 1.11 | 1.45 | 6.33 MB |
| 500 | 1.65 | 2.31 | 15.12 MB |
| 1000 | 2.63 | 3.41 | 29.77 MB |

The tracked-memory metric covers the largest explicitly owned numerical arrays (`X`, full/naive influence matrices, covariance matrices, Gaussian draws). OS RSS also includes Python/NumPy/process overhead and shared libraries.

## Optimization decisions

- Analytic derivatives rather than autodiff or numerical derivatives in production.
- `24 + 3` block system; no repeated full `27 x 27` inverse in production.
- Direct scalar influence construction; full 27-vector scores are QA-only.
- Responses generated only for the 11 panels that actually enter state/final equations; only `X` is stored for all 120 panels because `W` requires all historical designs.
- Common Gaussian max-statistic draws cached once per worker.
- Vectorized NumPy kernels and moment calculations.
- Parallelization across independent Monte Carlo replications.
- BLAS threads capped at one per worker to avoid oversubscription.
- Compressed atomic per-replication checkpoints and deterministic resume.

## Frozen formal command

```bash
python test_experiment2_acceptance.py
python run_experiment2.py --formal --n-jobs 6 --outdir experiment2_formal_R500
```

`--formal` locks `R=500`, `N={200,500,1000}`, and seed family `20260816`.
