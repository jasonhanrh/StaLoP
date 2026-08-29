# Experiment 2 — Frozen Scientific Protocol

## Objective

Validate the **final Section 3 joint response-adaptive covariance** for StaLoP against a deliberately naive covariance that conditions on the realized panel weights. This is an inference experiment, not a tuning experiment.

## Frozen DGP

- Historical panels: `T=120`; future target panel: `121`.
- Within-panel sample sizes: `N in {200,500,1000}`.
- `X_ti iid N(0,I_2)` and `epsilon_ti iid N(0,1)`.
- Globally linear response: `Y_ti = beta0_t + 0.8 X1_ti - 0.5 X2_ti + epsilon_ti`, hence the target-local linear representation is exact.

### Prespecified nonadjacent analogue panels

Eligible final-fit panels are `{21,41,61,81,101}`. Their one-sided histories are `{20,40,60,80,100}`, respectively, and the target state uses panel `120`. No response from panel 121 is generated or used.

History-panel intercept offsets are `gamma=(-0.60,-0.30,0,0.30,0.60)`. Eligible next-panel intercepts obey the exact bridge `beta0_t=0.50+0.75 gamma_t`; target panel 121 has intercept `0.50`. Population discrepancy/softmax symmetry makes the weighted bridge offset exactly zero. Thus local-polynomial bias and population transfer bias are both zero, while individual analogue panels remain heterogeneous, so the first-order adaptive-weight term is nondegenerate.

## Frozen StaLoP configuration

Experiment 2 performs **no tuning**.

- basis: `phi(x,x*)=(1,x1-x1*,x2-x2*)`;
- isotropic Gaussian kernel;
- bandwidth `h_N = 1.50 N^{-1/6}`;
- `eta=1`;
- `delta_theta=1e-6`;
- all five prespecified nonadjacent panels are used;
- no GDF and no Algorithm-2 selection.

Inference grid: 15 points `(u,0)`, `u in [-0.75,0.75]`. Nominal pointwise and simultaneous coverage: 90%.

## Covariance estimators

FULL uses the empirical Section-3 joint influence function with nuisance block `gamma=(six predictive states, vech(W))`:

`psi_c_eff = psi_c - A_c,gamma A_gamma,gamma^{-1} psi_gamma`.

NAIVE uses the exact same fitted StaLoP point estimator but drops the indirect nuisance term, i.e. treats realized panel weights as fixed.

All derivatives through `z,W -> theta -> standardized theta -> alpha -> c` are analytic.

## Simultaneous inference

For each replication, the influence covariance is standardized to a correlation matrix and the 0.90 quantile of `max |G(x)|` is approximated with 4096 prespecified common Gaussian draws. A tiny eigenvalue floor is used only for Gaussian correlation simulation; reported covariance matrices and SEs are not altered.

## Formal Monte Carlo

Frozen count: `R=500` for each `N=200,500,1000`. Formal seed family: `20260816`.

## Primary outputs

For FULL and NAIVE: pointwise Z mean/SD/quantiles, 90% pointwise coverage and width, simultaneous 90% coverage and width, empirical q90 of max-|Z|, mean Gaussian q90, runtime, and tracked numerical-array memory.

## Non-negotiable gates

- no target-response leakage;
- analytic derivative finite-difference parity;
- empirical score roots;
- optimized influence covariance equals explicit 27-dimensional sandwich;
- FULL and NAIVE share the exact same point estimate;
- deterministic reproducibility;
- no silent ridge, pseudoinverse, failure replacement, or coverage-driven DGP/tuning modification.
