# Frozen Section 7 Application Protocol

Date frozen: 2026-08-16

## Common rules

1. Formal Monte Carlo replications: **R=200** for each controlled application setting.
2. Do not change DGPs, candidate grids, seeds, method definitions, or evaluation sets after inspecting formal results.
3. All response-derived predictive states are one-sided.
4. Algorithm-2 candidate comparison is the final GDF/MSPE criterion from manuscript Section 4.
5. Deterministic machine-scale ties use a prespecified parsimony rule; substantive ties are not broken using future risk.
6. BLAS threads are capped at one per worker to avoid oversubscription.

## Application 1

### DGP
- `A=0.75`, `sigma=0.1`.
- `t=0,...,1.60` by `0.01`.
- rolling origins `t_s >= 1.10`.
- target is noise-free `f(t_{s+1})`.

### Method
- observable one-sided input: local level/slope from the last 7 raw observations through `t_s`;
- Algorithm-1 history length: 7 prior input/response panels;
- Gaussian covariate kernel;
- `q_h=(0.10,0.20,0.30,0.50,0.75,1.00)`;
- `eta=(1e-6,0.01,0.05,0.10,0.25,0.50,1,2,5)`;
- K fractions `(0.25,0.50,0.75,1.00)`;
- noise variance estimated strictly pre-response by the second-difference estimator;
- legacy Time-Local and KNN grids retained and tuned before the evaluation interval.

## Application 2

### DGP
Exactly preserves the legacy CES calibration DGP. Seeds remain `910000+rep` for real data and `920000+rep` for simulator design, preserving paired comparisons across noise levels.

### Method
- generated simulator order defines the sequential panel order;
- one simulator output per response panel;
- `H_a` is the previous 60 simulator runs;
- Gaussian kernel;
- response-free singleton bandwidth rule `q_h=60/500=0.12`;
- eta grid `(0.25,0.50,0.75,1,1.5,2,3)`;
- K fractions `(0.20,0.30,0.40,0.50,0.70,1.00)`;
- deterministic simulator conditional variance `sigma2_hat=0`;
- evaluation grid `20x20`.

## Application 3

### DGP
- exact current Experiment-1 predictable four-regime cycle;
- `rho_x=0.6`, `rho_eta=0.5`, `sigma_mu=1`, burn-in 50;
- `sigma_a=0`, `sigma_u=1`;
- add two iid standard-normal noise covariates independent of the DGP;
- simulate T historical panels plus target panel T+1.

### Method
- historical response-free standardization of all four candidate variables once per replication;
- all 16 subsets, including empty and full;
- K_state=1, matching final Experiment 1;
- q_h grid `(0.05,0.075,0.10,0.15,0.20,0.30)`;
- eta grid `(0.25,0.50,0.75,1,1.5,2,3)`;
- K fractions `(0.30,0.40,0.50,0.60,0.70,0.80)`;
- known controlled-DGP variance `sigma2=1` held common across all subsets;
- each target input independently minimizes Algorithm-2 criterion;
- subset score is the average minimized criterion across target units;
- ties: criterion, smaller subset size, lexical label;
- test MSPE is against the noise-free target-panel mean.

No formal-results-driven modification is permitted after launch.
