# Section 5 — Experiment 1 final implementation suite

This package extends the audited StaLoP production engine to implement the **full Experiment 1 used in the final manuscript**. The audited estimator/tuning engine in `simulation_core.py` is preserved; Experiment-1-specific diagnostics are implemented in `experiment1_core.py`.

## Frozen primary design

- Settings: `(N,T) = (100,30), (200,30), (100,120), (200,120)`.
- DGP: recurrent predictable four-regime dynamic panel, `sigma_a=0`, observational variance `sigma_u^2=1`.
- Predictive state: `K_state=1`, so every historical state is strictly pre-response.
- StaLoP: target-specific exhaustive finite-grid Algorithm-2 tuning over `(K,h,eta)`, with `q_h` only indexing the realized target-specific bandwidth `h`.
- Time-Local and KNN-Covariate: independently tuned by pre-target rolling-origin validation.
- KNN final prediction: uniform arithmetic mean of the selected nearest responses.

## Experiment 1 main comparison

Retained manuscript run:

```bash
python test_acceptance.py
python test_experiment1_acceptance.py
python run_experiment1_main.py --R 200 --seed0 20260814 --n-jobs 6 --part all --outdir experiment1_main_R200
```

Methods:

1. Global Linear
2. Time-Local
3. KNN-Covariate
4. StaLoP
5. Oracle-Compatible StaLoP

Reported quantities include MSPE, MAE, total method runtime, and (for StaLoP/oracle) true-regime compatibility of retained panels.

### Oracle-Compatible StaLoP definition

The oracle uses the same eligible historical panels as StaLoP (panels with a valid pre-response state), but true regime labels identify the panels whose subsequently realized response surface matches the target regime. All truly compatible eligible panels are retained with uniform panel weights. The covariate bandwidth is selected from the same `q_h` grid by the same GDF-corrected MSPE criterion. Thus the oracle isolates perfect compatibility recovery; it does not use future responses for fitting.

### Compatibility metric

`compatible_retained_pct` is the percentage of **retained panels that truly share the target regime** (precision). The tuning output also stores recall, baseline compatible prevalence, enrichment, and mean discrepancy among compatible/incompatible panels.

## Experiment 1 theory/risk diagnostics

The candidate-level diagnostics are intentionally separated from the full N-target main comparison to keep computation and memory modest. By default they use the representative settings `(N,T)=(100,30)` and `(200,120)` and three fixed central target inputs `(-0.5,-0.5)`, `(0,0)`, `(0.5,0.5)`.

Retained manuscript theory run:

```bash
python run_experiment1_theory.py --R 200 --seed0 20260815 --n-jobs 6 --settings representative --outdir experiment1_theory_R200
```

The retained theory run uses `seed0=20260815`, distinct from the main run seed family and inspected theory-pilot seeds. The runners retain an earlier `--formal` R=300 convenience preset; that preset was not used for the R=200 manuscript results in this release. Because the retained R=200 main run was invoked explicitly rather than through `--formal`, its preserved `launch.json` records `run_mode: "pilot"`; this is a legacy runner-mode label only.

Outputs:

- `candidate_risk_tracking.csv`: Monte Carlo true future MSPE, mean GDF-corrected estimated MSPE, and mean uncorrected residual risk for every finite-grid candidate, stratified by target point and true regime.
- `risk_tracking_summary.csv`: RMSE/MAE/correlation with true risk for corrected versus uncorrected criteria.
- `k_bias_variance.csv`: empirical squared bias, variance, true future MSPE, estimated MSPE, and uncorrected risk over retained-panel count K. For each fixed K, `(h,eta)` are profiled by the same GDF criterion.
- `oracle_configuration_detail.csv`: finite-grid true-risk oracle configuration versus the configuration selected by average estimated MSPE in each fixed target-point/regime stratum.
- `oracle_configuration_summary.csv`: compact setting-level oracle MSPE, selected MSPE, regret, and selected tuning summaries.
- `adaptive_selected_rows.csv`: the operational target-wise Algorithm-2 selections for reference.

### Bias-variance definition

Bias and variance are calculated within fixed target-point/true-regime strata, where the true target mean is constant. Variance uses denominator `n`, so `bias^2 + variance = MSPE` algebraically within each stratum (up to machine rounding). Stratum summaries are then sample-size weighted.

### Oracle-configuration definition

For the theory diagnostic only, the Monte Carlo finite-grid oracle is the fixed candidate `(K,q_h,eta)` with the smallest empirical true future MSPE within a fixed target-point/regime stratum. The criterion-selected fixed candidate minimizes mean estimated MSPE in the same stratum. This makes the reported finite-grid regret nonnegative by construction. The operational adaptive Algorithm-2 MSPE is reported separately and is not mislabeled as that fixed-configuration regret.

## Reproducibility and safety

- deterministic signed-int64 seeds keyed by `(seed0,T,N,replicate_id)`;
- per-replication atomic checkpointing and resume;
- protocol/source hashes stored in manifests;
- BLAS threads forced to one per worker;
- no simulation result is allowed to change the DGP, tuning grids, or method definitions.

Legacy May/June runners and pre-Experiment-1 protocols are not included in this production release.
