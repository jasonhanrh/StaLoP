# Experiment 1 implementation QA report

## Assessment

The package contains the completed Experiment-1 R=200 production results. The underlying StaLoP estimator/tuner remains the previously audited `simulation_core.py`; the Experiment-1 diagnostics do not alter the estimated StaLoP mathematics.

Confidence values below are engineering/scientific audit assessments, not probabilities.

| Component | Assessment | Evidence |
|---|---:|---|
| Frozen Algorithm-1 implementation | 99% | original acceptance suite; strict pre-response states, panel-1 exclusion, no silent future fallback |
| Frozen Algorithm-2 `(K,h,eta)` tuning | 99% | exhaustive finite-grid parity and explicit GDF tests in original suite |
| Main Experiment-1 parity for existing four methods | 99% | same-seed extended replication reproduces Global/Time-Local/KNN/StaLoP MSPE exactly |
| Oracle-Compatible StaLoP | 98% | true regime only defines retained subset; explicit observation-level D/H/GDF parity passes |
| MAE implementation | 99% | direct mean absolute error against the same noise-free target used for MSPE |
| Compatibility recovery metric | 98% | exact retained panel reconstruction at selected q_h/K; precision, recall, enrichment and theta separation stored |
| Bias-variance-vs-K diagnostic | 97% | fixed point/regime strata keep truth constant; `bias^2 + variance = MSPE` identity verified numerically; h/eta profiled by the same GDF criterion conditional on K |
| Estimated-MSPE vs true-risk tracking | 97% | candidate-level Monte Carlo aggregation; full-coverage candidate restriction prevents missing-validity selection bias |
| Criterion-selected vs oracle configuration regret | 97% | oracle and criterion-selected fixed candidates defined within the same fixed point/regime finite-grid stratum; adaptive Algorithm-2 risk reported separately |
| Baseline independent tuning | 98% | inherited audited rolling-origin tuning; no methodological changes |
| Reproducibility / checkpointing | 98% | deterministic `(seed0,T,N,rep)` seeds; atomic hashed checkpoints; resume acceptance test passes |
| Runtime optimization | 96% | frozen vectorized StaLoP core; Experiment-1 diagnostics separated from reported StaLoP runtime; theory candidate aggregation streams into compact accumulators |
| Memory optimization | 97% | six-worker main smoke peak process-tree RSS ~685 MB; theory diagnostic ~770 MB; no all-candidate main-table storage |
| Code structure / auditability | 98% | frozen estimator core separated from Experiment-1 extensions and theory runner; legacy runners/protocols excluded from this release |

No component is assessed below 95%.

## Acceptance tests

Both suites pass:

- `python test_acceptance.py` -> `ALL_V3_ACCEPTANCE_TESTS_PASS`
- `python test_experiment1_acceptance.py` -> `ALL_EXPERIMENT1_ACCEPTANCE_TESTS_PASS`

Experiment-1-specific gates include:

1. exact same-seed MSPE parity for Global Linear, Time-Local, KNN-Covariate and StaLoP;
2. Oracle-Compatible explicit D/H/GDF parity;
3. candidate-selection minimum-criterion check;
4. K bias-variance algebraic identity check;
5. hashed checkpoint/resume equivalence.

## Runtime / memory benchmark

Test environment: Linux container, single-threaded BLAS per process.

- Main Experiment-1 smoke, R=2 in all four cells, 4 workers: about 12.3 s internal elapsed; process peak from `/usr/bin/time` ~143 MB for parent. A process-tree monitor with 6 requested workers measured peak total RSS ~685 MB.
- Theory diagnostics, R=20 in representative settings `(N,T)=(100,30),(200,120)`, 6 workers: about 7.9 s internal elapsed; separate process-tree monitor measured peak total RSS ~770 MB.

The main experiment is intentionally more expensive than the earlier four-method pilot because it additionally computes Oracle-Compatible StaLoP and compatibility-recovery diagnostics. The reported StaLoP runtime excludes the diagnostic-only retained-set reconstruction, so the runtime table remains a fair method runtime rather than a data-analysis overhead measurement.

## Scientific definitions fixed in code

### Oracle-Compatible StaLoP

Uses the same eligible historical panel set as StaLoP, true regime labels to retain exactly compatible panels, uniform panel weights among those panels, and the same GDF-corrected MSPE criterion to select the covariate bandwidth. It is an oracle compatibility benchmark, not a deployable method.

### Compatible Panels Retained (%)

Precision: fraction of retained panels whose realized regime equals the target regime. Oracle-Compatible is 100% by construction. Recall and enrichment are stored as additional diagnostics.

### Bias-variance over K

For each fixed K, h and eta are reselected by the same GDF criterion. Bias and variance are computed only within fixed target-point/target-regime strata, where the true future mean is constant, then sample-size weighted across strata.

### Estimated future MSPE / oracle configuration

Candidate-level true future risk is Monte Carlo squared error. Estimated MSPE is the GDF-corrected criterion; uncorrected risk is R. Risk summaries use only candidates with maximal replication coverage in each fixed point/regime stratum. The finite-grid oracle is the fixed candidate with minimum Monte Carlo true risk in that stratum; the criterion-selected fixed candidate minimizes mean estimated MSPE. Operational adaptive Algorithm-2 risk is reported in a separate column and is not conflated with fixed-grid regret.

## Retained manuscript execution

The R=200 manuscript results were generated with the explicit commands below:

```bash
python test_acceptance.py
python test_experiment1_acceptance.py
python run_experiment1_main.py --R 200 --seed0 20260814 --n-jobs 6 --part all --outdir experiment1_main_R200
python run_experiment1_theory.py --R 200 --seed0 20260815 --n-jobs 6 --settings representative --outdir experiment1_theory_R200
```

The runners retain an earlier `--formal` R=300 convenience preset; use the explicit R=200 commands above to reproduce the retained manuscript results.
