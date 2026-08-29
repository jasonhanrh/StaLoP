# Section 7 Applications — QA Report

## Assessment

The code has passed independent algebraic parity, information-ordering, reproducibility, numerical-finiteness, and production-size pilot checks. The percentages below are engineering/scientific audit assessments, not literal posterior probabilities.

| Component | Assessment | Main evidence |
|---|---:|---|
| Shared GDF/MSPE algebra | 99% | optimized sufficient-statistic result matches explicit D/H/GDF construction candidate-by-candidate |
| Sequence no-future leakage | 98% | perturbing one-step response leaves its observed input and its Algorithm-1 predictive state unchanged |
| Sequence final Algorithm-2 implementation | 97% | h-specific states/W/ranking rebuilt; joint K/eta minimization; R=30 pilot stable |
| CES DGP parity | 99% | same CES formula, distributions, calibration bounds, seeds, sample sizes, and 20x20 formal evaluation design as legacy |
| CES no-future leakage | 98% | response panel is excluded from its own 60-panel predictive-state history |
| CES Algorithm-2 specialization | 97% | singleton response-free h permitted by Section 4; exact K/eta GDF block; deterministic sigma2=0 explicit |
| Multidimensional Algorithm-1/2 algebra | 99% | scalar d=2 candidate matches frozen Experiment-1 reference for prediction, R, GDF, and criterion |
| Vectorized variable-selection engine | 98% | batch result matches scalar reference; d=0 empty subset tested |
| Variable-selection full recomputation | 98% | every subset independently rebuilds distances, local bases, states, W, discrepancies, rankings and tuning |
| Reproducibility | 98% | deterministic seed families and exact same-seed sequence outputs (excluding wall-clock timing) |
| Runtime / memory engineering | 96% | vectorized target blocks, cumulative sufficient stats, process-level BLAS cap, no all-candidate tensor retained across subsets |

No assessed production component is below 95%.

## Acceptance suite

Final command:

```bash
python test_applications_acceptance.py
```

Final result:

```text
PASS test_point_panel_gdf_explicit_parity
PASS test_sine_dgp_and_no_future_leakage
PASS test_sine_reproducibility_and_finiteness
PASS test_ces_dgp_and_strict_states
PASS test_ces_small_prediction_finite
PASS test_nd_scalar_parity_with_frozen_reference
PASS test_nd_batch_matches_scalar_and_empty_subset
PASS test_final_dynamic_dgp_contract
ALL_APPLICATIONS_ACCEPTANCE_TESTS_PASS
```

## Scientific pilots (not manuscript results)

These pilots were used only to test whether the unchanged DGPs remain scientifically meaningful under the final implementation. They did not trigger result-driven DGP changes.

### Sequence, R=30

- Global Linear MSPE: 0.005930
- Time-Local: 0.009627
- KNN-Covariate: 0.009261
- StaLoP: 0.005892

The intended qualitative comparison remains viable without changing the sine DGP.

### CES, R=10 at each noise level, n_eval=400

StaLoP / KNN / Global mean MSPE:

- sigma=0.1: 0.005009 / 0.021726 / 0.041504
- sigma=0.2: 0.005622 / 0.024675 / 0.044210
- sigma=0.3: 0.008699 / 0.031772 / 0.049199

### Variable selection

Production-size one-replication pilots passed at all four settings. The correct `{x1,x2}` subset was selected in each inspected setting. A five-replication `(N,T)=(100,30)` pilot selected `{x1,x2}` in 5/5 replications. These are QA pilots only and must not be reported as final evidence.

## Performance benchmarks

Container benchmark, single-process unless noted:

- Sequence R=30 with 6 workers: ~20 s wall; max reported RSS ~125 MB.
- CES R=10 x 3 noise levels with 6 workers: ~12 s wall; max reported RSS ~166 MB.
- Variable selection `(N,T)=(100,30)`: ~5.6 s per replication.
- Variable selection `(N,T)=(200,120)`: ~27.6 s per replication at block size 20; max single-process RSS ~221 MB.

`block_size=20` is the selected speed/memory compromise. Larger blocks increase memory; smaller blocks increase repeated overhead. Actual wall time depends strongly on available physical cores and BLAS/runtime configuration.

## Important interpretation note for CES

The calibrated simulator is deterministic conditional on theta_hat. Therefore the observational variance entering the Section-4 optimism correction is zero in this application. The code does not manufacture a positive pseudo-noise estimate merely to make GDF nonzero. This deterministic-simulator specialization is reflected in the current manuscript.
