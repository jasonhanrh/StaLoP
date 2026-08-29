# Section 6 — Experiment 2 final implementation suite

This package implements the **new** Experiment 2 for the current StaLoP Section-3 joint response-adaptive inference theory. It is independent of the legacy simultaneous-band code and Experiment 1.

## Files

- `experiment2_core.py` — DGP, StaLoP fit, analytic derivatives, joint/naive influence functions, simultaneous covariance.
- `run_experiment2.py` — parallel checkpointed Monte Carlo runner.
- `test_experiment2_acceptance.py` — algebraic/numerical acceptance suite.
- `EXPERIMENT2_PROTOCOL.md` — frozen scientific protocol.
- `QA_REPORT_EXPERIMENT2.md` — audit/benchmark report.

## Acceptance

```bash
python test_experiment2_acceptance.py
```

## Implementation pilot

```bash
python run_experiment2.py --R 30 --seed0 2026081601 --n-jobs 6 --N 200 500 1000 --outdir experiment2_pilot_R30
```

The pilot is only for derivative integrity, conditioning, numerical stability, runtime and memory. Do not alter the DGP/configuration based on FULL-versus-NAIVE coverage.

## Formal run

```bash
python run_experiment2.py --formal --n-jobs 6 --outdir experiment2_formal_R500
```

`--formal` locks `R=500`, `N={200,500,1000}`, and `seed0=20260816`.

## Runtime/memory architecture

- analytic derivatives; no autodiff;
- 24-dimensional nuisance block + 3-dimensional final coefficient;
- block solves instead of repeatedly inverting the full 27x27 Jacobian;
- scalar influence vectors constructed directly; explicit 27x27 sandwich exists only for QA parity;
- full `T x N` storage only for X; Y is generated only for panels entering states/final fit;
- common reusable Gaussian integration draws;
- per-replication compressed atomic checkpoints;
- multiprocessing across replications;
- BLAS thread caps to prevent worker oversubscription.

The runner records tracked large-array memory. For OS-level RSS, use `/usr/bin/time -v` or the operating-system process monitor.
