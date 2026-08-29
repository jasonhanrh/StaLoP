# Section 7 Applications — FINAL Algorithm 1/2 Rebuild

This package rebuilds the three manuscript Section 7 applications against the **final** StaLoP method and **final Section 4 GDF/MSPE tuning rule**.

## Statistical contract

- Algorithm 1: one-sided predictive states; no response from a panel may enter that panel's predictive state.
- Algorithm 2: minimize
  `R + 2 * sigma2_hat * GDF - sigma2_hat` over the prespecified finite candidate set.
- Whenever `h` is tuned, predictive states, `W`, discrepancies, ranking, retained panels, and final weights are recomputed for every `h`.
- `K` and `eta` are selected jointly conditional on `h`; no legacy two-stage `eta`-then-`K` tuning remains.
- No future target response is used in fitting or tuning.
- No pseudoinverse, silent fallback fit, or coverage/result-driven parameter change is used in the new production engines.

The frozen Experiment-1 `simulation_core.py` is included as a reference implementation. Its SHA-256 is
`08d9df9195003193e6485ebea779c4a71193d61bc8c33bffa36101c054b07038`.

## Application 1 — Sequence Data

DGP is unchanged: `f(t)=0.75 sin(2*pi*t)`, noise SD `0.1`, grid `0:0.01:1.60`, rolling origins from `1.10`.

The legacy seven-observation level/slope summary is retained as the observable one-sided input for predicting the next response. The final StaLoP predictive state is then constructed from the preceding seven input/next-response panels, strictly excluding the current panel response. `h`, `K`, and `eta` are jointly selected target-by-target by Algorithm 2.

Formal run:

```bash
python run_application1_sine.py --formal --n-jobs 6 --outdir application1_sine_R200
```

## Application 2 — CES Simulator Calibration

The legacy CES DGP/calibration mechanism is unchanged:

- `(A,delta,rho)=(1,0.6,-0.5)`;
- real sample `n=300`;
- calibrated simulator sample `n=500`;
- multiplicative log-normal noise levels `0.1,0.2,0.3`;
- constrained calibration `rho in [-0.95,0.95]`;
- evaluation grid `20 x 20 = 400`.

The 500 simulator runs retain their generated order and are treated as sequential one-observation panels. The predictive state for a response panel uses only the preceding 60 simulator runs; the target state uses the final 60. The target-specific bandwidth is prespecified by the response-free `60/500=0.12` distance quantile, so Section 4's `h` candidate set is a singleton; `K` and `eta` are jointly selected.

Conditional on the calibrated parameter, simulator outputs are deterministic, so the observation-noise variance in the Section-4 criterion is exactly zero. The GDF term is computed but has zero multiplier rather than introducing artificial simulator noise.

Formal run:

```bash
python run_application2_ces.py --formal --n-jobs 6 --n-eval 400 --outdir application2_ces_R200
```

## Application 3 — Target-Relevant Variable Selection

The DGP is the **current final Experiment-1 recurrent dynamic-panel DGP**, plus two independent `N(0,1)` noise covariates. Hence the four final settings are

- `(N,T)=(100,30)`
- `(200,30)`
- `(100,120)`
- `(200,120)`

rather than the obsolete `N=40` cells in the earlier table.

For every one of the 16 subsets of `{x1,x2,z1,z2}`, the entire StaLoP pipeline is recomputed. The production grid preserves the legacy variable-selection bandwidth and eta grids and adds the legacy retained-panel fraction grid so that `K,h,eta` are jointly tuned. The controlled DGP has known `sigma_u^2=1`; that known variance is used for the GDF criterion so the application isolates variable-subset selection rather than re-testing variance estimation already studied in Experiment 1.

Formal run:

```bash
python run_application3_variable_selection.py --formal --n-jobs 6 \
  --settings 100x30,200x30,100x120,200x120 --block-size 20 \
  --save-candidates --outdir application3_variable_selection_R200
```

## Required first step

Before any production run:

```bash
python test_applications_acceptance.py
```

The final line must be:

```text
ALL_APPLICATIONS_ACCEPTANCE_TESTS_PASS
```

See `QA_REPORT_APPLICATIONS.md` and `APPLICATIONS_PROTOCOL.md` for the audit and frozen protocol.
