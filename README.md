# StaLoP final production release (2026-08-28)

This repository contains only the currently frozen final production code and the corresponding final result artifacts.

## Included suites

1. `01_experiment1_simulation`: final R=200 simulation and theory/fixed-K result summaries.
2. `02_experiment2_prediction_bands`: final R=500 simultaneous-confidence-band outputs for N=200, 500, 1000, plus the frozen N=2000, 5000, and 10000 extensions and the 15-grid coverage addendum.
3. `03_applications`: final R=200 sine, CES, and variable-selection applications.
4. `04_realdata`: final migration-analysis code and final Texas, Georgia, and Florida outputs.

## Deliberately excluded

- all legacy/audit/development versions;
- pilot runs and obsolete diagnostics;
- per-task checkpoint trees, because the included top-level aggregate raw outputs, summaries, manifests, and source hashes are the final retained provenance;
- the 7.8 GB raw migration CSV and the three frozen semiannual binned migration inputs. Their filenames, hashes, provenance, and loading specification are documented in `04_realdata/code/DATA_PROVENANCE.md` and `04_realdata/code/REALDATA_PROTOCOL.md`.

Each suite's code directory includes its frozen runner(s), core implementation, acceptance test(s), protocol, requirements, and source-hash/provenance files. Results directories contain only final aggregate output files and manifests, not resumable checkpoints.
