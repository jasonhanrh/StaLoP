# Final county-to-county migration analysis

This package replaces the legacy real-data StaLoP residual predictor with the final manuscript Algorithms 1--2. It includes the three frozen processed binned inputs needed for the formal migration analysis; it does not include the 7.8-GB monthly raw source.

## 1. Acceptance tests

```bash
python test_realdata_acceptance.py
```

Required final line:

```text
ALL_REALDATA_ACCEPTANCE_TESTS_PASS
```

## 2. Use the frozen binned inputs

The three required semiannual binned inputs are bundled in `../data/`:

- Georgia: `../data/realdata_edge_bins.csv`
- Florida: `../data/fl_od_binned.csv`
- Texas: `../data/tx_od_binned.csv`

Before running the analysis, verify each file against the approved SHA-256 value recorded in `REALDATA_PROTOCOL.md` and `PROVENANCE_REALDATA.json`.

## 3. Formal multi-state run

```bash
python run_realdata_multistate.py \
  --ga ../data/realdata_edge_bins.csv \
  --fl ../data/fl_od_binned.csv \
  --tx ../data/tx_od_binned.csv \
  --output-dir realdata_FINAL_results
```

Do not alter grids or rerun selectively based on results.

For a single state:

```bash
python run_realdata_migration.py \
  --input "<PATH_TO_BINNED_FILE>" \
  --state-name Georgia \
  --output-dir realdata_ga_FINAL
```

## 4. Georgia manuscript figure

```bash
python make_realdata_figures.py \
  --results realdata_FINAL_results/georgia \
  --output realdata_main_combined_mspe_retrieval_FINAL.png
```

The script also writes a PDF.

## 5. Descriptive-only audit

```bash
python audit_realdata_descriptives.py \
  --input "<PATH_TO_GA_BINNED>" \
  --out ga_data_audit
```

## Production rules

- Do not modify method, grids, baselines, active-edge definition, or target bins after launch.
- Do not use target responses for model fitting or tuning.
- Do not replace an invalid StaLoP candidate with a ridge/pseudoinverse/fallback.
- Do not describe the risk-percentile covariate as a destination flood-event count.
- Do not use legacy real-data outputs in the revised manuscript after the final rerun.
