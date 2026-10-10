# Data provenance

## Released frozen binned inputs

The final real-data analysis uses the following frozen semiannual binned inputs,
which are included under `../data/` in this release:

- Georgia: `../data/realdata_edge_bins.csv` (27.43 MiB)
- Florida: `../data/fl_od_binned.csv` (45.32 MiB)
- Texas: `../data/tx_od_binned.csv` (59.30 MiB)

Their approved SHA-256 values are recorded in `REALDATA_PROTOCOL.md` and
`PROVENANCE_REALDATA.json`. The repository preserves their original bytes via
the data-specific rule in `.gitattributes`; do not rewrite their line endings.

## Raw-source provenance

Primary monthly source recorded by the production audit:

`sampleOD_events_dist_covars_risk.csv` (external 7.8-GB data file; not included in this release)

Audit size: 7,782,682,713 bytes.

The final model rerun uses the preserved semiannual binned inputs above rather than repeatedly ingesting the 7.8-GB monthly source. This keeps the raw-to-binned layer identical to the audited production aggregation.

The binned source fields used by the final analysis are `tbin`, county IDs, `flow`, distance/log-distance, `x1_i/x1_j`, and `x2_i/x2_j`. The production raw loader documents `x1` as log population (income fallback) and `x2` as the FEMA-style risk-percentile field `riskPctl`, not event counts.
