# Data provenance

## External inputs not included in this release

The final real-data analysis uses the following frozen semiannual binned inputs:

- Georgia: `realdata_edge_bins.csv`
- Florida: `fl_od_binned.csv`
- Texas: `tx_od_binned.csv`

Their approved SHA-256 values are recorded in `REALDATA_PROTOCOL.md` and `PROVENANCE_REALDATA.json`. These input files are external to this code release and must be supplied to the real-data runners through the command-line paths documented in `README.md`.

## Raw-source provenance

Primary monthly source recorded by the production audit:

`sampleOD_events_dist_covars_risk.csv` (external data file; not included in this release)

Audit size: 7,782,682,713 bytes.

The final model rerun uses the preserved semiannual binned inputs above rather than repeatedly ingesting the 7.8-GB monthly source. This keeps the raw-to-binned layer identical to the audited production aggregation.

The binned source fields used by the final analysis are `tbin`, county IDs, `flow`, distance/log-distance, `x1_i/x1_j`, and `x2_i/x2_j`. The production raw loader documents `x1` as log population (income fallback) and `x2` as the FEMA-style risk-percentile field `riskPctl`, not event counts.
