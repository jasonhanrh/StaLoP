# Frozen real-data migration protocol

## Purpose

Re-run the county-to-county migration analysis with the final manuscript Algorithms 1 and 2 while preserving the legacy empirical design wherever it is not methodologically inconsistent.

No result-driven modification is permitted after the formal launch.

## Data protocol retained from the earlier manuscript version

- Semiannual county-to-county OD bins, 2000--2019.
- Response on each edge: `log1p(max(flow,0))`.
- Main detailed state: Georgia.
- Robustness states: Florida and Texas.
- Formal target bins: `201601,201602,201701,201702,201801,201802`.
- Training for target bin `tb`: all rows with `tbin < tb`.
- Active evaluation edges: OD pairs with positive flow in at least one earlier bin. The target-bin flow is allowed to be zero.
- Target-bin responses are evaluation-only and are not available to model fitting or tuning.
- Target-bin county covariates are prediction inputs and may be used because they are observed at forecast time.

The binned covariates inherited from the legacy production loader are:

- `x1_i`, `x1_j`: log population (with the historical income fallback in raw preprocessing);
- `x2_i`, `x2_j`: county risk-percentile covariates (`riskPctl_i`, `riskPctl_j`);
- `logdist`: log OD distance.

The final manuscript must not describe `x2` as the number of flood events; the production source uses the risk-percentile fields.

## Preserved separable gravity application layer

For every rolling target and for origin and destination sides separately:

1. Aggregate historical edge flow to a time-by-county side panel.
2. Fit the historical-only nuisance regression `side_y ~ 1 + x1 + x2`.
3. Estimate a historical-only county effect `alpha_i` as the mean residual for that county.
4. Define the dynamic residual panel response by removing the nuisance regression and county effect.
5. Apply final StaLoP Algorithms 1--2 to this residual panel.
6. Add the nuisance component back to the StaLoP residual prediction.

The final edge prediction remains

`origin prediction + destination prediction + gamma * logdist`,

where `gamma` is estimated from historical edges only. This preserves the legacy real-data gravity decomposition; the StaLoP residual predictor is the component replaced by the final manuscript algorithm.

## Final Algorithm 1 mapping

All target-local calculations use training-standardized `(x1,x2)` coordinates. Standardization is historical-only and response-free.

- Basis: centered local linear `[1, x1-x1*, x2-x2*]`.
- Covariate kernel: binary local kernel `1{||x-x*|| <= h}`. This is the same production kernel used by the frozen Experiment-1 Algorithm-2 reference.
- `H_a`: up to the preceding six semiannual panels, strictly before panel `a`.
- Historical eligible panels: labels 2 through T; panel 1 is excluded because it has no preceding predictive state.
- For each `h`, predictive states are pooled local-linear fits over `H_a`.
- `W_h`: kernel-weighted empirical local-basis Gram matrix over historical residual-panel observations, with the manuscript-allowed `1e-8` numerical ridge only on W.
- Discrepancy: `(z_t-z_target)' W_h (z_t-z_target)`.
- Discrepancies are standardized with `delta_theta=1e-8`.
- Panel factors: `exp(-eta * theta_tilde)` over the retained panels.
- Observation weights: retained-panel factor times the same covariate kernel, normalized over all retained observations.
- Final residual prediction: intercept of the weighted local-linear fit.
- No current/target response enters a predictive state.

## Final Algorithm 2 mapping

Frozen grid:

- bandwidth quantile indices `q_h = (0.10, 0.20, 0.30)`;
- `eta = (0.25, 0.50, 0.75, 1.00)`;
- K fractions `(0.20, 0.30, 0.40, 0.50, 0.70, 1.00)`, converted to integer retained-panel counts and deduplicated.

For every `q_h`, the realized target-specific `h` is the corresponding historical covariate-distance quantile. **The complete state/W/discrepancy/ranking path is rebuilt for every h.**

For each candidate `(K,h,eta)`:

- `R` is the normalized weighted historical residual loss;
- `GDF = tr(D H)` is evaluated exactly from sufficient statistics;
- criterion is `R + 2*sigma2_hat*GDF - sigma2_hat`;
- state and final ridges are exactly zero;
- rank-deficient or numerically ill-conditioned candidates are rejected, not silently regularized or replaced;
- deterministic numerical ties use smaller `q_h`, then smaller K, then eta closest to 1.

`sigma2_hat` is a training-only within-panel nearest-neighbor pair-difference estimate, using the closest 50% of unique NN pairs and extrapolating the pair-difference variance to zero covariate distance. This is the same variance-estimation principle used in the frozen Experiment-1 production core.

## Baselines

- **Global Linear:** the historical-only nuisance regression plus historical county effect, with no dynamic residual borrowing.
- **Time-Local:** the same nuisance layer plus a one-sided local-linear level from the last six observed residuals for the county. Any baseline fallback is explicitly recorded; it does not alter StaLoP.
- **KNN-Covariate:** uniform arithmetic mean of the k nearest historical residual observations in standardized covariate space. k is selected separately for origin/destination using six historical rolling-validation panels and a frozen grid `(5,10,20,40,80,160,320)`. Each validation fold re-estimates its nuisance layer from data strictly preceding that validation panel.

## Formal data inputs

Use the preserved semiannual binned inputs. This avoids repeatedly loading the 7.8-GB monthly raw CSV and preserves exact raw-to-binned provenance.

Known SHA-256 values from the audit bundle:

- Georgia `realdata_edge_bins.csv`: `3e58bbec5f4c1450be43a4dde1fbd5868c91170d559ff7a089296d0cb8d34e4b`
- Florida `fl_od_binned.csv`: `94103c9531f3d17aaeacf3e2c5bc212542e6f7ab3671e2bf8b02f6abed36fcb4`
- Texas `tx_od_binned.csv`: `7a9d245332081bffd459887d60b74b312e90592033e7fb3ccbc5abe5d4a20651`

The monthly raw source remains the provenance source; it is not needed for the formal model rerun.

## Frozen Georgia descriptive gates

Before any scientific interpretation, the normalized binned Georgia input must reproduce:

- 237,950 semiannual OD rows;
- 159 origin counties;
- 1,616 destination counties;
- 40 bins, 200001--201902;
- rows/bin mean 5,948.75, min 5,939, max 5,959;
- active evaluation counts `86,90,90,93,97,105`;
- historical training counts `190278,196237,202196,208155,214114,220073`.

The previous audit value 3,231 destination counties was an identifier-normalization artifact; FIPS normalization recovers 1,616.

## Formal outputs required

For each origin state:

- bin-level metrics;
- edge-level predictions;
- selected `(K,q_h,h,eta)` by target/side/county;
- historical-period relevance mass;
- top historical periods;
- KNN validation diagnostics;
- data audit;
- runtime;
- source/input hashes in manifest.

For Georgia, the manuscript figure is regenerated from the frozen final outputs only.
