# Final real-data production QA report

## Assessment

The package contains the completed final rerun using the frozen binned migration inputs. The production method is aligned with the final Algorithm 1 and GDF-MSPE Algorithm 2, not the obsolete pseudo-target/future-panel tuning versions.

Engineering/scientific confidence assessments (not probabilistic guarantees):

| Component | Assessment | Main evidence |
|---|---:|---|
| Final Algorithm-1 mapping | 98% | strict one-sided history tests; target response not an input; balanced reference parity |
| Final Algorithm-2 joint tuning | 99% | each h rebuilds states/W/theta/ranking; exhaustive K/eta search; frozen Experiment-1 parity |
| GDF/MSPE algebra | 99% | direct observation-level D/H parity on synthetic ragged panels |
| Legacy real-data protocol preservation | 98% | same six targets, active-edge rule, separable gravity layer, side-panel definitions, distance term |
| Baseline fairness | 97% | training-only Time-Local; uniform-mean KNN with rolling historical validation and fold-specific nuisance refit |
| Leakage control | 99% | target response is separated before side-panel construction; mutation-invariance test |
| Data identifiers / sample-size audit | 99% | Georgia normalized count exactly reproduces 159 origins, 1,616 destinations and all six n_eval/n_train counts |
| Numerical fail-fast behavior | 98% | zero state/final ridge, no pinv/lstsq in production core, invalid candidates rejected |
| Relevance-mass accounting | 99% | selected normalized mass sums to one exactly within tolerance |
| Runtime / memory | 96% | panel sufficient statistics; candidate cumulative algebra; binned inputs avoid 7.8-GB raw load; target-level arrays not accumulated |

No production component is assessed below 95%.

## Acceptance tests

The clean acceptance suite passes:

- strict one-sided `H_a` construction;
- county-FIPS normalization;
- explicit observation-level `D,H` GDF parity;
- exact parity with the frozen Experiment-1 Algorithm-2 core on a balanced synthetic panel;
- h-dependent recomputation of the state path;
- period-mass normalization;
- target-response invariance;
- binned-loader row/key preservation.

Required line:

`ALL_REALDATA_ACCEPTANCE_TESTS_PASS`

## Legacy issues that required correction

The May real-data implementation could not be reused as-is because:

1. its state was an ID-specific level/slope history summary rather than the final target-local predictive-state estimator;
2. it used robust-scaled Euclidean state distances instead of the final W-induced discrepancy;
3. changing `q_h` did not recompute states, W, discrepancy or ranking;
4. state/final solves silently escalated ridge and had fallbacks;
5. KNN used a separate GDF-local regression rather than the locked uniform-mean nearest-neighbor baseline;
6. the manuscript described a flood-event-count covariate although the production loader used `riskPctl_i/riskPctl_j`.

The final package changes only these method/implementation mismatches and keeps the empirical design and gravity decomposition.

## Actual-data QA performed before packaging

Using the preserved Georgia binned input, descriptive code reproduces exactly:

- 237,950 rows;
- 159 origins;
- 1,616 destinations;
- 40 semiannual bins;
- 5,948.75 mean rows/bin (5,939--5,959);
- target n_eval `86,90,90,93,97,105`;
- training rows `190278,196237,202196,208155,214114,220073`.

A complete six-bin Georgia run finished with finite outputs, no invalid final target, and all three h values valid for all displayed selected-target rows. It verified execution/mechanism; the statistical grids were not changed after inspecting its performance. The selected relevance-mass output also assigns meaningful mass to nonadjacent historical bins, consistent with the manuscript's cautious mechanism statement.

Fresh single-target Florida and Texas checks also completed without numerical failure. No outcome-driven DGP, grid, or baseline change was made.

## Manuscript/release consistency safeguards

The final manuscript and release should:

- keep the verified dataset and evaluation sample sizes explicit;
- describe `x2` as the county risk-percentile covariate, not event count;
- say Algorithm 2 selects `(K,h,eta)` rather than only K;
- describe the gravity decomposition explicitly rather than implying StaLoP is fit directly to raw edge Y with all covariates in one local regression;
- treat nonadjacent relevance-mass patterns as evidence consistent with non-temporal borrowing, not as a causal explanation of performance;
- report heterogeneous bin-level performance without claiming uniform dominance;
- regenerate every table and figure from the final frozen outputs.
