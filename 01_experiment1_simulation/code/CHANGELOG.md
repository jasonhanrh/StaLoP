# Final production changes relative to the May--June Section 5 code

- aligned the simulation estimand with `m_t(x)` by removing the unit random intercept from the primary DGP;
- replaced the stochastic regime transition in the primary design by a predictable recurrent cycle so the simulation directly instantiates the revised predictive-compatibility mechanism;
- enforced strictly pre-response predictive states and removed panel-1 self-use;
- removed silent full-history/identity fallbacks;
- replaced rolling-CV StaLoP tuning by target-specific joint GDF-MSPE selection of `(K,h,eta)`;
- recompute the complete state/discrepancy/ranking path for each bandwidth candidate;
- made the final local fit/GDF hat matrix unpenalized, matching the revised manuscript equations; ridge is retained only for `W` stabilization;
- introduced a pre-target nearest-neighbor pair-difference noise-variance estimator with zero-distance extrapolation;
- independently tune Time-Local and KNN with pre-target rolling-origin validation; KNN is the uniform nearest-response mean;
- added broad, nonbinding tuning grids and an explicit near-uniform positive eta limit;
- vectorized/batched the dominant StaLoP and baseline computations;
- fixed exact seed provenance, added atomic checkpoints/resume, and guaranteed execution-order/worker-count invariance;
- separated pilot and untouched formal seed families;
- excluded the legacy simultaneous-band implementation because it is not the final Section 3 joint-adaptive inference procedure.
