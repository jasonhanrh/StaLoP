from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

import simulation_core as sim


def get_case(N=40, T=30, seed=20260506, mode="predictable_cycle"):
    rng = np.random.default_rng(seed)
    data = sim.simulate_panel(N, T, sim.DGPConfig(sigma_a=0.0, regime_mode=mode), rng)
    t_star_idx = int(np.floor(0.70 * T)) - 1
    pre = sim.stack_prearrival(data["x1"], data["x2"], data["y"], t_star_idx)
    x_star = np.array([data["x1"][0, t_star_idx], data["x2"][0, t_star_idx]])
    return data, x_star, pre


def test_pre_response_histories():
    T_hist = 10
    assert sim.history_indices_for_state(1, T_hist, 1).size == 0
    assert np.array_equal(sim.history_indices_for_state(2, T_hist, 1), np.array([0]))
    assert np.array_equal(sim.history_indices_for_state(8, T_hist, 1), np.array([6]))
    assert np.array_equal(sim.history_indices_for_state(11, T_hist, 1), np.array([9]))
    assert np.array_equal(sim.history_indices_for_state(8, T_hist, 5), np.array([2,3,4,5,6]))


def test_panel1_excluded_and_qh_recompute():
    _, x_star, pre = get_case(N=200)
    x1p,x2p,yp,tp,Th = pre
    cfg = sim.MethodConfig(K_state=1)
    comps=[]
    for q in (0.05,0.20):
        try: comps.append(sim.build_target_components(x1p,x2p,yp,tp,x_star,Th,q,cfg))
        except sim.InvalidCandidate: pass
    assert len(comps) == 2
    for c in comps:
        elig=np.asarray(c["eligible_panel_idx"])
        assert 0 not in elig
        assert np.array_equal(elig,np.arange(1,Th))
    assert comps[0]["h"] != comps[1]["h"]
    assert not np.allclose(comps[0]["W"], comps[1]["W"]) or not np.allclose(comps[0]["theta"], comps[1]["theta"])


def test_batched_state_solver_parity():
    _, x_star, pre = get_case(N=200)
    x1p, x2p, yp, tp, Th = pre
    cfg = sim.MethodConfig(K_state=1)
    static = sim.prepare_target_static(x1p, x2p, x_star)
    for q in (0.05, 0.10, 0.20, 0.40):
        dist = np.asarray(static["dist"])
        h = max(float(np.quantile(dist, q)), cfg.min_bandwidth)
        kval = (dist <= h).astype(float)
        A, b, w, _ = sim.kernel_panel_sufficient_stats(
            np.asarray(static["Phi"]), yp, kval, tp, Th
        )
        try:
            z_fast, zt_fast, e_fast = sim.estimate_predictive_states_from_panel_stats(A, b, w, Th, cfg)
            z_ref, zt_ref, e_ref = sim.estimate_predictive_states_from_panel_stats_reference(A, b, w, Th, cfg)
        except sim.InvalidCandidate:
            # Invalidity must agree; verify the reference is also invalid if production rejected.
            try:
                sim.estimate_predictive_states_from_panel_stats_reference(A, b, w, Th, cfg)
            except sim.InvalidCandidate:
                continue
            raise AssertionError((q, "production/reference state validity mismatch"))
        assert np.array_equal(e_fast, e_ref)
        assert np.max(np.abs(z_fast-z_ref)) < 1e-10, (q, np.max(np.abs(z_fast-z_ref)))
        assert np.max(np.abs(zt_fast-zt_ref)) < 1e-10, (q, np.max(np.abs(zt_fast-zt_ref)))


def test_fast_gdf_reference_parity():
    _, x_star, pre = get_case(N=200)
    x1p,x2p,yp,tp,Th=pre
    cfg=sim.MethodConfig(K_state=1)
    sigma2,_=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,cfg.noise_pair_fraction,cfg.min_noise_pairs)
    comp=None
    for q in (0.05,0.10,0.20):
        try:
            comp=sim.build_target_components(x1p,x2p,yp,tp,x_star,Th,q,cfg)
            break
        except sim.InvalidCandidate: continue
    assert comp is not None
    Ks=sim.rank_candidate_counts(len(comp["theta"]),(0.2,0.5,0.8,0.9))
    for eta in (0.05,0.75,3.0,12.0):
        fast={int(r["K"]):r for r in sim.evaluate_eta_over_K_fast(comp,eta,Ks,sigma2,cfg)}
        for K in Ks:
            try: ref=sim.evaluate_gdf_candidate_reference(comp,tp,eta,K,sigma2,cfg)
            except sim.InvalidCandidate:
                assert K not in fast
                continue
            f=fast[K]
            for key in ("pred","criterion","R","GDF","ESS"):
                assert abs(float(ref[key])-float(f[key])) < 1e-9, (eta,K,key,ref[key],f[key])


def test_explicit_DH_gdf_parity():
    """Directly reconstruct normalized D and the ridge-stabilized H matrix."""
    _, x_star, pre = get_case(N=40)
    x1p, x2p, yp, tp, Th = pre
    cfg = sim.MethodConfig(K_state=1)
    sigma2, _ = sim.estimate_noise_variance_within_panel_nn(
        x1p, x2p, yp, tp, Th, cfg.noise_pair_fraction, cfg.min_noise_pairs
    )
    comp = sim.build_target_components(x1p, x2p, yp, tp, x_star, Th, 0.20, cfg)
    eta = 1.5
    K = sim.rank_candidate_counts(len(comp["theta"]), (0.50,), (5,))[0]
    ref = sim.evaluate_gdf_candidate_reference(comp, tp, eta, K, sigma2, cfg)

    eligible = np.asarray(comp["eligible_panel_idx"], dtype=int)
    order = np.asarray(comp["order"], dtype=int)
    theta_std = np.asarray(comp["theta_std"], dtype=float)
    retained_pos = order[:int(K)]
    log_score = -eta * theta_std
    shift = float(np.max(log_score[retained_pos]))
    beta_full = np.zeros(Th, dtype=float)
    beta_full[eligible[retained_pos]] = np.exp(log_score[retained_pos] - shift)
    obs_raw = beta_full[tp - 1] * np.asarray(comp["K_vals"], dtype=float)
    Q = float(obs_raw.sum())
    assert Q > 0
    w = obs_raw / Q
    Phi = np.asarray(comp["Phi"], dtype=float)
    D = np.diag(w)
    M = Phi.T @ D @ Phi + cfg.final_ridge * np.eye(Phi.shape[1])
    H = Phi @ np.linalg.solve(M, Phi.T @ D)
    resid = yp - H @ yp
    R_direct = float(resid @ D @ resid)
    GDF_direct = float(np.trace(D @ H))
    pred_direct = float((H @ yp)[0] * 0.0 + np.linalg.solve(M, Phi.T @ D @ yp)[0])
    crit_direct = R_direct + 2.0 * sigma2 * GDF_direct - sigma2

    assert abs(R_direct - ref["R"]) < 1e-9, (R_direct, ref["R"])
    assert abs(GDF_direct - ref["GDF"]) < 1e-9, (GDF_direct, ref["GDF"])
    assert abs(pred_direct - ref["pred"]) < 1e-9, (pred_direct, ref["pred"])
    assert abs(crit_direct - ref["criterion"]) < 1e-9, (crit_direct, ref["criterion"])


def test_knn_kdtree_production_parity_continuous():
    """Production KD-tree KNN equals direct Euclidean KNN on continuous data."""
    data, _, _ = get_case(N=40, T=30, seed=91731)
    x1, x2, y = data["x1"], data["x2"], data["y"]
    t_star_idx = int(np.floor(0.70 * 30)) - 1
    x1p, x2p, yp, _, _ = sim.stack_prearrival(x1, x2, y, t_star_idx)
    targets = np.arange(0, 40, 3)
    for k in (1, 2, 5, 20, 80, 320, 1280):
        batch = sim.predict_knn_covariate_batch(
            x1p, x2p, yp, x1[targets, t_star_idx], x2[targets, t_star_idx], k
        )
        direct = np.array([
            sim.predict_knn_covariate(
                x1p, x2p, yp, np.array([x1[i, t_star_idx], x2[i, t_star_idx]]), k
            )
            for i in targets
        ])
        assert np.max(np.abs(batch - direct)) < 1e-12, (k, np.max(np.abs(batch-direct)))

    small_grid = sim.TuningGrid(
        q_h_grid=(0.30,), eta_grid=(1.0,), rank_count_grid=(1,), rank_fraction_grid=(),
        time_window_grid=(3,), knn_k_grid=(1, 2, 5, 20, 80, 320, 1280)
    )
    tuned = sim.tune_knn_covariate(x1, x2, y, t_star_idx, small_grid)

    # Independent brute-force rolling-origin reference.
    scores = {}
    for k in small_grid.knn_k_grid:
        se = 0.0; count = 0
        for v_idx in sim.validation_times(t_star_idx):
            xa, xb, ya, _, _ = sim.stack_prearrival(x1, x2, y, v_idx)
            pred = np.array([
                sim.predict_knn_covariate(
                    xa, xb, ya, np.array([x1[i, v_idx], x2[i, v_idx]]), int(k)
                )
                for i in range(x1.shape[0])
            ])
            se += float(np.sum((pred - y[:, v_idx]) ** 2)); count += len(pred)
        scores[int(k)] = se / count
    k_ref = min(scores, key=lambda kk: (scores[kk], kk))
    assert int(tuned["knn_k"]) == int(k_ref), (tuned, k_ref, scores)
    assert abs(float(tuned["cv_mse"]) - float(scores[k_ref])) < 1e-12


def test_all_eta_k_batch_parity():
    _, x_star, pre = get_case(N=200)
    x1p,x2p,yp,tp,Th=pre
    cfg=sim.MethodConfig(K_state=1)
    sigma2,_=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,cfg.noise_pair_fraction,cfg.min_noise_pairs)
    comp=sim.build_target_components(x1p,x2p,yp,tp,x_star,Th,0.10,cfg)
    Ks=sim.rank_candidate_counts(len(comp["theta"]),(0.25,0.5),(1,2,3,5))
    etas=(0.01,0.25,1.0,5.0,20.0)
    batch={(r["eta"],int(r["K"])):r for r in sim.evaluate_all_eta_k_fast(comp,etas,Ks,sigma2,cfg)}
    for eta in etas:
        one={(r["eta"],int(r["K"])):r for r in sim.evaluate_eta_over_K_fast(comp,eta,Ks,sigma2,cfg)}
        for key,r in one.items():
            b=batch[key]
            for fld in ("pred","criterion","R","GDF","ESS"):
                assert abs(float(r[fld])-float(b[fld]))<1e-9,(key,fld,r[fld],b[fld])


def test_joint_argmin_matches_exhaustive_reference():
    _, x_star, pre=get_case(N=200)
    x1p,x2p,yp,tp,Th=pre
    cfg=sim.MethodConfig(K_state=1)
    sigma2,_=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,cfg.noise_pair_fraction,cfg.min_noise_pairs)
    grid=sim.TuningGrid(q_h_grid=(0.05,0.10),eta_grid=(0.1,1.0,5.0),rank_fraction_grid=(0.3,0.7))
    best,allc=sim.tune_stalop_gdf_target(x1p,x2p,yp,tp,x_star,Th,sigma2,grid,cfg,True)
    assert len(allc)>0
    rows=[r._asdict() for r in allc.itertuples(index=False)]
    manual=None
    for row in rows:
        if sim._candidate_is_better(row, manual): manual=row
    assert manual is not None
    assert abs(best["criterion"]-float(manual["criterion"]))<1e-12
    assert best["q_h"]==float(manual["q_h"])
    assert best["eta"]==float(manual["eta"])
    assert int(best["K"])==int(manual["K"])
    assert (Th-1) in set(allc["K"].astype(int))


def test_baseline_tuning_is_strictly_pre_target():
    for ti in (20,83):
        v=sim.validation_times(ti)
        assert v and max(v)<ti and min(v)>=1


def test_vectorized_time_local_matches_scalar():
    data,_,_=get_case(N=40)
    x1,x2,y=data["x1"],data["x2"],data["y"]
    v_idx=10
    x1p,x2p,yp,tp,Th=sim.stack_prearrival(x1,x2,y,v_idx)
    beta=sim.fit_time_local(x1p,x2p,yp,tp,Th,5)
    pv=sim.predict_linear_beta(beta,x1[:,v_idx],x2[:,v_idx])
    ps=np.array([sim.predict_time_local(x1p,x2p,yp,tp,np.array([x1[i,v_idx],x2[i,v_idx]]),Th,5) for i in range(x1.shape[0])])
    assert np.max(np.abs(pv-ps))<1e-10


def test_knn_partial_sort_exact_parity_with_ties():
    x1=np.array([0.,0.,1.,-1.,2.,-2.,3.,-3.])
    x2=np.zeros_like(x1)
    y=np.arange(len(x1),dtype=float)
    xs=np.array([0.,0.])
    x1_sd=max(float(np.std(x1,ddof=1)),1e-12); x2_sd=max(float(np.std(x2,ddof=1)),1e-12)
    for k in (1,2,3,4,5,8):
        got=sim._knn_sorted_prefix_values(x1,x2,y,xs,k,x1_sd,x2_sd)
        d2=((x1-xs[0])/x1_sd)**2+((x2-xs[1])/x2_sd)**2
        idx=np.lexsort((np.arange(len(d2)),d2))[:k]
        exp=y[idx]
        assert np.array_equal(got,exp),(k,got,exp)


def test_configuration_validation():
    sim.validate_configuration(sim.DGPConfig(), sim.MethodConfig(), sim.TuningGrid())
    bad = sim.TuningGrid(q_h_grid=(0.0,), eta_grid=(1.0,))
    try:
        sim.validate_configuration(sim.DGPConfig(), sim.MethodConfig(), bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid q_h grid was not rejected")


def test_deterministic_seed_mapping():
    s=[]
    for T in (30,120):
        for N in (40,200):
            for r in range(300):
                s.append(sim.deterministic_task_seed(20260506,T,N,r))
    assert len(s)==len(set(s))
    assert sim.deterministic_task_seed(20260506,120,200,17)==sim.deterministic_task_seed(20260506,120,200,17)
    assert sim.deterministic_task_seed(20260506,30,40,0)!=sim.deterministic_task_seed(20260506,120,40,0)


def test_checkpoint_resume_and_part_invariance():
    tmp=Path(tempfile.mkdtemp(prefix="s5v3test_"))
    try:
        small_grid=sim.TuningGrid(q_h_grid=(0.30,0.40),eta_grid=(0.5,1.0),rank_fraction_grid=(0.5,),time_window_grid=(3,),knn_k_grid=(10,))
        cfg=sim.MethodConfig(K_state=1)
        dgp=sim.DGPConfig(regime_mode="predictable_cycle")
        r1,t1,_=sim.run_monte_carlo_parallel([(40,30)],1,dgp_config=dgp,method_config=cfg,grid=small_grid,n_jobs=1,checkpoint_dir=tmp/"cp",resume=True)
        # Resume must perform zero recomputation.
        worker0=sim._worker
        def _should_not_run(task):
            raise AssertionError("resume recomputed a completed checkpoint")
        sim._worker=_should_not_run
        try:
            r2,t2,_=sim.run_monte_carlo_parallel([(40,30)],1,dgp_config=dgp,method_config=cfg,grid=small_grid,n_jobs=1,checkpoint_dir=tmp/"cp",resume=True)
        finally:
            sim._worker=worker0
        pd.testing.assert_frame_equal(r1,r2,check_dtype=False)
        pd.testing.assert_frame_equal(t1,t2,check_dtype=False)

        # Corrupt one cached payload. The hash/integrity checks must reject it and
        # recompute the task, restoring the exact deterministic result.
        task_id = sim._task_id(30,40,0)
        methods_path, _, _ = sim._checkpoint_paths(tmp/"cp", task_id)
        with open(methods_path, "a", encoding="utf-8") as f:
            f.write("\n")
        r3,t3,_=sim.run_monte_carlo_parallel([(40,30)],1,dgp_config=dgp,method_config=cfg,grid=small_grid,n_jobs=1,checkpoint_dir=tmp/"cp",resume=True)
        nonruntime = [c for c in r1.columns if c != "runtime_sec"]
        pd.testing.assert_frame_equal(r1[nonruntime],r3[nonruntime],check_dtype=False,rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(t1,t3,check_dtype=False,rtol=0,atol=1e-12)

        # Seed for the same cell/rep does not depend on whether other cells are present.
        assert int(r1.iloc[0]["seed"])==sim.deterministic_task_seed(20260506,30,40,0)
    finally:
        shutil.rmtree(tmp,ignore_errors=True)


def test_execution_order_and_parallel_invariance():
    grid=sim.TuningGrid(
        q_h_grid=(0.30,0.40,0.50), eta_grid=(0.5,1.0), rank_count_grid=(1,2,3),
        rank_fraction_grid=(0.5,), time_window_grid=(3,5), knn_k_grid=(1,5,10)
    )
    cfg=sim.MethodConfig(K_state=1)
    dgp=sim.DGPConfig(regime_mode="predictable_cycle")
    settings=[(40,30),(40,120)]
    r1,t1,_=sim.run_monte_carlo_parallel(settings,1,dgp_config=dgp,method_config=cfg,grid=grid,n_jobs=1)
    r2,t2,_=sim.run_monte_carlo_parallel(list(reversed(settings)),1,dgp_config=dgp,method_config=cfg,grid=grid,n_jobs=2)
    cols=[c for c in r1.columns if c!="runtime_sec"]
    a=r1[cols].sort_values(["T","N","replicate_id","method"],kind="mergesort").reset_index(drop=True)
    b=r2[cols].sort_values(["T","N","replicate_id","method"],kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=0,atol=1e-12)
    pd.testing.assert_frame_equal(
        t1.sort_values(["T","N","replicate_id","target_unit"],kind="mergesort").reset_index(drop=True),
        t2.sort_values(["T","N","replicate_id","target_unit"],kind="mergesort").reset_index(drop=True),
        check_dtype=False,rtol=0,atol=1e-12
    )


def sigma_sanity(nseed=20):
    rows=[]
    for N,T in ((40,30),(200,30),(40,120),(200,120)):
        vals=[]
        for r in range(nseed):
            seed=20260506+r
            d=sim.simulate_panel(N,T,sim.DGPConfig(sigma_u=1.0,regime_mode="predictable_cycle"),np.random.default_rng(seed))
            ti=int(np.floor(.7*T))-1
            x1p,x2p,yp,tp,Th=sim.stack_prearrival(d["x1"],d["x2"],d["y"],ti)
            est,npairs=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,0.5,20)
            assert npairs>=20 and np.isfinite(est) and est>0
            vals.append(est)
        m=float(np.mean(vals)); sd=float(np.std(vals,ddof=1))
        # Strong finite-sample gate: mean within 10% of truth in every formal cell.
        assert abs(m-1.0)<=0.10, (N,T,m,sd)
        rows.append((N,T,m,sd))
    return rows


def compatibility_sanity(nseed=5):
    rows=[]
    for N,T in ((40,30),(200,30),(40,120),(200,120)):
        enrich=[]; rhos=[]
        for r in range(nseed):
            df=sim.predictive_compatibility_diagnostic(
                N,T,sim.deterministic_task_seed(20260506,T,N,r),q_h=0.30,max_units=min(N,30)
            )
            if df.empty: continue
            for _,g in df.groupby("unit"):
                k=max(1,int(np.ceil(.25*len(g))))
                gs=g.sort_values("theta_hat",kind="mergesort")
                top=float(gs.iloc[:k]["same_future_regime"].mean())
                base=float(g["same_future_regime"].mean())
                if base>0: enrich.append(top/base)
                if g["theta_hat"].nunique()>1 and g["true_surface_discrepancy"].nunique()>1:
                    rho=float(g[["theta_hat","true_surface_discrepancy"]].corr(method="spearman").iloc[0,1])
                    if np.isfinite(rho): rhos.append(rho)
        mean_enrich=float(np.mean(enrich)); mean_rho=float(np.mean(rhos)); pos_rate=float(np.mean(np.asarray(rhos)>0))
        # Directional mechanism gates: state proximity enriches truly compatible
        # regimes and ranks true local-surface discrepancy in the right direction.
        assert mean_enrich>1.05, (N,T,mean_enrich)
        assert mean_rho>0.15 and pos_rate>0.65, (N,T,mean_rho,pos_rate)
        rows.append((N,T,mean_enrich,mean_rho,pos_rate))
    return rows


def test_seed_csv_roundtrip_exact():
    seeds=[sim.deterministic_task_seed(20260506,T,N,r) for T in (30,120) for N in (40,200) for r in range(10)]
    assert all(0 <= s < (1<<63) for s in seeds)
    df=pd.DataFrame({"seed":seeds,"x":np.arange(len(seeds),dtype=int)})
    tmp=Path(tempfile.mkdtemp(prefix="s5seed_"))
    try:
        path=tmp/"seed.csv"; df.to_csv(path,index=False)
        back=pd.read_csv(path)
        assert back["seed"].dtype.kind in "iu", back.dtypes
        assert back["seed"].astype("int64").tolist()==seeds
    finally:
        shutil.rmtree(tmp,ignore_errors=True)


def test_zero_final_ridge_matches_manuscript_hat_matrix():
    assert sim.MethodConfig().final_ridge == 0.0
    _,x_star,pre=get_case(N=200)
    x1p,x2p,yp,tp,Th=pre; cfg=sim.MethodConfig(K_state=1)
    sig,_=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,cfg.noise_pair_fraction,cfg.min_noise_pairs)
    comp=sim.build_target_components(x1p,x2p,yp,tp,x_star,Th,0.20,cfg)
    K=min(3,len(comp["theta"]))
    out=sim.evaluate_gdf_candidate_reference(comp,tp,1.0,K,sig,cfg)
    order=np.asarray(comp["order"]); elig=np.asarray(comp["eligible_panel_idx"]); th=np.asarray(comp["theta_std"])
    keep=order[:K]; beta=np.zeros(Th); scores=np.exp(-th[keep]); beta[elig[keep]]=scores
    kval=np.asarray(comp["K_vals"]); obs=beta[tp-1]*kval; D=obs/obs.sum(); Phi=np.asarray(comp["Phi"]); y=yp
    M=Phi.T@(D[:,None]*Phi); H=Phi@np.linalg.solve(M,Phi.T*D[None,:])
    R=float(np.sum(D*(y-H@y)**2)); G=float(np.trace(np.diag(D)@H))
    crit=R+2*sig*G-sig
    assert abs(out["R"]-R)<1e-9 and abs(out["GDF"]-G)<1e-9 and abs(out["criterion"]-crit)<1e-9


def test_eta_lower_grid_is_near_uniform_limit():
    grid=sim.TuningGrid()
    eta0=min(grid.eta_grid)
    assert eta0 <= 1e-6
    # For standardized scores from at most 82 eligible panels, the maximum
    # attainable z-score spread is O(sqrt(n)); this representative worst-case
    # range makes eta0 effectively uniform to far below simulation precision.
    z=np.linspace(-10.0,10.0,82)
    w=np.exp(-eta0*z); w/=w.sum()
    u=np.full_like(w,1.0/len(w))
    assert np.max(np.abs(w-u)) < 1e-6


def test_best_only_eta_k_block_parity():
    _,x_star,pre=get_case(N=200)
    x1p,x2p,yp,tp,Th=pre; cfg=sim.MethodConfig(K_state=1)
    sig,_=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,cfg.noise_pair_fraction,cfg.min_noise_pairs)
    comp=sim.build_target_components(x1p,x2p,yp,tp,x_star,Th,0.20,cfg)
    Ks=sim.rank_candidate_counts(len(comp["theta"]),(0.25,0.5,0.75),(1,2,3,4,5,8))
    etas=(1e-6,0.01,0.25,1.0,5.0,50.0)
    rows=sim.evaluate_all_eta_k_fast(comp,etas,Ks,sig,cfg)
    best,nvalid=sim.evaluate_all_eta_k_fast(comp,etas,Ks,sig,cfg,best_only=True)
    assert isinstance(rows,list) and best is not None and nvalid==len(rows)
    manual=None
    for row in rows:
        rr={**row,"q_h":0.20}
        if sim._candidate_is_better(rr, manual): manual=rr
    bb={**best,"q_h":0.20}
    assert manual is not None
    for fld in ("criterion","pred","R","GDF","ESS","eta","K"):
        assert abs(float(bb[fld])-float(manual[fld]))<1e-12,(fld,bb[fld],manual[fld])


def test_production_tuner_matches_full_exhaustive_on_multiple_targets():
    data,_,_=get_case(N=40,T=30,seed=20260509)
    ti=int(np.floor(.7*30))-1
    x1p,x2p,yp,tp,Th=sim.stack_prearrival(data["x1"],data["x2"],data["y"],ti)
    cfg=sim.MethodConfig(K_state=1)
    sig,_=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,cfg.noise_pair_fraction,cfg.min_noise_pairs)
    grid=sim.TuningGrid(q_h_grid=(0.10,0.20,0.40),eta_grid=(1e-6,0.05,1.0,8.0),
                        rank_count_grid=(1,2,3,5),rank_fraction_grid=(0.5,))
    for i in (0,3,7,13,21):
        xs=np.array([data["x1"][i,ti],data["x2"][i,ti]])
        prod=sim.tune_stalop_gdf_target(x1p,x2p,yp,tp,xs,Th,sig,grid,cfg,False)
        full,allc=sim.tune_stalop_gdf_target(x1p,x2p,yp,tp,xs,Th,sig,grid,cfg,True)
        assert len(allc)>0
        for fld in ("criterion","pred","R","GDF","ESS","q_h","eta","K"):
            assert abs(float(prod[fld])-float(full[fld]))<1e-12,(i,fld,prod[fld],full[fld])

def main():
    tests=[
        ("pre_response_histories",test_pre_response_histories),
        ("panel1_excluded_qh_recompute",test_panel1_excluded_and_qh_recompute),
        ("batched_state_solver_parity",test_batched_state_solver_parity),
        ("fast_gdf_reference_parity",test_fast_gdf_reference_parity),
        ("explicit_DH_gdf_parity",test_explicit_DH_gdf_parity),
        ("all_eta_k_batch_parity",test_all_eta_k_batch_parity),
        ("best_only_eta_k_block_parity",test_best_only_eta_k_block_parity),
        ("joint_argmin_exhaustive",test_joint_argmin_matches_exhaustive_reference),
        ("production_vs_full_tuner",test_production_tuner_matches_full_exhaustive_on_multiple_targets),
        ("baseline_pre_target",test_baseline_tuning_is_strictly_pre_target),
        ("time_local_vectorization_parity",test_vectorized_time_local_matches_scalar),
        ("knn_partial_sort_exact_parity",test_knn_partial_sort_exact_parity_with_ties),
        ("knn_kdtree_production_parity",test_knn_kdtree_production_parity_continuous),
        ("configuration_validation",test_configuration_validation),
        ("deterministic_seed_mapping",test_deterministic_seed_mapping),
        ("seed_csv_roundtrip_exact",test_seed_csv_roundtrip_exact),
        ("zero_final_ridge_hat_matrix",test_zero_final_ridge_matches_manuscript_hat_matrix),
        ("eta_near_uniform_limit",test_eta_lower_grid_is_near_uniform_limit),
        ("checkpoint_resume",test_checkpoint_resume_and_part_invariance),
        ("execution_order_parallel_invariance",test_execution_order_and_parallel_invariance),
    ]
    for name,fn in tests:
        fn(); print("PASS",name,flush=True)
    sig=sigma_sanity(); print("PASS sigma_sanity",sig,flush=True)
    comp=compatibility_sanity(); print("PASS compatibility_sanity",comp,flush=True)
    print("ALL_V3_ACCEPTANCE_TESTS_PASS",flush=True)


if __name__=="__main__":
    main()
