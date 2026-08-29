from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import simulation_core as sim
import experiment1_core as exp1

DGP = sim.DGPConfig(sigma_a=0.0, regime_mode="predictable_cycle")
METHOD = sim.MethodConfig(K_state=1)
GRID = sim.TuningGrid()


def assert_close(a, b, tol=1e-11, msg=""):
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(a)-float(b)) > tol:
        raise AssertionError(f"{msg}: {a} != {b}")


def test_main_parity():
    N, T = 100, 30
    seed = sim.deterministic_task_seed(2026081401, T, N, 0)
    base, _ = sim.run_one_replication(N, T, seed, 0, DGP, METHOD, GRID)
    ext, tuning = exp1.run_experiment1_replication(N, T, seed, 0, DGP, METHOD, GRID)
    for method in ["Global Linear", "Time-Local", "KNN-Covariate", "StaLoP"]:
        a = float(base.loc[base.method == method, "mspe"].iloc[0])
        b = float(ext.loc[ext.method == method, "mspe"].iloc[0])
        assert_close(a, b, 1e-12, f"main parity {method}")
    if len(tuning) != N:
        raise AssertionError("one StaLoP tuning row per target is required")
    if not tuning["compatible_retained_pct"].between(0, 100).all():
        raise AssertionError("compatibility precision outside [0,100]")
    oracle = ext.loc[ext.method == "Oracle-Compatible StaLoP"].iloc[0]
    assert_close(float(oracle["compatible_retained_pct"]), 100.0, 0.0, "oracle compatibility")
    if not np.isfinite(ext["mae"].to_numpy(float)).all():
        raise AssertionError("MAE must be finite")
    print("PASS main-method exact parity + MAE + compatibility")


def test_oracle_gdf_explicit():
    N, T = 100, 30
    seed = sim.deterministic_task_seed(2026081501, T, N, 0)
    data = sim.simulate_panel(N, T, DGP, np.random.default_rng(seed))
    tstar = int(np.floor(.7*T))-1
    x1p,x2p,yp,tp,Th = sim.stack_prearrival(data['x1'],data['x2'],data['y'],tstar)
    sigma2,_ = sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,METHOD.noise_pair_fraction,METHOD.min_noise_pairs)
    xs = np.array([0.0,0.0])
    static=sim.prepare_target_static(x1p,x2p,xs)
    h=float(np.quantile(static['dist'], .1)); kval=(static['dist']<=h).astype(float)
    A,b,w,y2=sim.kernel_panel_sufficient_stats(static['Phi'],yp,kval,tp,Th)
    regimes=data['S'][0]; target_reg=int(regimes[tstar]); eligible=np.arange(1,Th); keep=eligible[regimes[eligible]==target_reg]
    row=exp1._oracle_subset_candidate(A,b,w,y2,keep,sigma2,METHOD)

    raw=np.isin(tp-1,keep).astype(float)*kval
    D=raw/raw.sum()
    Phi=static['Phi']; M=Phi.T@(D[:,None]*Phi); H=Phi@np.linalg.solve(M,Phi.T*D[None,:])
    pred=float(np.array([1.,0.,0.])@np.linalg.solve(M,Phi.T@(D*yp)))
    R=float(np.sum(D*(yp-H@yp)**2)); G=float(np.trace(np.diag(D)@H)); crit=R+2*sigma2*G-sigma2
    assert_close(row['pred'],pred,1e-10,'oracle pred')
    assert_close(row['R'],R,1e-10,'oracle R')
    assert_close(row['GDF'],G,1e-10,'oracle GDF')
    assert_close(row['criterion'],crit,1e-10,'oracle criterion')
    print("PASS oracle-compatible explicit D/H/GDF parity")


def test_theory_outputs():
    N,T=100,30; seed=sim.deterministic_task_seed(2026081501,T,N,0)
    c,k,s=exp1.run_theory_diagnostic_replication(N,T,seed,0,((-0.5,-0.5),(0.,0.),(.5,.5)),DGP,METHOD,GRID)
    if c.empty or k.empty or s.empty:
        raise AssertionError("theory diagnostics must be nonempty")
    # Each selected row must equal the minimum criterion in its point-specific candidate set.
    for pid,row in s.groupby('point_id'):
        cand=c[c.point_id==pid]
        assert_close(float(row.iloc[0].criterion),float(cand.criterion.min()),1e-10,'selected min criterion')
    # With at least 8 independent replications there should be >=2 replicates in
    # typical regime strata, enabling a nondegenerate bias/variance smoke.
    parts=[]
    for rep in range(8):
        sd=sim.deterministic_task_seed(2026081501,T,N,rep)
        _,kk,_=exp1.run_theory_diagnostic_replication(N,T,sd,rep,((0.,0.),),DGP,METHOD,GRID)
        parts.append(kk)
    bv=exp1.summarize_k_bias_variance(pd.concat(parts,ignore_index=True))
    if bv.empty or float(bv.max_abs_identity_error.max())>1e-10:
        raise AssertionError("bias^2 + variance identity failed")
    print("PASS candidate selection + K bias/variance identity")


def test_checkpoint_resume_main():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); pref=str(p/'out'); cp=p/'cp'
        a=exp1.run_experiment1_main_parallel([(100,30)],1,2026081502,DGP,METHOD,GRID,n_jobs=1,save_prefix=pref,checkpoint_dir=cp,resume=True)
        b=exp1.run_experiment1_main_parallel([(100,30)],1,2026081502,DGP,METHOD,GRID,n_jobs=1,save_prefix=pref,checkpoint_dir=cp,resume=True)
        cols=['seed','replicate_id','N','T','method','mspe','mae']
        pd.testing.assert_frame_equal(a[0][cols].reset_index(drop=True), b[0][cols].reset_index(drop=True), check_exact=False, rtol=0.0, atol=1e-14)
    print("PASS Experiment-1 checkpoint/resume exactness")


if __name__ == '__main__':
    test_main_parity()
    test_oracle_gdf_explicit()
    test_theory_outputs()
    test_checkpoint_resume_main()
    print('ALL_EXPERIMENT1_ACCEPTANCE_TESTS_PASS')
