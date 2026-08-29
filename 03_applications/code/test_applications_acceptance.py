"""Acceptance suite for the final Section-6 application package."""
from __future__ import annotations
import numpy as np
import pandas as pd

from applications_common import PointPanelGrid,tune_point_panel_gdf,explicit_point_panel_candidate
import run_application1_sine as sine
import run_application2_ces as ces
import dynamic_panel_multidim as nd
import simulation_core as sim


def test_point_panel_gdf_explicit_parity():
    rng=np.random.default_rng(11); n=40; X=rng.normal(size=(n,2)); y=rng.normal(size=n)
    states=rng.normal(size=(n,3)); xs=np.array([.2,-.1]); zs=np.array([.1,.2,-.3]); kern=np.exp(-.2*np.sum((X-xs)**2,axis=1))
    grid=PointPanelGrid((.5,1.0),rank_count_grid=(8,15,25))
    best,rows=tune_point_panel_gdf(X,y,states,xs,zs,.7,grid,kernel_weight=kern,return_all=True)
    for row in rows:
        ref=explicit_point_panel_candidate(X,y,states,xs,zs,.7,row['eta'],row['K'],kernel_weight=kern)
        for k in ('pred','R','GDF','criterion'):
            assert np.allclose(row[k],ref[k],rtol=2e-11,atol=2e-12),(k,row[k],ref[k])
    m=min(r['criterion'] for r in rows)
    assert best['criterion'] <= m+1e-12


def test_sine_dgp_and_no_future_leakage():
    cfg=sine.SineConfig(); rng=np.random.default_rng(123)
    t=np.arange(cfg.t_start,cfg.t_end+cfg.dt/2,cfg.dt); truth=sine.f_sine(t,cfg.amp); y=truth+rng.normal(0,cfg.noise_sd,len(t))
    assert np.allclose(truth,cfg.amp*np.sin(2*np.pi*t))
    panel=sine.build_sequence_panels(t,y,cfg.K_state)
    a=30; sidx=int(panel['state_idx'][a])
    # The observed one-sided input X_s may use y_s but never y_{s+1}.
    y2=y.copy(); y2[sidx+1]+=999.0
    panel2=sine.build_sequence_panels(t,y2,cfg.K_state)
    assert np.array_equal(panel['X'][a],panel2['X'][a]),"future one-step response leaked into X_s"
    # Final Algorithm-1 predictive state for response panel j excludes y_next[j].
    X=panel['X'][:a]; yy=panel['y_next'][:a]; center=X.mean(0);scale=X.std(0,ddof=1);Xs=(X-center)/scale;xs=(panel['X'][a]-center)/scale
    j=15; h=.8
    z1=sine._predictive_state_window(Xs,yy,j,xs,h,cfg.predictive_history,cfg.max_condition)
    yy2=yy.copy(); yy2[j]+=999.0
    z2=sine._predictive_state_window(Xs,yy2,j,xs,h,cfg.predictive_history,cfg.max_condition)
    assert np.array_equal(z1,z2),"panel response leaked into its predictive state"

def test_sine_reproducibility_and_finiteness():
    cfg=sine.SineConfig(R=1)
    a,b=sine.run_one_rep(cfg,0); c,d=sine.run_one_rep(cfg,0)
    pd.testing.assert_frame_equal(a.drop(columns=['runtime_sec']).reset_index(drop=True),c.drop(columns=['runtime_sec']).reset_index(drop=True),check_exact=True)
    pd.testing.assert_frame_equal(b.reset_index(drop=True),d.reset_index(drop=True),check_exact=True)
    assert np.isfinite(a[['se','pred','target']].to_numpy()).all()
    assert np.isfinite(b[['K','eta','q_h','criterion','GDF','ESS','sigma2_hat']].to_numpy()).all()


def test_ces_dgp_and_strict_states():
    cfg=ces.CESConfig(n_eval=16); cfg.validate()
    d=ces.simulate_real(cfg,1,.2)
    # Exact legacy DGP identity: positive inputs and multiplicative response.
    assert (d[['K','L','Y']].to_numpy()>0).all()
    theta=ces.calibrate_theta(d,cfg); X,y=ces.simulate_simulator(d,theta,cfg,1); Xs,center,scale=ces.standardize(X)
    xstar=np.array([0.0,0.0])
    Z,zs,eligible,k,W,h=ces.sequential_states_one_target(Xs,y,xstar,cfg.state_window,cfg.h_quantile,cfg.max_condition,cfg.w_ridge)
    j=int(eligible[10]); row=10
    y2=y.copy(); y2[j]+=1000.0
    Z2,zs2,eligible2,*_=ces.sequential_states_one_target(Xs,y2,xstar,cfg.state_window,cfg.h_quantile,cfg.max_condition,cfg.w_ridge)
    assert np.array_equal(eligible,eligible2)
    assert np.allclose(Z[row],Z2[row],rtol=0,atol=1e-12),"CES response leaked into its own state"
    assert np.isfinite(Z).all() and np.isfinite(zs).all() and np.isfinite(W).all() and h>0


def test_ces_small_prediction_finite():
    cfg=ces.CESConfig(n_eval=16)
    raw,tune=ces.run_one_rep(cfg,.1,1)
    assert len(raw)==3 and len(tune)==16
    assert np.isfinite(raw[['mspe','rmse']].to_numpy()).all()
    assert np.isfinite(tune[['K','eta','h','criterion','GDF','ESS']].to_numpy()).all()
    assert np.all(tune['sigma2_hat'].to_numpy()==0.0)


def test_nd_scalar_parity_with_frozen_reference():
    rng=np.random.default_rng(44); T=8; N=45
    X=rng.normal(size=(T,N,2)); y=rng.normal(size=(T,N)); xstar=np.array([.15,-.2])
    q=.4; eta=.75; K=3; sigma=.9
    ref_nd=nd.explicit_candidate_one_target(X,y,xstar,q,eta,K,sigma,max_condition=1e8)
    # Same panel ordering/algebra through the frozen two-dimensional reference.
    x1=X[:,:,0].reshape(-1); x2=X[:,:,1].reshape(-1); yy=y.reshape(-1); times=np.repeat(np.arange(1,T+1),N)
    comp=sim.build_target_components(x1,x2,yy,times,xstar,T,q,sim.MethodConfig(K_state=1,max_design_condition=1e8))
    ref=sim.evaluate_gdf_candidate_reference(comp,times,eta,K,sigma,sim.MethodConfig(K_state=1,max_design_condition=1e8))
    for a,b in [('pred','pred'),('criterion','criterion'),('R','R'),('GDF','GDF')]:
        assert np.allclose(ref_nd[a],ref[b],rtol=2e-10,atol=2e-11),(a,ref_nd[a],ref[b])


def test_nd_batch_matches_scalar_and_empty_subset():
    rng=np.random.default_rng(55);T=8;N=45;X=rng.normal(size=(T,N,2));y=rng.normal(size=(T,N));Xt=rng.normal(size=(3,2))
    grid=nd.VariableTuningGrid(q_h_grid=(.4,),eta_grid=(.75,),rank_fraction_grid=(.5,))
    pred,crit,diag=nd.tune_subset_all_targets(X,y,Xt,.9,grid,block_size=2,max_condition=1e8)
    K=int(np.floor(.5*(T-1)))
    for i in range(3):
        ref=nd.explicit_candidate_one_target(X,y,Xt[i],.4,.75,K,.9,max_condition=1e8)
        assert np.allclose(pred[i],ref['pred'],rtol=2e-10,atol=2e-11)
        assert np.allclose(crit[i],ref['criterion'],rtol=2e-10,atol=2e-11)
    Xe=np.empty((T,N,0)); Xte=np.empty((3,0))
    pe,ce,de=nd.tune_subset_all_targets(Xe,y,Xte,.9,grid,block_size=2,max_condition=1e8)
    assert np.isfinite(pe).all() and np.isfinite(ce).all()


def test_final_dynamic_dgp_contract():
    rng=np.random.default_rng(777)
    d=sim.simulate_panel(30,13,sim.DGPConfig(regime_mode='predictable_cycle',sigma_a=0,sigma_u=1),rng)
    S=d['S'][0]
    assert np.all(d['S']==S[None,:])
    assert np.all(S[1:]==1+(S[:-1]%4))
    assert np.all(d['a']==0)


def main():
    tests=[v for k,v in globals().copy().items() if k.startswith('test_') and callable(v)]
    for f in tests:
        f(); print(f"PASS {f.__name__}")
    print("ALL_APPLICATIONS_ACCEPTANCE_TESTS_PASS")

if __name__=='__main__': main()
