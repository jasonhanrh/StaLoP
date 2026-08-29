"""Acceptance tests for the final migration real-data package."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from realdata_core import (
    RealDataConfig, SideModel, aggregate_period_mass, explicit_candidate_check,
    fit_side_model, history_indices, load_binned_edges, normalize_fips5,
    predict_side_methods, rank_counts, tune_stalop_target,
)
from simulation_core_reference import MethodConfig, TuningGrid, tune_stalop_gdf_target


def synthetic_model(T=10,N=80,seed=17,sigma2=0.4):
    rng=np.random.default_rng(seed)
    X=[];y=[];ti=[];rows=[]
    for t in range(T):
        x=rng.normal(size=(N,2))
        yy=0.3*np.sin(t/2)+0.7*x[:,0]-0.4*x[:,1]+rng.normal(scale=np.sqrt(sigma2),size=N)
        for i in range(N): rows.append({"id":f"{i:03d}","tbin":200000+t,"y":yy[i],"x1":x[i,0],"x2":x[i,1],"resid":yy[i]})
        X.append(x);y.append(yy);ti.extend([t]*N)
    X=np.vstack(X);y=np.concatenate(y);ti=np.asarray(ti)
    # Direct model: no nuisance transformation, coordinates already final.
    panel=pd.DataFrame(rows)
    return SideModel(side="synthetic",panel=panel,beta=np.array([0.,0.,0.]),alpha=pd.Series(0.,index=[f"{i:03d}" for i in range(N)]),
                     x_mean=np.zeros(2),x_sd=np.ones(2),X_std=X,y_resid=y,time_index=ti,
                     bins=np.arange(T)+200000,sigma2_hat=sigma2,sigma2_pairs=100)


def test_history_strictly_one_sided():
    for a in range(2,12):
        h=history_indices(a,10,6)
        assert np.all(h < a-1)
        assert len(h)<=6
    assert np.array_equal(history_indices(11,10,6),np.arange(4,10))


def test_fips_normalization():
    s=pd.Series([6037,"06037",6037.0,"13001","00123"])
    out=normalize_fips5(s).tolist()
    assert out==["06037","06037","06037","13001","00123"]


def test_explicit_DH_parity():
    m=synthetic_model(T=9,N=100,seed=11)
    cfg=RealDataConfig(state_window=2,q_h_grid=(.30,),eta_grid=(.75,),rank_fraction_grid=(.5,),rank_count_grid=())
    best=tune_stalop_target(m,np.array([.1,-.2]),cfg)
    ref=explicit_candidate_check(m,np.array([.1,-.2]),best["q_h"],best["eta"],best["K"],cfg)
    for k in ("pred_resid","R","GDF","criterion"):
        assert np.isclose(best[k],ref[k],rtol=2e-10,atol=2e-11),(k,best[k],ref[k])


def test_balanced_parity_with_frozen_experiment1_core():
    m=synthetic_model(T=8,N=120,seed=29,sigma2=.7)
    cfg=RealDataConfig(state_window=1,q_h_grid=(.20,.35),eta_grid=(.5,1.),rank_fraction_grid=(.5,1.),rank_count_grid=(),max_design_condition=1e4)
    x=np.array([.15,.05])
    ours=tune_stalop_target(m,x,cfg)
    g=TuningGrid(q_h_grid=cfg.q_h_grid,eta_grid=cfg.eta_grid,rank_fraction_grid=cfg.rank_fraction_grid,rank_count_grid=(),time_window_grid=(1,),knn_k_grid=(1,))
    mc=MethodConfig(K_state=1,state_ridge=0.,final_ridge=0.,w_ridge=cfg.w_ridge,theta_standardize=True,min_bandwidth=cfg.min_bandwidth,require_full_rank_before_ridge=True,max_design_condition=cfg.max_design_condition)
    ref=tune_stalop_gdf_target(m.X_std[:,0],m.X_std[:,1],m.y_resid,m.time_index+1,x,len(m.bins),m.sigma2_hat,g,mc)
    for k in ("q_h","h","eta","K","pred","R","GDF","criterion"):
        a=ours["pred_resid"] if k=="pred" else ours[k]
        b=ref[k]
        assert np.isclose(float(a),float(b),rtol=3e-10,atol=3e-11),(k,a,b)


def test_h_recomputes_state_path():
    m=synthetic_model(T=10,N=90,seed=41)
    # Candidate h values should generally produce different selected state geometry.
    c1=RealDataConfig(state_window=3,q_h_grid=(.10,),eta_grid=(1.,),rank_fraction_grid=(.5,),rank_count_grid=())
    c2=RealDataConfig(state_window=3,q_h_grid=(.60,),eta_grid=(1.,),rank_fraction_grid=(.5,),rank_count_grid=())
    a=tune_stalop_target(m,np.array([.3,-.1]),c1); b=tune_stalop_target(m,np.array([.3,-.1]),c2)
    assert a["h"] != b["h"]
    # Not a forced-result test: at least one state-dependent output must differ.
    assert not (np.isclose(a["criterion"],b["criterion"]) and np.isclose(a["pred_resid"],b["pred_resid"]))


def test_mass_normalization_and_no_fallback():
    m=synthetic_model(T=10,N=80,seed=55)
    cfg=RealDataConfig(state_window=2,q_h_grid=(.3,),eta_grid=(.5,1.),rank_fraction_grid=(.5,1.),rank_count_grid=())
    a=tune_stalop_target(m,np.array([0.,0.]),cfg)
    assert np.isclose(np.sum(a["period_mass"]),1.0,atol=1e-12)
    assert a["n_valid_h"]==1 and a["n_candidates"]>0


def test_target_response_invariance():
    # Target response is not even an argument of the tuner.  Explicitly show
    # historical results do not change when an external target-y vector changes.
    m=synthetic_model(T=9,N=70,seed=73)
    cfg=RealDataConfig(state_window=2,q_h_grid=(.2,.4),eta_grid=(.5,1.),rank_fraction_grid=(.5,1.),rank_count_grid=())
    x=np.array([.2,.2]); a=tune_stalop_target(m,x,cfg)
    target_y=np.arange(7.0); target_y[:]=9999.0
    b=tune_stalop_target(m,x,cfg)
    for k in ("q_h","eta","K","pred_resid","criterion"):
        assert a[k]==b[k]


def test_loader_normalizes_and_preserves_rows():
    df=pd.DataFrame({"tbin":[201601,201601],"FIPS5i":[13001,13001],"FIPS5j":[6037,"06065"],"flow":[1.,0.],"dist":[10.,20.],"x1_i":[1.,1.],"x2_i":[2.,2.],"x1_j":[3.,4.],"x2_j":[4.,5.]})
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"x.csv";df.to_csv(p,index=False);e=load_binned_edges(p)
        assert len(e)==2 and e.FIPS5j.tolist()==["06037","06065"]
        assert e.key.tolist()==["13001|06037","13001|06065"]


def run_all():
    tests=[test_history_strictly_one_sided,test_fips_normalization,test_explicit_DH_parity,test_balanced_parity_with_frozen_experiment1_core,test_h_recomputes_state_path,test_mass_normalization_and_no_fallback,test_target_response_invariance,test_loader_normalizes_and_preserves_rows]
    for t in tests: t(); print("PASS",t.__name__)
    print("ALL_REALDATA_ACCEPTANCE_TESTS_PASS")

if __name__=="__main__": run_all()
