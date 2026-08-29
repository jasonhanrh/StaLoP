"""Algebraic/numerical acceptance tests for FINAL Experiment 2."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
import subprocess, sys, tempfile, json
from pathlib import Path
from experiment2_core import (DEFAULT_CONFIG,analytical_population_balance,bandwidth_for_N,
    explicit_joint_sandwich_debug,finite_difference_alpha_derivative,infer_one_point,run_one_replication,
    simulate_replication)

def assert_close(a,b,tol,msg):
    if not np.allclose(a,b,rtol=tol,atol=tol): raise AssertionError(f"{msg}: {a} vs {b}")

def test_design_and_no_leakage():
    c=DEFAULT_CONFIG; c.validate(); assert c.target_panel_1based==121 and c.target_history_panel_1based==120
    assert all(h<t for h,t in zip(c.history_panels_1based,c.eligible_panels_1based))
    s=simulate_replication(24,0,999,c); assert c.target_panel_1based not in s["Y"]
    assert set(s["Y"].keys())==(set(c.history_panels_1based)|{c.target_history_panel_1based}|set(c.eligible_panels_1based))

def test_population_balance():
    b=analytical_population_balance(DEFAULT_CONFIG); assert abs(b["weighted_state_offset"])<2e-15
    assert abs(b["weighted_next_panel_offset"])<2e-15 and b["max_pair_weight_asymmetry"]<2e-15

def test_bandwidth_rule():
    h=[bandwidth_for_N(n) for n in (200,500,1000)]; assert h[0]>h[1]>h[2]>0
    assert_close(h[0],1.5*200**(-1/6),1e-14,"bandwidth")

def test_alpha_derivative_fd():
    c=DEFAULT_CONFIG
    states=np.array([[-.60,.12,-.08],[-.30,.10,-.05],[0,.11,-.07],[.30,.09,-.06],[.60,.13,-.04],[.02,.10,-.06]])
    W=np.array([[1,.04,-.03],[.04,.8,.05],[-.03,.05,1.1]])
    d=finite_difference_alpha_derivative(states,W,c,eps=2e-6)
    if d["max_abs_error"]>2e-7 or d["relative_fro_error"]>2e-6: raise AssertionError(f"alpha Jacobian mismatch {d}")

def test_fast_vs_explicit_sandwich():
    c=DEFAULT_CONFIG; s=simulate_replication(36,2,123456,c); x=np.array([.10,0.]); h=bandwidth_for_N(36,c)
    d=explicit_joint_sandwich_debug(s["X"],s["Y"],x,h,c)
    assert_close(d["var_full_fast"],d["var_full_explicit"],3e-10,"full sandwich parity")
    assert_close(d["var_naive_fast"],d["var_naive_explicit"],3e-10,"naive sandwich parity")
    if d["score_root_max"]>2e-10: raise AssertionError(f"score root too large {d}")

def test_root_and_adaptivity():
    c=DEFAULT_CONFIG; s=simulate_replication(60,3,654321,c)
    a=infer_one_point(s["X"],s["Y"],np.array([0.,0.]),bandwidth_for_N(60,c),c)
    if max(a["root_z_max"],a["root_W_max"],a["root_c_max"])>2e-10: raise AssertionError("root equation failed")
    for k in ("J_alpha_gamma_norm","Acg_norm","indirect_row_norm"):
        if a[k]<=1e-8: raise AssertionError(f"adaptive term unexpectedly vanished: {k}={a[k]}")

def test_finite_outputs():
    r=run_one_replication(40,4,777777,DEFAULT_CONFIG)
    for k in ("mhat","se_full","se_naive","z_full","z_naive","Sigma_full","Sigma_naive"):
        if not np.all(np.isfinite(r[k])): raise AssertionError(f"nonfinite {k}")
    if np.any(r["se_full"]<=0) or np.any(r["se_naive"]<=0): raise AssertionError("nonpositive SE")
    if r["root_score_max"]>2e-10: raise AssertionError("root gate failed")

def test_reproducibility():
    a=run_one_replication(28,5,888888,DEFAULT_CONFIG); b=run_one_replication(28,5,888888,DEFAULT_CONFIG)
    for k in ("mhat","se_full","se_naive","Sigma_full","Sigma_naive","crit_full","crit_naive","maxz_full","maxz_naive"):
        assert_close(a[k],b[k],0.0,f"reproducibility {k}")

def test_checkpoint_resume_runner():
    here=Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as td:
        out=Path(td)/"resume_test"
        cmd=[sys.executable,str(here/"run_experiment2.py"),"--R","1","--seed0","919191","--n-jobs","1","--N","24","--outdir",str(out)]
        subprocess.run(cmd,cwd=here,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        cp=out/"checkpoints"/"N24_rep0000.npz"
        if not cp.exists(): raise AssertionError("checkpoint was not created")
        before=cp.read_bytes()
        subprocess.run(cmd,cwd=here,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        after=cp.read_bytes()
        if before!=after: raise AssertionError("valid checkpoint was rewritten on resume")
        m=json.loads((out/"experiment2_manifest.json").read_text())
        if m["reused_checkpoints"]!=1 or m["new_tasks_completed"]!=0:
            raise AssertionError(f"resume counters incorrect: {m['reused_checkpoints']}, {m['new_tasks_completed']}")

def main():
    tests=[test_design_and_no_leakage,test_population_balance,test_bandwidth_rule,test_alpha_derivative_fd,
           test_fast_vs_explicit_sandwich,test_root_and_adaptivity,test_finite_outputs,test_reproducibility,test_checkpoint_resume_runner]
    for fn in tests: fn(); print("PASS",fn.__name__)
    print("ALL_EXPERIMENT2_ACCEPTANCE_TESTS_PASS")
if __name__=="__main__": main()
