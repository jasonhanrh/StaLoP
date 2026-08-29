"""Final Section-6 Application 3: target-relevant variable selection.

DGP: the FINAL Experiment-1 recurrent dynamic-panel design, augmented by two
independent N(0,1) noise variables.  This preserves the scientific DGP in the
JASA application while updating N=40 to the approved N=100 settings used by the
final simulation section.

For every one of the 16 subsets of {x1,x2,z1,z2}, the full StaLoP pipeline is
recomputed from scratch.  At every future target input Algorithm 2 jointly
selects (K,h,eta) by the final GDF-corrected MSPE criterion.  The variable subset
is then selected by the average minimized target-local criterion.  No rolling
validation or reuse of full-model states/rankings is used.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import simulation_core as sim
from applications_common import sha256_file
from dynamic_panel_multidim import VariableTuningGrid, tune_subset_all_targets

VARIABLES=("x1","x2","z1","z2")
TRUE_SUBSET=("x1","x2")
FULL_SUBSET=VARIABLES


def all_subsets() -> tuple[tuple[str,...],...]:
    out=[]
    for k in range(5): out.extend(tuple(s) for s in combinations(VARIABLES,k))
    return tuple(out)

CANDIDATE_SUBSETS=all_subsets()


def subset_label(s:tuple[str,...])->str:
    return "empty" if not s else "+".join(s)


@dataclass(frozen=True)
class VariableSelectionConfig:
    R:int=200
    settings:tuple[tuple[int,int],...]=((100,30),(200,30),(100,120),(200,120)) # (N,T)
    seed0:int=2026081603
    block_size:int=20
    theta_delta:float=1e-8
    w_ridge:float=1e-8
    max_condition:float=1e4
    # The controlled DGP has known observational variance sigma_u^2=1.
    # Using it here isolates variable-subset/Algorithm-2 behavior; Experiment 1
    # already validates the feasible variance estimator.
    sigma2_known:float=1.0
    grid:VariableTuningGrid=VariableTuningGrid()


def make_design(data:dict[str,np.ndarray],rng_noise:np.random.Generator)->np.ndarray:
    x1=np.asarray(data["x1"],float); x2=np.asarray(data["x2"],float)
    N,TT=x1.shape
    X=np.empty((N,TT,4),float)
    X[:,:,0]=x1; X[:,:,1]=x2
    X[:,:,2:]=rng_noise.normal(0.0,1.0,size=(N,TT,2))
    return X


def response_free_standardize(Xhist:np.ndarray,Xtarget:np.ndarray)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    center=Xhist.reshape(-1,Xhist.shape[-1]).mean(axis=0)
    scale=Xhist.reshape(-1,Xhist.shape[-1]).std(axis=0,ddof=1)
    if np.any((~np.isfinite(scale))|(scale<1e-12)):
        raise RuntimeError("degenerate variable-selection covariate scale")
    return (Xhist-center)/scale,(Xtarget-center)/scale,center,scale


def _seed_for(cfg:VariableSelectionConfig,N:int,T:int,rep:int)->int:
    # deterministic non-overlapping setting families
    setting_id=cfg.settings.index((N,T))
    return int(cfg.seed0 + setting_id*1_000_000 + rep)


def evaluate_subset(
    Xhist_full:np.ndarray,Xtarget_full:np.ndarray,yhist:np.ndarray,target_truth:np.ndarray,
    subset:tuple[str,...],sigma2_hat:float,cfg:VariableSelectionConfig,
)->dict:
    idx=[VARIABLES.index(v) for v in subset]
    Xh=Xhist_full[:,:,idx] if idx else np.empty((*Xhist_full.shape[:2],0),float)
    Xt=Xtarget_full[:,idx] if idx else np.empty((len(Xtarget_full),0),float)
    tic=perf_counter()
    pred,criterion,diag=tune_subset_all_targets(
        Xh,yhist,Xt,sigma2_hat,cfg.grid,block_size=cfg.block_size,
        theta_delta=cfg.theta_delta,w_ridge=cfg.w_ridge,max_condition=cfg.max_condition,
    )
    runtime=float(perf_counter()-tic)
    return {
        "subset":subset_label(subset),"subset_size":len(subset),
        "has_x1":int("x1" in subset),"has_x2":int("x2" in subset),
        "has_z1":int("z1" in subset),"has_z2":int("z2" in subset),
        "is_true_subset":int(subset==TRUE_SUBSET),"is_full_subset":int(subset==FULL_SUBSET),
        "includes_both_true":int(all(v in subset for v in TRUE_SUBSET)),
        "includes_any_noise":int(any(v in subset for v in ("z1","z2"))),
        "local_mspe_criterion":float(np.mean(criterion)),
        "testing_mspe":float(np.mean((pred-target_truth)**2)),
        "K_mean":float(np.mean(diag["K"])),"K_median":float(np.median(diag["K"])),
        "q_h_mean":float(np.mean(diag["q_h"])),"eta_mean":float(np.mean(diag["eta"])),
        "GDF_mean":float(np.mean(diag["GDF"])),"ESS_mean":float(np.mean(diag["ESS"])),
        "n_valid_candidate_mean":float(np.mean(diag["n_valid_candidates"])),
        "runtime_sec":runtime,
    }


def run_one_task(cfg:VariableSelectionConfig,N:int,T:int,rep:int)->tuple[pd.DataFrame,pd.DataFrame]:
    seed=_seed_for(cfg,N,T,rep)
    rng=np.random.default_rng(seed)
    # T historical panels + one future target panel, exactly matching the final
    # Experiment-1 recurrent predictable-cycle DGP.
    dgp=sim.DGPConfig(regime_mode="predictable_cycle",sigma_a=0.0,sigma_u=1.0)
    data=sim.simulate_panel(N=N,T=T+1,config=dgp,rng=rng)
    rng_noise=np.random.default_rng(seed+10_000_000)
    X=make_design(data,rng_noise)
    # transpose to [time,unit,variable]
    Xhist_raw=np.transpose(X[:,:T,:],(1,0,2)); Xtarget_raw=X[:,T,:]
    yhist=np.asarray(data["y"][:,:T],float).T
    truth=np.asarray(data["m_true"][:,T],float)
    Xhist,Xtarget,center,scale=response_free_standardize(Xhist_raw,Xtarget_raw)
    sigma2_hat=float(cfg.sigma2_known)
    n_pairs=0
    sigma_clipped=False

    rows=[]
    for subset in CANDIDATE_SUBSETS:
        row=evaluate_subset(Xhist,Xtarget,yhist,truth,subset,sigma2_hat,cfg)
        row.update({"N":N,"T":T,"rep_id":rep,"seed":seed,"sigma2_hat":sigma2_hat,
                    "sigma2_pairs":n_pairs,"sigma2_clipped":int(sigma_clipped)})
        rows.append(row)
    cand=pd.DataFrame(rows)
    sel=cand.sort_values(["local_mspe_criterion","subset_size","subset"],kind="mergesort").iloc[0]
    full=cand[cand.is_full_subset==1].iloc[0]
    summary=pd.DataFrame([{
        "N":N,"T":T,"rep_id":rep,"seed":seed,"n_candidate_models":len(CANDIDATE_SUBSETS),
        "selected_subset":sel.subset,"selects_true_subset":int(sel.is_true_subset),
        "selected_includes_both_true":int(sel.includes_both_true),"selected_includes_any_noise":int(sel.includes_any_noise),
        "selected_size":int(sel.subset_size),"selected_model_mspe":float(sel.testing_mspe),"full_model_mspe":float(full.testing_mspe),
        "selected_local_mspe_criterion":float(sel.local_mspe_criterion),"full_local_mspe_criterion":float(full.local_mspe_criterion),
        "selected_K_mean":float(sel.K_mean),"selected_q_h_mean":float(sel.q_h_mean),"selected_eta_mean":float(sel.eta_mean),
        "full_K_mean":float(full.K_mean),"full_q_h_mean":float(full.q_h_mean),"full_eta_mean":float(full.eta_mean),
        "sigma2_hat":sigma2_hat,"sigma2_pairs":n_pairs,"sigma2_clipped":int(sigma_clipped),
        "all_subsets_runtime_sec":float(cand.runtime_sec.sum()),
    }])
    return cand,summary


def summarize(rep:pd.DataFrame)->pd.DataFrame:
    def sd(x): return float(np.std(x,ddof=1))
    def mcse(x): return float(np.std(x,ddof=1)/np.sqrt(len(x)))
    return rep.groupby(["N","T"],as_index=False).agg(
        correct_input_selected=("selects_true_subset","mean"),
        correct_input_mcse=("selects_true_subset",mcse),
        both_true_selected=("selected_includes_both_true","mean"),
        any_noise_selected=("selected_includes_any_noise","mean"),
        selected_size_mean=("selected_size","mean"),
        selected_model_mspe=("selected_model_mspe","mean"),selected_model_mspe_sd=("selected_model_mspe",sd),
        full_model_mspe=("full_model_mspe","mean"),full_model_mspe_sd=("full_model_mspe",sd),
        selected_K_mean=("selected_K_mean","mean"),selected_q_h_mean=("selected_q_h_mean","mean"),selected_eta_mean=("selected_eta_mean","mean"),
        sigma2_mean=("sigma2_hat","mean"),runtime_mean_sec=("all_subsets_runtime_sec","mean"),n_rep=("rep_id","size"),
    )


def parse_settings(text:str)->tuple[tuple[int,int],...]:
    out=[]
    for part in text.split(","):
        n,t=part.split("x"); out.append((int(n),int(t)))
    return tuple(out)


def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--R",type=int,default=200)
    ap.add_argument("--seed0",type=int,default=VariableSelectionConfig().seed0)
    ap.add_argument("--n-jobs",type=int,default=6)
    ap.add_argument("--settings",type=parse_settings,default=VariableSelectionConfig().settings,
                    help="comma list NxT, e.g. 100x30,200x30,100x120,200x120")
    ap.add_argument("--block-size",type=int,default=20)
    ap.add_argument("--outdir",type=str,default="application3_variable_selection_R200")
    ap.add_argument("--formal",action="store_true")
    ap.add_argument("--save-candidates",action="store_true")
    args=ap.parse_args()
    R=200 if args.formal else int(args.R)
    cfg=VariableSelectionConfig(R=R,settings=tuple(args.settings),seed0=int(args.seed0),block_size=int(args.block_size))
    tasks=[(N,T,r) for N,T in cfg.settings for r in range(R)]
    n_jobs=max(1,min(int(args.n_jobs),len(tasks),mp.cpu_count() or 1))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    tic=perf_counter(); cands=[]; reps=[]
    if n_jobs==1:
        for done,(N,T,r) in enumerate(tasks,1):
            a,b=run_one_task(cfg,N,T,r); cands.append(a); reps.append(b)
            if done%5==0 or done==len(tasks): print(f"Completed {done}/{len(tasks)}",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            fut={ex.submit(run_one_task,cfg,N,T,r):(N,T,r) for N,T,r in tasks}
            for done,f in enumerate(as_completed(fut),1):
                a,b=f.result(); cands.append(a); reps.append(b)
                if done%5==0 or done==len(tasks): print(f"Completed {done}/{len(tasks)}",flush=True)
    cand=pd.concat(cands,ignore_index=True).sort_values(["N","T","rep_id","subset_size","subset"])
    rep=pd.concat(reps,ignore_index=True).sort_values(["N","T","rep_id"])
    agg=summarize(rep)
    if args.save_candidates: cand.to_csv(out/"application3_variable_selection_candidates.csv",index=False)
    rep.to_csv(out/"application3_variable_selection_rep_summary.csv",index=False)
    agg.to_csv(out/"application3_variable_selection_summary.csv",index=False)
    manifest={"application":"variable_selection","config":asdict(cfg),"R":R,"valid_tasks":int(len(rep)),
              "wall_seconds":float(perf_counter()-tic),"source_sha256":sha256_file(__file__),
              "simulation_core_sha256":sha256_file(Path(__file__).with_name("simulation_core.py"))}
    (out/"application3_variable_selection_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(agg.to_string(index=False)); print(f"wall_seconds={manifest['wall_seconds']:.3f}")


if __name__=="__main__":
    mp.freeze_support(); main()
