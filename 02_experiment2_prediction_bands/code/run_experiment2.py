"""Parallel/checkpointed runner for the FINAL Experiment 2."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v,"1")

import argparse, concurrent.futures as cf, csv, hashlib, json, math, tempfile, time
from pathlib import Path
import numpy as np
from experiment2_core import (
    CODE_VERSION, DEFAULT_CONFIG, analytical_population_balance,
    fixed_gaussian_draws, run_one_replication, target_grid,
    bandwidth_for_N,
)

FORMAL_R=500
FORMAL_SEED0=20260816
FORMAL_NS=(200,500,1000)


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()


def checkpoint_path(outdir,N,rep):
    return outdir/"checkpoints"/f"N{N}_rep{rep:04d}.npz"


def checkpoint_valid(path,N,rep,seed0):
    if not path.exists(): return False
    try:
        with np.load(path,allow_pickle=False) as z:
            return (str(z["version"].item())==CODE_VERSION and int(z["N"].item())==N and
                    int(z["rep"].item())==rep and int(z["seed0"].item())==seed0 and
                    np.all(np.isfinite(z["mhat"])) and np.all(np.isfinite(z["se_full"])) and
                    np.all(np.isfinite(z["se_naive"])))
    except Exception:
        return False


def save_checkpoint_atomic(path,result):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmpname=tempfile.mkstemp(prefix=path.stem+"_",suffix=".npz",dir=path.parent); os.close(fd)
    tmp=Path(tmpname)
    try:
        np.savez_compressed(tmp,**result); os.replace(tmp,path)
    finally:
        if tmp.exists(): tmp.unlink()


_WORKER_DRAWS = None

def worker(args):
    global _WORKER_DRAWS
    N,rep,seed0=args
    if _WORKER_DRAWS is None:
        _WORKER_DRAWS = fixed_gaussian_draws(DEFAULT_CONFIG)
    return run_one_replication(N,rep,seed0,DEFAULT_CONFIG,_WORKER_DRAWS)


def load_checkpoint(path):
    out={}
    with np.load(path,allow_pickle=False) as z:
        for k in z.files:
            a=z[k]; out[k]=a.item() if a.ndim==0 else a.copy()
    return out


def quant(x,p): return float(np.quantile(np.asarray(x,dtype=float),p,method="linear"))


def aggregate_results(results,outdir,cfg=DEFAULT_CONFIG):
    outdir.mkdir(parents=True,exist_ok=True)
    grid=target_grid(cfg); zcrit=1.6448536269514722
    rep_fields=["N","T","rep","seed0","h","runtime_seconds","tracked_peak_array_bytes","crit_full","crit_naive",
                "maxz_full","maxz_naive","simcover_full","simcover_naive","avgwidth_full","avgwidth_naive",
                "center_mhat","center_truth","center_se_full","center_se_naive","center_z_full","center_z_naive",
                "center_cover_full","center_cover_naive","avg_point_cover_full","avg_point_cover_naive",
                "mean_se_ratio_full_over_naive","condition_W_max","condition_state_max","condition_final_max",
                "root_score_max","Acg_norm_mean","indirect_row_norm_mean","J_alpha_gamma_norm_mean",
                "min_eig_corr_full_before_repair","min_eig_corr_naive_before_repair"]
    with (outdir/"experiment2_replication_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rep_fields); w.writeheader()
        for r in sorted(results,key=lambda x:(int(x["N"]),int(x["rep"]))): w.writerow({k:r[k] for k in rep_fields})

    point_rows=[]; sim_rows=[]; runtime_rows=[]
    for N in sorted(set(int(r["N"]) for r in results)):
        rs=[r for r in results if int(r["N"])==N]; R=len(rs); center=cfg.grid_size//2
        for method in ("full","naive"):
            Z=np.stack([r[f"z_{method}"] for r in rs]); C=np.stack([r[f"point_cover_{method}"] for r in rs]); S=np.stack([r[f"se_{method}"] for r in rs])
            for scope,zv,cv,sv in (("center_x0",Z[:,center],C[:,center],S[:,center]),("pooled_grid",Z.ravel(),C.ravel(),S.ravel())):
                point_rows.append({"N":N,"R":R,"method":method,"scope":scope,"z_mean":float(np.mean(zv)),
                                   "z_sd":float(np.std(zv,ddof=1)) if np.asarray(zv).size>1 else 0.0,"z_q025":quant(zv,.025),"z_q50":quant(zv,.5),
                                   "z_q975":quant(zv,.975),"coverage_90":float(np.mean(cv)),"mean_se":float(np.mean(sv)),
                                   "mean_pointwise_width_90":float(np.mean(2*zcrit*sv))})
        for method in ("full","naive"):
            sc=np.array([r[f"simcover_{method}"] for r in rs],float); widths=np.array([r[f"avgwidth_{method}"] for r in rs],float)
            mz=np.array([r[f"maxz_{method}"] for r in rs],float); cr=np.array([r[f"crit_{method}"] for r in rs],float)
            sim_rows.append({"N":N,"R":R,"method":method,"simultaneous_coverage_90":float(sc.mean()),
                             "simultaneous_coverage_mcse":float(math.sqrt(sc.mean()*(1-sc.mean())/R)),
                             "avg_band_width":float(widths.mean()),"empirical_q90_max_abs_Z":quant(mz,.9),
                             "mean_gaussian_q90":float(cr.mean()),"sd_gaussian_q90":float(cr.std(ddof=1)) if R>1 else 0.0})
        rt=np.array([r["runtime_seconds"] for r in rs],float); mem=np.array([r["tracked_peak_array_bytes"] for r in rs],float)
        runtime_rows.append({"N":N,"R":R,"runtime_mean_seconds":float(rt.mean()),"runtime_sd_seconds":float(rt.std(ddof=1)) if R>1 else 0.0,
                             "runtime_mcse_seconds":float(rt.std(ddof=1)/math.sqrt(R)) if R>1 else 0.0,
                             "tracked_peak_arrays_mean_MB":float(mem.mean()/1024**2),"tracked_peak_arrays_max_MB":float(mem.max()/1024**2)})
    def write_rows(name,rows):
        with (outdir/name).open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_rows("experiment2_pointwise_summary.csv",point_rows); write_rows("experiment2_simultaneous_summary.csv",sim_rows); write_rows("experiment2_runtime_summary.csv",runtime_rows)

    ordered=sorted(results,key=lambda x:(int(x["N"]),int(x["rep"])))
    np.savez_compressed(outdir/"experiment2_grid_results.npz",N=np.array([r["N"] for r in ordered],np.int32),
                        rep=np.array([r["rep"] for r in ordered],np.int32),grid=grid,
                        truth=np.stack([r["truth"] for r in ordered]),mhat=np.stack([r["mhat"] for r in ordered]),
                        se_full=np.stack([r["se_full"] for r in ordered]),se_naive=np.stack([r["se_naive"] for r in ordered]),
                        z_full=np.stack([r["z_full"] for r in ordered]),z_naive=np.stack([r["z_naive"] for r in ordered]),
                        Sigma_full=np.stack([r["Sigma_full"] for r in ordered]),Sigma_naive=np.stack([r["Sigma_naive"] for r in ordered]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--R",type=int,default=30); ap.add_argument("--seed0",type=int,default=2026081601)
    ap.add_argument("--N",nargs="+",type=int,default=[200,500,1000]); ap.add_argument("--n-jobs",type=int,default=1)
    ap.add_argument("--outdir",type=str,default="experiment2_pilot_R30"); ap.add_argument("--formal",action="store_true")
    args=ap.parse_args()
    if args.formal: R,seed0,Ns=FORMAL_R,FORMAL_SEED0,list(FORMAL_NS)
    else: R,seed0,Ns=args.R,args.seed0,args.N
    if R<=0 or args.n_jobs<=0 or any(N<=0 for N in Ns): raise ValueError("R, n-jobs and N must be positive")
    cfg=DEFAULT_CONFIG; cfg.validate(); outdir=Path(args.outdir); (outdir/"checkpoints").mkdir(parents=True,exist_ok=True)
    tasks=[]; reused=0
    for N in Ns:
        for rep in range(R):
            p=checkpoint_path(outdir,N,rep)
            if checkpoint_valid(p,N,rep,seed0): reused+=1
            else: tasks.append((N,rep,seed0))
    t0=time.perf_counter(); completed=0; failed=[]
    if args.n_jobs==1:
        for task in tasks:
            try:
                ans=worker(task); save_checkpoint_atomic(checkpoint_path(outdir,task[0],task[1]),ans); completed+=1
                print(f"[{completed}/{len(tasks)}] N={task[0]} rep={task[1]}",flush=True)
            except Exception as e:
                failed.append((task,repr(e))); print(f"FAILED {task}: {e}",flush=True)
    else:
        with cf.ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            futures={ex.submit(worker,t):t for t in tasks}
            for fut in cf.as_completed(futures):
                task=futures[fut]
                try:
                    ans=fut.result(); save_checkpoint_atomic(checkpoint_path(outdir,task[0],task[1]),ans); completed+=1
                    print(f"[{completed}/{len(tasks)}] N={task[0]} rep={task[1]}",flush=True)
                except Exception as e:
                    failed.append((task,repr(e))); print(f"FAILED {task}: {e}",flush=True)
    elapsed=time.perf_counter()-t0
    results=[]; hashes={}
    for N in Ns:
        for rep in range(R):
            p=checkpoint_path(outdir,N,rep)
            if not checkpoint_valid(p,N,rep,seed0): failed.append(((N,rep,seed0),"missing_or_invalid_checkpoint")); continue
            results.append(load_checkpoint(p)); hashes[p.name]=sha256_file(p)
    if failed:
        fp=outdir/"FAILED_TASKS.json"; fp.write_text(json.dumps(failed,indent=2),encoding="utf-8"); raise RuntimeError(f"{len(failed)} tasks failed; see {fp}")
    aggregate_results(results,outdir,cfg)
    here=Path(__file__).resolve().parent
    src={p.name:sha256_file(p) for p in [here/"experiment2_core.py",here/"run_experiment2.py",here/"test_experiment2_acceptance.py"] if p.exists()}
    manifest={"code_version":CODE_VERSION,"formal":bool(args.formal),"R":R,"N":Ns,"seed0":seed0,"n_jobs":args.n_jobs,
              "total_tasks":len(Ns)*R,"new_tasks_completed":completed,"reused_checkpoints":reused,"elapsed_wall_seconds":elapsed,
              "config":cfg.__dict__,"bandwidths":{str(N):bandwidth_for_N(N,cfg) for N in Ns},
              "population_balance":analytical_population_balance(cfg),
              "bias_correction":"zero_by_exact_linear_and_bridge_balanced_DGP_design",
              "tuning":"NONE; prespecified fixed eta, histories, eligible panels, and h_N rule",
              "full_covariance":"joint z/W -> theta -> standardized discrepancy -> alpha -> c influence",
              "naive_covariance":"same point estimator; realized alpha treated as fixed",
              "source_sha256":src,"checkpoint_sha256":hashes}
    (outdir/"experiment2_manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
    print(f"COMPLETE: {len(results)}/{len(Ns)*R} valid tasks"); print(f"Elapsed wall time: {elapsed:.3f} s"); print(f"Output: {outdir.resolve()}")

if __name__=="__main__": main()
