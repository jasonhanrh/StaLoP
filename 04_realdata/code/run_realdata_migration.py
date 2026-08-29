"""Final rolling real-data analysis for one origin state."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

from realdata_core import (
    RealDataConfig, aggregate_period_mass, build_side_panel, config_json, data_audit,
    estimate_gamma, fit_side_model, load_binned_edges, metrics, predict_side_methods,
    target_covariates, tune_knn_k, validate_config,
)


def sha256_file(p: str|Path) -> str:
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame,p: Path) -> None:
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=p.parent,prefix=p.name+".",suffix=".tmp"); os.close(fd)
    try: df.to_csv(tmp,index=False); os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def run(input_path: Path,outdir: Path,state_name: str,cfg: RealDataConfig) -> None:
    validate_config(cfg); outdir.mkdir(parents=True,exist_ok=True); t0=perf_counter()
    edge=load_binned_edges(input_path)
    audit_rows,audit_summary=data_audit(edge,cfg.target_bins)
    bins_all=np.array(sorted(edge.tbin.unique()),int)
    missing=sorted(set(cfg.target_bins)-set(bins_all))
    if missing: raise RuntimeError(f"target bins absent: {missing}")

    positive=edge.loc[edge.flow>0].groupby("tbin")["key"].apply(list).to_dict()
    active:set[str]=set(); last=-np.inf
    for b in bins_all[bins_all<min(cfg.target_bins)]: active.update(positive.get(int(b),[])); last=max(last,int(b))

    bin_rows=[]; edge_rows=[]; tuning_rows=[]; mass_rows=[]; similar_rows=[]; knn_rows=[]; candidate_rows=[]; runtime_rows=[]
    for tb in cfg.target_bins:
        bt=perf_counter()
        for b in bins_all[(bins_all>last)&(bins_all<tb)]: active.update(positive.get(int(b),[])); last=max(last,int(b))
        tr=edge[edge.tbin<tb].copy(); te0=edge[edge.tbin==tb].copy(); te=te0[te0.key.isin(active)].copy() if cfg.active_only else te0.copy()
        if len(te)<cfg.min_eval_edges: raise RuntimeError(f"target {tb} has only {len(te)} eval edges")
        gamma=estimate_gamma(tr)
        side_objs={}; side_preds={}; side_tuning={}; side_mass={}; side_time_diag={}; side_knn={}
        for side,target_ids in (("origin",te.FIPS5i.astype(str).unique()),("dest",te.FIPS5j.astype(str).unique())):
            panel=build_side_panel(tr,side); model=fit_side_model(panel,cfg,estimate_sigma2=True); model.side=side
            cov=target_covariates(te0,side)
            k,knndf=tune_knn_k(panel,cfg); knndf.insert(0,"side",side); knndf.insert(0,"tbin",tb); knn_rows.append(knndf)
            preds,tune,mass,tdiag,cands=predict_side_methods(model,cov,target_ids,cfg,k)
            tune.insert(0,"side",side); tune.insert(0,"tbin",tb); tuning_rows.append(tune)
            if not cands.empty: cands.insert(0,"side",side); cands.insert(0,"tbin",tb); candidate_rows.append(cands)
            pm=aggregate_period_mass(mass,len(target_ids),model.bins)
            side_objs[side]=model; side_preds[side]=preds; side_tuning[side]=tune; side_mass[side]=pm; side_time_diag[side]=tdiag; side_knn[side]=k

        merged=te[["tbin","FIPS5i","FIPS5j","key","y","logdist"]].copy()
        for method in ("global","time","knn","stalop"):
            a=side_preds["origin"][method].rename(columns={"id":"FIPS5i","value":"fo"})
            b=side_preds["dest"][method].rename(columns={"id":"FIPS5j","value":"fd"})
            mm=merged[["key"]].merge(te[["key","FIPS5i","FIPS5j"]],on="key",how="left").merge(a,on="FIPS5i",how="left").merge(b,on="FIPS5j",how="left")
            if mm[["fo","fd"]].isna().any().any(): raise RuntimeError(f"missing {method} side prediction")
            merged[f"pred_{method}"]=mm.fo.to_numpy(float)+mm.fd.to_numpy(float)+gamma*merged.logdist.to_numpy(float)
            merged[f"sqerr_{method}"]=(merged[f"pred_{method}"].to_numpy(float)-merged.y.to_numpy(float))**2
        y=merged.y.to_numpy(float); mets={m:metrics(y,merged[f"pred_{m}"].to_numpy(float)) for m in ("global","time","knn","stalop")}
        origin_t=side_tuning["origin"]; dest_t=side_tuning["dest"]
        bin_rows.append({"tbin":tb,"n_train":len(tr),"n_target_rows":len(te0),"n_eval":len(te),"gamma":gamma,
                         **{f"{m}_{k}":v for m,d in mets.items() for k,v in d.items()},
                         "sigma2_origin":side_objs["origin"].sigma2_hat,"sigma2_dest":side_objs["dest"].sigma2_hat,
                         "selected_K_origin_mean":float(origin_t.K.mean()),"selected_K_dest_mean":float(dest_t.K.mean()),
                         "selected_q_h_origin_mean":float(origin_t.q_h.mean()),"selected_q_h_dest_mean":float(dest_t.q_h.mean()),
                         "selected_eta_origin_mean":float(origin_t.eta.mean()),"selected_eta_dest_mean":float(dest_t.eta.mean()),
                         "knn_k_origin":side_knn["origin"],"knn_k_dest":side_knn["dest"]})
        edge_rows.append(merged.drop(columns=["logdist"]))

        periods=sorted(set(side_mass["origin"].index).union(side_mass["dest"].index))
        op=side_mass["origin"].reindex(periods,fill_value=0.0); dp=side_mass["dest"].reindex(periods,fill_value=0.0); cp=.5*(op+dp)
        for p in periods: mass_rows.append({"tbin":tb,"period":int(p),"origin_mass":float(op.loc[p]),"dest_mass":float(dp.loc[p]),"combined_mass":float(cp.loc[p])})
        top=cp.sort_values(ascending=False).head(cfg.top_periods)
        similar_rows.append({"tbin":tb,"top_periods_combined":",".join(str(int(x)) for x in top.index),"mass_top_periods_combined":float(top.sum()),
                             "top_periods_origin":",".join(str(int(x)) for x in op.sort_values(ascending=False).head(cfg.top_periods).index),
                             "top_periods_dest":",".join(str(int(x)) for x in dp.sort_values(ascending=False).head(cfg.top_periods).index)})
        bin_seconds=perf_counter()-bt
        runtime_rows.append({"tbin":tb,"seconds":bin_seconds})
        print(f"{state_name} {tb}: StaLoP={mets['stalop']['mspe']:.6f}, KNN={mets['knn']['mspe']:.6f}, Global={mets['global']['mspe']:.6f}, Time={mets['time']['mspe']:.6f}, n={len(te)}, sec={bin_seconds:.3f}",flush=True)
        # Release target-specific panel/statistic arrays before the next rolling origin.
        del side_objs, side_preds, side_tuning, side_mass, side_time_diag, side_knn, tr, te0, te

    bin_df=pd.DataFrame(bin_rows); edge_df=pd.concat(edge_rows,ignore_index=True); tune_df=pd.concat(tuning_rows,ignore_index=True); mass_df=pd.DataFrame(mass_rows); sim_df=pd.DataFrame(similar_rows); knn_df=pd.concat(knn_rows,ignore_index=True); run_df=pd.DataFrame(runtime_rows)
    overall=pd.DataFrame([{"origin_state":state_name,"n_bins":len(bin_df),"mean_n_eval":float(bin_df.n_eval.mean()),
                           **{f"{m}_mspe_mean":float(bin_df[f'{m}_mspe'].mean()) for m in ('global','time','knn','stalop')},
                           "stalop_win_vs_global":float((bin_df.stalop_mspe<bin_df.global_mspe).mean()),
                           "stalop_win_vs_time":float((bin_df.stalop_mspe<bin_df.time_mspe).mean()),
                           "stalop_win_vs_knn":float((bin_df.stalop_mspe<bin_df.knn_mspe).mean())}])
    atomic_csv(bin_df,outdir/"realdata_bin_summary.csv"); atomic_csv(edge_df,outdir/"realdata_edge_predictions.csv"); atomic_csv(tune_df,outdir/"realdata_stalop_selected_tuning.csv"); atomic_csv(mass_df,outdir/"realdata_stalop_period_mass.csv"); atomic_csv(sim_df,outdir/"realdata_stalop_similar_periods.csv"); atomic_csv(knn_df,outdir/"realdata_knn_validation.csv"); atomic_csv(overall,outdir/"realdata_overall_summary.csv"); atomic_csv(audit_rows,outdir/"realdata_target_data_audit.csv"); atomic_csv(run_df,outdir/"realdata_runtime.csv")
    if candidate_rows: atomic_csv(pd.concat(candidate_rows,ignore_index=True),outdir/"realdata_stalop_candidate_rows.csv")
    with open(outdir/"realdata_data_summary.json","w",encoding="utf8") as f: json.dump(audit_summary,f,indent=2)
    sources={p.name:sha256_file(p) for p in [Path(__file__).resolve(),Path(__file__).with_name("realdata_core.py")]}
    manifest={"origin_state":state_name,"input_path":str(input_path),"input_sha256":sha256_file(input_path),"config":config_json(cfg),"sources":sources,"wall_seconds":perf_counter()-t0,"data_summary":audit_summary}
    with open(outdir/"realdata_manifest.json","w",encoding="utf8") as f: json.dump(manifest,f,indent=2)
    print(f"DONE {state_name}: {outdir} wall={manifest['wall_seconds']:.3f}s")


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True,type=Path); p.add_argument("--output-dir",required=True,type=Path); p.add_argument("--state-name",required=True)
    p.add_argument("--save-candidates",action="store_true"); return p.parse_args()

if __name__=="__main__":
    a=parse_args(); cfg=RealDataConfig(save_candidate_rows=a.save_candidates); run(a.input,a.output_dir,a.state_name,cfg)
