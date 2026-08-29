"""Final Section-6 Application 1: rolling sequence prediction.

The sine DGP is unchanged from the legacy JASA application:
    f(t)=0.75 sin(2*pi*t), y(t)=f(t)+eps, eps~N(0,0.1^2),
    t=0,0.01,...,1.60.

Exact final-method specialization
---------------------------------
The one-sided level/slope summary from the most recent seven raw observations
is the observed input X_s for predicting y_{s+1}; it is available before the
one-step response.  StaLoP then applies Algorithms 1--2 to these sequential
input-response panels.  For panel a, its predictive state at the current
input x* is a local-linear fit of previous input/next-response pairs in the
prespecified seven-panel history H_a.  Thus y_{a+1} never enters its own
predictive state.  For every h candidate, states, W, discrepancies, ranking,
and the full (eta,K) grid are recomputed before the GDF/MSPE minimization.
"""
from __future__ import annotations

import argparse,json,multiprocessing as mp,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import asdict,dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS","1");os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1");os.environ.setdefault("VECLIB_MAXIMUM_THREADS","1");os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

import numpy as np
import pandas as pd
from applications_common import PointPanelGrid,tune_point_panel_gdf,InvalidCandidate,sha256_file


@dataclass(frozen=True)
class SineConfig:
    R:int=200
    seed0:int=2026081601
    amp:float=.75
    noise_sd:float=.1
    dt:float=.01
    t_start:float=0.0
    t_end:float=1.60
    eval_start:float=1.10
    K_state:int=7                 # first-level one-sided sequence summary
    predictive_history:int=7      # H_a length in final Algorithm 1
    time_window_grid:tuple[int,...]=(5,7,9,11,15,20)
    knn_frac_grid:tuple[float,...]=(.05,.10,.20,.30)
    min_train:int=20
    max_validation_points:int=25
    q_h_grid:tuple[float,...]=(.10,.20,.30,.50,.75,1.00)
    eta_grid:tuple[float,...]=(1e-6,.01,.05,.10,.25,.50,1.00,2.00,5.00)
    rank_count_grid:tuple[int,...]=()
    rank_fraction_grid:tuple[float,...]=(.25,.50,.75,1.00)
    theta_delta:float=1e-8
    w_ridge:float=1e-8
    max_condition:float=1e10


def f_sine(t,amp): return amp*np.sin(2*np.pi*np.asarray(t,float))


def _fit_sequence_summary(tw:np.ndarray,yw:np.ndarray)->np.ndarray:
    dx=np.asarray(tw,float)-float(tw[-1]); P=np.column_stack([np.ones(len(dx)),dx]); A=P.T@P
    eig=np.linalg.eigvalsh(A/len(P))
    if eig[0]<=1e-14: raise RuntimeError("rank-deficient sequence state")
    return np.linalg.solve(A,P.T@yw)


def build_sequence_panels(t:np.ndarray,y:np.ndarray,K_state:int)->dict[str,np.ndarray]:
    # X_s uses observations through time s and predicts y_{s+1}.
    state_idx=np.arange(K_state-1,len(t)-1,dtype=int)
    X=np.empty((len(state_idx),2),float)
    for j,s in enumerate(state_idx):
        idx=np.arange(s-K_state+1,s+1,dtype=int)
        X[j]=_fit_sequence_summary(t[idx],y[idx])
    return {"state_idx":state_idx,"X":X,"y_next":y[state_idx+1].copy(),"target_time":t[state_idx+1].copy(),"origin_time":t[state_idx].copy()}


def estimate_sigma2_pre_response(y_available:np.ndarray)->float:
    d2=np.diff(np.asarray(y_available,float),n=2)
    if len(d2)<3: raise RuntimeError("too few observations for sigma2")
    return max(float(np.mean(d2*d2)/6.0),1e-12)


def _linear_predict(X:np.ndarray,y:np.ndarray,xstar:np.ndarray)->float:
    P=np.column_stack([np.ones(len(X)),X]); A=P.T@P
    eig=np.linalg.eigvalsh(A/len(P))
    if eig[0]<=1e-12: raise RuntimeError("rank-deficient sequence baseline")
    b=np.linalg.solve(A,P.T@y); return float(b[0]+b[1:]@xstar)


def _time_predict(X,y,xstar,window):
    w=min(max(int(window),3),len(y)); return _linear_predict(X[-w:],y[-w:],xstar)


def _knn_predict(X,y,xstar,frac):
    scale=np.std(X,axis=0,ddof=1); scale=np.maximum(scale,1e-8); d=np.linalg.norm((X-xstar)/scale,axis=1)
    k=min(max(int(np.ceil(float(frac)*len(y))),3),len(y)); idx=np.argpartition(d,k-1)[:k]; return float(np.mean(y[idx]))


def _validation_positions(n_train,cfg):
    v=np.arange(cfg.min_train,n_train,dtype=int)
    if len(v)<=cfg.max_validation_points:return v
    pos=np.linspace(0,len(v)-1,cfg.max_validation_points);return v[np.unique(np.round(pos).astype(int))]


def tune_baselines_once(panel,first_eval,cfg):
    X=panel["X"];y=panel["y_next"];vals=_validation_positions(first_eval,cfg)
    bw=(np.inf,cfg.time_window_grid[0])
    for w in cfg.time_window_grid:
        se=[]
        for v in vals:
            if v<=w+2:continue
            se.append((_time_predict(X[:v],y[:v],X[v],w)-y[v])**2)
        if se and float(np.mean(se))<bw[0]:bw=(float(np.mean(se)),int(w))
    bk=(np.inf,cfg.knn_frac_grid[0])
    for f in cfg.knn_frac_grid:
        se=[(_knn_predict(X[:v],y[:v],X[v],f)-y[v])**2 for v in vals if v>=cfg.min_train]
        if se and float(np.mean(se))<bk[0]:bk=(float(np.mean(se)),float(f))
    return int(bw[1]),float(bk[1])


def _gaussian(dist,h):
    h=max(float(h),1e-10); logw=-.5*(np.asarray(dist,float)/h)**2; logw-=float(np.max(logw)); return np.exp(logw)


def _predictive_state_window(X:np.ndarray,y:np.ndarray,j:int,xstar:np.ndarray,h:float,m:int,max_condition:float)->np.ndarray:
    idx=np.arange(j-m,j,dtype=int); dx=X[idx]-xstar; dist=np.linalg.norm(dx,axis=1); w=_gaussian(dist,h)
    P=np.column_stack([np.ones(m),dx]); A=P.T@(w[:,None]*P)/float(w.sum()); eig=np.linalg.eigvalsh(A)
    if eig[0]<=1e-12 or eig[-1]/eig[0]>max_condition: raise InvalidCandidate("invalid sequence predictive state")
    b=P.T@(w*y[idx])/float(w.sum()); return np.linalg.solve(A,b)


def _stalop_one_target(Xraw:np.ndarray,y:np.ndarray,xstar_raw:np.ndarray,sigma2_hat:float,cfg:SineConfig)->dict:
    center=Xraw.mean(axis=0);scale=Xraw.std(axis=0,ddof=1)
    if np.any(scale<1e-10):raise RuntimeError("degenerate sequence input scale")
    X=(Xraw-center)/scale;xstar=(xstar_raw-center)/scale;dist=np.linalg.norm(X-xstar,axis=1)
    m=cfg.predictive_history;eligible=np.arange(m,len(y),dtype=int)
    inner=PointPanelGrid(cfg.eta_grid,cfg.rank_count_grid,cfg.rank_fraction_grid)
    best=None
    for q in cfg.q_h_grid:
        h=max(float(np.quantile(dist,float(q))),1e-6); kernel=_gaussian(dist,h)
        try:
            Z=np.vstack([_predictive_state_window(X,y,int(j),xstar,h,m,cfg.max_condition) for j in eligible])
            zstar=_predictive_state_window(X,y,len(y),xstar,h,m,cfg.max_condition)
        except InvalidCandidate:
            continue
        P=np.column_stack([np.ones(len(X)),X-xstar]);W=P.T@(kernel[:,None]*P)/float(kernel.sum())+cfg.w_ridge*np.eye(3)
        try:
            cand=tune_point_panel_gdf(X[eligible],y[eligible],Z,xstar,zstar,sigma2_hat,inner,theta_delta=cfg.theta_delta,
                                      w_ridge=cfg.w_ridge,max_condition=cfg.max_condition,kernel_weight=kernel[eligible],W_override=W)
        except InvalidCandidate:
            continue
        row={**cand,"q_h":float(q),"h":float(h)}
        if best is None:best=row;continue
        c,b=float(row["criterion"]),float(best["criterion"]);tol=1e-14+1e-12*max(1,abs(c),abs(b))
        key=(row["q_h"],int(row["K"]),abs(np.log(max(row["eta"],1e-300))),row["eta"])
        bkey=(best["q_h"],int(best["K"]),abs(np.log(max(best["eta"],1e-300))),best["eta"])
        if c<b-tol or (abs(c-b)<=tol and key<bkey):best=row
    if best is None:raise RuntimeError("no valid sine Algorithm-2 candidate")
    return best


def run_one_rep(cfg:SineConfig,rep_id:int):
    seed=cfg.seed0+rep_id;rng=np.random.default_rng(seed);t=np.arange(cfg.t_start,cfg.t_end+cfg.dt/2,cfg.dt)
    truth=f_sine(t,cfg.amp);y=truth+rng.normal(0,cfg.noise_sd,len(t));panel=build_sequence_panels(t,y,cfg.K_state)
    ev=np.flatnonzero(panel["origin_time"]>=cfg.eval_start-1e-12)
    if len(ev)==0:raise RuntimeError("empty sequence evaluation")
    tw,kf=tune_baselines_once(panel,int(ev[0]),cfg);rows=[];tunes=[]
    for a in ev:
        if a<cfg.min_train:continue
        Xtr=panel["X"][:a];yn=panel["y_next"][:a];xs=panel["X"][a]
        raw_origin=int(panel["state_idx"][a]);sigma2=estimate_sigma2_pre_response(y[:raw_origin+1])
        tic=perf_counter();best=_stalop_one_target(Xtr,yn,xs,sigma2,cfg);rt=perf_counter()-tic
        target=float(truth[raw_origin+1])
        preds={"Global Linear":_linear_predict(Xtr,yn,xs),"Time-Local":_time_predict(Xtr,yn,xs,tw),
               "KNN-Covariate":_knn_predict(Xtr,yn,xs,kf),"StaLoP":float(best["pred"])}
        for method,p in preds.items():
            rows.append({"rep_id":rep_id,"seed":seed,"target_time":float(panel["target_time"][a]),"method":method,
                         "se":float((p-target)**2),"pred":p,"target":target,"time_window":tw if method=="Time-Local" else np.nan,
                         "knn_frac":kf if method=="KNN-Covariate" else np.nan,"runtime_sec":rt if method=="StaLoP" else np.nan})
        tunes.append({"rep_id":rep_id,"seed":seed,"target_time":float(panel["target_time"][a]),"K":int(best["K"]),"eta":best["eta"],
                      "q_h":best["q_h"],"h":best["h"],"criterion":best["criterion"],"R":best["R"],"GDF":best["GDF"],"ESS":best["ESS"],
                      "sigma2_hat":sigma2,"n_train":a,"n_eligible":a-cfg.predictive_history})
    return pd.DataFrame(rows),pd.DataFrame(tunes)


def summarize(raw,tuning):
    rep=raw.groupby(["rep_id","method"],as_index=False).agg(overall_se=("se","mean"))
    overall=rep.groupby("method",as_index=False).agg(overall_mspe=("overall_se","mean"),overall_sd=("overall_se",lambda x:float(np.std(x,ddof=1))),
                                                     overall_mcse=("overall_se",lambda x:float(np.std(x,ddof=1)/np.sqrt(len(x)))),n_rep=("overall_se","size"))
    bytime=raw.groupby(["target_time","method"],as_index=False).agg(mspe=("se","mean"),mcse=("se",lambda x:float(np.std(x,ddof=1)/np.sqrt(len(x)))))
    ts=pd.DataFrame([{"K_mean":tuning.K.mean(),"K_median":tuning.K.median(),"eta_mean":tuning.eta.mean(),"eta_median":tuning.eta.median(),
                      "q_h_mean":tuning.q_h.mean(),"h_mean":tuning.h.mean(),"sigma2_mean":tuning.sigma2_hat.mean(),"ESS_mean":tuning.ESS.mean()}])
    return overall,bytime,ts


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--R",type=int,default=200);ap.add_argument("--seed0",type=int,default=SineConfig().seed0)
    ap.add_argument("--n-jobs",type=int,default=6);ap.add_argument("--outdir",default="application1_sine_R200");ap.add_argument("--formal",action="store_true");args=ap.parse_args()
    R=200 if args.formal else args.R;cfg=SineConfig(R=R,seed0=args.seed0);out=Path(args.outdir);out.mkdir(parents=True,exist_ok=True);nj=max(1,min(args.n_jobs,R,mp.cpu_count() or 1))
    tic=perf_counter();rp=[];tp=[]
    if nj==1:
        for done,r in enumerate(range(R),1):a,b=run_one_rep(cfg,r);rp.append(a);tp.append(b);print(f"Completed {done}/{R}",flush=True) if done%20==0 or done==R else None
    else:
        with ProcessPoolExecutor(max_workers=nj) as ex:
            fut={ex.submit(run_one_rep,cfg,r):r for r in range(R)}
            for done,f in enumerate(as_completed(fut),1):a,b=f.result();rp.append(a);tp.append(b);print(f"Completed {done}/{R}",flush=True) if done%20==0 or done==R else None
    raw=pd.concat(rp,ignore_index=True).sort_values(["rep_id","target_time","method"]);tun=pd.concat(tp,ignore_index=True).sort_values(["rep_id","target_time"])
    overall,bytime,ts=summarize(raw,tun);raw.to_csv(out/"application1_sine_raw.csv",index=False);tun.to_csv(out/"application1_sine_tuning.csv",index=False)
    overall.to_csv(out/"application1_sine_summary.csv",index=False);bytime.to_csv(out/"application1_sine_by_time.csv",index=False);ts.to_csv(out/"application1_sine_tuning_summary.csv",index=False)
    man={"application":"sine","config":asdict(cfg),"R":R,"valid_replications":int(raw.rep_id.nunique()),"wall_seconds":perf_counter()-tic,"source_sha256":sha256_file(__file__)}
    (out/"application1_sine_manifest.json").write_text(json.dumps(man,indent=2),encoding="utf-8");print(overall.to_string(index=False));print(ts.to_string(index=False));print(f"wall_seconds={man['wall_seconds']:.3f}")

if __name__=="__main__":mp.freeze_support();main()
