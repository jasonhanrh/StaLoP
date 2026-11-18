from __future__ import annotations
import argparse, hashlib, json, os, sys, zipfile, urllib.request
from pathlib import Path
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('NUMEXPR_NUM_THREADS','1')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from shapely.geometry import shape as shapely_shape
import shapefile

ROOT = Path(__file__).resolve().parents[1]
SIMDIR = ROOT / '01_experiment1_simulation' / 'code'
sys.path.insert(0, str(SIMDIR))
import simulation_core as sim

DATA_URL='https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
ZONE_ZIP='https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
ZONE_LOOKUP='https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv'

YEAR=2024
TARGET_START=pd.Timestamp('2024-10-01')
TARGET_END=pd.Timestamp('2024-12-31')
Q_GRID=(0.10,0.20,0.30)
ETA_GRID=(0.25,0.50,0.75,1.00)
K_FRACS=(0.20,0.30,0.40,0.50,0.70,1.00)
TIME_WINDOWS=(1,2,3,6,12)
KNN_GRID=(1,2,3,5,10,20,40)

METHOD=sim.MethodConfig(K_state=1,state_ridge=0.0,final_ridge=0.0,w_ridge=1e-8,
                        theta_standardize=True,min_bandwidth=1e-10,
                        noise_pair_fraction=0.50,min_noise_pairs=20,
                        require_full_rank_before_ridge=True,max_design_condition=1e4)
GRID=sim.TuningGrid(q_h_grid=Q_GRID,eta_grid=ETA_GRID,rank_count_grid=(),rank_fraction_grid=K_FRACS,
                    time_window_grid=TIME_WINDOWS,knn_k_grid=KNN_GRID)

def sha256(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def download(url:str,path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size>0: return
    print('DOWNLOAD',url,'->',path,flush=True)
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 StaLoP-repro/1.0'})
    with urllib.request.urlopen(req,timeout=180) as r, path.open('wb') as f:
        while True:
            b=r.read(1<<20)
            if not b: break
            f.write(b)

def load_manhattan_zones(data_dir:Path):
    zzip=data_dir/'taxi_zones.zip'; lookup=data_dir/'taxi_zone_lookup.csv'
    download(ZONE_ZIP,zzip); download(ZONE_LOOKUP,lookup)
    ex=data_dir/'taxi_zones'; ex.mkdir(exist_ok=True)
    with zipfile.ZipFile(zzip) as z: z.extractall(ex)
    shp=next(ex.glob('*.shp'))
    rdr=shapefile.Reader(str(shp))
    fields=[x[0] for x in rdr.fields[1:]]
    records=[]
    for sr in rdr.iterShapeRecords():
        d=dict(zip(fields,sr.record))
        low={str(k).lower():v for k,v in d.items()}
        lid=int(low.get('locationid',low.get('location_i',low.get('objectid'))))
        geom=shapely_shape(sr.shape.__geo_interface__)
        c=geom.centroid
        records.append((lid,float(c.x),float(c.y)))
    cent=pd.DataFrame(records,columns=['LocationID','cx','cy']).drop_duplicates('LocationID')
    lu=pd.read_csv(lookup)
    lu.columns=[c.strip() for c in lu.columns]
    man=lu[lu['Borough'].astype(str).str.casefold()=='manhattan'][['LocationID','Borough','Zone']].copy()
    man['LocationID']=man['LocationID'].astype(int)
    man=man.merge(cent,on='LocationID',how='inner').sort_values('LocationID').reset_index(drop=True)
    if len(man)<50: raise RuntimeError(f'unexpected Manhattan zone count {len(man)}')
    for c in ['cx','cy']:
        man[c]=(man[c]-man[c].mean())/man[c].std(ddof=1)
    return man,zzip,lookup

def build_daily_panel(data_dir:Path,out_dir:Path):
    man,zzip,lookup=load_manhattan_zones(data_dir)
    zone_ids=set(man.LocationID.astype(int))
    pieces=[]; manifest=[]
    for m in range(1,13):
        p=data_dir/f'yellow_tripdata_{YEAR}-{m:02d}.parquet'
        download(DATA_URL.format(year=YEAR,month=m),p)
        df=pd.read_parquet(p,columns=['tpep_pickup_datetime','PULocationID'])
        dt=pd.to_datetime(df['tpep_pickup_datetime'],errors='coerce')
        good=(dt>=pd.Timestamp(f'{YEAR}-01-01'))&(dt<pd.Timestamp(f'{YEAR+1}-01-01'))
        loc=pd.to_numeric(df['PULocationID'],errors='coerce')
        good &= loc.isin(zone_ids)
        z=pd.DataFrame({'date':dt[good].dt.floor('D'),'LocationID':loc[good].astype(int)})
        g=z.groupby(['date','LocationID'],observed=True).size().rename('pickups').reset_index()
        pieces.append(g)
        manifest.append({'file':p.name,'bytes':p.stat().st_size,'sha256':sha256(p),'n_raw':len(df),'n_accepted_manhattan':int(g.pickups.sum())})
        print('MONTH',m,'raw',len(df),'accepted',int(g.pickups.sum()),flush=True)
        del df,z,g
    counts=pd.concat(pieces,ignore_index=True).groupby(['date','LocationID'],as_index=False).pickups.sum()
    dates=pd.date_range(f'{YEAR}-01-01',f'{YEAR}-12-31',freq='D')
    full=pd.MultiIndex.from_product([dates,man.LocationID.tolist()],names=['date','LocationID']).to_frame(index=False)
    full=full.merge(counts,on=['date','LocationID'],how='left').fillna({'pickups':0})
    full=full.merge(man[['LocationID','cx','cy','Zone']],on='LocationID',how='left')
    full['pickups']=full.pickups.astype(int)
    full['y']=np.log1p(full.pickups.to_numpy(float))
    full['panel']=(full.date-full.date.min()).dt.days.astype(int)
    out_dir.mkdir(parents=True,exist_ok=True)
    full.to_csv(out_dir/'daily_manhattan_panel.csv.gz',index=False,compression='gzip')
    pd.DataFrame(manifest).to_csv(out_dir/'raw_manifest.csv',index=False)
    meta={'year':YEAR,'n_zones':int(len(man)),'n_days':int(len(dates)),'n_rows':int(len(full)),
          'target_start':str(TARGET_START.date()),'target_end':str(TARGET_END.date()),
          'zone_zip_sha256':sha256(zzip),'lookup_sha256':sha256(lookup),
          'total_pickups':int(full.pickups.sum()),'zero_zone_days':int((full.pickups==0).sum())}
    (out_dir/'data_audit.json').write_text(json.dumps(meta,indent=2))
    return full,man,meta

def baseline_validation_panels(target_idx:int,max_folds=6,min_train=6):
    return list(range(max(min_train,target_idx-max_folds),target_idx))

def fit_linear(X,y):
    P=np.column_stack([np.ones(len(X)),X])
    return np.linalg.lstsq(P,y,rcond=None)[0]

def pred_linear(beta,X): return beta[0]+X@beta[1:]

def tune_time(Xmat,Ymat,target_idx):
    rows=[]
    for w in TIME_WINDOWS:
        sse=0.; n=0
        for v in baseline_validation_panels(target_idx):
            start=max(0,v-w)
            Xh=np.tile(Xmat,(v-start,1)); yh=Ymat[:,start:v].T.reshape(-1)
            b=fit_linear(Xh,yh); p=pred_linear(b,Xmat); e=p-Ymat[:,v]
            sse+=float(e@e); n+=len(e)
        rows.append((sse/n,w))
    return min(rows)[1]

def spatial_knn_predict(Xmat, hist_zone_mean, k):
    ke=min(int(k),len(Xmat))
    _,ix=cKDTree(Xmat).query(Xmat,k=ke,workers=1)
    ix=np.asarray(ix,dtype=int)
    return hist_zone_mean[ix] if ke==1 else hist_zone_mean[ix].mean(axis=1)

def tune_knn(Xmat,Ymat,target_idx):
    rows=[]
    for k in KNN_GRID:
        sse=0.; n=0
        for v in baseline_validation_panels(target_idx):
            hist_mean=Ymat[:,:v].mean(axis=1)
            p=spatial_knn_predict(Xmat,hist_mean,k)
            e=p-Ymat[:,v];sse+=float(e@e);n+=len(e)
        rows.append((sse/n,k))
    return min(rows)[1]

def recent_candidate(comp,K,sigma2):
    eligible=np.asarray(comp['eligible_panel_idx'],int)
    panels=np.sort(eligible)[::-1][:int(K)]
    pA=np.asarray(comp['panel_A']); pb=np.asarray(comp['panel_b']); pw=np.asarray(comp['panel_weight']); py2=np.asarray(comp['panel_y2'])
    Q=float(pw[panels].sum())
    if Q<=0: raise sim.InvalidCandidate('recent zero support')
    A=pA[panels].sum(axis=0); b=pb[panels].sum(axis=0)
    c=sim.solve_sufficient_stats_no_fallback(A,b,Q,METHOD.final_ridge,METHOD.require_full_rank_before_ridge,METHOD.max_design_condition)
    y2=float(py2[panels].sum()); R=max(float((y2-2*c@b+c@A@c)/Q),0.0)
    V=pA[panels].sum(axis=0); M=A+METHOD.final_ridge*Q*np.eye(A.shape[0])
    G=float(np.trace(np.linalg.solve(M,V))/Q)
    crit=R+2*sigma2*G-sigma2
    pm=np.zeros(len(pw)); pm[panels]=pw[panels]/Q
    return {'pred':float(c[0]),'R':R,'GDF':G,'criterion':float(crit),'K':int(K),'period_mass':pm}

def tune_recent(x1p,x2p,yp,tp,xstar,Th,sigma2):
    static=sim.prepare_target_static(x1p,x2p,xstar)
    best=None; nvalid=0
    for q in Q_GRID:
        try: comp=sim.build_target_components(x1p,x2p,yp,tp,xstar,Th,q,METHOD,static=static)
        except sim.InvalidCandidate: continue
        for K in sim.rank_candidate_counts(len(comp['theta']),K_FRACS,()):
            try: r=recent_candidate(comp,K,sigma2)
            except sim.InvalidCandidate: continue
            nvalid+=1; row={**r,'q_h':float(q),'h':float(comp['h'])}
            if best is None or row['criterion']<best['criterion']-1e-12 or (abs(row['criterion']-best['criterion'])<=1e-12 and (row['q_h'],row['K'])<(best['q_h'],best['K'])):
                best=row
    if best is None: raise RuntimeError('no valid Recent-Local candidate')
    best['n_candidates']=nvalid
    return best

def stalop_period_mass(x1p,x2p,yp,tp,xstar,Th,best):
    comp=sim.build_target_components(x1p,x2p,yp,tp,xstar,Th,float(best['q_h']),METHOD,h_override=float(best['h']))
    eligible=np.asarray(comp['eligible_panel_idx'],int); order=np.asarray(comp['order'],int); th=np.asarray(comp['theta_std'],float)
    keep=order[:int(best['K'])]; panels=eligible[keep]
    scores=np.exp(-float(best['eta'])*th[keep]); pw=np.asarray(comp['panel_weight'],float)
    denom=float(np.sum(scores*pw[panels])); pm=np.zeros(Th)
    if denom>0: pm[panels]=scores*pw[panels]/denom
    return pm

def evaluate_day(target_idx,dates,Xmat,Ymat):
    N=len(Xmat); Th=target_idx
    x1p=np.tile(Xmat[:,0],Th); x2p=np.tile(Xmat[:,1],Th)
    yp=Ymat[:,:Th].T.reshape(-1); tp=np.repeat(np.arange(1,Th+1),N)
    sigma2,npairs=sim.estimate_noise_variance_within_panel_nn(x1p,x2p,yp,tp,Th,METHOD.noise_pair_fraction,METHOD.min_noise_pairs)
    Xh=np.tile(Xmat,(Th,1)); bg=fit_linear(Xh,yp); pg=pred_linear(bg,Xmat)
    tw=tune_time(Xmat,Ymat,target_idx); start=max(0,Th-tw); bt=fit_linear(np.tile(Xmat,(Th-start,1)),Ymat[:,start:Th].T.reshape(-1)); pt=pred_linear(bt,Xmat)
    kk=tune_knn(Xmat,Ymat,target_idx); pk=spatial_knn_predict(Xmat,Ymat[:,:Th].mean(axis=1),kk)
    truth=Ymat[:,target_idx]
    ps=np.full(N,np.nan); pr=np.full(N,np.nan); masses=[]; tuning=[]
    for i in range(N):
        xs=Xmat[i]
        try:
            b=sim.tune_stalop_gdf_target(x1p,x2p,yp,tp,xs,Th,sigma2,GRID,METHOD,False)
            rr=tune_recent(x1p,x2p,yp,tp,xs,Th,sigma2)
            pm=stalop_period_mass(x1p,x2p,yp,tp,xs,Th,b)
        except (RuntimeError,sim.InvalidCandidate,np.linalg.LinAlgError):
            continue
        ps[i]=float(b['pred']);pr[i]=float(rr['pred']);masses.append(pm)
        tuning.append((float(b['q_h']),float(b['eta']),int(b['K']),float(b['criterion'])))
    ok=np.isfinite(ps)&np.isfinite(pr)
    preds={'Global Linear':pg,'Time-Local':pt,'KNN-Covariate':pk,'Recent-Local':pr,'StaLoP':ps}
    rows=[]
    for name,p in preds.items():
        e=p[ok]-truth[ok]; rows.append({'date':str(dates[target_idx].date()),'target_panel':int(target_idx),'method':name,'mspe':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'n':int(ok.sum())})
    if masses:
        mm=np.mean(np.vstack(masses),axis=0); lag=Th-np.arange(Th); weighted=float(np.sum(lag*mm)); mass1=float(mm[lag>1].sum());mass7=float(mm[lag>7].sum());mass14=float(mm[lag>14].sum());mass28=float(mm[lag>28].sum())
    else: weighted=mass1=mass7=mass14=mass28=np.nan
    tune=np.asarray(tuning,float) if tuning else np.empty((0,4))
    diag={'date':str(dates[target_idx].date()),'target_panel':int(target_idx),'n_target_full':N,'n_supported':int(ok.sum()),'support_rate':float(ok.mean()),
          'sigma2_hat':float(sigma2),'sigma2_pairs':int(npairs),'time_window':int(tw),'knn_k':int(kk),
          'mass_lag_gt1':mass1,'mass_lag_gt7':mass7,'mass_lag_gt14':mass14,'mass_lag_gt28':mass28,'weighted_lag_days':weighted,
          'qh_boundary_rate':float(np.mean(np.isin(tune[:,0],[min(Q_GRID),max(Q_GRID)]))) if len(tune) else np.nan,
          'eta_boundary_rate':float(np.mean(np.isin(tune[:,1],[min(ETA_GRID),max(ETA_GRID)]))) if len(tune) else np.nan,
          'K_full_rate':float(np.mean(tune[:,2]==(Th-1))) if len(tune) else np.nan}
    return rows,diag

def compatibility_audit(Xmat,Ymat,dates,target_indices):
    surf=(Ymat-Ymat.mean(axis=0,keepdims=True))/(Ymat.std(axis=0,ddof=1,keepdims=True)+1e-8)
    out=[]
    for t in target_indices:
        if t<8: continue
        zt=surf[:,t-1]
        hist=np.arange(1,t)
        dstate=np.mean((surf[:,hist-1]-zt[:,None])**2,axis=0)
        isel=int(hist[np.argmin(dstate)])
        dsel=float(np.mean((surf[:,isel]-surf[:,t])**2))
        drecent=float(np.mean((surf[:,t-1]-surf[:,t])**2))
        out.append({'date':str(dates[t].date()),'matched_day':str(dates[isel].date()),'state_lag_days':int(t-isel),
                    'matched_next_surface_mse':dsel,'recent_surface_mse':drecent,'matched_beats_recent':int(dsel<drecent)})
    return pd.DataFrame(out)

def summarize(metrics,diag):
    piv=metrics.pivot(index='target_panel',columns='method',values='mspe')
    pooled={}
    for method,g in metrics.groupby('method'):
        pooled[method]=float(np.average(g.mspe,weights=g.n))
    s=pooled['StaLoP']; r=pooled['Recent-Local']; k=pooled['KNN-Covariate']; tm=pooled['Time-Local']; gl=pooled['Global Linear']
    rs=piv['StaLoP']/piv['Recent-Local']; rk=piv['StaLoP']/piv['KNN-Covariate']
    out={'dataset':'NYC_TLC_Yellow_Manhattan_2024','role':'final_candidate','stalop_mspe':s,'recent_local_mspe':r,'knn_mspe':k,'time_local_mspe':tm,'global_mspe':gl,
         'rel_gain_vs_recent_local':(r-s)/r,'rel_gain_vs_knn':(k-s)/k,'rel_gain_vs_time_local':(tm-s)/tm,
         'median_panel_ratio_vs_recent_local':float(rs.median()),'median_panel_ratio_vs_knn':float(rk.median()),
         'panel_win_rate_vs_recent_local':float((rs<1).mean()),'panel_win_rate_vs_knn':float((rk<1).mean()),
         'mean_support_rate':float(diag.support_rate.mean()),'median_support_rate':float(diag.support_rate.median()),'min_support_rate':float(diag.support_rate.min()),
         'median_mass_lag_gt1_panel':float(diag.mass_lag_gt1.median()),'median_mass_lag_gt_7d':float(diag.mass_lag_gt7.median()),
         'median_mass_lag_gt_14d':float(diag.mass_lag_gt14.median()),'median_mass_lag_gt_28d':float(diag.mass_lag_gt28.median()),
         'median_weighted_lag_days':float(diag.weighted_lag_days.median()),
         'mean_stalop_qh_boundary_concentration':float(diag.qh_boundary_rate.mean()),'mean_stalop_eta_boundary_concentration':float(diag.eta_boundary_rate.mean()),
         'mean_stalop_K_full_rate':float(diag.K_full_rate.mean()),'n_target_obs_supported':int(diag.n_supported.sum()),'n_target_panels':int(len(diag))}
    out['prediction_gate']=bool(s<r and s<k and out['median_panel_ratio_vs_recent_local']<1 and out['median_panel_ratio_vs_knn']<1 and out['panel_win_rate_vs_recent_local']>0.5 and out['panel_win_rate_vs_knn']>0.5)
    out['non_temporal_borrowing_gate']=bool(out['median_mass_lag_gt_7d']>0.5)
    out['support_gate']=bool(out['mean_support_rate']>=0.80)
    out['main_paper_gate']=bool(out['prediction_gate'] and out['non_temporal_borrowing_gate'] and out['support_gate'])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workdir',default='tmp_nyc_tlc/work');ap.add_argument('--outdir',default='tmp_nyc_tlc/results');ap.add_argument('--n-jobs',type=int,default=4);args=ap.parse_args()
    work=ROOT/args.workdir; out=ROOT/args.outdir; work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    t0=perf_counter(); full,man,meta=build_daily_panel(work,out)
    dates=pd.date_range(f'{YEAR}-01-01',f'{YEAR}-12-31',freq='D'); zone_ids=man.LocationID.tolist(); N=len(zone_ids);T=len(dates)
    Xmat=man[['cx','cy']].to_numpy(float); Ymat=np.zeros((N,T))
    pos={z:i for i,z in enumerate(zone_ids)}
    for rr in full.itertuples(index=False): Ymat[pos[int(rr.LocationID)],int(rr.panel)]=float(rr.y)
    targets=np.array([i for i,d in enumerate(dates) if TARGET_START<=d<=TARGET_END],int)
    comp=compatibility_audit(Xmat,Ymat,dates,targets);comp.to_csv(out/'compatibility_audit.csv',index=False)
    print('COMPAT matched win rate',comp.matched_beats_recent.mean(),'median lag',comp.state_lag_days.median(),flush=True)
    res=Parallel(n_jobs=args.n_jobs,backend='loky',verbose=10)(delayed(evaluate_day)(int(t),dates,Xmat,Ymat) for t in targets)
    metrics=pd.DataFrame([x for rr,_ in res for x in rr]);diag=pd.DataFrame([d for _,d in res])
    metrics.to_csv(out/'panel_metrics.csv',index=False);diag.to_csv(out/'panel_diagnostics.csv',index=False)
    summary=summarize(metrics,diag); summary.update({'compatibility_matched_win_rate':float(comp.matched_beats_recent.mean()),'compatibility_median_lag_days':float(comp.state_lag_days.median()),'runtime_sec':float(perf_counter()-t0)})
    pd.DataFrame([summary]).to_csv(out/'application_gate_summary.csv',index=False)
    (out/'application_gate_summary.json').write_text(json.dumps(summary,indent=2))
    protocol={'design_locked_before_run':True,'panel':'calendar day','unit':'Manhattan TLC taxi zone','response':'log1p daily Yellow Taxi pickups','covariates':'response-free standardized native-CRS taxi-zone centroid x,y','initial_history':'2024-01-01..2024-09-30','rolling_targets':'2024-10-01..2024-12-31','state_window':1,'q_h_grid':Q_GRID,'eta_grid':ETA_GRID,'K_fractions':K_FRACS,'knn_definition':'nearest unique zones applied to pre-target historical zone means','gate_definition':'StaLoP pooled MSPE < Recent-Local and KNN; both median panel ratios <1; both win rates >.5; median mass lag>7d >.5; mean support >=.8'}
    (out/'DESIGN_LOCK.json').write_text(json.dumps(protocol,indent=2))
    print('FINAL_SUMMARY',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
