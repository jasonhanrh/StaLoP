"""Multidimensional exact StaLoP/GDF engine for Application 3.

This module generalizes the frozen Experiment-1 Algorithm-1/2 algebra from two
covariates to d=0,...,4 without changing the statistical criterion.  It is
specialized to the production K_state=1 design and batches target inputs so the
16-subset variable-selection application remains computationally feasible.

For every q_h candidate and every target input, the kernel, panel sufficient
statistics, predictive states, W, discrepancies, ranking, and all (eta,K)
candidates are recomputed.  No candidate reuses response-adaptive quantities
from another variable subset or bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.spatial import cKDTree


class InvalidCandidate(RuntimeError):
    pass


@dataclass(frozen=True)
class VariableTuningGrid:
    q_h_grid: tuple[float, ...] = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    eta_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
    rank_fraction_grid: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

    def validate(self) -> None:
        if not self.q_h_grid or any(not (0.0 < q <= 1.0) for q in self.q_h_grid):
            raise ValueError("invalid q_h grid")
        if not self.eta_grid or any(e <= 0.0 for e in self.eta_grid):
            raise ValueError("invalid eta grid")
        if not self.rank_fraction_grid or any(not (0.0 < f <= 1.0) for f in self.rank_fraction_grid):
            raise ValueError("invalid K-fraction grid")


def standardize_theta_batch(theta: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    theta = np.asarray(theta, float)
    mean = np.mean(theta, axis=1, keepdims=True)
    if theta.shape[1] <= 1:
        return np.zeros_like(theta)
    sd = np.std(theta, axis=1, ddof=1, keepdims=True)
    return (theta - mean) / (sd + float(delta))


def rank_counts(n_eligible: int, fractions: Iterable[float]) -> np.ndarray:
    vals = sorted({min(max(int(np.floor(float(f) * n_eligible)), 1), n_eligible) for f in fractions})
    if not vals:
        raise ValueError("empty K grid")
    return np.asarray(vals, dtype=int)


def _safe_batched_solve(M: np.ndarray, rhs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    M = np.asarray(M, float); rhs = np.asarray(rhs, float)
    L = M.shape[-1]
    Ms = M.copy(); rs = rhs.copy()
    if np.any(~valid):
        Ms[~valid] = np.eye(L)
        rs[~valid] = 0.0
    try:
        return np.linalg.solve(Ms, rs[..., None])[..., 0]
    except np.linalg.LinAlgError as exc:
        raise InvalidCandidate("batched local solve failed") from exc


def _gram_valid_batch(A_raw: np.ndarray, weight: np.ndarray, max_condition: float) -> np.ndarray:
    weight = np.asarray(weight, float)
    valid = np.isfinite(weight) & (weight > 0.0)
    scaled = A_raw / np.maximum(weight[..., None, None], 1e-300)
    eig = np.linalg.eigvalsh(scaled)
    lo = eig[..., 0]; hi = eig[..., -1]
    valid &= np.isfinite(lo) & np.isfinite(hi) & (lo > 1e-12)
    valid &= (hi / np.maximum(lo, 1e-300)) <= float(max_condition)
    return valid


def panel_stats_block(dx: np.ndarray, y_hist: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Binary-kernel panel sufficient statistics.

    dx: [B,T,N,d], y_hist: [T,N], mask: [B,T,N].
    Returns A[B,T,L,L], b[B,T,L], w[B,T], y2[B,T].
    """
    B,T,N,d = dx.shape
    L=d+1
    m=np.asarray(mask,float)
    y=np.asarray(y_hist,float)
    w=m.sum(axis=2)
    b=np.zeros((B,T,L),float)
    A=np.zeros((B,T,L,L),float)
    y2=np.einsum('btn,tn->bt',m,y*y,optimize=True)
    b[:,:,0]=np.einsum('btn,tn->bt',m,y,optimize=True)
    A[:,:,0,0]=w
    if d:
        sx=np.einsum('btn,btnd->btd',m,dx,optimize=True)
        b[:,:,1:]=np.einsum('btn,btnd,tn->btd',m,dx,y,optimize=True)
        A[:,:,0,1:]=sx; A[:,:,1:,0]=sx
        A[:,:,1:,1:]=np.einsum('btn,btnd,btne->btde',m,dx,dx,optimize=True)
    return A,b,w,y2


def _candidate_tie_better(
    crit: np.ndarray, K: int, eta: float, q_h: float,
    best_crit: np.ndarray, best_K: np.ndarray, best_eta: np.ndarray, best_q: np.ndarray,
) -> np.ndarray:
    finite=np.isfinite(crit)
    best_finite=np.isfinite(best_crit)
    scale=np.maximum(1.0,np.maximum(np.abs(crit),np.where(best_finite,np.abs(best_crit),0.0)))
    tol=1e-14+1e-12*scale
    best_cmp=np.where(best_finite,best_crit,0.0)
    better=finite & (~best_finite | (crit < best_cmp-tol))
    tied=finite & best_finite & (np.abs(crit-best_cmp)<=tol)
    # final Section-5 production tie convention: smaller h-index/q, smaller K,
    # then eta closest to one, then smaller eta.
    qbetter=q_h < best_q-1e-15
    qtie=np.abs(q_h-best_q)<=1e-15
    kbetter=K < best_K
    ktie=K==best_K
    ekey=abs(float(np.log(max(float(eta),1e-300))))
    best_ekey=np.abs(np.log(np.maximum(best_eta,1e-300)))
    ebetter=(ekey < best_ekey-1e-15) | ((np.abs(ekey-best_ekey)<=1e-15)&(eta<best_eta))
    secondary=qbetter | (qtie & (kbetter | (ktie & ebetter)))
    return better | (tied & secondary)


def tune_subset_all_targets(
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    X_target: np.ndarray,
    sigma2_hat: float,
    grid: VariableTuningGrid = VariableTuningGrid(),
    *,
    block_size: int = 20,
    theta_delta: float = 1e-8,
    w_ridge: float = 1e-8,
    max_condition: float = 1e4,
) -> tuple[np.ndarray, np.ndarray, dict[str,np.ndarray]]:
    """Target-specific exhaustive Algorithm-2 tuning for one variable subset.

    X_hist is [T,N,d] and X_target is [N_target,d].  Inputs should already be
    standardized using response-free historical scales.  K_state is exactly 1.
    Returns predictions, minimized criteria, and selected tuning diagnostics.
    """
    grid.validate()
    X_hist=np.asarray(X_hist,float); y_hist=np.asarray(y_hist,float); X_target=np.asarray(X_target,float)
    if X_hist.ndim!=3 or y_hist.shape!=X_hist.shape[:2] or X_target.ndim!=2 or X_target.shape[1]!=X_hist.shape[2]:
        raise ValueError("dimension mismatch in multidimensional tuner")
    T,N,d=X_hist.shape; nt=len(X_target); L=d+1; P=T-1
    if T<3: raise ValueError("T must be at least 3")
    if not np.isfinite(sigma2_hat) or sigma2_hat<0: raise ValueError("invalid sigma2_hat")
    Kvals=rank_counts(P,grid.rank_fraction_grid)
    etas=np.asarray(grid.eta_grid,float)
    qvals=tuple(float(q) for q in grid.q_h_grid)

    pred_out=np.full(nt,np.nan); crit_out=np.full(nt,np.inf)
    K_out=np.full(nt,-1,int); eta_out=np.full(nt,np.nan); q_out=np.full(nt,np.nan)
    R_out=np.full(nt,np.nan); GDF_out=np.full(nt,np.nan); ESS_out=np.full(nt,np.nan); h_out=np.full(nt,np.nan)
    nvalid_out=np.zeros(nt,int)

    for start in range(0,nt,max(1,int(block_size))):
        stop=min(nt,start+max(1,int(block_size))); Xt=X_target[start:stop]; B=len(Xt)
        if d:
            dx=X_hist[None,:,:,:]-Xt[:,None,None,:]
            dist=np.sqrt(np.sum(dx*dx,axis=-1))
        else:
            dx=np.empty((B,T,N,0),float); dist=np.zeros((B,T,N),float)

        bestcrit=np.full(B,np.inf); bestpred=np.full(B,np.nan)
        bestK=np.full(B,np.iinfo(np.int32).max,int); besteta=np.full(B,np.inf); bestq=np.full(B,np.inf)
        bestR=np.full(B,np.nan); bestG=np.full(B,np.nan); bestESS=np.full(B,np.nan); besth=np.full(B,np.nan)
        nvalid=np.zeros(B,int)

        flatdist=dist.reshape(B,-1)
        hmatrix=np.quantile(flatdist,np.asarray(qvals),axis=1).T if d else np.zeros((B,len(qvals)))
        hmatrix=np.maximum(hmatrix,1e-10)

        for iq,q_h in enumerate(qvals):
            h=hmatrix[:,iq]
            mask=dist<=h[:,None,None]
            A,b,w,y2=panel_stats_block(dx,y_hist,mask)
            valid_state=_gram_valid_batch(A,w,max_condition)
            valid_target=np.all(valid_state,axis=1)
            coeff=_safe_batched_solve(A/np.maximum(w[:,:,None,None],1e-300), b/np.maximum(w[:,:,None],1e-300), valid_state)
            Zhist=coeff[:,:-1,:]; zstar=coeff[:,-1,:]
            W=A.sum(axis=1)/np.maximum(w.sum(axis=1)[:,None,None],1e-300)+float(w_ridge)*np.eye(L)[None,:,:]
            diff=Zhist-zstar[:,None,:]
            theta=np.maximum(np.einsum('bpi,bij,bpj->bp',diff,W,diff,optimize=True),0.0)
            tstd=standardize_theta_batch(theta,theta_delta)
            order=np.argsort(theta,axis=1,kind='stable')

            # final response panels are labels 2,...,T => panel indices 1,...,T-1
            Ael=A[:,1:,:,:]; bel=b[:,1:,:]; wel=w[:,1:]; y2el=y2[:,1:]
            idxA=np.broadcast_to(order[:,:,None,None],(B,P,L,L))
            idxb=np.broadcast_to(order[:,:,None],(B,P,L))
            Aord=np.take_along_axis(Ael,idxA,axis=1)
            bord=np.take_along_axis(bel,idxb,axis=1)
            word=np.take_along_axis(wel,order,axis=1)
            y2ord=np.take_along_axis(y2el,order,axis=1)
            tsord=np.take_along_axis(tstd,order,axis=1)

            logbeta=-etas[None,:,None]*tsord[:,None,:]
            logbeta-=np.max(logbeta,axis=2,keepdims=True)
            beta=np.exp(logbeta)
            cumQ=np.cumsum(beta*word[:,None,:],axis=2)
            cumA=np.cumsum(beta[:,:,:,None,None]*Aord[:,None,:,:,:],axis=2)
            cumb=np.cumsum(beta[:,:,:,None]*bord[:,None,:,:],axis=2)
            cumy2=np.cumsum(beta*y2ord[:,None,:],axis=2)
            beta2=beta*beta
            cumV=np.cumsum(beta2[:,:,:,None,None]*Aord[:,None,:,:,:],axis=2)
            # binary kernel => within-panel sum K_i^2 = panel weight
            cumobs2=np.cumsum(beta2*word[:,None,:],axis=2)

            for ie,eta in enumerate(etas):
                for K in Kvals:
                    j=int(K)-1
                    Q=cumQ[:,ie,j]; Ar=cumA[:,ie,j]; br=cumb[:,ie,j]; yr=cumy2[:,ie,j]; Vr=cumV[:,ie,j]; obs2=cumobs2[:,ie,j]
                    valid=valid_target & _gram_valid_batch(Ar,Q,max_condition) & np.isfinite(obs2) & (obs2>0)
                    c=_safe_batched_solve(Ar,br,valid)
                    sse=yr-2*np.einsum('bi,bi->b',c,br,optimize=True)+np.einsum('bi,bij,bj->b',c,Ar,c,optimize=True)
                    R=np.maximum(sse/np.maximum(Q,1e-300),0.0)
                    M=Ar.copy(); Ms=M.copy(); Vs=Vr.copy()
                    if np.any(~valid):
                        Ms[~valid]=np.eye(L); Vs[~valid]=0.0
                    solved=np.linalg.solve(Ms,Vs)
                    G=np.trace(solved,axis1=1,axis2=2)/np.maximum(Q,1e-300)
                    crit=R+2*float(sigma2_hat)*G-float(sigma2_hat)
                    ESS=Q*Q/np.maximum(obs2,1e-300)
                    crit=np.where(valid,crit,np.inf)
                    nvalid += valid.astype(int)
                    upd=_candidate_tie_better(crit,int(K),float(eta),float(q_h),bestcrit,bestK,besteta,bestq)
                    if np.any(upd):
                        bestcrit[upd]=crit[upd]; bestpred[upd]=c[upd,0]; bestK[upd]=int(K); besteta[upd]=float(eta); bestq[upd]=float(q_h)
                        bestR[upd]=R[upd]; bestG[upd]=G[upd]; bestESS[upd]=ESS[upd]; besth[upd]=h[upd]

        if np.any(~np.isfinite(bestcrit)):
            bad=np.flatnonzero(~np.isfinite(bestcrit))+start
            raise InvalidCandidate(f"no valid Algorithm-2 candidate for targets {bad[:10].tolist()}")
        pred_out[start:stop]=bestpred; crit_out[start:stop]=bestcrit; K_out[start:stop]=bestK; eta_out[start:stop]=besteta; q_out[start:stop]=bestq
        R_out[start:stop]=bestR; GDF_out[start:stop]=bestG; ESS_out[start:stop]=bestESS; h_out[start:stop]=besth; nvalid_out[start:stop]=nvalid

    diagnostics={"K":K_out,"eta":eta_out,"q_h":q_out,"h":h_out,"R":R_out,"GDF":GDF_out,"ESS":ESS_out,"n_valid_candidates":nvalid_out}
    return pred_out,crit_out,diagnostics


def estimate_noise_variance_nn_nd(
    X_hist: np.ndarray,
    y_hist: np.ndarray,
    *,
    pair_fraction: float=0.50,
    min_pairs: int=20,
) -> tuple[float,int,bool]:
    """Response-only pre-target sigma^2 estimate using within-panel NN pairs.

    X_hist must be [T,N,d] in a fixed common coordinate system.  For d=0 the
    estimate reduces to unique adjacent-index pairs, but production calls this
    function once on the full four-dimensional design and reuses that estimate
    for every subset, preventing the variance estimate itself from selecting a
    variable model.
    """
    X=np.asarray(X_hist,float); y=np.asarray(y_hist,float)
    T,N,d=X.shape
    pair_d2=[]; pair_v=[]
    for t in range(T):
        if N<2: continue
        if d:
            tree=cKDTree(X[t]); dist,nn=tree.query(X[t],k=2)
            pairs={(min(i,int(j)),max(i,int(j))) for i,j in enumerate(nn[:,1]) if i!=int(j)}
        else:
            pairs={(i,i+1) for i in range(N-1)}
        for i,j in pairs:
            pair_d2.append(float(np.sum((X[t,i]-X[t,j])**2)) if d else 0.0)
            pair_v.append(0.5*float((y[t,i]-y[t,j])**2))
    if len(pair_v)<max(3,int(min_pairs)):
        raise RuntimeError("too few NN pairs for sigma2")
    d2=np.asarray(pair_d2); vals=np.asarray(pair_v); order=np.argsort(d2,kind='stable')
    keep=min(len(order),max(int(min_pairs),int(np.ceil(float(pair_fraction)*len(order)))))
    d2=d2[order[:keep]]; vals=vals[order[:keep]]
    if float(np.var(d2))<=1e-20:
        raw=float(np.mean(vals))
    else:
        Xreg=np.column_stack([np.ones(keep),d2]); beta=np.linalg.solve(Xreg.T@Xreg,Xreg.T@vals); raw=float(beta[0])
    if not np.isfinite(raw): raise RuntimeError("non-finite sigma2 estimate")
    clipped=raw<=1e-12
    return max(raw,1e-12),int(keep),bool(clipped)


def explicit_candidate_one_target(
    X_hist: np.ndarray,y_hist:np.ndarray,xstar:np.ndarray,q_h:float,eta:float,K:int,sigma2_hat:float,
    *,theta_delta:float=1e-8,w_ridge:float=1e-8,max_condition:float=1e4,
) -> dict[str,float]:
    """Slow scalar reference for acceptance/parity tests (K_state=1)."""
    X=np.asarray(X_hist,float); y=np.asarray(y_hist,float); xstar=np.asarray(xstar,float)
    T,N,d=X.shape; L=d+1
    dx=X-xstar[None,None,:] if d else np.empty((T,N,0))
    dist=np.sqrt(np.sum(dx*dx,axis=-1)) if d else np.zeros((T,N))
    h=max(float(np.quantile(dist,q_h)),1e-10); mask=dist<=h
    A,b,w,y2=panel_stats_block(dx[None,...],y,mask[None,...]); A=A[0];b=b[0];w=w[0];y2=y2[0]
    valid=_gram_valid_batch(A,w,max_condition)
    if not bool(np.all(valid)): raise InvalidCandidate("invalid scalar state")
    coeff=np.linalg.solve(A/w[:,None,None],(b/w[:,None])[...,None])[...,0]
    Z=coeff[:-1]; zs=coeff[-1]
    W=A.sum(0)/w.sum()+w_ridge*np.eye(L); diff=Z-zs
    theta=np.maximum(np.einsum('pi,ij,pj->p',diff,W,diff),0); ts=standardize_theta_batch(theta[None,:],theta_delta)[0]
    order=np.argsort(theta,kind='stable'); kk=min(max(int(K),1),T-1); keep=order[:kk]
    scores=np.exp(-float(eta)*ts); beta=np.zeros(T); beta[1+keep]=scores[keep]
    Q=float(beta@w); Ar=np.tensordot(beta,A,axes=(0,0)); br=beta@b
    if not bool(_gram_valid_batch(Ar[None],np.array([Q]),max_condition)[0]): raise InvalidCandidate("invalid scalar final")
    c=np.linalg.solve(Ar,br); R=max(float((beta@y2-2*c@br+c@Ar@c)/Q),0.0)
    V=np.tensordot(beta*beta,A,axes=(0,0)); G=float(np.trace(np.linalg.solve(Ar,V))/Q)
    return {"pred":float(c[0]),"criterion":float(R+2*sigma2_hat*G-sigma2_hat),"R":R,"GDF":G,"h":h}
