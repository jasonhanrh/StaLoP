"""Final real-data StaLoP core for county-to-county migration flows.

Statistical contract
--------------------
This module preserves the legacy separable gravity application but replaces the
legacy state-local residual predictor by the final manuscript Algorithms 1--2:

* strict one-sided predictive states H_a contained in {1,...,a-1};
* target-local local-linear states estimated with the same covariate bandwidth h;
* empirical W and quadratic predictive-state discrepancy;
* standardized discrepancy and exp(-eta * theta_tilde) panel weighting;
* hard top-K retention;
* normalized observation weights and local-linear residual prediction;
* exact GDF criterion R + 2 sigma^2 GDF - sigma^2;
* exhaustive finite-grid joint tuning over (K,h,eta), with the full state/W/
  discrepancy/ranking path rebuilt whenever h changes.

The application-specific gravity decomposition is a nuisance layer fitted from
historical data only.  StaLoP predicts the dynamic origin/destination residual
components; the edge prediction is origin + destination + gamma*log(distance).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

# Keep BLAS single-threaded.  Parallelism, if used by an outer driver, should be
# at the task level rather than nested inside linear algebra calls.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy.spatial import cKDTree


class InvalidCandidate(RuntimeError):
    """A prescribed local candidate is not estimable without changing method."""


@dataclass(frozen=True)
class RealDataConfig:
    # Exact legacy six-target evaluation protocol.
    target_bins: tuple[int, ...] = (201601, 201602, 201701, 201702, 201801, 201802)
    active_only: bool = True
    min_eval_edges: int = 50

    # H_a: most recent six semiannual panels strictly before a.
    state_window: int = 6

    # Final Algorithm-2 candidate grids.  These retain the legacy real-data
    # grid values, but h now correctly rebuilds the entire Algorithm-1 path.
    q_h_grid: tuple[float, ...] = (0.10, 0.20, 0.30)
    eta_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    rank_fraction_grid: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.70, 1.00)
    rank_count_grid: tuple[int, ...] = ()

    # Manuscript-compatible numerical settings.
    theta_delta: float = 1e-8
    w_ridge: float = 1e-8
    state_ridge: float = 0.0
    final_ridge: float = 0.0
    min_bandwidth: float = 1e-10
    max_design_condition: float = 1e6

    # Training-only variance estimator for the GDF correction.
    noise_pair_fraction: float = 0.50
    min_noise_pairs: int = 20

    # Baselines.
    time_window: int = 6
    time_min_obs: int = 3
    knn_validation_bins: int = 6
    knn_k_grid: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)

    # Diagnostics / output.
    top_periods: int = 6
    save_candidate_rows: bool = False


def validate_config(cfg: RealDataConfig) -> None:
    if cfg.state_window < 1:
        raise ValueError("state_window must be positive")
    if not cfg.q_h_grid or any(q <= 0 or q > 1 for q in cfg.q_h_grid):
        raise ValueError("q_h_grid must lie in (0,1]")
    if not cfg.eta_grid or any(e <= 0 for e in cfg.eta_grid):
        raise ValueError("eta_grid must be positive")
    if any(f <= 0 or f > 1 for f in cfg.rank_fraction_grid):
        raise ValueError("rank_fraction_grid must lie in (0,1]")
    if any(k <= 0 for k in cfg.rank_count_grid):
        raise ValueError("rank_count_grid must be positive")
    if not cfg.rank_fraction_grid and not cfg.rank_count_grid:
        raise ValueError("K grid may not be empty")
    if cfg.state_ridge != 0.0 or cfg.final_ridge != 0.0:
        raise ValueError("final manuscript implementation requires zero state/final ridge")
    if cfg.w_ridge < 0:
        raise ValueError("w_ridge must be nonnegative")
    if cfg.max_design_condition <= 1:
        raise ValueError("max_design_condition must exceed 1")
    if not 0 < cfg.noise_pair_fraction <= 1:
        raise ValueError("noise_pair_fraction must lie in (0,1]")
    if cfg.min_noise_pairs < 3:
        raise ValueError("min_noise_pairs must be at least 3")
    if cfg.time_window < 1 or cfg.time_min_obs < 1:
        raise ValueError("invalid Time-Local window")
    if not cfg.knn_k_grid or any(k < 1 for k in cfg.knn_k_grid):
        raise ValueError("knn_k_grid must be positive")


def normalize_fips5(series: pd.Series) -> pd.Series:
    """Normalize county identifiers before any unique-count or key operation."""
    s = series.astype(str).str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"[^0-9]", "", regex=True)
    missing = s.str.len().eq(0)
    out = s.str.lstrip("0").replace("", "0").str.zfill(5)
    out = out.mask(missing)
    return out


def load_binned_edges(path: str | Path) -> pd.DataFrame:
    """Load a pre-aggregated semiannual OD file with strict schema checks."""
    path = Path(path)
    required = {"tbin", "FIPS5i", "FIPS5j", "flow", "x1_i", "x2_i", "x1_j", "x2_j"}
    header = pd.read_csv(path, nrows=0)
    if not required.issubset(header.columns):
        raise RuntimeError(
            f"{path} is not a supported binned input; missing {sorted(required-set(header.columns))}"
        )
    use = ["tbin", "FIPS5i", "FIPS5j", "flow", "dist", "logdist",
           "x1_i", "x2_i", "x1_j", "x2_j", "y"]
    edge = pd.read_csv(path, usecols=lambda c: c in use, low_memory=False)
    edge["FIPS5i"] = normalize_fips5(edge["FIPS5i"])
    edge["FIPS5j"] = normalize_fips5(edge["FIPS5j"])
    edge["tbin"] = pd.to_numeric(edge["tbin"], errors="coerce")
    for col in ("flow", "x1_i", "x2_i", "x1_j", "x2_j"):
        edge[col] = pd.to_numeric(edge[col], errors="coerce")
    edge["flow"] = edge["flow"].fillna(0.0).clip(lower=0.0)
    if "logdist" in edge:
        edge["logdist"] = pd.to_numeric(edge["logdist"], errors="coerce")
    elif "dist" in edge:
        edge["dist"] = pd.to_numeric(edge["dist"], errors="coerce")
        edge["logdist"] = np.log(np.maximum(edge["dist"].to_numpy(float), 1e-8))
    else:
        raise RuntimeError("binned input must contain dist or logdist")
    if "y" in edge:
        edge["y"] = pd.to_numeric(edge["y"], errors="coerce")
    else:
        edge["y"] = np.log1p(edge["flow"].to_numpy(float))
    edge = edge.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["tbin", "FIPS5i", "FIPS5j", "x1_i", "x2_i", "x1_j", "x2_j", "logdist", "y"]
    )
    edge["tbin"] = edge["tbin"].astype(int)
    edge["key"] = edge["FIPS5i"] + "|" + edge["FIPS5j"]
    # Exact one-row-per-(bin,edge) contract.
    if edge.duplicated(["tbin", "FIPS5i", "FIPS5j"]).any():
        raise RuntimeError("binned input contains duplicate tbin/origin/destination rows")
    return edge.sort_values(["tbin", "FIPS5i", "FIPS5j"]).reset_index(drop=True)


def data_audit(edge: pd.DataFrame, target_bins: Sequence[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descriptive audit only; no model fitting."""
    bins = np.array(sorted(edge.tbin.unique()), dtype=int)
    positive = edge.loc[edge.flow > 0].groupby("tbin")["key"].apply(list).to_dict()
    active: set[str] = set()
    last = -np.inf
    if len(target_bins):
        first_target = min(target_bins)
        for b in bins[bins < first_target]:
            active.update(positive.get(int(b), [])); last = max(last, int(b))
    rows = []
    for tb in target_bins:
        for b in bins[(bins > last) & (bins < tb)]:
            active.update(positive.get(int(b), [])); last = max(last, int(b))
        tr = edge[edge.tbin < int(tb)]
        te0 = edge[edge.tbin == int(tb)]
        te = te0[te0.key.isin(active)]
        rows.append({"tbin": int(tb), "n_train": int(len(tr)), "n_target_rows": int(len(te0)),
                     "n_active_eval": int(len(te)), "n_eval_response": int(len(te))})
    per_bin = edge.groupby("tbin").size()
    summary = {
        "semiannual_rows": int(len(edge)),
        "n_origin_counties": int(edge.FIPS5i.nunique()),
        "n_destination_counties": int(edge.FIPS5j.nunique()),
        "n_bins": int(edge.tbin.nunique()),
        "first_bin": int(edge.tbin.min()),
        "last_bin": int(edge.tbin.max()),
        "rows_per_bin_mean": float(per_bin.mean()),
        "rows_per_bin_min": int(per_bin.min()),
        "rows_per_bin_max": int(per_bin.max()),
    }
    return pd.DataFrame(rows), summary


def build_side_panel(edge_train: pd.DataFrame, side: str) -> pd.DataFrame:
    """Legacy separable-gravity side panel, preserved exactly in definition."""
    if side == "origin":
        out = (edge_train.groupby(["tbin", "FIPS5i"], as_index=False)
               .agg(flow=("flow", "sum"), x1=("x1_i", "mean"), x2=("x2_i", "mean"))
               .rename(columns={"FIPS5i": "id"}))
    elif side == "dest":
        out = (edge_train.groupby(["tbin", "FIPS5j"], as_index=False)
               .agg(flow=("flow", "sum"), x1=("x1_j", "mean"), x2=("x2_j", "mean"))
               .rename(columns={"FIPS5j": "id"}))
    else:
        raise ValueError("side must be origin or dest")
    out["id"] = out["id"].astype(str)
    out["y"] = np.log1p(np.maximum(out["flow"].to_numpy(float), 0.0))
    return out.sort_values(["tbin", "id"]).reset_index(drop=True)


def target_covariates(edge_target_all: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == "origin":
        out = (edge_target_all.groupby("FIPS5i", as_index=False)
               .agg(x1=("x1_i", "mean"), x2=("x2_i", "mean"))
               .rename(columns={"FIPS5i": "id"}))
    elif side == "dest":
        out = (edge_target_all.groupby("FIPS5j", as_index=False)
               .agg(x1=("x1_j", "mean"), x2=("x2_j", "mean"))
               .rename(columns={"FIPS5j": "id"}))
    else:
        raise ValueError("side must be origin or dest")
    out["id"] = out["id"].astype(str)
    return out


def _ols_no_fallback(X: np.ndarray, y: np.ndarray, max_condition: float = 1e10) -> np.ndarray:
    A = X.T @ X
    eig = np.linalg.eigvalsh(A / max(len(X), 1))
    if not np.all(np.isfinite(eig)) or eig.min() <= 1e-12 or eig.max()/eig.min() > max_condition:
        raise RuntimeError("invalid nuisance OLS design")
    return np.linalg.solve(A, X.T @ y)


@dataclass
class SideModel:
    side: str
    panel: pd.DataFrame
    beta: np.ndarray
    alpha: pd.Series
    x_mean: np.ndarray
    x_sd: np.ndarray
    X_std: np.ndarray
    y_resid: np.ndarray
    time_index: np.ndarray  # zero-based panel index
    bins: np.ndarray
    sigma2_hat: float
    sigma2_pairs: int

    def base_prediction(self, ids: Sequence[str], x_raw: np.ndarray) -> np.ndarray:
        ids_arr = np.asarray(ids, dtype=str)
        alpha = self.alpha.reindex(ids_arr)
        if alpha.isna().any():
            missing = ids_arr[alpha.isna().to_numpy()]
            raise RuntimeError(f"active target contains side IDs absent from history: {missing[:5].tolist()}")
        X = np.column_stack([np.ones(len(x_raw)), x_raw])
        return X @ self.beta + alpha.to_numpy(float)

    def standardize_x(self, x_raw: np.ndarray) -> np.ndarray:
        return (np.asarray(x_raw, float) - self.x_mean[None, :]) / self.x_sd[None, :]


def fit_side_model(panel: pd.DataFrame, cfg: RealDataConfig, estimate_sigma2: bool = True) -> SideModel:
    dd = panel[["id", "tbin", "y", "x1", "x2"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(dd) < 20:
        raise RuntimeError("too few side-panel rows")
    Xraw = dd[["x1", "x2"]].to_numpy(float)
    Xn = np.column_stack([np.ones(len(dd)), Xraw])
    y = dd.y.to_numpy(float)
    beta = _ols_no_fallback(Xn, y)
    partial = y - Xn @ beta
    tmp = pd.DataFrame({"id": dd.id.astype(str), "partial": partial})
    alpha = tmp.groupby("id")["partial"].mean().sort_index()
    resid = partial - alpha.reindex(dd.id.astype(str)).to_numpy(float)

    mean = Xraw.mean(axis=0)
    sd = Xraw.std(axis=0, ddof=1)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 1e-12):
        raise RuntimeError("degenerate side covariate scale")
    Xstd = (Xraw - mean) / sd
    bins = np.array(sorted(dd.tbin.unique()), dtype=int)
    bmap = {int(b): j for j, b in enumerate(bins)}
    tidx = dd.tbin.map(bmap).to_numpy(int)

    sigma2, n_pairs = (np.nan, 0)
    if estimate_sigma2:
        sigma2, n_pairs = estimate_noise_variance_nn(
            Xstd, resid, tidx, len(bins), cfg.noise_pair_fraction, cfg.min_noise_pairs
        )
    dd["resid"] = resid
    return SideModel(side="", panel=dd.reset_index(drop=True), beta=beta, alpha=alpha,
                     x_mean=mean, x_sd=sd, X_std=Xstd, y_resid=resid,
                     time_index=tidx, bins=bins, sigma2_hat=float(sigma2), sigma2_pairs=int(n_pairs))


def estimate_noise_variance_nn(
    X: np.ndarray, y: np.ndarray, time_index: np.ndarray, T_hist: int,
    pair_fraction: float, min_pairs: int,
) -> tuple[float, int]:
    pair_d2: list[float] = []
    pair_v: list[float] = []
    for t in range(T_hist):
        idx = np.flatnonzero(time_index == t)
        if len(idx) < 2:
            continue
        Xt = X[idx]
        tree = cKDTree(Xt)
        dist, nn = tree.query(Xt, k=2)
        pairs = {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nn[:, 1]) if i != int(j)}
        for i, j in pairs:
            pair_d2.append(float(np.sum((Xt[i]-Xt[j])**2)))
            pair_v.append(0.5 * float((y[idx[i]]-y[idx[j]])**2))
    if len(pair_v) < max(3, min_pairs):
        raise RuntimeError("too few within-panel nearest-neighbor pairs for sigma2")
    d2 = np.asarray(pair_d2); v = np.asarray(pair_v)
    order = np.argsort(d2, kind="mergesort")
    keep = min(len(order), max(min_pairs, int(math.ceil(pair_fraction*len(order)))))
    d2 = d2[order[:keep]]; v = v[order[:keep]]
    Xreg = np.column_stack([np.ones(keep), d2])
    beta = _ols_no_fallback(Xreg, v, max_condition=1e12)
    sigma2 = float(beta[0])
    if not np.isfinite(sigma2):
        raise RuntimeError("non-finite sigma2 estimate")
    return max(sigma2, 1e-12), int(keep)


def _check_design(A_raw: np.ndarray, Q: float, cfg: RealDataConfig) -> None:
    if not np.isfinite(Q) or Q <= 0:
        raise InvalidCandidate("zero local support")
    eig = np.linalg.eigvalsh(A_raw / Q)
    if not np.all(np.isfinite(eig)) or eig.min() <= 1e-12:
        raise InvalidCandidate("rank-deficient local design")
    if eig.max()/eig.min() > cfg.max_design_condition:
        raise InvalidCandidate("ill-conditioned local design")


def _solve_stats(A: np.ndarray, b: np.ndarray, Q: float, cfg: RealDataConfig, ridge: float) -> np.ndarray:
    _check_design(A, Q, cfg)
    # state_ridge/final_ridge are both zero in the final protocol; this generic
    # expression keeps the code visibly tied to the manuscript equations.
    return np.linalg.solve(A/Q + ridge*np.eye(A.shape[0]), b/Q)


def history_indices(a_one_based: int, T_hist: int, window: int) -> np.ndarray:
    if a_one_based < 2 or a_one_based > T_hist+1:
        return np.empty(0, dtype=int)
    end = a_one_based-1
    start = max(0, end-window)
    return np.arange(start, end, dtype=int)


def _panel_sufficient_stats(
    X: np.ndarray, y: np.ndarray, time_index: np.ndarray, T_hist: int,
    x_star: np.ndarray, h: float,
) -> dict[str, np.ndarray | float]:
    d = X - x_star[None, :]
    dist = np.sqrt(np.sum(d*d, axis=1))
    kval = (dist <= h).astype(float)
    Phi = np.column_stack([np.ones(len(X)), d])
    L = Phi.shape[1]
    pw = np.bincount(time_index, weights=kval, minlength=T_hist).astype(float)
    py2 = np.bincount(time_index, weights=kval*y*y, minlength=T_hist).astype(float)
    pb = np.zeros((T_hist,L), float)
    pA = np.zeros((T_hist,L,L), float)
    for j in range(L):
        pb[:,j] = np.bincount(time_index, weights=kval*Phi[:,j]*y, minlength=T_hist)
        for k in range(j,L):
            v = np.bincount(time_index, weights=kval*Phi[:,j]*Phi[:,k], minlength=T_hist)
            pA[:,j,k]=v; pA[:,k,j]=v
    return {"dist":dist,"kval":kval,"Phi":Phi,"panel_weight":pw,"panel_y2":py2,
            "panel_b":pb,"panel_A":pA}


def _states_and_discrepancy(stats: dict[str, Any], T_hist: int, cfg: RealDataConfig) -> dict[str, Any]:
    pA=np.asarray(stats["panel_A"]); pb=np.asarray(stats["panel_b"]); pw=np.asarray(stats["panel_weight"])
    states=[]; eligible=[]
    # Historical panel labels 2,...,T_hist, exactly as in the final simulation core.
    for panel_idx in range(1,T_hist):
        H=history_indices(panel_idx+1,T_hist,cfg.state_window)
        A=pA[H].sum(axis=0); b=pb[H].sum(axis=0); Q=float(pw[H].sum())
        try:
            z=_solve_stats(A,b,Q,cfg,cfg.state_ridge)
        except (InvalidCandidate, LinAlgError):
            raise InvalidCandidate(f"invalid predictive state for historical panel index {panel_idx}")
        states.append(z); eligible.append(panel_idx)
    Ht=history_indices(T_hist+1,T_hist,cfg.state_window)
    try:
        zt=_solve_stats(pA[Ht].sum(axis=0),pb[Ht].sum(axis=0),float(pw[Ht].sum()),cfg,cfg.state_ridge)
    except (InvalidCandidate, LinAlgError) as exc:
        raise InvalidCandidate("invalid target predictive state") from exc
    states=np.asarray(states); eligible=np.asarray(eligible,dtype=int)
    Qall=float(pw.sum()); Aall=pA.sum(axis=0)
    _check_design(Aall,Qall,cfg)
    W=Aall/Qall + cfg.w_ridge*np.eye(Aall.shape[0])
    diff=states-zt[None,:]
    theta=np.maximum(np.einsum("ti,ij,tj->t",diff,W,diff),0.0)
    sd=float(np.std(theta,ddof=1)) if len(theta)>1 else 0.0
    theta_std=(theta-float(theta.mean()))/(sd+cfg.theta_delta)
    order=np.argsort(theta,kind="mergesort")
    return {"Z_hist":states,"z_target":zt,"eligible":eligible,"W":W,
            "theta":theta,"theta_std":theta_std,"order":order}


def rank_counts(n: int, cfg: RealDataConfig) -> tuple[int,...]:
    counts={min(max(int(math.floor(f*n)),1),n) for f in cfg.rank_fraction_grid}
    counts.update(min(max(int(k),1),n) for k in cfg.rank_count_grid)
    counts.add(n)
    return tuple(sorted(counts))


def _tie_secondary(row: dict[str,Any]) -> tuple[float,int,float,float]:
    eta=max(float(row["eta"]),1e-300)
    return (float(row["q_h"]),int(row["K"]),abs(math.log(eta)),eta)


def _better(row: dict[str,Any], best: dict[str,Any] | None) -> bool:
    if best is None: return True
    a=float(row["criterion"]); b=float(best["criterion"])
    tol=1e-14+1e-12*max(1.0,abs(a),abs(b))
    if a < b-tol: return True
    if abs(a-b)<=tol: return _tie_secondary(row)<_tie_secondary(best)
    return False


def evaluate_candidate_from_stats(
    stats: dict[str,Any], state: dict[str,Any], eta: float, K: int,
    sigma2: float, q_h: float, h: float, cfg: RealDataConfig,
) -> dict[str,Any]:
    pA=np.asarray(stats["panel_A"]); pb=np.asarray(stats["panel_b"]); pw=np.asarray(stats["panel_weight"])
    py2=np.asarray(stats["panel_y2"])
    elig=np.asarray(state["eligible"],int); order=np.asarray(state["order"],int)
    theta_std=np.asarray(state["theta_std"])
    K=min(max(int(K),1),len(elig)); pos=order[:K]; panels=elig[pos]
    logs=-float(eta)*theta_std[pos]; logs-=float(np.max(logs)); beta=np.exp(logs)
    Q=float(beta @ pw[panels])
    A=np.tensordot(beta,pA[panels],axes=(0,0)); b=beta @ pb[panels]
    try: c=_solve_stats(A,b,Q,cfg,cfg.final_ridge)
    except (InvalidCandidate,LinAlgError) as exc: raise InvalidCandidate("invalid final local fit") from exc
    y2=float(beta @ py2[panels])
    sse=max(y2-2*float(c@b)+float(c@A@c),0.0); R=sse/Q
    V=np.tensordot(beta*beta,pA[panels],axes=(0,0))
    GDF=float(np.trace(np.linalg.solve(A,V))/Q)
    crit=float(R+2*float(sigma2)*GDF-float(sigma2))
    obs2=float((beta*beta) @ pw[panels]); ESS=float(Q*Q/obs2)
    mass=np.zeros(len(pw),float); mass[panels]=beta*pw[panels]/Q
    return {"pred_resid":float(c[0]),"criterion":crit,"R":float(R),"GDF":GDF,
            "K":int(K),"eta":float(eta),"q_h":float(q_h),"h":float(h),"ESS":ESS,
            "period_mass":mass}


def tune_stalop_target(
    model: SideModel, x_star_raw: np.ndarray, cfg: RealDataConfig,
    return_candidates: bool=False,
) -> dict[str,Any] | tuple[dict[str,Any],pd.DataFrame]:
    """Exact exhaustive Algorithm 2 for one target side input."""
    X=model.X_std; y=model.y_resid; ti=model.time_index; T=len(model.bins)
    xstar=model.standardize_x(np.asarray(x_star_raw,float).reshape(1,2))[0]
    dist=np.sqrt(np.sum((X-xstar[None,:])**2,axis=1))
    hs=np.maximum(np.quantile(dist,np.asarray(cfg.q_h_grid,float)),cfg.min_bandwidth)
    best=None; rows=[]; n_valid_h=0; n_valid=0
    for q,h in zip(cfg.q_h_grid,hs):
        stats=_panel_sufficient_stats(X,y,ti,T,xstar,float(h))
        try: state=_states_and_discrepancy(stats,T,cfg)
        except InvalidCandidate: continue
        n_valid_h+=1
        Ks=rank_counts(len(state["eligible"]),cfg)
        for eta in cfg.eta_grid:
            for K in Ks:
                try:
                    row=evaluate_candidate_from_stats(stats,state,eta,K,model.sigma2_hat,float(q),float(h),cfg)
                except InvalidCandidate:
                    continue
                n_valid+=1
                if return_candidates:
                    rows.append({k:v for k,v in row.items() if k!="period_mass"})
                if _better(row,best): best=row
    if best is None:
        raise RuntimeError("no valid StaLoP candidate for target side input")
    best={**best,"n_valid_h":int(n_valid_h),"n_candidates":int(n_valid),"sigma2_hat":model.sigma2_hat}
    if return_candidates: return best,pd.DataFrame(rows)
    return best


def explicit_candidate_check(
    model: SideModel, x_star_raw: np.ndarray, q_h: float, eta: float, K: int,
    cfg: RealDataConfig,
) -> dict[str,float]:
    """Slow observation-level D/H reference for acceptance testing."""
    X=model.X_std; y=model.y_resid; ti=model.time_index; T=len(model.bins)
    xs=model.standardize_x(np.asarray(x_star_raw,float).reshape(1,2))[0]
    dist=np.sqrt(np.sum((X-xs)**2,axis=1)); h=max(float(np.quantile(dist,q_h)),cfg.min_bandwidth)
    stats=_panel_sufficient_stats(X,y,ti,T,xs,h); state=_states_and_discrepancy(stats,T,cfg)
    elig=state["eligible"]; order=state["order"]; theta_std=state["theta_std"]
    K=min(max(int(K),1),len(elig)); pos=order[:K]; panels=elig[pos]
    logs=-eta*theta_std[pos]; logs-=logs.max(); beta=np.exp(logs)
    beta_full=np.zeros(T); beta_full[panels]=beta
    kval=np.asarray(stats["kval"]); Phi=np.asarray(stats["Phi"])
    raw=beta_full[ti]*kval; Q=float(raw.sum()); D=np.diag(raw/Q)
    A=Phi.T@D@Phi; b=Phi.T@D@y; c=np.linalg.solve(A,b)
    resid=y-Phi@c; R=float(np.sum(np.diag(D)*resid*resid))
    H=Phi@np.linalg.solve(A,Phi.T@D); GDF=float(np.trace(D@H))
    return {"pred_resid":float(c[0]),"R":R,"GDF":GDF,
            "criterion":float(R+2*model.sigma2_hat*GDF-model.sigma2_hat)}


def tricube(u: np.ndarray) -> np.ndarray:
    a=np.abs(np.asarray(u,float)); w=np.zeros_like(a); ok=a<1; w[ok]=(1-a[ok]**3)**3; return w


def time_local_residual(model: SideModel, unit_id: str, cfg: RealDataConfig) -> tuple[float,bool]:
    dd=model.panel[model.panel.id.astype(str)==str(unit_id)].sort_values("tbin")
    y=dd.resid.to_numpy(float)
    if len(y)<cfg.time_min_obs: return 0.0,True
    y=y[-cfg.time_window:]; n=len(y)
    s=np.arange(n,dtype=float); center=s[-1]; h=max(float(n),1.0)
    w=tricube((s-center)/h); X=np.column_stack([np.ones(n),s-center])
    A=X.T@(w[:,None]*X); b=X.T@(w*y)
    try:
        eig=np.linalg.eigvalsh(A/max(w.sum(),1e-12))
        if eig.min()<=1e-12: return float(np.average(y,weights=w)),True
        c=np.linalg.solve(A,b); return float(c[0]),False
    except LinAlgError:
        return float(np.average(y,weights=w)),True


def _knn_prefix_predictions(trainX: np.ndarray, trainy: np.ndarray, queryX: np.ndarray, kgrid: Sequence[int]) -> dict[int,np.ndarray]:
    kmax=min(max(kgrid),len(trainX)); tree=cKDTree(trainX); _,idx=tree.query(queryX,k=kmax)
    if kmax==1: idx=np.asarray(idx)[:,None]
    vals=trainy[np.asarray(idx)]; csum=np.cumsum(vals,axis=1)
    return {int(k): csum[:,min(int(k),kmax)-1]/min(int(k),kmax) for k in kgrid if int(k)<=len(trainX)}


def tune_knn_k(panel: pd.DataFrame, cfg: RealDataConfig) -> tuple[int,pd.DataFrame]:
    """Training-only rolling-origin tuning of uniform-mean KNN."""
    bins=np.array(sorted(panel.tbin.unique()),int)
    eligible=bins[1:]
    vals=eligible[-cfg.knn_validation_bins:]
    accum={int(k):[0.0,0] for k in cfg.knn_k_grid}
    for v in vals:
        tr=panel[panel.tbin<v].copy(); va=panel[panel.tbin==v].copy()
        if len(tr)<20 or len(va)==0: continue
        m=fit_side_model(tr,cfg,estimate_sigma2=False)
        seen=va.id.astype(str).isin(m.alpha.index)
        va=va.loc[seen].copy()
        if len(va)==0:
            continue
        xva=va[["x1","x2"]].to_numpy(float); ids=va.id.astype(str).to_numpy()
        base=m.base_prediction(ids,xva); rtrue=va.y.to_numpy(float)-base
        q=m.standardize_x(xva)
        preds=_knn_prefix_predictions(m.X_std,m.y_resid,q,cfg.knn_k_grid)
        for k,p in preds.items():
            e=p-rtrue; accum[k][0]+=float(e@e); accum[k][1]+=len(e)
    rows=[]
    for k,(sse,n) in accum.items():
        if n>0: rows.append({"k":k,"validation_mse":sse/n,"n_validation":n})
    if not rows: raise RuntimeError("KNN rolling validation produced no valid fold")
    df=pd.DataFrame(rows).sort_values(["validation_mse","k"],kind="mergesort").reset_index(drop=True)
    return int(df.iloc[0].k),df


def predict_side_methods(
    model: SideModel, target_cov: pd.DataFrame, target_ids: Sequence[str], cfg: RealDataConfig,
    knn_k: int,
) -> tuple[dict[str,pd.DataFrame],pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    cov=target_cov.assign(id=target_cov.id.astype(str)).set_index("id")
    ids=np.asarray([str(i) for i in target_ids])
    if not set(ids).issubset(cov.index):
        raise RuntimeError("target covariates missing for active side ID")
    xr=cov.loc[ids,["x1","x2"]].to_numpy(float)
    base=model.base_prediction(ids,xr)

    global_pred=base.copy(); time_pred=np.empty(len(ids)); stalop=np.empty(len(ids))
    time_fb=[]; tuning=[]; masses=[]; candidate_frames=[]
    for j,uid in enumerate(ids):
        rt,fb=time_local_residual(model,uid,cfg); time_pred[j]=base[j]+rt; time_fb.append(fb)
        if cfg.save_candidate_rows:
            best,cand=tune_stalop_target(model,xr[j],cfg,return_candidates=True)
            cand.insert(0,"id",uid); candidate_frames.append(cand)
        else:
            best=tune_stalop_target(model,xr[j],cfg)
        stalop[j]=base[j]+float(best["pred_resid"])
        tuning.append({"id":uid,**{k:v for k,v in best.items() if k!="period_mass"}})
        pm=np.asarray(best["period_mass"])
        for t,mass in enumerate(pm):
            if mass>0:
                masses.append({"id":uid,"period":int(model.bins[t]),"mass":float(mass)})

    Xq=model.standardize_x(xr); knn_res=_knn_prefix_predictions(model.X_std,model.y_resid,Xq,(knn_k,))[knn_k]
    knn_pred=base+knn_res
    pred={m:pd.DataFrame({"id":ids,"value":v}) for m,v in
          {"global":global_pred,"time":time_pred,"knn":knn_pred,"stalop":stalop}.items()}
    tune=pd.DataFrame(tuning); mass=pd.DataFrame(masses)
    time_diag=pd.DataFrame({"id":ids,"time_fallback":np.asarray(time_fb,int)})
    candidates=pd.concat(candidate_frames,ignore_index=True) if candidate_frames else pd.DataFrame()
    return pred,tune,mass,pd.concat([time_diag],ignore_index=True),candidates


def estimate_gamma(edge_train: pd.DataFrame) -> float:
    """Legacy training-only distance coefficient, preserved."""
    mi=edge_train.groupby("FIPS5i",as_index=False).agg(mi=("y","mean"))
    mj=edge_train.groupby("FIPS5j",as_index=False).agg(mj=("y","mean"))
    tmp=edge_train.merge(mi,on="FIPS5i",how="left").merge(mj,on="FIPS5j",how="left")
    yt=tmp.y.to_numpy(float)-tmp.mi.to_numpy(float)-tmp.mj.to_numpy(float)
    x=tmp.logdist.to_numpy(float); ok=np.isfinite(x)&np.isfinite(yt)
    if ok.sum()<10: raise RuntimeError("too few rows for distance coefficient")
    x=x[ok]-x[ok].mean(); yt=yt[ok]-yt[ok].mean(); den=float(x@x)
    if den<=1e-12: raise RuntimeError("degenerate distance design")
    return float(x@yt/den)


def metrics(y: np.ndarray,p: np.ndarray) -> dict[str,float]:
    e=np.asarray(p)-np.asarray(y); out={"mspe":float(np.mean(e*e)),"rmse":float(np.sqrt(np.mean(e*e))),"mae":float(np.mean(np.abs(e)))}
    out["corr"]=float(np.corrcoef(y,p)[0,1]) if np.std(y)>1e-12 and np.std(p)>1e-12 else np.nan
    return out


def aggregate_period_mass(mass: pd.DataFrame, n_ids: int, bins: np.ndarray) -> pd.Series:
    if mass.empty: return pd.Series(0.0,index=bins,dtype=float)
    s=mass.groupby("period").mass.sum().reindex(bins,fill_value=0.0)/max(int(n_ids),1)
    # Each ID's selected observation weights sum to 1; the averaged mass must too.
    if not np.isclose(float(s.sum()),1.0,rtol=1e-8,atol=1e-10):
        raise RuntimeError(f"period mass does not sum to one: {s.sum()}")
    return s


def config_json(cfg: RealDataConfig) -> dict[str,Any]:
    d=asdict(cfg)
    return d
