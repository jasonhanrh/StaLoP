"""Final Section-6 Application 2: CES simulator calibration.

The CES DGP/calibration mechanism is unchanged from the legacy JASA code.
The StaLoP implementation is rebuilt to obey the final sequential Algorithm 1
and the final GDF/MSPE Algorithm 2 supplied on 2026-08-16.

Sequential specialization
-------------------------
The 500 simulator runs retain their generated order and are treated as
one-observation historical panels.  For response panel a, its predictive state
uses only the previous ``state_window=60`` simulator runs; the target state uses
the final 60 runs.  All local coefficients are estimated at the same evaluation
input x*, so the state vector and final local-linear coefficient use identical
basis coordinates.  The covariate bandwidth h is prespecified by the response-
free target-specific 60/500 distance quantile (a singleton H_N, explicitly
allowed by final Section 4).  K and eta are selected jointly by the exact
GDF/MSPE criterion.

Conditional on the calibrated parameter, simulator responses are deterministic.
Accordingly their observation-noise variance is exactly zero and the GDF
correction has zero multiplier in this deterministic-simulator specialization;
the same Section-4 criterion is still evaluated without inventing artificial
simulator noise.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from applications_common import PointPanelGrid, tune_point_panel_gdf, sha256_file


@dataclass(frozen=True)
class CESConfig:
    R: int = 200
    n_real: int = 300
    n_sim: int = 500
    n_eval: int = 400
    sigma_eps_values: tuple[float, ...] = (0.1, 0.2, 0.3)
    seed_real_base: int = 910000
    seed_sim_base: int = 920000
    theta_true: tuple[float, float, float] = (1.0, 0.6, -0.5)
    theta_init: tuple[float, float, float] = (1.0, 0.5, -0.3)
    rho_bounds: tuple[float, float] = (-0.95, 0.95)
    q_lo: float = 0.05
    q_hi: float = 0.95
    state_window: int = 60
    # Legacy neighborhood size 60/500 is retained as a singleton h rule.
    h_quantile: float = 0.12
    knn_frac: float = 0.10
    eta_grid: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
    retained_fraction_grid: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.70, 1.00)
    theta_delta: float = 1e-8
    w_ridge: float = 1e-8
    max_condition: float = 1e8

    def validate(self) -> None:
        if self.n_real <= 10 or self.n_sim <= self.state_window + 5:
            raise ValueError("invalid CES sample sizes/state window")
        n_side = int(round(np.sqrt(self.n_eval)))
        if n_side * n_side != self.n_eval:
            raise ValueError("n_eval must be a perfect square")
        if not (0.0 < self.h_quantile <= 1.0):
            raise ValueError("h_quantile must lie in (0,1]")
        if abs(self.h_quantile - self.state_window / self.n_sim) > 1e-12:
            raise ValueError("frozen CES h_quantile must equal state_window/n_sim")


def ces_fun(K: np.ndarray, L: np.ndarray, theta: tuple[float, float, float]) -> np.ndarray:
    A, delta, rho = theta
    K = np.asarray(K, dtype=float); L = np.asarray(L, dtype=float)
    if abs(float(rho)) < 1e-8:
        return A * (K ** delta) * (L ** (1.0 - delta))
    inside = delta * (K ** rho) + (1.0 - delta) * (L ** rho)
    if np.any(inside <= 0.0):
        raise FloatingPointError("nonpositive CES interior")
    return A * inside ** (1.0 / rho)


def simulate_real(cfg: CESConfig, rep_id: int, sigma_eps: float) -> pd.DataFrame:
    # Same seed across sigma values preserves the legacy paired DGP: the same X
    # and the same underlying standard-normal draw are scaled by sigma_eps.
    rng = np.random.default_rng(cfg.seed_real_base + int(rep_id))
    Sigma = np.array([[0.5, 0.2], [0.2, 0.5]], dtype=float)
    log_kl = rng.multivariate_normal(np.zeros(2), Sigma, size=cfg.n_real)
    K = np.exp(log_kl[:, 0]); L = np.exp(log_kl[:, 1])
    y_true = ces_fun(K, L, cfg.theta_true)
    z = rng.normal(0.0, 1.0, size=cfg.n_real)
    y = y_true * np.exp(float(sigma_eps) * z)
    return pd.DataFrame({"K": K, "L": L, "Y": y, "Y_true": y_true})


def calibrate_theta(data: pd.DataFrame, cfg: CESConfig) -> tuple[float, float, float]:
    K = data["K"].to_numpy(); L = data["L"].to_numpy(); y = data["Y"].to_numpy()
    A0, delta0, rho0 = cfg.theta_init

    def unpack(par: np.ndarray) -> tuple[float, float, float]:
        A = float(np.exp(par[0]))
        # stable logistic
        x = float(par[1])
        delta = float(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))
        return A, delta, float(par[2])

    def objective(par: np.ndarray) -> float:
        try:
            mu = ces_fun(K, L, unpack(par))
        except (FloatingPointError, ValueError, OverflowError):
            return 1e100
        if not np.all(np.isfinite(mu)):
            return 1e100
        return float(np.mean((y - mu) ** 2))

    par0 = np.array([np.log(A0), np.log(delta0 / (1.0 - delta0)), rho0], dtype=float)
    res = minimize(
        objective, par0, method="L-BFGS-B",
        bounds=[(None, None), (None, None), cfg.rho_bounds],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    if (not bool(res.success)) or (not np.all(np.isfinite(res.x))):
        raise RuntimeError(f"CES calibration failed: {res.message}")
    theta = unpack(res.x)
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("non-finite calibrated CES parameter")
    return theta


def simulate_simulator(data_real: pd.DataFrame, theta_hat: tuple[float, float, float], cfg: CESConfig, rep_id: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed_sim_base + int(rep_id))
    K_lo, K_hi = np.quantile(data_real["K"], [cfg.q_lo, cfg.q_hi])
    L_lo, L_hi = np.quantile(data_real["L"], [cfg.q_lo, cfg.q_hi])
    K = rng.uniform(K_lo, K_hi, size=cfg.n_sim)
    L = rng.uniform(L_lo, L_hi, size=cfg.n_sim)
    X = np.column_stack([K, L])
    y = ces_fun(K, L, theta_hat)
    return X, np.asarray(y, float)


def make_eval_grid(data_real: pd.DataFrame, cfg: CESConfig) -> np.ndarray:
    n_side = int(round(np.sqrt(cfg.n_eval)))
    K_grid = np.quantile(data_real["K"], np.linspace(cfg.q_lo, cfg.q_hi, n_side))
    L_grid = np.quantile(data_real["L"], np.linspace(cfg.q_lo, cfg.q_hi, n_side))
    KK, LL = np.meshgrid(K_grid, L_grid, indexing="xy")
    return np.column_stack([KK.ravel(), LL.ravel()])


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.asarray(X, float).mean(axis=0)
    scale = np.asarray(X, float).std(axis=0, ddof=1)
    if np.any((~np.isfinite(scale)) | (scale < 1e-12)):
        raise RuntimeError("degenerate CES simulator input scale")
    return (X - center) / scale, center, scale


def _check_batch_gram(A: np.ndarray, weight: np.ndarray, max_condition: float) -> np.ndarray:
    scaled = A / weight[:, None, None]
    eig = np.linalg.eigvalsh(scaled)
    lo = eig[:, 0]; hi = eig[:, -1]
    valid = np.isfinite(lo) & np.isfinite(hi) & (lo > 1e-12)
    valid &= (hi / np.maximum(lo, 1e-300)) <= float(max_condition)
    return valid


def sequential_states_one_target(
    Xs: np.ndarray,
    y: np.ndarray,
    xstar: np.ndarray,
    state_window: int,
    h_quantile: float,
    max_condition: float,
    w_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Strict pre-response states at one evaluation input.

    Historical response panel j (zero-based) uses only panels j-state_window,...,j-1.
    Eligible response panels are j=state_window,...,n-1.  The future target state
    uses the final state_window simulator panels.
    """
    Xs = np.asarray(Xs, float); y = np.asarray(y, float); xstar = np.asarray(xstar, float)
    n, d = Xs.shape; L = d + 1
    if n <= state_window:
        raise RuntimeError("state window exceeds simulator history")
    dx = Xs - xstar[None, :]
    dist = np.sqrt(np.sum(dx * dx, axis=1))
    h = max(float(np.quantile(dist, float(h_quantile))), 1e-10)
    kernel = np.exp(-0.5 * (dist / h) ** 2)
    Phi = np.column_stack([np.ones(n), dx])
    Aobs = kernel[:, None, None] * (Phi[:, :, None] * Phi[:, None, :])
    bobs = kernel[:, None] * Phi * y[:, None]
    wobs = kernel
    Ap = np.concatenate([np.zeros((1, L, L)), np.cumsum(Aobs, axis=0)], axis=0)
    bp = np.concatenate([np.zeros((1, L)), np.cumsum(bobs, axis=0)], axis=0)
    wp = np.concatenate([[0.0], np.cumsum(wobs)])

    eligible = np.arange(state_window, n, dtype=int)
    starts = eligible - state_window
    Awin = Ap[eligible] - Ap[starts]
    bwin = bp[eligible] - bp[starts]
    wwin = wp[eligible] - wp[starts]
    if np.any(wwin <= 1e-12):
        raise RuntimeError("zero CES state-kernel mass")
    valid = _check_batch_gram(Awin, wwin, max_condition)
    if not bool(np.all(valid)):
        raise RuntimeError("invalid CES historical state local design")
    states = np.linalg.solve(Awin / wwin[:, None, None], (bwin / wwin[:, None])[..., None])[..., 0]

    At = Ap[n] - Ap[n - state_window]
    bt = bp[n] - bp[n - state_window]
    wt = float(wp[n] - wp[n - state_window])
    if wt <= 1e-12:
        raise RuntimeError("zero CES target-state kernel mass")
    eval_t = np.linalg.eigvalsh(At / wt)
    if float(eval_t[0]) <= 1e-12 or float(eval_t[-1] / eval_t[0]) > max_condition:
        raise RuntimeError("invalid CES target-state local design")
    zstar = np.linalg.solve(At / wt, bt / wt)

    W = (Phi.T @ (kernel[:, None] * Phi)) / float(kernel.sum()) + float(w_ridge) * np.eye(L)
    return states, zstar, eligible, kernel, W, h


def predict_global(X: np.ndarray, y: np.ndarray, Xe: np.ndarray) -> np.ndarray:
    Phi = np.column_stack([np.ones(len(X)), X])
    eig = np.linalg.eigvalsh(Phi.T @ Phi / len(Phi))
    if float(eig[0]) <= 1e-12:
        raise RuntimeError("rank-deficient CES global baseline")
    beta = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
    return np.column_stack([np.ones(len(Xe)), Xe]) @ beta


def predict_knn(Xs: np.ndarray, y: np.ndarray, Xes: np.ndarray, frac: float) -> np.ndarray:
    k = min(max(int(np.ceil(float(frac) * len(y))), 5), len(y))
    tree = cKDTree(Xs)
    _, idx = tree.query(Xes, k=k)
    if k == 1:
        idx = idx[:, None]
    return np.mean(y[np.asarray(idx, dtype=int)], axis=1)


def predict_stalop(
    Xs: np.ndarray, y: np.ndarray, Xes: np.ndarray, cfg: CESConfig
) -> tuple[np.ndarray, pd.DataFrame, float]:
    grid = PointPanelGrid(
        eta_grid=cfg.eta_grid,
        rank_fraction_grid=cfg.retained_fraction_grid,
    )
    pred = np.empty(len(Xes), dtype=float)
    tuning_rows: list[dict] = []
    tic = perf_counter()
    for i, xstar in enumerate(Xes):
        states, zstar, eligible, kernel_all, W, h = sequential_states_one_target(
            Xs, y, xstar, cfg.state_window, cfg.h_quantile,
            cfg.max_condition, cfg.w_ridge,
        )
        best = tune_point_panel_gdf(
            X=Xs[eligible], y=y[eligible],
            states_target_coordinates=states,
            x_star=np.asarray(xstar, float), z_star=zstar,
            sigma2_hat=0.0,
            grid=grid,
            theta_delta=cfg.theta_delta,
            w_ridge=cfg.w_ridge,
            max_condition=cfg.max_condition,
            kernel_weight=kernel_all[eligible],
            W_override=W,
        )
        pred[i] = float(best["pred"])
        tuning_rows.append({
            "eval_id": int(i), "K": int(best["K"]), "eta": float(best["eta"]),
            "h": float(h), "h_quantile": float(cfg.h_quantile),
            "criterion": float(best["criterion"]), "R": float(best["R"]),
            "GDF": float(best["GDF"]), "ESS": float(best["ESS"]),
            "sigma2_hat": 0.0, "n_eligible": int(len(eligible)),
        })
    return pred, pd.DataFrame(tuning_rows), float(perf_counter() - tic)


def run_one_rep(cfg: CESConfig, sigma_eps: float, rep_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_real = simulate_real(cfg, rep_id, sigma_eps)
    theta_hat = calibrate_theta(data_real, cfg)
    Xsim, ysim = simulate_simulator(data_real, theta_hat, cfg, rep_id)
    Xeval = make_eval_grid(data_real, cfg)
    truth = ces_fun(Xeval[:, 0], Xeval[:, 1], cfg.theta_true)
    Xs, center, scale = standardize(Xsim)
    Xes = (Xeval - center) / scale

    pred_global = predict_global(Xsim, ysim, Xeval)
    pred_knn = predict_knn(Xs, ysim, Xes, cfg.knn_frac)
    pred_stalop, tuning, stalop_runtime = predict_stalop(Xs, ysim, Xes, cfg)
    preds = {
        "Global Linear": pred_global,
        "KNN-Covariate": pred_knn,
        "StaLoP": pred_stalop,
    }
    rows = []
    for method, p in preds.items():
        se = (np.asarray(p) - truth) ** 2
        rows.append({
            "rep_id": int(rep_id), "sigma_eps": float(sigma_eps), "method": method,
            "mspe": float(np.mean(se)), "rmse": float(np.sqrt(np.mean(se))),
            "theta_A": float(theta_hat[0]), "theta_delta": float(theta_hat[1]),
            "theta_rho": float(theta_hat[2]),
            "runtime_sec": stalop_runtime if method == "StaLoP" else np.nan,
        })
    tuning.insert(0, "rep_id", int(rep_id)); tuning.insert(1, "sigma_eps", float(sigma_eps))
    return pd.DataFrame(rows), tuning


def summarize(raw: pd.DataFrame, tuning: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = raw.groupby(["sigma_eps", "method"], as_index=False).agg(
        mspe_mean=("mspe", "mean"),
        mspe_sd=("mspe", lambda x: float(np.std(x, ddof=1))),
        mspe_mcse=("mspe", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))),
        rmse_mean=("rmse", "mean"), n_rep=("mspe", "size"),
    )
    stalop = raw[raw.method == "StaLoP"][["rep_id", "sigma_eps", "mspe"]].rename(columns={"mspe":"stalop_mspe"})
    wins=[]
    for method in ("Global Linear", "KNN-Covariate"):
        other=raw[raw.method==method][["rep_id","sigma_eps","mspe"]].rename(columns={"mspe":"other_mspe"})
        d=stalop.merge(other,on=["rep_id","sigma_eps"])
        for sig,g in d.groupby("sigma_eps"):
            wins.append({"sigma_eps":float(sig),"baseline":method,"win_rate":float((g.stalop_mspe<g.other_mspe).mean()),"n":len(g)})
    tune_summary=tuning.groupby("sigma_eps",as_index=False).agg(
        K_mean=("K","mean"),K_median=("K","median"),eta_mean=("eta","mean"),eta_median=("eta","median"),
        h_mean=("h","mean"),ESS_mean=("ESS","mean"),GDF_mean=("GDF","mean"),criterion_mean=("criterion","mean")
    )
    return summary,pd.DataFrame(wins),tune_summary


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--R",type=int,default=200)
    ap.add_argument("--n-jobs",type=int,default=6)
    ap.add_argument("--n-eval",type=int,default=400)
    ap.add_argument("--outdir",type=str,default="application2_ces_R200")
    ap.add_argument("--formal",action="store_true")
    args=ap.parse_args()
    R=200 if args.formal else int(args.R)
    cfg=CESConfig(R=R,n_eval=int(args.n_eval)); cfg.validate()
    tasks=[(float(sig),r) for sig in cfg.sigma_eps_values for r in range(1,R+1)]
    n_jobs=max(1,min(int(args.n_jobs),len(tasks),mp.cpu_count() or 1))
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    tic=perf_counter(); raws=[]; tunes=[]
    if n_jobs==1:
        for done,(sig,r) in enumerate(tasks,1):
            a,b=run_one_rep(cfg,sig,r); raws.append(a); tunes.append(b)
            if done%10==0 or done==len(tasks): print(f"Completed {done}/{len(tasks)}",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            fut={ex.submit(run_one_rep,cfg,sig,r):(sig,r) for sig,r in tasks}
            for done,f in enumerate(as_completed(fut),1):
                a,b=f.result(); raws.append(a); tunes.append(b)
                if done%10==0 or done==len(tasks): print(f"Completed {done}/{len(tasks)}",flush=True)
    raw=pd.concat(raws,ignore_index=True).sort_values(["sigma_eps","rep_id","method"])
    tuning=pd.concat(tunes,ignore_index=True).sort_values(["sigma_eps","rep_id","eval_id"])
    summary,wins,tune_summary=summarize(raw,tuning)
    raw.to_csv(out/"application2_ces_raw.csv",index=False)
    tuning.to_csv(out/"application2_ces_tuning.csv",index=False)
    summary.to_csv(out/"application2_ces_summary.csv",index=False)
    wins.to_csv(out/"application2_ces_winrate.csv",index=False)
    tune_summary.to_csv(out/"application2_ces_tuning_summary.csv",index=False)
    manifest={"application":"ces","config":asdict(cfg),"R":R,"valid_tasks":int(raw[["sigma_eps","rep_id"]].drop_duplicates().shape[0]),
              "wall_seconds":float(perf_counter()-tic),"source_sha256":sha256_file(__file__)}
    (out/"application2_ces_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(summary.to_string(index=False)); print(tune_summary.to_string(index=False)); print(f"wall_seconds={manifest['wall_seconds']:.3f}")


if __name__=="__main__":
    mp.freeze_support(); main()
