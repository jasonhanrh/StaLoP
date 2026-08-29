"""
Experiment 2 core for StaLoP inference.

Controlled validation of the FINAL Section-3 joint response-adaptive
sandwich covariance against a naive fixed-weight covariance.

This module is standalone. It does NOT import or modify Experiment 1 code
and it performs no Algorithm-2 tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import math
import time
import numpy as np

CODE_VERSION = "EX2_FINAL_V1_20260816"

# Standard lower-column vech order for a 3x3 symmetric matrix:
# (0,0), (1,0), (2,0), (1,1), (2,1), (2,2)
VECH_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 0), (1, 0), (2, 0), (1, 1), (2, 1), (2, 2)
)


@dataclass(frozen=True)
class Experiment2Config:
    T: int = 120
    eligible_panels_1based: Tuple[int, ...] = (21, 41, 61, 81, 101)
    history_panels_1based: Tuple[int, ...] = (20, 40, 60, 80, 100)
    target_history_panel_1based: int = 120
    target_panel_1based: int = 121

    # Exact linear DGP.
    state_base_intercept: float = -0.25
    target_base_intercept: float = 0.50
    slope_x1: float = 0.80
    slope_x2: float = -0.50
    state_offsets: Tuple[float, ...] = (-0.60, -0.30, 0.0, 0.30, 0.60)
    bridge_slope: float = 0.75
    sigma_eps: float = 1.0

    # FINAL StaLoP fixed inference configuration. No EX2 tuning.
    eta: float = 1.0
    delta_theta: float = 1.0e-6
    bandwidth_scale: float = 1.50

    # Simultaneous inference.
    alpha: float = 0.10
    grid_size: int = 15
    grid_x1_min: float = -0.75
    grid_x1_max: float = 0.75
    grid_x2: float = 0.0
    gaussian_draws: int = 4096
    gaussian_seed: int = 481516234

    # Hard numerical gates. No silent ridge/pseudoinverse/fallback.
    max_gram_condition: float = 1.0e10
    min_variance: float = 1.0e-18
    derivative_sd_floor: float = 1.0e-14

    def validate(self) -> None:
        if self.T != 120:
            raise ValueError("The frozen Experiment-2 design requires T=120.")
        if len(self.eligible_panels_1based) != 5 or len(self.history_panels_1based) != 5:
            raise ValueError("Exactly five eligible/history panels are required.")
        if len(self.state_offsets) != 5:
            raise ValueError("Exactly five state offsets are required.")
        if not np.allclose(np.asarray(self.state_offsets), -np.asarray(self.state_offsets)[::-1], rtol=0, atol=0):
            raise ValueError("state_offsets must be exactly symmetric around zero.")
        for h, t in zip(self.history_panels_1based, self.eligible_panels_1based):
            if h >= t:
                raise ValueError("Every predictive-state history must precede its panel.")
        if self.target_history_panel_1based >= self.target_panel_1based:
            raise ValueError("Target state must use only pre-target responses.")
        if max(self.eligible_panels_1based) > self.T or self.target_history_panel_1based > self.T:
            raise ValueError("Historical panel exceeds T.")
        if self.delta_theta <= 0 or self.eta <= 0:
            raise ValueError("eta and delta_theta must be positive.")
        if self.grid_size < 3 or self.grid_size % 2 != 1:
            raise ValueError("Use an odd grid_size >=3 so x=0 is included.")
        if not (0 < self.alpha < 1):
            raise ValueError("alpha must lie in (0,1).")


DEFAULT_CONFIG = Experiment2Config()


def bandwidth_for_N(N: int, cfg: Experiment2Config = DEFAULT_CONFIG) -> float:
    """Prespecified d=2 local-linear rate h_N = c_h N^{-1/(d+4)}."""
    if N <= 0:
        raise ValueError("N must be positive.")
    return float(cfg.bandwidth_scale * (float(N) ** (-1.0 / 6.0)))


def target_grid(cfg: Experiment2Config = DEFAULT_CONFIG) -> np.ndarray:
    x1 = np.linspace(cfg.grid_x1_min, cfg.grid_x1_max, cfg.grid_size)
    x2 = np.full(cfg.grid_size, cfg.grid_x2, dtype=float)
    return np.column_stack([x1, x2])


def gaussian_kernel(X: np.ndarray, xstar: np.ndarray, h: float) -> np.ndarray:
    """Unnormalized isotropic Gaussian kernel; common constants cancel."""
    D = (X - xstar[None, :]) / h
    return np.exp(-0.5 * np.einsum("ij,ij->i", D, D, optimize=True))


def local_basis(X: np.ndarray, xstar: np.ndarray) -> np.ndarray:
    out = np.empty((X.shape[0], 3), dtype=float)
    out[:, 0] = 1.0
    out[:, 1:] = X - xstar[None, :]
    return out


def vech_matrix_rows(phi: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Rows are vech(phi_i phi_i' - W) in VECH_PAIRS order."""
    out = np.empty((phi.shape[0], 6), dtype=float)
    for j, (r, c) in enumerate(VECH_PAIRS):
        out[:, j] = phi[:, r] * phi[:, c] - W[r, c]
    return out


def theta_derivative_wrt_vech(delta: np.ndarray) -> np.ndarray:
    out = np.empty(6, dtype=float)
    for j, (r, c) in enumerate(VECH_PAIRS):
        out[j] = delta[r] * delta[c] if r == c else 2.0 * delta[r] * delta[c]
    return out


def _solve_checked(G: np.ndarray, rhs: np.ndarray, max_cond: float, label: str):
    cond = float(np.linalg.cond(G))
    if not np.isfinite(cond) or cond > max_cond:
        raise np.linalg.LinAlgError(f"{label} ill-conditioned: cond={cond}")
    sol = np.linalg.solve(G, rhs)
    if not np.all(np.isfinite(sol)):
        raise FloatingPointError(f"{label} solve produced non-finite values.")
    return sol, cond


def weighted_local_linear(X, y, xstar, h, max_cond):
    phi = local_basis(X, xstar)
    k = gaussian_kernel(X, xstar, h)
    G = phi.T @ (k[:, None] * phi)
    rhs = phi.T @ (k * y)
    coef, cond = _solve_checked(G, rhs, max_cond, "local-linear Gram")
    return coef, phi, k, cond


def weighted_W_from_all(Xflat, xstar, h, max_cond):
    phi = local_basis(Xflat, xstar)
    k = gaussian_kernel(Xflat, xstar, h)
    denom = float(k.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise FloatingPointError("Nonpositive W-kernel denominator.")
    W = (phi.T @ (k[:, None] * phi)) / denom
    cond = float(np.linalg.cond(W))
    if not np.isfinite(cond) or cond > max_cond:
        raise np.linalg.LinAlgError(f"W ill-conditioned: cond={cond}")
    if abs(W[0, 0] - 1.0) > 2e-13:
        raise AssertionError("W[0,0] must equal one numerically.")
    return W, phi, k, cond


def standardized_discrepancy(theta, delta_theta, sd_floor):
    """Returns standardized theta, its Jacobian wrt theta, and sample SD."""
    m = theta.size
    if m < 2:
        raise ValueError("Need at least two discrepancies.")
    centered = theta - float(theta.mean())
    s2 = float(centered @ centered) / float(m - 1)
    s = math.sqrt(max(s2, 0.0))
    D = s + delta_theta
    P = np.eye(m) - np.ones((m, m), dtype=float) / float(m)
    if s > sd_floor:
        ds_dtheta = centered / (float(m - 1) * s)
    else:
        ds_dtheta = np.zeros(m, dtype=float)
    J = P / D - np.outer(centered, ds_dtheta) / (D * D)
    z = centered / D
    return z, J, s


def softmax_weights_and_jacobian(ztheta, eta):
    logits = -eta * ztheta
    logits -= float(logits.max())
    e = np.exp(logits)
    alpha = e / float(e.sum())
    J = -eta * (np.diag(alpha) - np.outer(alpha, alpha))
    return alpha, J


def alpha_and_derivative(states, W, eta, delta_theta, sd_floor):
    """
    states shape (6,3): first 5 eligible pre-states, last target pre-state.
    gamma stacks six state vectors (18) then vech(W) (6): total 24.
    """
    if states.shape != (6, 3):
        raise ValueError("states must have shape (6,3).")
    J_theta_gamma = np.zeros((5, 24), dtype=float)
    theta = np.empty(5, dtype=float)
    z0 = states[5]
    for j in range(5):
        d = states[j] - z0
        theta[j] = float(d @ W @ d)
        grad_z = 2.0 * (W @ d)
        J_theta_gamma[j, 3*j:3*(j+1)] = grad_z
        J_theta_gamma[j, 15:18] = -grad_z
        J_theta_gamma[j, 18:24] = theta_derivative_wrt_vech(d)
    ztheta, J_std_theta, theta_sd = standardized_discrepancy(theta, delta_theta, sd_floor)
    alpha, J_soft = softmax_weights_and_jacobian(ztheta, eta)
    J_alpha_gamma = J_soft @ J_std_theta @ J_theta_gamma
    if abs(float(alpha.sum()) - 1.0) > 5e-14:
        raise AssertionError("Panel weights do not sum to one.")
    if np.max(np.abs(J_alpha_gamma.sum(axis=0))) > 5e-11:
        raise AssertionError("Softmax derivative violates simplex identity.")
    return alpha, theta, ztheta, J_alpha_gamma, theta_sd


def analytical_population_balance(cfg=DEFAULT_CONFIG):
    gamma = np.asarray(cfg.state_offsets, dtype=float)
    theta = gamma * gamma  # W00=1; population state differences are intercept-only.
    ztheta, _, _ = standardized_discrepancy(theta, cfg.delta_theta, cfg.derivative_sd_floor)
    alpha, _ = softmax_weights_and_jacobian(ztheta, cfg.eta)
    weighted_offset = float(alpha @ gamma)
    return {
        "weighted_state_offset": weighted_offset,
        "weighted_next_panel_offset": cfg.bridge_slope * weighted_offset,
        "max_pair_weight_asymmetry": float(max(abs(alpha[0]-alpha[4]), abs(alpha[1]-alpha[3]))),
    }


def _panel_global_coef(panel_1based, cfg):
    if panel_1based in cfg.history_panels_1based:
        j = cfg.history_panels_1based.index(panel_1based)
        intercept = cfg.state_base_intercept + cfg.state_offsets[j]
    elif panel_1based == cfg.target_history_panel_1based:
        intercept = cfg.state_base_intercept
    elif panel_1based in cfg.eligible_panels_1based:
        j = cfg.eligible_panels_1based.index(panel_1based)
        intercept = cfg.target_base_intercept + cfg.bridge_slope * cfg.state_offsets[j]
    elif panel_1based == cfg.target_panel_1based:
        intercept = cfg.target_base_intercept
    else:
        intercept = cfg.state_base_intercept
    return np.asarray([intercept, cfg.slope_x1, cfg.slope_x2], dtype=float)


def target_truth(grid, cfg=DEFAULT_CONFIG):
    return cfg.target_base_intercept + cfg.slope_x1*grid[:,0] + cfg.slope_x2*grid[:,1]


def simulate_replication(N, rep, seed0, cfg=DEFAULT_CONFIG):
    """Generate all X (needed by W), but Y only for panels entering z or c."""
    cfg.validate()
    ss = np.random.SeedSequence([int(seed0), int(N), int(rep)])
    rng = np.random.default_rng(ss)
    X = rng.standard_normal((cfg.T, N, 2), dtype=np.float64)
    needed = sorted(set(cfg.history_panels_1based) | {cfg.target_history_panel_1based} | set(cfg.eligible_panels_1based))
    Y: Dict[int, np.ndarray] = {}
    for p in needed:
        b = _panel_global_coef(p, cfg)
        Xp = X[p-1]
        eps = rng.normal(0.0, cfg.sigma_eps, size=N)
        Y[p] = b[0] + Xp @ b[1:] + eps
    return {"X": X, "Y": Y}


def _build_state_blocks(X, Y, xstar, h, cfg):
    states = np.empty((6,3), dtype=float)
    phis=[]; ks=[]; resids=[]; Gmeans=[]; conds=[]
    hist = list(cfg.history_panels_1based) + [cfg.target_history_panel_1based]
    N = X.shape[1]
    for j,p in enumerate(hist):
        coef,phi,k,cond = weighted_local_linear(X[p-1],Y[p],xstar,h,cfg.max_gram_condition)
        states[j] = coef
        resid = Y[p] - phi @ coef
        Gmean = (phi.T @ (k[:,None]*phi)) / float(N)
        phis.append(phi); ks.append(k); resids.append(resid); Gmeans.append(Gmean); conds.append(cond)
    return states,phis,ks,resids,Gmeans,conds


def _fit_final(X,Y,xstar,h,alpha,cfg):
    G=np.zeros((3,3)); rhs=np.zeros(3); phis=[]; ks=[]
    for j,p in enumerate(cfg.eligible_panels_1based):
        phi=local_basis(X[p-1],xstar); k=gaussian_kernel(X[p-1],xstar,h)
        G += alpha[j]*(phi.T @ (k[:,None]*phi))
        rhs += alpha[j]*(phi.T @ (k*Y[p]))
        phis.append(phi); ks.append(k)
    c,cond=_solve_checked(G,rhs,cfg.max_gram_condition,"final StaLoP Gram")
    resids=[Y[p]-phis[j]@c for j,p in enumerate(cfg.eligible_panels_1based)]
    return c,phis,ks,resids,cond


def infer_one_point(X,Y,xstar,h,cfg=DEFAULT_CONFIG,return_debug=False):
    """Fit StaLoP and compute full/naive scalar influence vectors at x*."""
    T,N,d=X.shape
    if T!=cfg.T or d!=2: raise ValueError("Unexpected X dimensions.")
    M=T*N; Xflat=X.reshape(M,2)
    W,phi_all,k_all,cond_W=weighted_W_from_all(Xflat,xstar,h,cfg.max_gram_condition)
    states,state_phi,state_k,state_resid,state_G,state_cond=_build_state_blocks(X,Y,xstar,h,cfg)
    alpha,theta,ztheta,J_alpha_gamma,theta_sd=alpha_and_derivative(states,W,cfg.eta,cfg.delta_theta,cfg.derivative_sd_floor)
    c,final_phi,final_k,final_resid,cond_c=_fit_final(X,Y,xstar,h,alpha,cfg)

    # A_gamma,gamma: six independent z blocks + W block.
    Agg=np.zeros((24,24))
    for j in range(6):
        Agg[3*j:3*(j+1),3*j:3*(j+1)] = -state_G[j]
    mean_k_all=float(k_all.mean())
    Agg[18:24,18:24] = -mean_k_all*np.eye(6)

    # Final equation Jacobian.
    Acc=np.zeros((3,3)); Acg=np.zeros((3,24)); individual_S=[]
    for j in range(5):
        phi=final_phi[j]; k=final_k[j]; r=final_resid[j]
        Gt=phi.T @ (k[:,None]*phi)
        Acc -= (alpha[j]/float(M))*Gt
        St=phi.T @ (k*r)
        individual_S.append(St)
        Acg += np.outer(St,J_alpha_gamma[j]) / float(M)

    q=np.array([1.0,0.0,0.0])
    q_Acc_inv=np.linalg.solve(Acc.T,q)     # q' A_cc^{-1}
    row_c=-q_Acc_inv
    tmp=q_Acc_inv @ Acg
    row_gamma=np.linalg.solve(Agg.T,tmp)   # q'A_cc^{-1}A_cg A_gg^{-1}

    IF_full=np.zeros(M); IF_naive=np.zeros(M)
    scoreW=k_all[:,None]*vech_matrix_rows(phi_all,W)
    IF_full += scoreW @ row_gamma[18:24]

    history_all=list(cfg.history_panels_1based)+[cfg.target_history_panel_1based]
    state_scale=float(M)/float(N)  # N_total/N_a = T
    for j,p in enumerate(history_all):
        sl=slice((p-1)*N,p*N)
        psi_z=state_scale*(state_k[j][:,None]*state_phi[j]*state_resid[j][:,None])
        IF_full[sl] += psi_z @ row_gamma[3*j:3*(j+1)]

    for j,p in enumerate(cfg.eligible_panels_1based):
        sl=slice((p-1)*N,p*N)
        psi_c=alpha[j]*(final_k[j][:,None]*final_phi[j]*final_resid[j][:,None])
        direct=psi_c @ row_c
        IF_full[sl] += direct
        IF_naive[sl] += direct

    var_full=float(IF_full@IF_full)/float(M*M)
    var_naive=float(IF_naive@IF_naive)/float(M*M)
    if (not np.isfinite(var_full) or not np.isfinite(var_naive) or
        var_full<=cfg.min_variance or var_naive<=cfg.min_variance):
        raise FloatingPointError(f"Invalid variances full={var_full}, naive={var_naive}")

    root_z=[]
    for j in range(6):
        root_z.append(float(np.max(np.abs((state_k[j][:,None]*state_phi[j]*state_resid[j][:,None]).mean(axis=0)))))
    root_W=float(np.max(np.abs(scoreW.mean(axis=0))))
    root_c_vec=np.zeros(3)
    for j in range(5): root_c_vec += alpha[j]*individual_S[j]
    root_c=float(np.max(np.abs(root_c_vec/float(M))))

    out={
        "mhat":float(c[0]), "se_full":math.sqrt(var_full), "se_naive":math.sqrt(var_naive),
        "IF_full":IF_full, "IF_naive":IF_naive, "alpha":alpha, "theta":theta,
        "ztheta":ztheta, "theta_sd":float(theta_sd), "states":states, "W":W, "c":c,
        "condition_W":cond_W, "condition_state_max":float(max(state_cond)), "condition_final":cond_c,
        "root_z_max":float(max(root_z)), "root_W_max":root_W, "root_c_max":root_c,
        "Acg_norm":float(np.linalg.norm(Acg)), "indirect_row_norm":float(np.linalg.norm(row_gamma)),
        "J_alpha_gamma_norm":float(np.linalg.norm(J_alpha_gamma)),
    }
    if return_debug:
        out.update({"Agg":Agg,"Acc":Acc,"Acg":Acg,"J_alpha_gamma":J_alpha_gamma,
                    "state_phi":state_phi,"state_k":state_k,"state_resid":state_resid,
                    "final_phi":final_phi,"final_k":final_k,"final_resid":final_resid,
                    "phi_all":phi_all,"k_all":k_all})
    return out


def explicit_joint_sandwich_debug(X,Y,xstar,h,cfg=DEFAULT_CONFIG):
    """Slow audit path: explicit observation x 27 score and full sandwich."""
    base=infer_one_point(X,Y,xstar,h,cfg,return_debug=True)
    T,N,_=X.shape; M=T*N; P=27
    Psi=np.zeros((M,P))
    history_all=list(cfg.history_panels_1based)+[cfg.target_history_panel_1based]
    scale=float(M)/float(N)
    for j,p in enumerate(history_all):
        sl=slice((p-1)*N,p*N)
        Psi[sl,3*j:3*(j+1)] = scale*(base["state_k"][j][:,None]*base["state_phi"][j]*base["state_resid"][j][:,None])
    scoreW=base["k_all"][:,None]*vech_matrix_rows(base["phi_all"],base["W"])
    Psi[:,18:24]=scoreW
    alpha=base["alpha"]
    for j,p in enumerate(cfg.eligible_panels_1based):
        sl=slice((p-1)*N,p*N)
        Psi[sl,24:27]=alpha[j]*(base["final_k"][j][:,None]*base["final_phi"][j]*base["final_resid"][j][:,None])
    A=np.zeros((P,P)); A[:24,:24]=base["Agg"]; A[24:27,:24]=base["Acg"]; A[24:27,24:27]=base["Acc"]
    Ainv=np.linalg.inv(A); B=(Psi.T@Psi)/float(M*M); V=Ainv@B@Ainv.T
    q=np.array([1.,0.,0.]); vfull=float(q@V[24:27,24:27]@q)
    Bcc=(Psi[:,24:27].T@Psi[:,24:27])/float(M*M); Acci=np.linalg.inv(base["Acc"])
    vnaive=float(q@(Acci@Bcc@Acci.T)@q)
    return {"var_full_fast":base["se_full"]**2,"var_full_explicit":vfull,
            "var_naive_fast":base["se_naive"]**2,"var_naive_explicit":vnaive,
            "score_root_max":float(np.max(np.abs(Psi.mean(axis=0)))),"A_condition":float(np.linalg.cond(A))}


def finite_difference_alpha_derivative(states,W,cfg=DEFAULT_CONFIG,eps=1e-6):
    _,_,_,Ja,_=alpha_and_derivative(states,W,cfg.eta,cfg.delta_theta,cfg.derivative_sd_floor)
    g0=np.concatenate([states.reshape(-1),np.array([W[r,c] for r,c in VECH_PAIRS])])
    def unpack(g):
        st=g[:18].reshape(6,3).copy(); vals=g[18:24]; W2=W.copy()
        for val,(r,c) in zip(vals,VECH_PAIRS): W2[r,c]=val; W2[c,r]=val
        return st,W2
    Jfd=np.empty_like(Ja)
    for k in range(g0.size):
        gp=g0.copy(); gm=g0.copy(); gp[k]+=eps; gm[k]-=eps
        sp,Wp=unpack(gp); sm,Wm=unpack(gm)
        ap=alpha_and_derivative(sp,Wp,cfg.eta,cfg.delta_theta,cfg.derivative_sd_floor)[0]
        am=alpha_and_derivative(sm,Wm,cfg.eta,cfg.delta_theta,cfg.derivative_sd_floor)[0]
        Jfd[:,k]=(ap-am)/(2*eps)
    return {"max_abs_error":float(np.max(np.abs(Jfd-Ja))),
            "relative_fro_error":float(np.linalg.norm(Jfd-Ja)/max(np.linalg.norm(Jfd),1e-15))}


def _nearest_psd_correlation(Sigma,min_eig=1e-12):
    d=np.sqrt(np.maximum(np.diag(Sigma),min_eig)); R=Sigma/np.outer(d,d); R=0.5*(R+R.T); np.fill_diagonal(R,1.)
    w,V=np.linalg.eigh(R); min_before=float(w.min()); w=np.maximum(w,min_eig)
    R2=(V*w)@V.T; dd=np.sqrt(np.diag(R2)); R2=R2/np.outer(dd,dd); R2=0.5*(R2+R2.T); np.fill_diagonal(R2,1.)
    return R2,min_before


def gaussian_max_critical(Sigma,base_draws,alpha):
    R,min_before=_nearest_psd_correlation(Sigma); w,V=np.linalg.eigh(R); w=np.maximum(w,0.)
    sqrtR=V@np.diag(np.sqrt(w))@V.T; G=base_draws@sqrtR.T; maxima=np.max(np.abs(G),axis=1)
    return float(np.quantile(maxima,1-alpha,method="linear")),min_before


def fixed_gaussian_draws(cfg=DEFAULT_CONFIG):
    rng=np.random.default_rng(cfg.gaussian_seed)
    return rng.standard_normal((cfg.gaussian_draws,cfg.grid_size),dtype=np.float64)


def run_one_replication(N,rep,seed0,cfg=DEFAULT_CONFIG,base_draws:Optional[np.ndarray]=None):
    t0=time.perf_counter(); sim=simulate_replication(N,rep,seed0,cfg); X=sim["X"]; Y=sim["Y"]
    grid=target_grid(cfg); truth=target_truth(grid,cfg); h=bandwidth_for_N(N,cfg); M=cfg.T*N; G=cfg.grid_size
    if base_draws is None: base_draws=fixed_gaussian_draws(cfg)
    mhat=np.empty(G); se_full=np.empty(G); se_naive=np.empty(G)
    IF_full=np.empty((G,M)); IF_naive=np.empty((G,M)); alpha_mat=np.empty((G,5)); theta_mat=np.empty((G,5))
    diag={"condition_W_max":0.,"condition_state_max":0.,"condition_final_max":0.,"root_score_max":0.,
          "Acg_norm_mean":0.,"indirect_row_norm_mean":0.,"J_alpha_gamma_norm_mean":0.}
    for g in range(G):
        a=infer_one_point(X,Y,grid[g],h,cfg)
        mhat[g]=a["mhat"]; se_full[g]=a["se_full"]; se_naive[g]=a["se_naive"]
        IF_full[g]=a["IF_full"]; IF_naive[g]=a["IF_naive"]; alpha_mat[g]=a["alpha"]; theta_mat[g]=a["theta"]
        diag["condition_W_max"]=max(diag["condition_W_max"],a["condition_W"])
        diag["condition_state_max"]=max(diag["condition_state_max"],a["condition_state_max"])
        diag["condition_final_max"]=max(diag["condition_final_max"],a["condition_final"])
        diag["root_score_max"]=max(diag["root_score_max"],a["root_z_max"],a["root_W_max"],a["root_c_max"])
        diag["Acg_norm_mean"]+=a["Acg_norm"]/G; diag["indirect_row_norm_mean"]+=a["indirect_row_norm"]/G
        diag["J_alpha_gamma_norm_mean"]+=a["J_alpha_gamma_norm"]/G
    Sigma_full=(IF_full@IF_full.T)/float(M*M); Sigma_naive=(IF_naive@IF_naive.T)/float(M*M)
    if np.max(np.abs(np.diag(Sigma_full)-se_full**2))>1e-12: raise AssertionError("Full covariance diagonal mismatch")
    if np.max(np.abs(np.diag(Sigma_naive)-se_naive**2))>1e-12: raise AssertionError("Naive covariance diagonal mismatch")
    crit_full,min_full=gaussian_max_critical(Sigma_full,base_draws,cfg.alpha)
    crit_naive,min_naive=gaussian_max_critical(Sigma_naive,base_draws,cfg.alpha)
    err=mhat-truth; z_full=err/se_full; z_naive=err/se_naive
    maxz_full=float(np.max(np.abs(z_full))); maxz_naive=float(np.max(np.abs(z_naive)))
    zcrit=1.6448536269514722
    pc_full=np.abs(z_full)<=zcrit; pc_naive=np.abs(z_naive)<=zcrit
    width_full=2*crit_full*se_full; width_naive=2*crit_naive*se_naive; center=G//2
    elapsed=float(time.perf_counter()-t0)
    tracked=int(X.nbytes+IF_full.nbytes+IF_naive.nbytes+Sigma_full.nbytes+Sigma_naive.nbytes+base_draws.nbytes)
    out={"version":CODE_VERSION,"N":int(N),"T":int(cfg.T),"rep":int(rep),"seed0":int(seed0),"h":float(h),
         "runtime_seconds":elapsed,"tracked_peak_array_bytes":tracked,"mhat":mhat,"truth":truth,"se_full":se_full,
         "se_naive":se_naive,"z_full":z_full,"z_naive":z_naive,"point_cover_full":pc_full.astype(np.int8),
         "point_cover_naive":pc_naive.astype(np.int8),"Sigma_full":Sigma_full,"Sigma_naive":Sigma_naive,
         "crit_full":crit_full,"crit_naive":crit_naive,"maxz_full":maxz_full,"maxz_naive":maxz_naive,
         "simcover_full":int(maxz_full<=crit_full),"simcover_naive":int(maxz_naive<=crit_naive),
         "avgwidth_full":float(width_full.mean()),"avgwidth_naive":float(width_naive.mean()),
         "center_mhat":float(mhat[center]),"center_truth":float(truth[center]),"center_se_full":float(se_full[center]),
         "center_se_naive":float(se_naive[center]),"center_z_full":float(z_full[center]),"center_z_naive":float(z_naive[center]),
         "center_cover_full":int(pc_full[center]),"center_cover_naive":int(pc_naive[center]),
         "avg_point_cover_full":float(pc_full.mean()),"avg_point_cover_naive":float(pc_naive.mean()),
         "mean_se_ratio_full_over_naive":float(np.mean(se_full/se_naive)),"mean_alpha":alpha_mat.mean(axis=0),
         "mean_theta":theta_mat.mean(axis=0),"min_eig_corr_full_before_repair":float(min_full),
         "min_eig_corr_naive_before_repair":float(min_naive)}
    out.update(diag)
    return out
