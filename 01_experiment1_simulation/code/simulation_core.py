"""Section 5 final-production dynamic-panel simulation core (v3).

Design goals
------------
1. Match the final manuscript Algorithms 1--2.
2. Enforce strict pre-response information ordering.
3. Tune StaLoP target-by-target with one GDF-corrected MSPE criterion over
   (K, h, eta), with q_h used only as a computational index for h.
4. Give Time-Local and KNN-Covariate independent pre-target tuning.
5. Use deterministic task seeds, atomic checkpoints, and resumable runs.
6. Preserve exact GDF mathematics while eliminating repeated candidate work.

The legacy simultaneous-band implementation is deliberately excluded because it
is not the joint response-adaptive inference procedure in the final Section 3.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import tempfile
from typing import Any

# Keep each process single-threaded. This is important when ProcessPoolExecutor
# is used on machines linked against multithreaded BLAS libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError, eigvalsh, lstsq, solve
from scipy.spatial import cKDTree


class InvalidCandidate(RuntimeError):
    """Raised when a prescribed local candidate is not estimable."""


@dataclass(frozen=True)
class DGPConfig:
    rho_x: float = 0.6
    rho_eta: float = 0.5
    sigma_mu: float = 1.0
    sigma_a: float = 0.0
    sigma_u: float = 1.0
    G: int = 4
    burnin: int = 50
    # Final primary design: a recurrent, predictable cycle. This directly
    # aligns the simulation mechanism with target-local predictive compatibility.
    regime_mode: str = "predictable_cycle"
    # Retained only for the legacy stochastic-cycle sensitivity mode.
    p_switch: float = 0.35


@dataclass(frozen=True)
class MethodConfig:
    # H_a consists of the most recent K_state panels before panel a.
    # K_state=1 makes the final primary DGP's next-state mapping transparent.
    K_state: int = 1
    # State/final local fits match the manuscript equations exactly; W may use
    # a tiny numerical ridge as explicitly allowed in Section 2.
    state_ridge: float = 0.0
    final_ridge: float = 0.0
    w_ridge: float = 1e-8
    theta_standardize: bool = True
    min_bandwidth: float = 1e-10
    # Noise estimator: fit the pair-difference bias trend using the closest
    # half of unique within-panel nearest-neighbor pairs and extrapolate to d=0.
    noise_pair_fraction: float = 0.50
    min_noise_pairs: int = 20
    require_full_rank_before_ridge: bool = True
    max_design_condition: float = 1e4


@dataclass(frozen=True)
class TuningGrid:
    # Broad prespecified grid. q_h only indexes the realized target-specific h.
    q_h_grid: tuple[float, ...] = (
        0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80
    )
    # 1e-6 is a numerically explicit near-uniform softmax limit; 50 covers the
    # near-hard-concentration limit without changing the method definition eta>0.
    eta_grid: tuple[float, ...] = (
        0.000001, 0.0001, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75,
        1.00, 1.50, 2.00, 3.00, 5.00, 8.00, 12.00, 20.00, 50.00
    )
    # Fractions only generate integer K candidates; K is the actual tuned quantity.
    rank_count_grid: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20)
    rank_fraction_grid: tuple[float, ...] = (0.25, 0.50, 0.75)
    time_window_grid: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 80, 120)
    knn_k_grid: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 80, 160, 320, 640, 1280)


def validate_configuration(
    dgp_config: DGPConfig, method_config: MethodConfig, grid: TuningGrid
) -> None:
    """Fail fast on malformed production/sensitivity configurations."""
    if int(dgp_config.G) != 4:
        raise ValueError("this Section 5 DGP defines exactly four regime surfaces")
    if not (-1.0 < float(dgp_config.rho_x) < 1.0):
        raise ValueError("rho_x must lie in (-1,1)")
    if not (-1.0 < float(dgp_config.rho_eta) < 1.0):
        raise ValueError("rho_eta must lie in (-1,1)")
    if float(dgp_config.sigma_mu) < 0 or float(dgp_config.sigma_a) < 0 or float(dgp_config.sigma_u) <= 0:
        raise ValueError("DGP scale parameters must be nonnegative and sigma_u must be positive")
    if int(dgp_config.burnin) < 0:
        raise ValueError("burnin must be nonnegative")
    if dgp_config.regime_mode not in {"predictable_cycle", "stochastic_cycle", "panel_time", "unit_time"}:
        raise ValueError("unknown regime_mode")
    if not (0.0 <= float(dgp_config.p_switch) <= 1.0):
        raise ValueError("p_switch must lie in [0,1]")

    if int(method_config.K_state) < 1:
        raise ValueError("K_state must be at least 1")
    if min(float(method_config.state_ridge), float(method_config.final_ridge), float(method_config.w_ridge)) < 0:
        raise ValueError("all ridge constants must be nonnegative")
    if float(method_config.min_bandwidth) <= 0:
        raise ValueError("min_bandwidth must be positive")
    if not (0.0 < float(method_config.noise_pair_fraction) <= 1.0):
        raise ValueError("noise_pair_fraction must lie in (0,1]")
    if int(method_config.min_noise_pairs) < 3:
        raise ValueError("min_noise_pairs must be at least 3")
    if float(method_config.max_design_condition) <= 1.0:
        raise ValueError("max_design_condition must exceed 1")

    if not grid.q_h_grid or any((float(q) <= 0.0 or float(q) > 1.0) for q in grid.q_h_grid):
        raise ValueError("q_h_grid must be nonempty with entries in (0,1]")
    if not grid.eta_grid or any(float(e) <= 0.0 for e in grid.eta_grid):
        raise ValueError("eta_grid must be nonempty and positive")
    if any(int(k) <= 0 for k in grid.rank_count_grid):
        raise ValueError("rank_count_grid entries must be positive")
    if any((float(f) <= 0.0 or float(f) > 1.0) for f in grid.rank_fraction_grid):
        raise ValueError("rank_fraction_grid entries must lie in (0,1]")
    if not grid.rank_count_grid and not grid.rank_fraction_grid:
        raise ValueError("at least one K-grid specification is required")
    if not grid.time_window_grid or any(int(w) <= 0 for w in grid.time_window_grid):
        raise ValueError("time_window_grid must be nonempty and positive")
    if not grid.knn_k_grid or any(int(k) <= 0 for k in grid.knn_k_grid):
        raise ValueError("knn_k_grid must be nonempty and positive")


REGIME_PARAMS = {
    1: dict(delta=0.0, beta=0.8, c=0.0),
    2: dict(delta=0.3, beta=0.4, c=0.5),
    3: dict(delta=-0.3, beta=1.2, c=-0.5),
    4: dict(delta=0.5, beta=-0.2, c=0.3),
}


def g_regime(x1: np.ndarray | float, x2: np.ndarray | float, s: int) -> np.ndarray | float:
    p = REGIME_PARAMS[int(s)]
    return (
        2.0 * np.sin(np.pi * (x1 + p["delta"])) * np.cos(np.pi * (x2 - p["delta"]))
        + 0.5 * (x1**2 + x2**2)
        + p["beta"] * x1 * x2
        + p["c"]
    )


def g_regime_local_coefficient(x_star: np.ndarray, s: int) -> np.ndarray:
    """Centered local-linear coefficient [level, grad_x1, grad_x2]."""
    x1, x2 = float(x_star[0]), float(x_star[1])
    p = REGIME_PARAMS[int(s)]
    d = p["delta"]
    beta = p["beta"]
    level = float(g_regime(x1, x2, s))
    grad1 = 2.0 * np.pi * np.cos(np.pi * (x1 + d)) * np.cos(np.pi * (x2 - d)) + x1 + beta * x2
    grad2 = -2.0 * np.pi * np.sin(np.pi * (x1 + d)) * np.sin(np.pi * (x2 - d)) + x2 + beta * x1
    return np.array([level, grad1, grad2], dtype=float)


def simulate_panel(
    N: int,
    T: int,
    config: DGPConfig = DGPConfig(),
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    mu = rng.normal(0.0, config.sigma_mu, size=(N, 2))
    Sigma_eta = np.array([[1.0, config.rho_eta], [config.rho_eta, 1.0]])
    L_eta = np.linalg.cholesky(Sigma_eta)

    TT = T + config.burnin
    x1_all = np.zeros((N, TT))
    x2_all = np.zeros((N, TT))
    x1_all[:, 0] = mu[:, 0] + rng.normal(0.0, 1.0, size=N)
    x2_all[:, 0] = mu[:, 1] + rng.normal(0.0, 1.0, size=N)
    for tt in range(1, TT):
        eta = rng.normal(0.0, 1.0, size=(N, 2)) @ L_eta.T
        x1_all[:, tt] = mu[:, 0] + config.rho_x * (x1_all[:, tt - 1] - mu[:, 0]) + eta[:, 0]
        x2_all[:, tt] = mu[:, 1] + config.rho_x * (x2_all[:, tt - 1] - mu[:, 1]) + eta[:, 1]
    x1 = x1_all[:, config.burnin:]
    x2 = x2_all[:, config.burnin:]

    if config.regime_mode == "predictable_cycle":
        # Random initial phase, then deterministic recurrent transition.
        S_time = np.empty(T, dtype=int)
        S_time[0] = int(rng.integers(1, config.G + 1))
        for tt in range(1, T):
            S_time[tt] = 1 + (S_time[tt - 1] % config.G)
        S = np.tile(S_time[None, :], (N, 1))
    elif config.regime_mode in {"stochastic_cycle", "panel_time"}:
        # Preserved only as a sensitivity/legacy mode; not the primary final DGP.
        S_time = np.empty(T, dtype=int)
        S_time[0] = int(rng.integers(1, config.G + 1))
        for tt in range(1, T):
            switch = int(rng.binomial(1, config.p_switch))
            S_time[tt] = 1 + ((S_time[tt - 1] - 1 + switch) % config.G)
        S = np.tile(S_time[None, :], (N, 1))
    elif config.regime_mode == "unit_time":
        S = np.zeros((N, T), dtype=int)
        S[:, 0] = rng.integers(1, config.G + 1, size=N)
        for tt in range(1, T):
            switch = rng.binomial(1, config.p_switch, size=N)
            S[:, tt] = 1 + ((S[:, tt - 1] - 1 + switch) % config.G)
    else:
        raise ValueError("unknown regime_mode")

    signal = np.zeros((N, T))
    for regime in range(1, config.G + 1):
        mask = S == regime
        signal[mask] = g_regime(x1[mask], x2[mask], regime)

    a = rng.normal(0.0, config.sigma_a, size=N)
    eps = rng.normal(0.0, config.sigma_u, size=(N, T))
    y = signal + a[:, None] + eps

    # Final Section 5 uses sigma_a=0, so m_t(x)=signal exactly.
    return {
        "x1": x1,
        "x2": x2,
        "S": S,
        "a": a,
        "signal": signal,
        "m_true": signal,
        "y": y,
    }


def stack_prearrival(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    t_star_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    if t_star_idx <= 0:
        raise ValueError("t_star_idx must be at least 1")
    N = x1.shape[0]
    T_hist = int(t_star_idx)
    x1_pre = x1[:, :T_hist].T.reshape(-1)
    x2_pre = x2[:, :T_hist].T.reshape(-1)
    y_pre = y[:, :T_hist].T.reshape(-1)
    times_pre = np.repeat(np.arange(1, T_hist + 1), N)
    return x1_pre, x2_pre, y_pre, times_pre, T_hist


def local_linear_design(x1_arr: np.ndarray, x2_arr: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(x1_arr)), x1_arr - x_star[0], x2_arr - x_star[1]])


def euclidean_distance_to_target(x1_arr: np.ndarray, x2_arr: np.ndarray, x_star: np.ndarray) -> np.ndarray:
    return np.sqrt((x1_arr - x_star[0]) ** 2 + (x2_arr - x_star[1]) ** 2)


def solve_normal_equations(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return solve(A, b)
    except LinAlgError:
        c, *_ = lstsq(A, b, rcond=None)
        return c


def _check_raw_local_design(
    A_raw: np.ndarray, weight_sum: float, require_full_rank: bool, max_condition: float = np.inf
) -> None:
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise InvalidCandidate("zero local kernel weight")
    if require_full_rank:
        scale = max(float(weight_sum), 1e-12)
        eig = eigvalsh(A_raw / scale)
        lam_min = float(np.min(eig))
        lam_max = float(np.max(eig))
        if (not np.isfinite(lam_min)) or (not np.isfinite(lam_max)) or lam_min <= 1e-12:
            raise InvalidCandidate("rank-deficient local design before ridge stabilization")
        if lam_max / lam_min > float(max_condition):
            raise InvalidCandidate("ill-conditioned local design before ridge stabilization")


def solve_sufficient_stats_no_fallback(
    A_raw: np.ndarray,
    b_raw: np.ndarray,
    weight_sum: float,
    ridge: float,
    require_full_rank: bool = True,
    max_condition: float = np.inf,
) -> np.ndarray:
    _check_raw_local_design(A_raw, weight_sum, require_full_rank, max_condition)
    A = A_raw / weight_sum + ridge * np.eye(A_raw.shape[0])
    b = b_raw / weight_sum
    return solve_normal_equations(A, b)


def kernel_panel_sufficient_stats(
    Phi: np.ndarray,
    y: np.ndarray,
    K_vals: np.ndarray,
    times_pre: np.ndarray,
    T_hist: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Panelwise sufficient statistics for the binary local kernel."""
    L = Phi.shape[1]
    panel_weight = np.bincount(times_pre - 1, weights=K_vals, minlength=T_hist).astype(float)
    panel_y2 = np.bincount(times_pre - 1, weights=K_vals * y * y, minlength=T_hist).astype(float)
    panel_b = np.zeros((T_hist, L))
    panel_A = np.zeros((T_hist, L, L))
    for j in range(L):
        panel_b[:, j] = np.bincount(
            times_pre - 1, weights=K_vals * Phi[:, j] * y, minlength=T_hist
        )
        for k in range(j, L):
            stat = np.bincount(
                times_pre - 1,
                weights=K_vals * Phi[:, j] * Phi[:, k],
                minlength=T_hist,
            )
            panel_A[:, j, k] = stat
            panel_A[:, k, j] = stat
    return panel_A, panel_b, panel_weight, panel_y2


def history_indices_for_state(a: int, T_hist: int, K_state: int) -> np.ndarray:
    """Zero-based panel indices in H_a; a is one-based and may equal T_hist+1."""
    if a < 2 or a > T_hist + 1:
        return np.empty(0, dtype=int)
    end_exclusive = a - 1
    start = max(0, end_exclusive - int(K_state))
    return np.arange(start, end_exclusive, dtype=int)


def estimate_predictive_states_from_panel_stats_reference(
    panel_A: np.ndarray,
    panel_b: np.ndarray,
    panel_weight: np.ndarray,
    T_hist: int,
    method_config: MethodConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar reference implementation of the strict pre-response states."""
    if T_hist < 2:
        raise InvalidCandidate("at least two historical panels are required")
    eligible_panel_idx = np.arange(1, T_hist, dtype=int)  # panel labels 2,...,T_hist
    Z_hist = np.zeros((len(eligible_panel_idx), panel_b.shape[1]))

    for pos, panel_idx in enumerate(eligible_panel_idx):
        a = int(panel_idx + 1)
        H_idx = history_indices_for_state(a, T_hist, method_config.K_state)
        A_raw = panel_A[H_idx].sum(axis=0)
        b_raw = panel_b[H_idx].sum(axis=0)
        weight_sum = float(panel_weight[H_idx].sum())
        Z_hist[pos] = solve_sufficient_stats_no_fallback(
            A_raw,
            b_raw,
            weight_sum,
            method_config.state_ridge,
            method_config.require_full_rank_before_ridge,
            method_config.max_design_condition,
        )

    H_target = history_indices_for_state(T_hist + 1, T_hist, method_config.K_state)
    z_target = solve_sufficient_stats_no_fallback(
        panel_A[H_target].sum(axis=0),
        panel_b[H_target].sum(axis=0),
        float(panel_weight[H_target].sum()),
        method_config.state_ridge,
        method_config.require_full_rank_before_ridge,
        method_config.max_design_condition,
    )
    return Z_hist, z_target, eligible_panel_idx


def _solve_panel_states_batch_k1(
    panel_A: np.ndarray,
    panel_b: np.ndarray,
    panel_weight: np.ndarray,
    method_config: MethodConfig,
) -> np.ndarray:
    """Exact batched local solves for K_state=1.

    With K_state=1, z_a is the local fit from panel a-1.  All state solves
    therefore share the same algebra and can be evaluated in one NumPy batch.
    No statistical approximation is introduced.
    """
    A_raw = np.asarray(panel_A, dtype=float)
    b_raw = np.asarray(panel_b, dtype=float)
    w = np.asarray(panel_weight, dtype=float)
    valid = np.isfinite(w) & (w > 0.0)
    if not bool(np.all(valid)):
        raise InvalidCandidate("zero local kernel weight in a predictive-state panel")

    scaled = A_raw / w[:, None, None]
    if method_config.require_full_rank_before_ridge:
        eig = np.linalg.eigvalsh(scaled)
        lam_min = np.min(eig, axis=1)
        lam_max = np.max(eig, axis=1)
        cond = lam_max / np.maximum(lam_min, 1e-300)
        valid &= np.isfinite(lam_min) & np.isfinite(lam_max) & (lam_min > 1e-12)
        valid &= cond <= float(method_config.max_design_condition)
        if not bool(np.all(valid)):
            raise InvalidCandidate("invalid local design in a predictive-state panel")

    L = A_raw.shape[1]
    M = scaled + float(method_config.state_ridge) * np.eye(L)[None, :, :]
    rhs = b_raw / w[:, None]
    try:
        return np.linalg.solve(M, rhs[..., None])[..., 0]
    except LinAlgError as exc:
        raise InvalidCandidate("singular predictive-state local solve") from exc


def estimate_predictive_states_from_panel_stats(
    panel_A: np.ndarray,
    panel_b: np.ndarray,
    panel_weight: np.ndarray,
    T_hist: int,
    method_config: MethodConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Historical pre-response states plus future-target pre-response state.

    The production K_state=1 path is batched exactly for speed.  The general
    K_state path is retained as a scalar reference/sensitivity implementation.
    """
    if T_hist < 2:
        raise InvalidCandidate("at least two historical panels are required")
    if int(method_config.K_state) == 1:
        states = _solve_panel_states_batch_k1(
            panel_A[:T_hist], panel_b[:T_hist], panel_weight[:T_hist], method_config
        )
        eligible_panel_idx = np.arange(1, T_hist, dtype=int)
        # Historical panel t=2,...,T_hist uses state from panel t-1; the future
        # target T_hist+1 uses the final observed historical panel.
        return states[:-1].copy(), states[-1].copy(), eligible_panel_idx
    return estimate_predictive_states_from_panel_stats_reference(
        panel_A, panel_b, panel_weight, T_hist, method_config
    )

def standardize_theta(theta: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if len(theta) <= 1:
        return np.zeros_like(theta)
    sd = float(np.std(theta, ddof=1))
    return (theta - float(np.mean(theta))) / (sd + eps)


def rank_candidate_counts(
    n_eligible: int,
    rank_fraction_grid: tuple[float, ...],
    rank_count_grid: tuple[int, ...] = (),
) -> tuple[int, ...]:
    if n_eligible <= 0:
        return tuple()
    counts = {
        min(max(int(np.floor(float(frac) * n_eligible)), 1), n_eligible)
        for frac in rank_fraction_grid
    }
    counts.update(min(max(int(k), 1), n_eligible) for k in rank_count_grid)
    counts.add(n_eligible)
    return tuple(sorted(counts))


def prepare_target_static(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    x_star: np.ndarray,
) -> dict[str, np.ndarray]:
    """Quantities independent of q_h, K and eta for one target input."""
    return {
        "dist": euclidean_distance_to_target(x1_pre, x2_pre, x_star),
        "Phi": local_linear_design(x1_pre, x2_pre, x_star),
    }


def build_target_components(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    x_star: np.ndarray,
    T_hist: int,
    q_h: float,
    method_config: MethodConfig,
    static: dict[str, np.ndarray] | None = None,
    h_override: float | None = None,
) -> dict[str, Any]:
    if not (0.0 < q_h <= 1.0):
        raise ValueError("q_h must be in (0,1]")
    if static is None:
        static = prepare_target_static(x1_pre, x2_pre, x_star)
    dist = np.asarray(static["dist"])
    Phi = np.asarray(static["Phi"])

    h = max(
        float(np.quantile(dist, q_h)) if h_override is None else float(h_override),
        method_config.min_bandwidth,
    )
    K_vals = (dist <= h).astype(float)
    if float(K_vals.sum()) <= 0.0:
        raise InvalidCandidate("zero kernel support")

    panel_A, panel_b, panel_weight, panel_y2 = kernel_panel_sufficient_stats(
        Phi, y_pre, K_vals, times_pre, T_hist
    )
    Z_hist, z_target, eligible_panel_idx = estimate_predictive_states_from_panel_stats(
        panel_A, panel_b, panel_weight, T_hist, method_config
    )

    weight_sum = float(panel_weight.sum())
    _check_raw_local_design(
        panel_A.sum(axis=0), weight_sum, method_config.require_full_rank_before_ridge,
        method_config.max_design_condition
    )
    W_mat = panel_A.sum(axis=0) / weight_sum
    W_mat = W_mat + method_config.w_ridge * np.eye(Phi.shape[1])
    diff = Z_hist - z_target[None, :]
    theta = np.maximum(np.einsum("ti,ij,tj->t", diff, W_mat, diff), 0.0)
    theta_std = standardize_theta(theta) if method_config.theta_standardize else theta.copy()
    order = np.argsort(theta, kind="mergesort")

    return {
        "q_h": float(q_h),
        "h": float(h),
        "K_vals": K_vals,
        "Phi": Phi,
        "panel_A": panel_A,
        "panel_b": panel_b,
        "panel_weight": panel_weight,
        "panel_y2": panel_y2,
        "Z_hist": Z_hist,
        "z_target": z_target,
        "eligible_panel_idx": eligible_panel_idx,
        "W": W_mat,
        "theta": theta,
        "theta_std": theta_std,
        "order": order,
    }


def evaluate_gdf_candidate_reference(
    components: dict[str, Any],
    times_pre: np.ndarray,
    eta: float,
    K: int,
    sigma2_hat: float,
    method_config: MethodConfig,
) -> dict[str, float]:
    """Slow direct reference implementation retained only for parity testing."""
    panel_A = np.asarray(components["panel_A"])
    panel_b = np.asarray(components["panel_b"])
    panel_weight = np.asarray(components["panel_weight"])
    panel_y2 = np.asarray(components["panel_y2"])
    eligible_panel_idx = np.asarray(components["eligible_panel_idx"], dtype=int)
    theta = np.asarray(components["theta"])
    theta_use = np.asarray(components["theta_std"])
    order = np.asarray(components["order"], dtype=int)
    K_vals = np.asarray(components["K_vals"])

    n_eligible = len(theta)
    K = min(max(int(K), 1), n_eligible)
    retained_pos = order[:K]
    log_score = -float(eta) * theta_use
    # One common shift has no effect after observation-weight normalization.
    shift = float(np.max(log_score[retained_pos]))
    exp_score = np.exp(log_score - shift)

    beta_full = np.zeros(panel_A.shape[0], dtype=float)
    retained_panels = eligible_panel_idx[retained_pos]
    beta_full[retained_panels] = exp_score[retained_pos]

    Q = float(beta_full @ panel_weight)
    if Q <= 0.0:
        raise InvalidCandidate("retained set has zero local kernel support")
    A_raw = np.tensordot(beta_full, panel_A, axes=(0, 0))
    b_raw = beta_full @ panel_b
    c_hat = solve_sufficient_stats_no_fallback(
        A_raw, b_raw, Q, method_config.final_ridge, method_config.require_full_rank_before_ridge
    )
    y2_raw = float(beta_full @ panel_y2)
    sse_raw = y2_raw - 2.0 * float(c_hat @ b_raw) + float(c_hat @ A_raw @ c_hat)
    R = max(float(sse_raw / Q), 0.0)

    V_raw = np.tensordot(beta_full * beta_full, panel_A, axes=(0, 0))
    M_reg = A_raw + method_config.final_ridge * Q * np.eye(A_raw.shape[0])
    GDF = float(np.trace(solve_normal_equations(M_reg, V_raw)) / Q)
    criterion = float(R + 2.0 * sigma2_hat * GDF - sigma2_hat)

    obs_raw = beta_full[times_pre - 1] * K_vals
    obs_sum = float(obs_raw.sum())
    if obs_sum <= 0.0:
        raise InvalidCandidate("zero final observation weight")
    ESS = float(obs_sum * obs_sum / np.sum(obs_raw * obs_raw))
    return {
        "pred": float(c_hat[0]),
        "criterion": criterion,
        "R": R,
        "GDF": GDF,
        "K": float(K),
        "eta": float(eta),
        "ESS": ESS,
    }


# Backward-compatible public name for tests/external inspection.
evaluate_gdf_candidate = evaluate_gdf_candidate_reference


def evaluate_eta_over_K_fast(
    components: dict[str, Any],
    eta: float,
    K_values: tuple[int, ...],
    sigma2_hat: float,
    method_config: MethodConfig,
) -> list[dict[str, float]]:
    """Exact GDF evaluation for all K at fixed (q_h, eta), using cumulative stats.

    The result is algebraically identical to repeated calls to the reference
    evaluator, but ranking/standardization/exponentiation and panel sums are
    performed once per eta rather than once per (eta,K).
    """
    panel_A = np.asarray(components["panel_A"])
    panel_b = np.asarray(components["panel_b"])
    panel_weight = np.asarray(components["panel_weight"])
    panel_y2 = np.asarray(components["panel_y2"])
    eligible = np.asarray(components["eligible_panel_idx"], dtype=int)
    order = np.asarray(components["order"], dtype=int)
    theta_std = np.asarray(components["theta_std"])

    ordered_panels = eligible[order]
    log_beta = -float(eta) * theta_std[order]
    log_beta -= float(np.max(log_beta))  # stable common scale; cancels from normalized D.
    beta = np.exp(log_beta)

    A_ord = panel_A[ordered_panels]
    b_ord = panel_b[ordered_panels]
    w_ord = panel_weight[ordered_panels]
    y2_ord = panel_y2[ordered_panels]

    cum_Q = np.cumsum(beta * w_ord)
    cum_A = np.cumsum(beta[:, None, None] * A_ord, axis=0)
    cum_b = np.cumsum(beta[:, None] * b_ord, axis=0)
    cum_y2 = np.cumsum(beta * y2_ord)
    cum_V = np.cumsum((beta * beta)[:, None, None] * A_ord, axis=0)
    # Binary kernel: sum_j (beta_t K_j)^2 = beta_t^2 * panel_weight_t.
    cum_obs2 = np.cumsum(beta * beta * w_ord)

    L = panel_A.shape[1]
    eye = np.eye(L)
    rows: list[dict[str, float]] = []
    for K in K_values:
        kk = min(max(int(K), 1), len(order))
        j = kk - 1
        Q = float(cum_Q[j])
        if Q <= 0.0:
            continue
        A_raw = cum_A[j]
        b_raw = cum_b[j]
        try:
            c_hat = solve_sufficient_stats_no_fallback(
                A_raw, b_raw, Q, method_config.final_ridge, method_config.require_full_rank_before_ridge
            )
        except InvalidCandidate:
            continue
        y2_raw = float(cum_y2[j])
        sse_raw = y2_raw - 2.0 * float(c_hat @ b_raw) + float(c_hat @ A_raw @ c_hat)
        R = max(float(sse_raw / Q), 0.0)
        M_reg = A_raw + method_config.final_ridge * Q * eye
        GDF = float(np.trace(solve_normal_equations(M_reg, cum_V[j])) / Q)
        criterion = float(R + 2.0 * sigma2_hat * GDF - sigma2_hat)
        obs2 = float(cum_obs2[j])
        if obs2 <= 0.0:
            continue
        ESS = float(Q * Q / obs2)
        rows.append({
            "pred": float(c_hat[0]),
            "criterion": criterion,
            "R": R,
            "GDF": GDF,
            "K": float(kk),
            "eta": float(eta),
            "ESS": ESS,
        })
    return rows



def evaluate_all_eta_k_fast(
    components: dict[str, Any],
    eta_values: tuple[float, ...],
    K_values: tuple[int, ...],
    sigma2_hat: float,
    method_config: MethodConfig,
    best_only: bool = False,
) -> list[dict[str, float]] | tuple[dict[str, float] | None, int]:
    """Vectorized exact evaluation of the full eta x K candidate block.

    This is algebraically identical to repeated reference evaluations. It batches
    the 3x3 local-linear solves over all eta/K combinations and is the production
    path for exhaustive Algorithm-2 tuning.
    """
    panel_A = np.asarray(components["panel_A"])
    panel_b = np.asarray(components["panel_b"])
    panel_weight = np.asarray(components["panel_weight"])
    panel_y2 = np.asarray(components["panel_y2"])
    eligible = np.asarray(components["eligible_panel_idx"], dtype=int)
    order = np.asarray(components["order"], dtype=int)
    theta_std = np.asarray(components["theta_std"])

    ordered_panels = eligible[order]
    A_ord = panel_A[ordered_panels]
    b_ord = panel_b[ordered_panels]
    w_ord = panel_weight[ordered_panels]
    y2_ord = panel_y2[ordered_panels]

    etas = np.asarray(tuple(float(e) for e in eta_values), dtype=float)
    if etas.size == 0 or len(K_values) == 0:
        return (None, 0) if best_only else []
    log_beta = -etas[:, None] * theta_std[order][None, :]
    log_beta -= np.max(log_beta, axis=1, keepdims=True)
    beta = np.exp(log_beta)  # [E, P]

    cum_Q = np.cumsum(beta * w_ord[None, :], axis=1)
    cum_A = np.cumsum(beta[:, :, None, None] * A_ord[None, :, :, :], axis=1)
    cum_b = np.cumsum(beta[:, :, None] * b_ord[None, :, :], axis=1)
    cum_y2 = np.cumsum(beta * y2_ord[None, :], axis=1)
    beta2 = beta * beta
    cum_V = np.cumsum(beta2[:, :, None, None] * A_ord[None, :, :, :], axis=1)
    cum_obs2 = np.cumsum(beta2 * w_ord[None, :], axis=1)

    k_array = np.asarray([min(max(int(k), 1), len(order)) for k in K_values], dtype=int)
    k_idx = k_array - 1
    Q = cum_Q[:, k_idx]                                  # [E,K]
    A_raw = cum_A[:, k_idx, :, :]                        # [E,K,L,L]
    b_raw = cum_b[:, k_idx, :]                           # [E,K,L]
    y2_raw = cum_y2[:, k_idx]                            # [E,K]
    V_raw = cum_V[:, k_idx, :, :]                        # [E,K,L,L]
    obs2 = cum_obs2[:, k_idx]                            # [E,K]

    valid = np.isfinite(Q) & (Q > 0.0) & np.isfinite(obs2) & (obs2 > 0.0)
    if method_config.require_full_rank_before_ridge:
        scaled_A = A_raw / np.maximum(Q[:, :, None, None], 1e-12)
        eig = np.linalg.eigvalsh(scaled_A)
        lam_min = np.min(eig, axis=-1)
        lam_max = np.max(eig, axis=-1)
        cond = lam_max / np.maximum(lam_min, 1e-300)
        valid &= np.isfinite(lam_min) & np.isfinite(lam_max) & (lam_min > 1e-12)
        valid &= cond <= float(method_config.max_design_condition)

    L = panel_A.shape[1]
    eye = np.eye(L)[None, None, :, :]
    M_reg = A_raw + method_config.final_ridge * Q[:, :, None, None] * eye
    # Invalid blocks may have Q=0 and a singular zero matrix. Replace only those
    # blocks by identity for the batched solve; they remain excluded by ``valid``.
    M_safe = M_reg.copy()
    b_safe = b_raw.copy()
    V_safe = V_raw.copy()
    if np.any(~valid):
        M_safe[~valid] = np.eye(L)
        b_safe[~valid] = 0.0
        V_safe[~valid] = 0.0
    c_hat = np.linalg.solve(M_safe, b_safe[..., None])[..., 0]
    fit_quad = np.einsum("...i,...ij,...j->...", c_hat, A_raw, c_hat)
    cb = np.einsum("...i,...i->...", c_hat, b_raw)
    R = np.maximum((y2_raw - 2.0 * cb + fit_quad) / np.maximum(Q, 1e-300), 0.0)
    solved_V = np.linalg.solve(M_safe, V_safe)
    GDF = np.trace(solved_V, axis1=-2, axis2=-1) / np.maximum(Q, 1e-300)
    criterion = R + 2.0 * float(sigma2_hat) * GDF - float(sigma2_hat)
    ESS = Q * Q / np.maximum(obs2, 1e-300)

    if best_only:
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            return None, 0
        cmin = float(np.min(criterion[valid]))
        tol = 1e-14 + 1e-12 * max(1.0, abs(cmin))
        tied = np.argwhere(valid & (criterion <= cmin + tol))
        # q_h is fixed within this block. Among machine-scale criterion ties,
        # use the same parsimony ordering as the global selector: smaller K,
        # then eta closest to 1.
        ie, ik = min(
            ((int(a), int(b)) for a, b in tied),
            key=lambda ij: (
                int(k_array[ij[1]]),
                abs(float(np.log(max(float(etas[ij[0]]), 1e-300)))),
                float(etas[ij[0]]),
            ),
        )
        return {
            "pred": float(c_hat[ie, ik, 0]),
            "criterion": float(criterion[ie, ik]),
            "R": float(R[ie, ik]),
            "GDF": float(GDF[ie, ik]),
            "K": float(k_array[ik]),
            "eta": float(etas[ie]),
            "ESS": float(ESS[ie, ik]),
        }, n_valid

    rows: list[dict[str, float]] = []
    for ie, eta in enumerate(etas):
        for ik, K in enumerate(k_array):
            if not bool(valid[ie, ik]):
                continue
            rows.append({
                "pred": float(c_hat[ie, ik, 0]),
                "criterion": float(criterion[ie, ik]),
                "R": float(R[ie, ik]),
                "GDF": float(GDF[ie, ik]),
                "K": float(K),
                "eta": float(eta),
                "ESS": float(ESS[ie, ik]),
            })
    return rows

def estimate_noise_variance_within_panel_nn(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    T_hist: int,
    pair_fraction: float = 0.50,
    min_pairs: int = 20,
) -> tuple[float, int]:
    """Pre-target local pair-difference estimate of sigma^2.

    For each historical panel, unique nearest-neighbor pairs are formed in
    standardized covariate space. Among the closest ``pair_fraction`` of these
    NN pairs, regress (Y_i-Y_j)^2/2 on squared covariate distance and use the
    intercept at zero distance. The intercept removes the leading finite-distance
    mean-function contribution while preserving a fully pre-target estimator.
    """
    if not (0.0 < pair_fraction <= 1.0):
        raise ValueError("pair_fraction must be in (0,1]")
    x1_sd = max(float(np.std(x1_pre, ddof=1)), 1e-12)
    x2_sd = max(float(np.std(x2_pre, ddof=1)), 1e-12)
    pair_d2: list[float] = []
    pair_v: list[float] = []

    for t in range(1, T_hist + 1):
        idx = np.flatnonzero(times_pre == t)
        if len(idx) < 2:
            continue
        X = np.column_stack([x1_pre[idx] / x1_sd, x2_pre[idx] / x2_sd])
        tree = cKDTree(X)
        dist, nn = tree.query(X, k=2)
        unique_pairs = {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nn[:, 1]) if i != int(j)}
        for i, j in unique_pairs:
            d2 = float(np.sum((X[i] - X[j]) ** 2))
            pair_d2.append(d2)
            pair_v.append(0.5 * float((y_pre[idx[i]] - y_pre[idx[j]]) ** 2))

    if len(pair_v) < max(3, min_pairs):
        raise RuntimeError("noise variance estimator found too few within-panel NN pairs")
    d2 = np.asarray(pair_d2)
    vals = np.asarray(pair_v)
    order = np.argsort(d2, kind="mergesort")
    keep = max(int(min_pairs), int(np.ceil(pair_fraction * len(order))))
    keep = min(keep, len(order))
    d2 = d2[order[:keep]]
    vals = vals[order[:keep]]
    Xreg = np.column_stack([np.ones(keep), d2])
    beta = solve_normal_equations(Xreg.T @ Xreg, Xreg.T @ vals)
    sigma2_hat = float(beta[0])
    if not np.isfinite(sigma2_hat):
        raise RuntimeError("non-finite noise variance estimate")
    # Positive-part truncation is explicit, deterministic, and tracked by callers.
    return max(sigma2_hat, 1e-12), int(keep)


def _secondary_candidate_key(row: dict[str, float]) -> tuple[float, int, float, float]:
    """Deterministic parsimony tie-break used only for numerically tied criteria."""
    eta = max(float(row["eta"]), 1e-300)
    return (float(row["q_h"]), int(row["K"]), abs(float(np.log(eta))), eta)


def _candidate_key(row: dict[str, float]) -> tuple[float, float, int, float, float]:
    """Backward-compatible full ordering for audit/reference code."""
    return (float(row["criterion"]), *_secondary_candidate_key(row))


def _candidate_is_better(
    row: dict[str, float], best: dict[str, float] | None, rtol: float = 1e-12, atol: float = 1e-14
) -> bool:
    """Criterion comparison robust to machine-scale batching differences."""
    if best is None:
        return True
    c = float(row["criterion"]); b = float(best["criterion"])
    tol = float(atol + rtol * max(1.0, abs(c), abs(b)))
    if c < b - tol:
        return True
    if abs(c - b) <= tol:
        return _secondary_candidate_key(row) < _secondary_candidate_key(best)
    return False


def tune_stalop_gdf_target(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    x_star: np.ndarray,
    T_hist: int,
    sigma2_hat: float,
    grid: TuningGrid,
    method_config: MethodConfig,
    return_all: bool = False,
) -> dict[str, float] | tuple[dict[str, float], pd.DataFrame]:
    """Final Algorithm 2: exact exhaustive finite-grid tuning over (K,h,eta).

    q_h indexes the realized target-specific bandwidth h. For every q_h, states,
    W, discrepancies and ranking are recomputed; then the full eta x K block is
    evaluated by the vectorized exact GDF engine.
    """
    static = prepare_target_static(x1_pre, x2_pre, x_star)
    q_values = tuple(float(q) for q in grid.q_h_grid)
    # One vectorized quantile call per target replaces one partition per q_h.
    h_values = np.maximum(
        np.asarray(np.quantile(np.asarray(static["dist"]), np.asarray(q_values)), dtype=float),
        float(method_config.min_bandwidth),
    )
    candidates: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    n_valid_qh = 0
    n_candidate = 0

    for q_h, h_value in zip(q_values, h_values):
        try:
            comp = build_target_components(
                x1_pre, x2_pre, y_pre, times_pre, x_star, T_hist,
                q_h, method_config, static=static, h_override=float(h_value),
            )
        except InvalidCandidate:
            continue
        n_valid_qh += 1
        K_values = rank_candidate_counts(
            len(np.asarray(comp["theta"])), grid.rank_fraction_grid, grid.rank_count_grid
        )
        if return_all:
            rows = evaluate_all_eta_k_fast(
                comp, tuple(float(e) for e in grid.eta_grid), K_values, sigma2_hat, method_config
            )
            assert isinstance(rows, list)
            n_candidate += len(rows)
            for out in rows:
                row = {**out, "q_h": q_h, "h": float(comp["h"]), "sigma2_hat": float(sigma2_hat)}
                candidates.append(row)
                if _candidate_is_better(row, best):
                    best = row
        else:
            block_best, n_valid = evaluate_all_eta_k_fast(
                comp, tuple(float(e) for e in grid.eta_grid), K_values, sigma2_hat, method_config,
                best_only=True,
            )
            n_candidate += int(n_valid)
            if block_best is not None:
                row = {**block_best, "q_h": q_h, "h": float(comp["h"]), "sigma2_hat": float(sigma2_hat)}
                if _candidate_is_better(row, best):
                    best = row

    if best is None:
        raise RuntimeError("no valid StaLoP candidate for target input")
    best = {**best, "n_valid_qh": int(n_valid_qh), "n_candidates": int(n_candidate)}
    if return_all:
        return best, pd.DataFrame(candidates)
    return best


# Audit alias: production is already exhaustive.
tune_stalop_gdf_target_exhaustive = tune_stalop_gdf_target


# -----------------------------------------------------------------------------
# Baselines and independent rolling-origin validation
# -----------------------------------------------------------------------------


def fit_global_linear(x1_pre: np.ndarray, x2_pre: np.ndarray, y_pre: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    X = np.column_stack([np.ones(len(y_pre)), x1_pre, x2_pre])
    return solve_normal_equations(X.T @ X + ridge * np.eye(3), X.T @ y_pre)


def predict_linear_beta(beta: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x1 + beta[2] * x2


def predict_global_linear(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    x_star: np.ndarray,
    ridge: float = 1e-8,
) -> float:
    beta = fit_global_linear(x1_pre, x2_pre, y_pre, ridge)
    return float(predict_linear_beta(beta, np.asarray(x_star[0]), np.asarray(x_star[1])))


def fit_time_local(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    T_hist: int,
    window: int,
    ridge: float = 1e-8,
) -> np.ndarray:
    start_time = max(1, T_hist - int(window) + 1)
    mask = times_pre >= start_time
    if not np.any(mask):
        raise RuntimeError("empty Time-Local training window")
    return fit_global_linear(x1_pre[mask], x2_pre[mask], y_pre[mask], ridge)


def predict_time_local(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    x_star: np.ndarray,
    T_hist: int,
    window: int,
    ridge: float = 1e-8,
) -> float:
    beta = fit_time_local(x1_pre, x2_pre, y_pre, times_pre, T_hist, window, ridge)
    return float(predict_linear_beta(beta, np.asarray(x_star[0]), np.asarray(x_star[1])))


def predict_knn_covariate(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    x_star: np.ndarray,
    k: int,
) -> float:
    n_obs = len(y_pre)
    k = min(max(int(k), 1), n_obs)
    x1_sd = max(float(np.std(x1_pre, ddof=1)), 1e-12)
    x2_sd = max(float(np.std(x2_pre, ddof=1)), 1e-12)
    d2 = ((x1_pre - x_star[0]) / x1_sd) ** 2 + ((x2_pre - x_star[1]) / x2_sd) ** 2
    idx = np.argpartition(d2, k - 1)[:k]
    return float(np.mean(y_pre[idx]))



def predict_knn_covariate_batch(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    x1_target: np.ndarray,
    x2_target: np.ndarray,
    k: int,
) -> np.ndarray:
    """Uniform-mean KNN predictions for a target batch using one KD-tree."""
    n_obs = len(y_pre)
    k_eff = min(max(int(k), 1), n_obs)
    x1_sd = max(float(np.std(x1_pre, ddof=1)), 1e-12)
    x2_sd = max(float(np.std(x2_pre, ddof=1)), 1e-12)
    X = np.column_stack([x1_pre / x1_sd, x2_pre / x2_sd])
    Xt = np.column_stack([np.asarray(x1_target) / x1_sd, np.asarray(x2_target) / x2_sd])
    tree = cKDTree(X)
    _, idx = tree.query(Xt, k=k_eff, workers=1)
    if k_eff == 1:
        return y_pre[np.asarray(idx, dtype=int)]
    return np.mean(y_pre[np.asarray(idx, dtype=int)], axis=1)


def validation_times(t_star_idx: int, min_train_time: int = 6) -> list[int]:
    if t_star_idx <= min_train_time + 1:
        return [t_star_idx - 1]
    start = max(min_train_time, int(np.floor(0.5 * t_star_idx)))
    return list(range(start, t_star_idx))


def tune_time_local(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    t_star_idx: int,
    grid: TuningGrid,
) -> dict[str, float]:
    rows = []
    # Values >= t_star_idx are all the same full-history model at every validation
    # origin; keep only the smallest representative to avoid redundant fits.
    windows = sorted(set(min(int(w), int(t_star_idx)) for w in grid.time_window_grid))
    for window in windows:
        se_sum = 0.0
        count = 0
        for v_idx in validation_times(t_star_idx):
            x1_pre, x2_pre, y_pre, times_pre, T_hist = stack_prearrival(x1, x2, y, v_idx)
            beta = fit_time_local(x1_pre, x2_pre, y_pre, times_pre, T_hist, int(window))
            pred = predict_linear_beta(beta, x1[:, v_idx], x2[:, v_idx])
            se_sum += float(np.sum((pred - y[:, v_idx]) ** 2))
            count += len(pred)
        rows.append({"window": int(window), "cv_mse": se_sum / count})
    return min(rows, key=lambda r: (r["cv_mse"], r["window"]))


def _knn_sorted_prefix_values(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    x_star: np.ndarray,
    max_k: int,
    x1_sd: float,
    x2_sd: float,
) -> np.ndarray:
    """Nearest-neighbor y values sorted to max_k; ties are deterministically index-broken."""
    d2 = ((x1_pre - x_star[0]) / x1_sd) ** 2 + ((x2_pre - x_star[1]) / x2_sd) ** 2
    max_k = min(max(int(max_k), 1), len(d2))
    if max_k == len(d2):
        idx = np.lexsort((np.arange(len(d2)), d2))
        return y_pre[idx]
    # Tie-safe partial selection: reproduce the first max_k elements of a stable
    # full distance sort without sorting the full history.
    tau = float(np.partition(d2, max_k - 1)[max_k - 1])
    idx_lt = np.flatnonzero(d2 < tau)
    idx_eq = np.flatnonzero(d2 == tau)
    need = max_k - len(idx_lt)
    chosen_eq = np.sort(idx_eq)[:need]
    part = np.concatenate([idx_lt, chosen_eq])
    local_order = np.lexsort((part, d2[part]))
    return y_pre[part[local_order]]


def tune_knn_covariate(
    x1: np.ndarray,
    x2: np.ndarray,
    y: np.ndarray,
    t_star_idx: int,
    grid: TuningGrid,
) -> dict[str, float]:
    """Independent rolling-origin tuning of the KNN neighbor count k."""
    k_specs = tuple(sorted(set(int(k) for k in grid.knn_k_grid if int(k) > 0)))
    se_sums = {k: 0.0 for k in k_specs}
    counts = {k: 0 for k in k_specs}
    for v_idx in validation_times(t_star_idx):
        x1_pre, x2_pre, y_pre, _, _ = stack_prearrival(x1, x2, y, v_idx)
        x1_sd = max(float(np.std(x1_pre, ddof=1)), 1e-12)
        x2_sd = max(float(np.std(x2_pre, ddof=1)), 1e-12)
        X = np.column_stack([x1_pre / x1_sd, x2_pre / x2_sd])
        Xt = np.column_stack([x1[:, v_idx] / x1_sd, x2[:, v_idx] / x2_sd])
        k_map = {k: min(k, len(y_pre)) for k in k_specs}
        max_k = max(k_map.values())
        tree = cKDTree(X)
        _, idx = tree.query(Xt, k=max_k, workers=1)
        if max_k == 1:
            y_sorted = y_pre[np.asarray(idx, dtype=int)][:, None]
        else:
            y_sorted = y_pre[np.asarray(idx, dtype=int)]
        csum = np.cumsum(y_sorted, axis=1)
        for k_spec in k_specs:
            k_eff = k_map[k_spec]
            pred = csum[:, k_eff - 1] / k_eff
            se_sums[k_spec] += float(np.sum((pred - y[:, v_idx]) ** 2))
            counts[k_spec] += len(pred)
    rows = [{"knn_k": k, "cv_mse": se_sums[k] / counts[k]} for k in k_specs]
    return min(rows, key=lambda r: (r["cv_mse"], r["knn_k"]))


# -----------------------------------------------------------------------------
# One replication / Monte Carlo
# -----------------------------------------------------------------------------


def run_one_replication(
    N: int,
    T: int,
    seed: int,
    replicate_id: int = 0,
    dgp_config: DGPConfig = DGPConfig(),
    method_config: MethodConfig = MethodConfig(),
    grid: TuningGrid = TuningGrid(),
    t_star_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_configuration(dgp_config, method_config, grid)
    if int(N) < 4 or int(T) < 5:
        raise ValueError("N>=4 and T>=5 are required for the Section 5 simulation")
    if not (0.0 < float(t_star_fraction) < 1.0):
        raise ValueError("t_star_fraction must lie in (0,1)")
    rng = np.random.default_rng(seed)
    data = simulate_panel(N, T, dgp_config, rng)
    x1, x2, y = data["x1"], data["x2"], data["y"]
    target = data["m_true"]

    t_star_idx = int(np.floor(t_star_fraction * T)) - 1
    t_star_idx = min(max(t_star_idx, 2), T - 1)
    x1_pre, x2_pre, y_pre, times_pre, T_hist = stack_prearrival(x1, x2, y, t_star_idx)
    sigma2_hat, sigma2_pairs = estimate_noise_variance_within_panel_nn(
        x1_pre, x2_pre, y_pre, times_pre, T_hist,
        method_config.noise_pair_fraction, method_config.min_noise_pairs,
    )

    method_rows: list[dict[str, Any]] = []

    # Global Linear: one fit, all final target units.
    tic = perf_counter()
    beta_global = fit_global_linear(x1_pre, x2_pre, y_pre)
    pred_global = predict_linear_beta(beta_global, x1[:, t_star_idx], x2[:, t_star_idx])
    runtime_global = perf_counter() - tic

    # Time-Local: independent rolling-origin tuning, then one final fit.
    tic = perf_counter()
    time_tune = tune_time_local(x1, x2, y, t_star_idx, grid)
    beta_time = fit_time_local(x1_pre, x2_pre, y_pre, times_pre, T_hist, int(time_tune["window"]))
    pred_time = predict_linear_beta(beta_time, x1[:, t_star_idx], x2[:, t_star_idx])
    runtime_time = perf_counter() - tic

    # KNN-Covariate: independent rolling-origin tuning, uniform mean final prediction.
    tic = perf_counter()
    knn_tune = tune_knn_covariate(x1, x2, y, t_star_idx, grid)
    pred_knn = predict_knn_covariate_batch(
        x1_pre, x2_pre, y_pre, x1[:, t_star_idx], x2[:, t_star_idx],
        int(knn_tune["knn_k"])
    )
    runtime_knn = perf_counter() - tic

    # StaLoP: target-specific Algorithm 2.
    tic = perf_counter()
    pred_stalop = np.zeros(N)
    tuning_rows: list[dict[str, Any]] = []
    q_min, q_max = min(grid.q_h_grid), max(grid.q_h_grid)
    e_min, e_max = min(grid.eta_grid), max(grid.eta_grid)
    for i in range(N):
        x_star = np.array([x1[i, t_star_idx], x2[i, t_star_idx]])
        best = tune_stalop_gdf_target(
            x1_pre, x2_pre, y_pre, times_pre, x_star, T_hist,
            sigma2_hat, grid, method_config, return_all=False,
        )
        pred_stalop[i] = float(best["pred"])
        tuning_rows.append({
            "seed": int(seed), "replicate_id": int(replicate_id),
            "T": int(T), "N": int(N), "target_unit": int(i),
            "q_h": float(best["q_h"]), "h": float(best["h"]),
            "eta": float(best["eta"]), "K": int(best["K"]),
            "criterion": float(best["criterion"]), "R": float(best["R"]),
            "GDF": float(best["GDF"]), "ESS": float(best["ESS"]),
            "sigma2_hat": float(sigma2_hat), "sigma2_pairs": int(sigma2_pairs),
            "n_valid_qh": int(best["n_valid_qh"]), "n_candidates": int(best["n_candidates"]),
            "q_h_boundary": int(float(best["q_h"]) in {q_min, q_max}),
            "q_h_lower": int(float(best["q_h"]) == q_min),
            "q_h_upper": int(float(best["q_h"]) == q_max),
            "eta_boundary": int(float(best["eta"]) in {e_min, e_max}),
            "eta_near_uniform": int(float(best["eta"]) == e_min),
            "eta_upper": int(float(best["eta"]) == e_max),
            "K_full": int(int(best["K"]) == (T_hist - 1)),
        })
    runtime_stalop = perf_counter() - tic

    target_vec = target[:, t_star_idx]
    tuning_df = pd.DataFrame(tuning_rows)
    specs = [
        ("Global Linear", pred_global, runtime_global, {}),
        ("Time-Local", pred_time, runtime_time, time_tune),
        ("KNN-Covariate", pred_knn, runtime_knn, knn_tune),
        ("StaLoP", pred_stalop, runtime_stalop, {}),
    ]

    for method, pred, runtime, tune in specs:
        row: dict[str, Any] = {
            "seed": int(seed), "replicate_id": int(replicate_id),
            "N": int(N), "T": int(T),
            "t_star_idx_0based": int(t_star_idx), "t_star_1based": int(t_star_idx + 1),
            "method": method,
            "mspe": float(np.mean((np.asarray(pred) - target_vec) ** 2)),
            "runtime_sec": float(runtime),
            "sigma2_hat": float(sigma2_hat),
            "sigma2_pairs": int(sigma2_pairs),
            "true_sigma2": float(dgp_config.sigma_u ** 2),
        }
        if method == "Time-Local":
            row.update({"time_window": int(tune["window"]), "time_cv_mse": float(tune["cv_mse"])})
        elif method == "KNN-Covariate":
            row.update({"knn_k": int(tune["knn_k"]), "knn_cv_mse": float(tune["cv_mse"])})
        elif method == "StaLoP":
            row.update({
                "K_mean": float(tuning_df["K"].mean()),
                "q_h_mean": float(tuning_df["q_h"].mean()),
                "eta_mean": float(tuning_df["eta"].mean()),
                "ESS_mean": float(tuning_df["ESS"].mean()),
                "q_h_boundary_rate": float(tuning_df["q_h_boundary"].mean()),
                "eta_boundary_rate": float(tuning_df["eta_boundary"].mean()),
                "K_full_rate": float(tuning_df["K_full"].mean()),
            })
        method_rows.append(row)

    return pd.DataFrame(method_rows), tuning_df


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["T", "N", "method"], as_index=False)
        .agg(
            mspe_mean=("mspe", "mean"),
            mspe_sd=("mspe", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
            mspe_mcse=("mspe", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            runtime_mean=("runtime_sec", "mean"),
            runtime_sd=("runtime_sec", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else np.nan),
            n_rep=("mspe", "size"),
        )
        .sort_values(["T", "N", "method"])
        .reset_index(drop=True)
    )


def deterministic_task_seed(seed0: int, T: int, N: int, replicate_id: int) -> int:
    """Stable signed-63-bit seed independent of task ordering, --part and n_jobs.

    Keeping seeds in signed int64 range prevents pandas from promoting mixed
    int64/uint64 task seeds to float64 during concatenation, which would lose
    exact seed provenance and break checkpoint validation.
    """
    payload = f"{int(seed0)}|{int(T)}|{int(N)}|{int(replicate_id)}".encode("utf-8")
    raw = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little", signed=False)
    return int(raw & ((1 << 63) - 1))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable_protocol(
    seed0: int,
    dgp_config: DGPConfig,
    method_config: MethodConfig,
    grid: TuningGrid,
    t_star_fraction: float,
) -> dict[str, Any]:
    return {
        "seed0": int(seed0),
        "dgp_config": asdict(dgp_config),
        "method_config": asdict(method_config),
        "grid": asdict(grid),
        "t_star_fraction": float(t_star_fraction),
        "core_sha256": sha256_file(__file__),
    }


def protocol_hash(protocol: dict[str, Any]) -> str:
    blob = json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, newline="") as f:
        tmp = Path(f.name)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, newline="") as f:
        tmp = Path(f.name)
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _task_id(T: int, N: int, replicate_id: int) -> str:
    return f"T{int(T)}_N{int(N)}_rep{int(replicate_id):04d}"


def _checkpoint_paths(checkpoint_dir: Path, task_id: str) -> tuple[Path, Path, Path]:
    return (
        checkpoint_dir / f"{task_id}_methods.csv",
        checkpoint_dir / f"{task_id}_tuning.csv",
        checkpoint_dir / f"{task_id}.done.json",
    )


def _save_checkpoint(
    checkpoint_dir: Path,
    task: dict[str, Any],
    methods: pd.DataFrame,
    tuning: pd.DataFrame,
    p_hash: str,
) -> None:
    task_id = _task_id(task["T"], task["N"], task["replicate_id"])
    methods_path, tuning_path, done_path = _checkpoint_paths(checkpoint_dir, task_id)
    _atomic_write_csv(methods, methods_path)
    _atomic_write_csv(tuning, tuning_path)
    done = {
        "task_id": task_id,
        "protocol_hash": p_hash,
        "seed": int(task["seed"]),
        "T": int(task["T"]),
        "N": int(task["N"]),
        "replicate_id": int(task["replicate_id"]),
        "method_rows": int(len(methods)),
        "tuning_rows": int(len(tuning)),
        "methods_sha256": sha256_file(methods_path),
        "tuning_sha256": sha256_file(tuning_path),
    }
    _atomic_write_text(done_path, json.dumps(done, indent=2, sort_keys=True))


def _load_valid_checkpoint(
    checkpoint_dir: Path,
    task: dict[str, Any],
    p_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    task_id = _task_id(task["T"], task["N"], task["replicate_id"])
    methods_path, tuning_path, done_path = _checkpoint_paths(checkpoint_dir, task_id)
    if not (methods_path.exists() and tuning_path.exists() and done_path.exists()):
        return None
    try:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("protocol_hash") != p_hash:
            return None
        if int(done.get("seed")) != int(task["seed"]):
            return None
        if sha256_file(methods_path) != done.get("methods_sha256"):
            return None
        if sha256_file(tuning_path) != done.get("tuning_sha256"):
            return None
        methods = pd.read_csv(methods_path)
        tuning = pd.read_csv(tuning_path)
        if len(methods) != int(done.get("method_rows")) or len(tuning) != int(done.get("tuning_rows")):
            return None
        expected_methods = {"Global Linear", "Time-Local", "KNN-Covariate", "StaLoP"}
        if len(methods) != 4 or set(methods.get("method", pd.Series(dtype=str)).astype(str)) != expected_methods:
            return None
        if len(tuning) != int(task["N"]):
            return None
        for df in (methods, tuning):
            if not {"seed", "replicate_id", "T", "N"}.issubset(df.columns):
                return None
            if not (df["seed"].astype("int64") == np.int64(task["seed"])).all():
                return None
            if not (df["replicate_id"].astype(int) == int(task["replicate_id"])).all():
                return None
            if not (df["T"].astype(int) == int(task["T"])).all():
                return None
            if not (df["N"].astype(int) == int(task["N"])).all():
                return None
        if set(tuning["target_unit"].astype(int)) != set(range(int(task["N"]))):
            return None
        return methods, tuning
    except Exception:
        return None


def _worker(task: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_one_replication(**task)


def run_monte_carlo_parallel(
    settings: list[tuple[int, int]],
    R: int,
    seed0: int = 20260506,
    dgp_config: DGPConfig = DGPConfig(),
    method_config: MethodConfig = MethodConfig(),
    grid: TuningGrid = TuningGrid(),
    t_star_fraction: float = 0.70,
    n_jobs: int = 1,
    save_prefix: str | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if R <= 0:
        raise ValueError("R must be positive")
    if not settings:
        raise ValueError("settings must be nonempty")
    validate_configuration(dgp_config, method_config, grid)
    protocol = _jsonable_protocol(seed0, dgp_config, method_config, grid, t_star_fraction)
    p_hash = protocol_hash(protocol)

    cp_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if cp_dir is not None:
        cp_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = cp_dir / "protocol_manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("protocol_hash") != p_hash:
                raise RuntimeError("checkpoint directory belongs to a different code/protocol hash")
        else:
            _atomic_write_text(
                manifest_path,
                json.dumps({"protocol_hash": p_hash, "protocol": protocol}, indent=2, sort_keys=True),
            )

    tasks: list[dict[str, Any]] = []
    for N, T in settings:
        for rep in range(R):
            tasks.append({
                "N": int(N), "T": int(T),
                "seed": deterministic_task_seed(seed0, T, N, rep),
                "replicate_id": int(rep),
                "dgp_config": dgp_config,
                "method_config": method_config,
                "grid": grid,
                "t_star_fraction": t_star_fraction,
            })

    method_parts: list[pd.DataFrame] = []
    tune_parts: list[pd.DataFrame] = []
    pending: list[dict[str, Any]] = []

    for task in tasks:
        cached = _load_valid_checkpoint(cp_dir, task, p_hash) if (resume and cp_dir is not None) else None
        if cached is None:
            pending.append(task)
        else:
            method_parts.append(cached[0])
            tune_parts.append(cached[1])

    if pending:
        n_jobs_eff = max(1, min(int(n_jobs), len(pending), os.cpu_count() or 1))
        if n_jobs_eff == 1:
            iterator = ((task, _worker(task)) for task in pending)
            for j, (task, out) in enumerate(iterator, 1):
                methods, tuning = out
                if cp_dir is not None:
                    _save_checkpoint(cp_dir, task, methods, tuning, p_hash)
                method_parts.append(methods)
                tune_parts.append(tuning)
                print(
                    f"completed {j}/{len(pending)} T={task['T']} N={task['N']} "
                    f"rep={task['replicate_id']} seed={task['seed']}", flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=n_jobs_eff) as ex:
                futs = {ex.submit(_worker, task): task for task in pending}
                for j, fut in enumerate(as_completed(futs), 1):
                    task = futs[fut]
                    try:
                        methods, tuning = fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Section 5 task failed: T={task['T']} N={task['N']} "
                            f"rep={task['replicate_id']} seed={task['seed']}"
                        ) from exc
                    if cp_dir is not None:
                        _save_checkpoint(cp_dir, task, methods, tuning, p_hash)
                    method_parts.append(methods)
                    tune_parts.append(tuning)
                    print(
                        f"completed {j}/{len(pending)} T={task['T']} N={task['N']} "
                        f"rep={task['replicate_id']} seed={task['seed']}", flush=True,
                    )

    if not method_parts or not tune_parts:
        raise RuntimeError("no Monte Carlo results available")
    results = pd.concat(method_parts, ignore_index=True).sort_values(
        ["T", "N", "replicate_id", "method"], kind="mergesort"
    ).reset_index(drop=True)
    tuning = pd.concat(tune_parts, ignore_index=True).sort_values(
        ["T", "N", "replicate_id", "target_unit"], kind="mergesort"
    ).reset_index(drop=True)
    summary = summarize_results(results)
    if save_prefix:
        _atomic_write_csv(results, Path(f"{save_prefix}_raw.csv"))
        _atomic_write_csv(tuning, Path(f"{save_prefix}_stalop_tuning.csv"))
        _atomic_write_csv(summary, Path(f"{save_prefix}_summary.csv"))
        _atomic_write_text(
            Path(f"{save_prefix}_manifest.json"),
            json.dumps({
                "protocol_hash": p_hash,
                "protocol": protocol,
                "settings": [[int(N), int(T)] for N, T in settings],
                "R": int(R),
                "n_tasks": int(len(tasks)),
            }, indent=2, sort_keys=True),
        )
    return results, tuning, summary


# -----------------------------------------------------------------------------
# Diagnostics / formal launch gates
# -----------------------------------------------------------------------------


def predictive_compatibility_diagnostic(
    N: int,
    T: int,
    seed: int,
    q_h: float = 0.10,
    dgp_config: DGPConfig = DGPConfig(),
    method_config: MethodConfig = MethodConfig(),
    t_star_fraction: float = 0.70,
    max_units: int = 40,
) -> pd.DataFrame:
    """Empirical state discrepancy versus true local-surface discrepancy."""
    rng = np.random.default_rng(seed)
    data = simulate_panel(N, T, dgp_config, rng)
    x1, x2, y, S = data["x1"], data["x2"], data["y"], data["S"]
    t_star_idx = int(np.floor(t_star_fraction * T)) - 1
    t_star_idx = min(max(t_star_idx, 2), T - 1)
    x1_pre, x2_pre, y_pre, times_pre, T_hist = stack_prearrival(x1, x2, y, t_star_idx)

    rows = []
    for i in range(min(N, max_units)):
        x_star = np.array([x1[i, t_star_idx], x2[i, t_star_idx]])
        try:
            comp = build_target_components(
                x1_pre, x2_pre, y_pre, times_pre, x_star, T_hist, q_h, method_config
            )
        except InvalidCandidate:
            continue
        theta = np.asarray(comp["theta"])
        W = np.asarray(comp["W"])
        eligible_idx = np.asarray(comp["eligible_panel_idx"], dtype=int)
        target_regime = int(S[i, t_star_idx])
        b_target = g_regime_local_coefficient(x_star, target_regime)
        for pos, panel_idx in enumerate(eligible_idx):
            hist_regime = int(S[i, panel_idx])
            b_hist = g_regime_local_coefficient(x_star, hist_regime)
            delta = b_hist - b_target
            true_disc = float(delta @ W @ delta)
            rows.append({
                "seed": int(seed), "unit": int(i), "panel_1based": int(panel_idx + 1),
                "theta_hat": float(theta[pos]),
                "true_surface_discrepancy": true_disc,
                "same_future_regime": int(hist_regime == target_regime),
                "hist_regime": hist_regime, "target_regime": target_regime,
            })
    return pd.DataFrame(rows)


def tuning_diagnostics(tuning: pd.DataFrame) -> pd.DataFrame:
    if tuning.empty:
        return pd.DataFrame()
    return (
        tuning.groupby(["T", "N"], as_index=False)
        .agg(
            n_targets=("target_unit", "size"),
            q_h_boundary_rate=("q_h_boundary", "mean"),
            q_h_lower_rate=("q_h_lower", "mean"),
            q_h_upper_rate=("q_h_upper", "mean"),
            eta_boundary_rate=("eta_boundary", "mean"),
            eta_near_uniform_rate=("eta_near_uniform", "mean"),
            eta_upper_rate=("eta_upper", "mean"),
            K_full_rate=("K_full", "mean"),
            K_mean=("K", "mean"),
            q_h_mean=("q_h", "mean"),
            eta_mean=("eta", "mean"),
            ESS_mean=("ESS", "mean"),
            sigma2_mean=("sigma2_hat", "mean"),
        )
        .sort_values(["T", "N"])
        .reset_index(drop=True)
    )



def baseline_tuning_diagnostics(results: pd.DataFrame, grid: TuningGrid = TuningGrid()) -> pd.DataFrame:
    """Summarize baseline tuning without conflating nominal-grid and effective boundaries.

    Time-Local windows larger than the available history all mean "use the full
    history".  Likewise, KNN k values larger than the available number of
    historical observations are clipped to the full history.  We therefore
    report both nominal grid-endpoint rates and effective full-history rates.
    """
    rows: list[dict[str, Any]] = []
    for (T, N), g in results.groupby(["T", "N"]):
        tl = g[g["method"] == "Time-Local"]
        kn = g[g["method"] == "KNN-Covariate"]

        if len(tl):
            tl_hist = tl["t_star_idx_0based"].astype(int)
            time_full = (tl["time_window"].astype(int).to_numpy() >= tl_hist.to_numpy())
            time_endpoint = tl["time_window"].isin({min(grid.time_window_grid), max(grid.time_window_grid)})
        else:
            time_full = np.array([], dtype=bool)
            time_endpoint = pd.Series(dtype=bool)

        if len(kn):
            kn_hist_obs = kn["N"].astype(int).to_numpy() * kn["t_star_idx_0based"].astype(int).to_numpy()
            knn_full = kn["knn_k"].astype(int).to_numpy() >= kn_hist_obs
            knn_endpoint = kn["knn_k"].isin({min(grid.knn_k_grid), max(grid.knn_k_grid)})
        else:
            knn_full = np.array([], dtype=bool)
            knn_endpoint = pd.Series(dtype=bool)

        rows.append({
            "T": int(T), "N": int(N),
            "time_window_mean": float(tl["time_window"].mean()) if len(tl) else np.nan,
            "time_grid_endpoint_rate": float(time_endpoint.mean()) if len(tl) else np.nan,
            "time_full_history_rate": float(np.mean(time_full)) if len(tl) else np.nan,
            "knn_k_mean": float(kn["knn_k"].mean()) if len(kn) else np.nan,
            "knn_grid_endpoint_rate": float(knn_endpoint.mean()) if len(kn) else np.nan,
            "knn_full_history_rate": float(np.mean(knn_full)) if len(kn) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["T", "N"]).reset_index(drop=True)
