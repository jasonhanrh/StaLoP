"""Shared production utilities for the final Section-6 applications.

This module implements the finite-grid GDF/MSPE inner selector required by
the FINAL Section 4 supplied on 2026-08-16.  It is used by the sequence and
CES applications after their application-specific predictive states and the
current bandwidth-specific kernel/W have been constructed.  It selects K and
eta exactly conditional on that h.  The sequence runner profiles this inner
selector over h; the CES deterministic-simulator application uses a
prespecified singleton h, which Section 4 explicitly permits.

The dynamic-panel variable-selection application uses the separate
multidimensional panel engine in ``dynamic_panel_multidim.py`` because there
h changes the predictive-state construction and must be recomputed for every
candidate bandwidth.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
import hashlib
import json
import os
import tempfile

# Prevent BLAS oversubscription under ProcessPoolExecutor.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from numpy.linalg import LinAlgError


class InvalidCandidate(RuntimeError):
    pass


@dataclass(frozen=True)
class PointPanelGrid:
    eta_grid: tuple[float, ...]
    rank_count_grid: tuple[int, ...] = ()
    rank_fraction_grid: tuple[float, ...] = ()

    def validate(self) -> None:
        if not self.eta_grid or any(float(e) <= 0.0 for e in self.eta_grid):
            raise ValueError("eta_grid must contain positive values")
        if any(int(k) <= 0 for k in self.rank_count_grid):
            raise ValueError("rank_count_grid must be positive")
        if any(not (0.0 < float(f) <= 1.0) for f in self.rank_fraction_grid):
            raise ValueError("rank_fraction_grid must lie in (0,1]")
        if not self.rank_count_grid and not self.rank_fraction_grid:
            raise ValueError("a K grid is required")


def standardize_theta(theta: np.ndarray, delta: float = 1e-8) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be one-dimensional")
    if theta.size <= 1:
        return np.zeros_like(theta)
    sd = float(np.std(theta, ddof=1))
    if not np.isfinite(sd):
        raise InvalidCandidate("non-finite discrepancy standard deviation")
    return (theta - float(np.mean(theta))) / (sd + float(delta))


def rank_candidate_counts(
    n_eligible: int,
    fractions: Iterable[float],
    counts: Iterable[int] = (),
    min_k: int = 1,
) -> tuple[int, ...]:
    if n_eligible < min_k:
        return tuple()
    out = {
        min(max(int(np.floor(float(f) * n_eligible)), min_k), n_eligible)
        for f in fractions
    }
    out.update(min(max(int(k), min_k), n_eligible) for k in counts)
    out.add(n_eligible)
    return tuple(sorted(out))


def _gram_valid(A_raw: np.ndarray, Q: float, max_condition: float) -> bool:
    if not np.isfinite(Q) or Q <= 0.0:
        return False
    A = np.asarray(A_raw, dtype=float) / Q
    try:
        eig = np.linalg.eigvalsh(A)
    except LinAlgError:
        return False
    lo, hi = float(eig[0]), float(eig[-1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 1e-12:
        return False
    return bool(hi / lo <= float(max_condition))


def _tie_key(row: dict[str, float]) -> tuple[int, float, float]:
    eta = max(float(row["eta"]), 1e-300)
    return (int(row["K"]), abs(float(np.log(eta))), eta)


def _is_better(row: dict[str, float], best: dict[str, float] | None) -> bool:
    if best is None:
        return True
    c, b = float(row["criterion"]), float(best["criterion"])
    tol = 1e-14 + 1e-12 * max(1.0, abs(c), abs(b))
    if c < b - tol:
        return True
    if abs(c - b) <= tol:
        return _tie_key(row) < _tie_key(best)
    return False


def tune_point_panel_gdf(
    X: np.ndarray,
    y: np.ndarray,
    states_target_coordinates: np.ndarray,
    x_star: np.ndarray,
    z_star: np.ndarray,
    sigma2_hat: float,
    grid: PointPanelGrid,
    *,
    theta_delta: float = 1e-8,
    w_ridge: float = 1e-8,
    final_ridge: float = 0.0,
    max_condition: float = 1e8,
    kernel_weight: np.ndarray | None = None,
    W_override: np.ndarray | None = None,
    return_all: bool = False,
) -> dict[str, float] | tuple[dict[str, float], list[dict[str, float]]]:
    """Exact Algorithm-1/2 specialization for one observation per panel.

    ``states_target_coordinates`` and ``z_star`` must be coefficient vectors in
    the same local basis coordinates used for the final fit.  ``kernel_weight``
    is the prespecified K_h factor.  Current sequence/CES production uses the
    singleton/global kernel, i.e. all ones.
    """
    grid.validate()
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    states = np.asarray(states_target_coordinates, dtype=float)
    x_star = np.asarray(x_star, dtype=float)
    z_star = np.asarray(z_star, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    L = d + 1
    if y.shape != (n,) or states.shape != (n, L) or z_star.shape != (L,) or x_star.shape != (d,):
        raise ValueError("dimension mismatch in point-panel tuner")
    if n < L:
        raise InvalidCandidate("insufficient historical panels")
    if not np.isfinite(sigma2_hat) or sigma2_hat < 0.0:
        raise ValueError("sigma2_hat must be finite and nonnegative")
    if kernel_weight is None:
        kernel = np.ones(n, dtype=float)
    else:
        kernel = np.asarray(kernel_weight, dtype=float)
        if kernel.shape != (n,) or np.any(kernel < 0.0) or not np.all(np.isfinite(kernel)):
            raise ValueError("invalid kernel weights")
    if float(kernel.sum()) <= 0.0:
        raise InvalidCandidate("zero kernel support")

    Phi = np.column_stack([np.ones(n), X - x_star])
    if W_override is None:
        W = (Phi.T @ (kernel[:, None] * Phi)) / float(kernel.sum())
        W = W + float(w_ridge) * np.eye(L)
    else:
        W = np.asarray(W_override, dtype=float)
        if W.shape != (L, L) or not np.all(np.isfinite(W)):
            raise ValueError("invalid W_override")
        W = 0.5 * (W + W.T)
    diff = states - z_star[None, :]
    theta = np.maximum(np.einsum("ni,ij,nj->n", diff, W, diff), 0.0)
    theta_std = standardize_theta(theta, theta_delta)
    order = np.argsort(theta, kind="mergesort")

    K_values = rank_candidate_counts(n, grid.rank_fraction_grid, grid.rank_count_grid, min_k=L)
    if not K_values:
        raise InvalidCandidate("no estimable K candidate")

    ordered = order
    Phi_o = Phi[ordered]
    y_o = y[ordered]
    kernel_o = kernel[ordered]
    outer_o = Phi_o[:, :, None] * Phi_o[:, None, :]

    etas = np.asarray(grid.eta_grid, dtype=float)
    log_beta = -etas[:, None] * theta_std[ordered][None, :]
    log_beta -= np.max(log_beta, axis=1, keepdims=True)
    beta = np.exp(log_beta)
    raw = beta * kernel_o[None, :]

    cum_Q = np.cumsum(raw, axis=1)
    cum_A = np.cumsum(raw[:, :, None, None] * outer_o[None, :, :, :], axis=1)
    cum_b = np.cumsum(raw[:, :, None] * Phi_o[None, :, :] * y_o[None, :, None], axis=1)
    cum_y2 = np.cumsum(raw * (y_o * y_o)[None, :], axis=1)
    raw2 = raw * raw
    cum_V = np.cumsum(raw2[:, :, None, None] * outer_o[None, :, :, :], axis=1)
    cum_obs2 = np.cumsum(raw2, axis=1)

    candidates: list[dict[str, float]] = []
    best: dict[str, float] | None = None
    eye = np.eye(L)
    for ie, eta in enumerate(etas):
        for K in K_values:
            j = int(K) - 1
            Q = float(cum_Q[ie, j])
            A_raw = cum_A[ie, j]
            if not _gram_valid(A_raw, Q, max_condition):
                continue
            M = A_raw + float(final_ridge) * Q * eye
            try:
                c = np.linalg.solve(M, cum_b[ie, j])
                solved_V = np.linalg.solve(M, cum_V[ie, j])
            except LinAlgError:
                continue
            sse = float(cum_y2[ie, j] - 2.0 * c @ cum_b[ie, j] + c @ A_raw @ c)
            R = max(sse / Q, 0.0)
            GDF = float(np.trace(solved_V) / Q)
            criterion = float(R + 2.0 * float(sigma2_hat) * GDF - float(sigma2_hat))
            obs2 = float(cum_obs2[ie, j])
            ESS = float(Q * Q / obs2) if obs2 > 0.0 else np.nan
            row = {
                "pred": float(c[0]),
                "criterion": criterion,
                "R": float(R),
                "GDF": float(GDF),
                "K": int(K),
                "eta": float(eta),
                "ESS": ESS,
                "sigma2_hat": float(sigma2_hat),
            }
            if return_all:
                candidates.append(row)
            if _is_better(row, best):
                best = row
    if best is None:
        raise InvalidCandidate("no valid GDF candidate")
    best = {**best, "n_candidates": int(len(candidates) if return_all else len(etas) * len(K_values))}
    return (best, candidates) if return_all else best


def explicit_point_panel_candidate(
    X: np.ndarray,
    y: np.ndarray,
    states_target_coordinates: np.ndarray,
    x_star: np.ndarray,
    z_star: np.ndarray,
    sigma2_hat: float,
    eta: float,
    K: int,
    *,
    theta_delta: float = 1e-8,
    w_ridge: float = 1e-8,
    kernel_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Slow observation-level D/H reference for acceptance testing."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    states = np.asarray(states_target_coordinates, float)
    x_star = np.asarray(x_star, float)
    z_star = np.asarray(z_star, float)
    n, d = X.shape; L = d + 1
    kernel = np.ones(n) if kernel_weight is None else np.asarray(kernel_weight, float)
    Phi = np.column_stack([np.ones(n), X - x_star])
    W = Phi.T @ (kernel[:, None] * Phi) / kernel.sum() + w_ridge * np.eye(L)
    diff = states - z_star[None, :]
    theta = np.maximum(np.einsum("ni,ij,nj->n", diff, W, diff), 0.0)
    ts = standardize_theta(theta, theta_delta)
    order = np.argsort(theta, kind="mergesort")
    K = min(max(int(K), L), n)
    keep = order[:K]
    raw = np.zeros(n)
    scores = np.exp(-float(eta) * ts)
    raw[keep] = scores[keep] * kernel[keep]
    Q = raw.sum()
    if Q <= 0:
        raise InvalidCandidate("zero weights")
    w = raw / Q
    D = np.diag(w)
    A = Phi.T @ D @ Phi
    c = np.linalg.solve(A, Phi.T @ (w * y))
    resid = y - Phi @ c
    R = float(np.sum(w * resid * resid))
    H = Phi @ np.linalg.solve(A, Phi.T @ D)
    GDF = float(np.trace(D @ H))
    crit = float(R + 2.0 * sigma2_hat * GDF - sigma2_hat)
    return {"pred": float(c[0]), "R": R, "GDF": GDF, "criterion": crit}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write_json(obj: Any, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        Path(tmp).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
