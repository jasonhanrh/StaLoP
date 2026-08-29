"""Experiment 1 extensions for the final Section 5 StaLoP simulation.

This module deliberately imports the frozen, audited ``simulation_core`` and
adds only the diagnostics requested by Professor Lin:

* Oracle-Compatible StaLoP using the true recurrent regime;
* MAE in addition to MSPE;
* retained-panel compatibility diagnostics;
* candidate-level true-risk / estimated-MSPE / uncorrected-risk tracking;
* K-profiled empirical bias-variance diagnostics.

The StaLoP estimator and Algorithm-2 tuning mathematics are not redefined here.
All estimated StaLoP fits call ``simulation_core.tune_stalop_gdf_target``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable
import hashlib
import json
import os

import numpy as np
import pandas as pd

import simulation_core as sim


# -----------------------------------------------------------------------------
# Oracle-compatible benchmark
# -----------------------------------------------------------------------------


def _oracle_subset_candidate(
    panel_A: np.ndarray,
    panel_b: np.ndarray,
    panel_weight: np.ndarray,
    panel_y2: np.ndarray,
    retained_panel_idx0: np.ndarray,
    sigma2_hat: float,
    method_config: sim.MethodConfig,
) -> dict[str, float]:
    """Evaluate the manuscript GDF criterion on a prespecified panel subset.

    The oracle benchmark uses true regime information only to define the subset.
    Within that subset panel weights are uniform, so this isolates perfect
    compatibility recovery from the estimated discrepancy/ranking step.
    """
    retained = np.asarray(retained_panel_idx0, dtype=int)
    if retained.size == 0:
        raise sim.InvalidCandidate("oracle-compatible retained set is empty")

    Q = float(np.sum(panel_weight[retained]))
    if Q <= 0.0:
        raise sim.InvalidCandidate("oracle-compatible subset has zero local support")

    A_raw = np.sum(panel_A[retained], axis=0)
    b_raw = np.sum(panel_b[retained], axis=0)
    c_hat = sim.solve_sufficient_stats_no_fallback(
        A_raw,
        b_raw,
        Q,
        method_config.final_ridge,
        method_config.require_full_rank_before_ridge,
        method_config.max_design_condition,
    )
    y2_raw = float(np.sum(panel_y2[retained]))
    sse_raw = y2_raw - 2.0 * float(c_hat @ b_raw) + float(c_hat @ A_raw @ c_hat)
    R = max(float(sse_raw / Q), 0.0)

    # Uniform panel multiplier beta_t=1 on the oracle-compatible subset.
    # Therefore V_raw = sum beta_t^2 A_t = sum A_t over the same subset.
    M_reg = A_raw + method_config.final_ridge * Q * np.eye(A_raw.shape[0])
    GDF = float(np.trace(sim.solve_normal_equations(M_reg, A_raw)) / Q)
    criterion = float(R + 2.0 * float(sigma2_hat) * GDF - float(sigma2_hat))

    # Binary kernel + uniform panel weights => observation-level ESS is the
    # number of locally supported observations in the compatible subset.
    obs2 = float(np.sum(panel_weight[retained]))
    ESS = float(Q * Q / obs2) if obs2 > 0.0 else np.nan
    return {
        "pred": float(c_hat[0]),
        "criterion": criterion,
        "R": R,
        "GDF": GDF,
        "ESS": ESS,
        "K": int(retained.size),
    }


def tune_oracle_compatible_target(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    x_star: np.ndarray,
    T_hist: int,
    historical_regime0: np.ndarray,
    target_regime: int,
    sigma2_hat: float,
    grid: sim.TuningGrid,
    method_config: sim.MethodConfig,
) -> dict[str, float]:
    """Oracle-compatible StaLoP benchmark.

    The benchmark uses the same eligible historical panels as estimated StaLoP
    (panels 2,...,T_hist), but retains exactly those whose *realized* regime is
    the target regime. All such compatible panels are retained. The bandwidth
    is selected over the same q_h grid using the same GDF-corrected MSPE
    criterion. No eta is needed because true compatibility replaces the
    estimated discrepancy weighting.
    """
    x_star = np.asarray(x_star, dtype=float)
    regimes = np.asarray(historical_regime0, dtype=int)
    if regimes.size < T_hist:
        raise ValueError("historical_regime0 must contain at least T_hist entries")

    eligible = np.arange(1, T_hist, dtype=int)  # same as Algorithm-1 eligible set
    compatible = eligible[regimes[eligible] == int(target_regime)]
    if compatible.size == 0:
        raise sim.InvalidCandidate("no oracle-compatible eligible historical panel")

    static = sim.prepare_target_static(x1_pre, x2_pre, x_star)
    dist = np.asarray(static["dist"], dtype=float)
    Phi = np.asarray(static["Phi"], dtype=float)
    q_values = np.asarray(tuple(float(q) for q in grid.q_h_grid), dtype=float)
    h_values = np.maximum(
        np.asarray(np.quantile(dist, q_values), dtype=float),
        float(method_config.min_bandwidth),
    )

    best: dict[str, float] | None = None
    for q_h, h in zip(q_values, h_values):
        K_vals = (dist <= float(h)).astype(float)
        if float(K_vals.sum()) <= 0.0:
            continue
        panel_A, panel_b, panel_weight, panel_y2 = sim.kernel_panel_sufficient_stats(
            Phi, y_pre, K_vals, times_pre, T_hist
        )
        try:
            row = _oracle_subset_candidate(
                panel_A,
                panel_b,
                panel_weight,
                panel_y2,
                compatible,
                sigma2_hat,
                method_config,
            )
        except sim.InvalidCandidate:
            continue
        row = {
            **row,
            "q_h": float(q_h),
            "h": float(h),
            "compatible_count": int(compatible.size),
        }
        if best is None:
            best = row
        else:
            c = float(row["criterion"])
            b = float(best["criterion"])
            tol = 1e-14 + 1e-12 * max(1.0, abs(c), abs(b))
            if c < b - tol or (abs(c - b) <= tol and float(row["q_h"]) < float(best["q_h"])):
                best = row

    if best is None:
        raise RuntimeError("no valid oracle-compatible bandwidth candidate")
    return best


# -----------------------------------------------------------------------------
# Selected-panel compatibility diagnostics
# -----------------------------------------------------------------------------


def selected_compatibility_diagnostics(
    x1_pre: np.ndarray,
    x2_pre: np.ndarray,
    y_pre: np.ndarray,
    times_pre: np.ndarray,
    x_star: np.ndarray,
    T_hist: int,
    best: dict[str, float],
    historical_regime0: np.ndarray,
    target_regime: int,
    method_config: sim.MethodConfig,
) -> dict[str, float]:
    """Recover the selected retained set and compare it with true regimes.

    This is a diagnostic-only recomputation at the already selected q_h. It is
    deliberately excluded from the reported StaLoP runtime.
    """
    static = sim.prepare_target_static(x1_pre, x2_pre, x_star)
    comp = sim.build_target_components(
        x1_pre,
        x2_pre,
        y_pre,
        times_pre,
        x_star,
        T_hist,
        float(best["q_h"]),
        method_config,
        static=static,
        h_override=float(best["h"]),
    )
    eligible = np.asarray(comp["eligible_panel_idx"], dtype=int)
    order = np.asarray(comp["order"], dtype=int)
    theta = np.asarray(comp["theta"], dtype=float)
    K = min(max(int(best["K"]), 1), len(order))
    retained_pos = order[:K]
    retained_panel_idx0 = eligible[retained_pos]

    regimes = np.asarray(historical_regime0, dtype=int)
    eligible_compatible = regimes[eligible] == int(target_regime)
    retained_compatible = regimes[retained_panel_idx0] == int(target_regime)

    n_true = int(np.sum(eligible_compatible))
    n_retained_true = int(np.sum(retained_compatible))
    precision = float(n_retained_true / K) if K > 0 else np.nan
    recall = float(n_retained_true / n_true) if n_true > 0 else np.nan

    theta_compatible = theta[eligible_compatible]
    theta_incompatible = theta[~eligible_compatible]
    mean_theta_compatible = float(np.mean(theta_compatible)) if theta_compatible.size else np.nan
    mean_theta_incompatible = float(np.mean(theta_incompatible)) if theta_incompatible.size else np.nan
    theta_ratio = (
        float(mean_theta_incompatible / max(mean_theta_compatible, 1e-15))
        if np.isfinite(mean_theta_compatible) and np.isfinite(mean_theta_incompatible)
        else np.nan
    )
    base_rate = float(n_true / len(eligible)) if len(eligible) else np.nan
    enrichment = float(precision / base_rate) if np.isfinite(base_rate) and base_rate > 0 else np.nan

    return {
        "compatible_retained_pct": 100.0 * precision,
        "compatible_recall_pct": 100.0 * recall,
        "compatible_base_pct": 100.0 * base_rate,
        "compatible_enrichment": enrichment,
        "theta_compatible_mean": mean_theta_compatible,
        "theta_incompatible_mean": mean_theta_incompatible,
        "theta_incompatible_to_compatible_ratio": theta_ratio,
        "n_compatible_eligible": n_true,
        "n_compatible_retained": n_retained_true,
    }


# -----------------------------------------------------------------------------
# Main Experiment-1 replication
# -----------------------------------------------------------------------------


def run_experiment1_replication(
    N: int,
    T: int,
    seed: int,
    replicate_id: int = 0,
    dgp_config: sim.DGPConfig = sim.DGPConfig(),
    method_config: sim.MethodConfig = sim.MethodConfig(),
    grid: sim.TuningGrid = sim.TuningGrid(),
    t_star_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One main Experiment-1 replication with Lin-requested diagnostics."""
    sim.validate_configuration(dgp_config, method_config, grid)
    rng = np.random.default_rng(int(seed))
    data = sim.simulate_panel(int(N), int(T), dgp_config, rng)
    x1, x2, y = data["x1"], data["x2"], data["y"]
    target = data["m_true"]
    regimes = np.asarray(data["S"][0], dtype=int)

    t_star_idx = int(np.floor(float(t_star_fraction) * T)) - 1
    t_star_idx = min(max(t_star_idx, 2), T - 1)
    x1_pre, x2_pre, y_pre, times_pre, T_hist = sim.stack_prearrival(x1, x2, y, t_star_idx)
    sigma2_hat, sigma2_pairs = sim.estimate_noise_variance_within_panel_nn(
        x1_pre,
        x2_pre,
        y_pre,
        times_pre,
        T_hist,
        method_config.noise_pair_fraction,
        method_config.min_noise_pairs,
    )

    # Baselines: exactly the same implementations and independent tuning as the
    # audited production engine.
    tic = perf_counter()
    beta_global = sim.fit_global_linear(x1_pre, x2_pre, y_pre)
    pred_global = sim.predict_linear_beta(beta_global, x1[:, t_star_idx], x2[:, t_star_idx])
    runtime_global = perf_counter() - tic

    tic = perf_counter()
    time_tune = sim.tune_time_local(x1, x2, y, t_star_idx, grid)
    beta_time = sim.fit_time_local(
        x1_pre, x2_pre, y_pre, times_pre, T_hist, int(time_tune["window"])
    )
    pred_time = sim.predict_linear_beta(beta_time, x1[:, t_star_idx], x2[:, t_star_idx])
    runtime_time = perf_counter() - tic

    tic = perf_counter()
    knn_tune = sim.tune_knn_covariate(x1, x2, y, t_star_idx, grid)
    pred_knn = sim.predict_knn_covariate_batch(
        x1_pre,
        x2_pre,
        y_pre,
        x1[:, t_star_idx],
        x2[:, t_star_idx],
        int(knn_tune["knn_k"]),
    )
    runtime_knn = perf_counter() - tic

    # Estimated StaLoP: exact Algorithm-2 tuner. Save best rows first so the
    # compatibility diagnostic recomputation is excluded from method runtime.
    tic = perf_counter()
    pred_stalop = np.empty(N, dtype=float)
    best_rows: list[dict[str, float]] = []
    q_min, q_max = min(grid.q_h_grid), max(grid.q_h_grid)
    e_min, e_max = min(grid.eta_grid), max(grid.eta_grid)
    for i in range(N):
        x_star = np.array([x1[i, t_star_idx], x2[i, t_star_idx]], dtype=float)
        best = sim.tune_stalop_gdf_target(
            x1_pre,
            x2_pre,
            y_pre,
            times_pre,
            x_star,
            T_hist,
            sigma2_hat,
            grid,
            method_config,
            return_all=False,
        )
        assert isinstance(best, dict)
        pred_stalop[i] = float(best["pred"])
        best_rows.append(best)
    runtime_stalop = perf_counter() - tic

    # Oracle-compatible benchmark. True regime information replaces estimated
    # ranking only; local prediction and GDF bandwidth tuning remain data-based.
    tic = perf_counter()
    pred_oracle = np.empty(N, dtype=float)
    oracle_rows: list[dict[str, float]] = []
    target_regime = int(regimes[t_star_idx])
    for i in range(N):
        x_star = np.array([x1[i, t_star_idx], x2[i, t_star_idx]], dtype=float)
        row = tune_oracle_compatible_target(
            x1_pre,
            x2_pre,
            y_pre,
            times_pre,
            x_star,
            T_hist,
            regimes[:T_hist],
            target_regime,
            sigma2_hat,
            grid,
            method_config,
        )
        pred_oracle[i] = float(row["pred"])
        oracle_rows.append(row)
    runtime_oracle = perf_counter() - tic

    # Diagnostic-only compatibility reconstruction, outside StaLoP runtime.
    tuning_rows: list[dict[str, Any]] = []
    for i, best in enumerate(best_rows):
        x_star = np.array([x1[i, t_star_idx], x2[i, t_star_idx]], dtype=float)
        compat = selected_compatibility_diagnostics(
            x1_pre,
            x2_pre,
            y_pre,
            times_pre,
            x_star,
            T_hist,
            best,
            regimes[:T_hist],
            target_regime,
            method_config,
        )
        tuning_rows.append({
            "seed": int(seed),
            "replicate_id": int(replicate_id),
            "T": int(T),
            "N": int(N),
            "target_unit": int(i),
            "target_regime": int(target_regime),
            "q_h": float(best["q_h"]),
            "h": float(best["h"]),
            "eta": float(best["eta"]),
            "K": int(best["K"]),
            "criterion": float(best["criterion"]),
            "R": float(best["R"]),
            "GDF": float(best["GDF"]),
            "ESS": float(best["ESS"]),
            "sigma2_hat": float(sigma2_hat),
            "sigma2_pairs": int(sigma2_pairs),
            "n_valid_qh": int(best["n_valid_qh"]),
            "n_candidates": int(best["n_candidates"]),
            "q_h_boundary": int(float(best["q_h"]) in {q_min, q_max}),
            "q_h_lower": int(float(best["q_h"]) == q_min),
            "q_h_upper": int(float(best["q_h"]) == q_max),
            "eta_boundary": int(float(best["eta"]) in {e_min, e_max}),
            "eta_near_uniform": int(float(best["eta"]) == e_min),
            "eta_upper": int(float(best["eta"]) == e_max),
            "K_full": int(int(best["K"]) == (T_hist - 1)),
            **compat,
        })
    tuning_df = pd.DataFrame(tuning_rows)

    target_vec = np.asarray(target[:, t_star_idx], dtype=float)
    method_rows: list[dict[str, Any]] = []
    specs = [
        ("Global Linear", pred_global, runtime_global, {}),
        ("Time-Local", pred_time, runtime_time, time_tune),
        ("KNN-Covariate", pred_knn, runtime_knn, knn_tune),
        ("StaLoP", pred_stalop, runtime_stalop, {}),
        ("Oracle-Compatible StaLoP", pred_oracle, runtime_oracle, {}),
    ]
    for method, pred, runtime, tune in specs:
        error = np.asarray(pred, dtype=float) - target_vec
        row: dict[str, Any] = {
            "seed": int(seed),
            "replicate_id": int(replicate_id),
            "N": int(N),
            "T": int(T),
            "t_star_idx_0based": int(t_star_idx),
            "t_star_1based": int(t_star_idx + 1),
            "target_regime": int(target_regime),
            "method": method,
            "mspe": float(np.mean(error * error)),
            "mae": float(np.mean(np.abs(error))),
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
                "compatible_retained_pct": float(tuning_df["compatible_retained_pct"].mean()),
                "compatible_recall_pct": float(tuning_df["compatible_recall_pct"].mean()),
                "compatible_enrichment": float(tuning_df["compatible_enrichment"].mean()),
                "q_h_boundary_rate": float(tuning_df["q_h_boundary"].mean()),
                "eta_boundary_rate": float(tuning_df["eta_boundary"].mean()),
                "K_full_rate": float(tuning_df["K_full"].mean()),
            })
        elif method == "Oracle-Compatible StaLoP":
            row.update({
                "compatible_retained_pct": 100.0,
                "oracle_K_mean": float(np.mean([r["K"] for r in oracle_rows])),
                "oracle_q_h_mean": float(np.mean([r["q_h"] for r in oracle_rows])),
                "oracle_h_mean": float(np.mean([r["h"] for r in oracle_rows])),
            })
        method_rows.append(row)

    return pd.DataFrame(method_rows), tuning_df


def summarize_experiment1_main(results: pd.DataFrame) -> pd.DataFrame:
    def _sd(x: pd.Series) -> float:
        return float(np.std(x, ddof=1)) if len(x) > 1 else np.nan

    out = (
        results.groupby(["T", "N", "method"], as_index=False)
        .agg(
            mspe_mean=("mspe", "mean"),
            mspe_sd=("mspe", _sd),
            mspe_mcse=("mspe", lambda x: _sd(x) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", _sd),
            runtime_mean=("runtime_sec", "mean"),
            runtime_sd=("runtime_sec", _sd),
            compatible_retained_pct=("compatible_retained_pct", "mean"),
            n_rep=("mspe", "size"),
        )
        .sort_values(["T", "N", "method"], kind="mergesort")
        .reset_index(drop=True)
    )
    return out


# -----------------------------------------------------------------------------
# Main Experiment-1 parallel/checkpoint runner
# -----------------------------------------------------------------------------


def _protocol_main(
    seed0: int,
    dgp_config: sim.DGPConfig,
    method_config: sim.MethodConfig,
    grid: sim.TuningGrid,
    t_star_fraction: float,
) -> dict[str, Any]:
    return {
        "seed0": int(seed0),
        "dgp_config": asdict(dgp_config),
        "method_config": asdict(method_config),
        "grid": asdict(grid),
        "t_star_fraction": float(t_star_fraction),
        "simulation_core_sha256": sim.sha256_file(sim.__file__),
        "experiment1_core_sha256": sim.sha256_file(__file__),
        "experiment": "Experiment1-main-v1",
    }


def _main_task_id(T: int, N: int, replicate_id: int) -> str:
    return f"T{int(T)}_N{int(N)}_rep{int(replicate_id):04d}"


def _main_cp_paths(cp: Path, task_id: str) -> tuple[Path, Path, Path]:
    return (
        cp / f"{task_id}_methods.csv",
        cp / f"{task_id}_tuning.csv",
        cp / f"{task_id}.done.json",
    )


def _save_main_cp(
    cp: Path,
    task: dict[str, Any],
    methods: pd.DataFrame,
    tuning: pd.DataFrame,
    p_hash: str,
) -> None:
    task_id = _main_task_id(task["T"], task["N"], task["replicate_id"])
    mp, tp, dp = _main_cp_paths(cp, task_id)
    sim._atomic_write_csv(methods, mp)
    sim._atomic_write_csv(tuning, tp)
    done = {
        "protocol_hash": p_hash,
        "seed": int(task["seed"]),
        "T": int(task["T"]),
        "N": int(task["N"]),
        "replicate_id": int(task["replicate_id"]),
        "method_rows": int(len(methods)),
        "tuning_rows": int(len(tuning)),
        "methods_sha256": sim.sha256_file(mp),
        "tuning_sha256": sim.sha256_file(tp),
    }
    sim._atomic_write_text(dp, json.dumps(done, indent=2, sort_keys=True))


def _load_main_cp(
    cp: Path,
    task: dict[str, Any],
    p_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    task_id = _main_task_id(task["T"], task["N"], task["replicate_id"])
    mp, tp, dp = _main_cp_paths(cp, task_id)
    if not (mp.exists() and tp.exists() and dp.exists()):
        return None
    try:
        done = json.loads(dp.read_text(encoding="utf-8"))
        if done.get("protocol_hash") != p_hash or int(done.get("seed")) != int(task["seed"]):
            return None
        if sim.sha256_file(mp) != done.get("methods_sha256") or sim.sha256_file(tp) != done.get("tuning_sha256"):
            return None
        methods = pd.read_csv(mp)
        tuning = pd.read_csv(tp)
        expected = {
            "Global Linear",
            "Time-Local",
            "KNN-Covariate",
            "StaLoP",
            "Oracle-Compatible StaLoP",
        }
        if len(methods) != 5 or set(methods["method"].astype(str)) != expected:
            return None
        if len(tuning) != int(task["N"]):
            return None
        if set(tuning["target_unit"].astype(int)) != set(range(int(task["N"]))):
            return None
        for df in (methods, tuning):
            if not (df["seed"].astype("int64") == np.int64(task["seed"])).all():
                return None
            if not (df["replicate_id"].astype(int) == int(task["replicate_id"])).all():
                return None
            if not (df["T"].astype(int) == int(task["T"])).all() or not (df["N"].astype(int) == int(task["N"])).all():
                return None
        return methods, tuning
    except Exception:
        return None


def _main_worker(task: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_experiment1_replication(**task)


def run_experiment1_main_parallel(
    settings: list[tuple[int, int]],
    R: int,
    seed0: int,
    dgp_config: sim.DGPConfig,
    method_config: sim.MethodConfig,
    grid: sim.TuningGrid,
    t_star_fraction: float = 0.70,
    n_jobs: int = 1,
    save_prefix: str | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if R <= 0 or not settings:
        raise ValueError("positive R and nonempty settings are required")
    sim.validate_configuration(dgp_config, method_config, grid)
    protocol = _protocol_main(seed0, dgp_config, method_config, grid, t_star_fraction)
    p_hash = sim.protocol_hash(protocol)

    cp = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if cp is not None:
        cp.mkdir(parents=True, exist_ok=True)
        manifest = cp / "protocol_manifest.json"
        if manifest.exists():
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("protocol_hash") != p_hash:
                raise RuntimeError("checkpoint directory belongs to a different Experiment-1 protocol")
        else:
            sim._atomic_write_text(manifest, json.dumps({"protocol_hash": p_hash, "protocol": protocol}, indent=2, sort_keys=True))

    tasks: list[dict[str, Any]] = []
    for N, T in settings:
        for rep in range(int(R)):
            tasks.append({
                "N": int(N),
                "T": int(T),
                "seed": sim.deterministic_task_seed(seed0, T, N, rep),
                "replicate_id": int(rep),
                "dgp_config": dgp_config,
                "method_config": method_config,
                "grid": grid,
                "t_star_fraction": float(t_star_fraction),
            })

    method_parts: list[pd.DataFrame] = []
    tune_parts: list[pd.DataFrame] = []
    pending: list[dict[str, Any]] = []
    for task in tasks:
        cached = _load_main_cp(cp, task, p_hash) if (resume and cp is not None) else None
        if cached is None:
            pending.append(task)
        else:
            method_parts.append(cached[0])
            tune_parts.append(cached[1])

    if pending:
        n_eff = max(1, min(int(n_jobs), len(pending), os.cpu_count() or 1))
        if n_eff == 1:
            for j, task in enumerate(pending, 1):
                methods, tuning = _main_worker(task)
                if cp is not None:
                    _save_main_cp(cp, task, methods, tuning, p_hash)
                method_parts.append(methods)
                tune_parts.append(tuning)
                print(f"completed {j}/{len(pending)} T={task['T']} N={task['N']} rep={task['replicate_id']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=n_eff) as ex:
                futs = {ex.submit(_main_worker, task): task for task in pending}
                for j, fut in enumerate(as_completed(futs), 1):
                    task = futs[fut]
                    try:
                        methods, tuning = fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Experiment-1 main task failed: T={task['T']} N={task['N']} rep={task['replicate_id']}"
                        ) from exc
                    if cp is not None:
                        _save_main_cp(cp, task, methods, tuning, p_hash)
                    method_parts.append(methods)
                    tune_parts.append(tuning)
                    print(f"completed {j}/{len(pending)} T={task['T']} N={task['N']} rep={task['replicate_id']}", flush=True)

    results = pd.concat(method_parts, ignore_index=True).sort_values(
        ["T", "N", "replicate_id", "method"], kind="mergesort"
    ).reset_index(drop=True)
    tuning = pd.concat(tune_parts, ignore_index=True).sort_values(
        ["T", "N", "replicate_id", "target_unit"], kind="mergesort"
    ).reset_index(drop=True)
    summary = summarize_experiment1_main(results)

    if save_prefix:
        sim._atomic_write_csv(results, Path(f"{save_prefix}_raw.csv"))
        sim._atomic_write_csv(tuning, Path(f"{save_prefix}_stalop_tuning.csv"))
        sim._atomic_write_csv(summary, Path(f"{save_prefix}_summary.csv"))
        sim._atomic_write_text(
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
# Theory diagnostics: candidate risk tracking + K bias/variance
# -----------------------------------------------------------------------------


def _best_row_for_fixed_K(candidates: pd.DataFrame, K: int) -> pd.Series:
    sub = candidates[candidates["K"].astype(int) == int(K)]
    if sub.empty:
        raise sim.InvalidCandidate("no valid candidate for requested K")
    best: dict[str, float] | None = None
    best_idx: int | None = None
    for idx, row in sub.iterrows():
        d = row.to_dict()
        if sim._candidate_is_better(d, best):
            best = d
            best_idx = int(idx)
    assert best_idx is not None
    return candidates.loc[best_idx]


def run_theory_diagnostic_replication(
    N: int,
    T: int,
    seed: int,
    replicate_id: int,
    target_points: tuple[tuple[float, float], ...],
    dgp_config: sim.DGPConfig,
    method_config: sim.MethodConfig,
    grid: sim.TuningGrid,
    t_star_fraction: float = 0.70,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Candidate-level diagnostics for one Monte Carlo replication.

    Returns three compact objects:
    1. candidate rows (one row per candidate and fixed target point);
    2. K-profiled prediction rows (h,eta reselected by GDF at each fixed K);
    3. adaptive Algorithm-2 selected rows.
    """
    rng = np.random.default_rng(int(seed))
    data = sim.simulate_panel(N, T, dgp_config, rng)
    x1, x2, y = data["x1"], data["x2"], data["y"]
    regimes = np.asarray(data["S"][0], dtype=int)
    t_star_idx = int(np.floor(float(t_star_fraction) * T)) - 1
    t_star_idx = min(max(t_star_idx, 2), T - 1)
    x1_pre, x2_pre, y_pre, times_pre, T_hist = sim.stack_prearrival(x1, x2, y, t_star_idx)
    sigma2_hat, _ = sim.estimate_noise_variance_within_panel_nn(
        x1_pre,
        x2_pre,
        y_pre,
        times_pre,
        T_hist,
        method_config.noise_pair_fraction,
        method_config.min_noise_pairs,
    )
    target_regime = int(regimes[t_star_idx])

    candidate_parts: list[pd.DataFrame] = []
    k_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for point_id, xy in enumerate(target_points):
        x_star = np.asarray(xy, dtype=float)
        truth = float(sim.g_regime(float(x_star[0]), float(x_star[1]), target_regime))
        best, candidates = sim.tune_stalop_gdf_target(
            x1_pre,
            x2_pre,
            y_pre,
            times_pre,
            x_star,
            T_hist,
            sigma2_hat,
            grid,
            method_config,
            return_all=True,
        )
        assert isinstance(best, dict) and isinstance(candidates, pd.DataFrame)
        c = candidates.copy()
        c["seed"] = int(seed)
        c["replicate_id"] = int(replicate_id)
        c["T"] = int(T)
        c["N"] = int(N)
        c["point_id"] = int(point_id)
        c["x1_star"] = float(x_star[0])
        c["x2_star"] = float(x_star[1])
        c["target_regime"] = int(target_regime)
        c["truth"] = truth
        c["error"] = c["pred"].astype(float) - truth
        c["sq_error"] = c["error"] ** 2
        candidate_parts.append(c[[
            "seed", "replicate_id", "T", "N", "point_id", "x1_star", "x2_star",
            "target_regime", "truth", "q_h", "h", "eta", "K", "pred", "criterion",
            "R", "GDF", "ESS", "sigma2_hat", "error", "sq_error"
        ]])

        selected_rows.append({
            "seed": int(seed),
            "replicate_id": int(replicate_id),
            "T": int(T),
            "N": int(N),
            "point_id": int(point_id),
            "x1_star": float(x_star[0]),
            "x2_star": float(x_star[1]),
            "target_regime": int(target_regime),
            "truth": truth,
            "pred": float(best["pred"]),
            "error": float(best["pred"] - truth),
            "sq_error": float((best["pred"] - truth) ** 2),
            "q_h": float(best["q_h"]),
            "h": float(best["h"]),
            "eta": float(best["eta"]),
            "K": int(best["K"]),
            "criterion": float(best["criterion"]),
            "R": float(best["R"]),
            "GDF": float(best["GDF"]),
        })

        for K in sorted(c["K"].astype(int).unique()):
            row = _best_row_for_fixed_K(c, int(K))
            pred = float(row["pred"])
            k_rows.append({
                "seed": int(seed),
                "replicate_id": int(replicate_id),
                "T": int(T),
                "N": int(N),
                "point_id": int(point_id),
                "x1_star": float(x_star[0]),
                "x2_star": float(x_star[1]),
                "target_regime": int(target_regime),
                "K": int(K),
                "q_h": float(row["q_h"]),
                "h": float(row["h"]),
                "eta": float(row["eta"]),
                "criterion": float(row["criterion"]),
                "R": float(row["R"]),
                "pred": pred,
                "truth": truth,
                "error": pred - truth,
                "sq_error": (pred - truth) ** 2,
            })

    return pd.concat(candidate_parts, ignore_index=True), pd.DataFrame(k_rows), pd.DataFrame(selected_rows)


def aggregate_candidate_risk(candidate_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate-level estimated and Monte Carlo future risks.

    Stratification by fixed target point and target regime keeps the future mean
    fixed within each stratum. q_h is the finite-grid tuning label; h is reported
    as its realized mean over Monte Carlo replicates.
    """
    keys = ["T", "N", "point_id", "x1_star", "x2_star", "target_regime", "K", "q_h", "eta"]
    return (
        candidate_rows.groupby(keys, as_index=False)
        .agg(
            true_future_mspe=("sq_error", "mean"),
            estimated_mspe=("criterion", "mean"),
            uncorrected_risk=("R", "mean"),
            h_mean=("h", "mean"),
            gdf_mean=("GDF", "mean"),
            ess_mean=("ESS", "mean"),
            n=("sq_error", "size"),
        )
        .sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
    )


def _full_coverage_candidates(candidate_agg: pd.DataFrame) -> pd.DataFrame:
    """Keep candidates observed in every replication available to their stratum.

    Local-design validity can remove a candidate in a particular replication.
    Risk comparisons must not reward candidates that are observed only on easy
    replications, so oracle/risk-tracking summaries use only maximal-coverage
    candidates within each fixed point/regime stratum.
    """
    strata = ["T", "N", "point_id", "x1_star", "x2_star", "target_regime"]
    tmp = candidate_agg.copy()
    tmp["max_n_stratum"] = tmp.groupby(strata)["n"].transform("max")
    return tmp[tmp["n"] == tmp["max_n_stratum"]].copy()


def summarize_risk_tracking(candidate_agg: pd.DataFrame) -> pd.DataFrame:
    """Compare GDF-corrected and uncorrected criteria with true future risk."""
    candidate_agg = _full_coverage_candidates(candidate_agg)
    rows: list[dict[str, Any]] = []
    for (T, N), g in candidate_agg.groupby(["T", "N"], sort=True):
        truth = g["true_future_mspe"].to_numpy(float)
        est = g["estimated_mspe"].to_numpy(float)
        unc = g["uncorrected_risk"].to_numpy(float)
        rmse_est = float(np.sqrt(np.mean((est - truth) ** 2)))
        rmse_unc = float(np.sqrt(np.mean((unc - truth) ** 2)))
        mae_est = float(np.mean(np.abs(est - truth)))
        mae_unc = float(np.mean(np.abs(unc - truth)))
        pearson_est = float(np.corrcoef(est, truth)[0, 1]) if len(g) > 1 else np.nan
        pearson_unc = float(np.corrcoef(unc, truth)[0, 1]) if len(g) > 1 else np.nan
        rows.append({
            "T": int(T),
            "N": int(N),
            "gdf_corrected_rmse_to_true_risk": rmse_est,
            "uncorrected_rmse_to_true_risk": rmse_unc,
            "gdf_corrected_mae_to_true_risk": mae_est,
            "uncorrected_mae_to_true_risk": mae_unc,
            "gdf_corrected_pearson": pearson_est,
            "uncorrected_pearson": pearson_unc,
            "gdf_rmse_improvement_pct": 100.0 * (rmse_unc - rmse_est) / max(rmse_unc, 1e-15),
        })
    return pd.DataFrame(rows)


def summarize_oracle_configuration(
    candidate_agg: pd.DataFrame,
    selected_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Criterion-selected fixed configuration versus true-risk oracle.

    For each fixed target-point/regime stratum, the Monte Carlo oracle is the
    finite-grid configuration minimizing empirical true future MSPE. The
    criterion-selected fixed configuration minimizes average estimated MSPE in
    that same stratum. This yields a nonnegative finite-grid regret by
    construction and directly validates the risk criterion. The operational
    target-wise adaptive Algorithm-2 MSPE is reported separately.
    """
    strata = ["T", "N", "point_id", "x1_star", "x2_star", "target_regime"]
    candidate_agg = _full_coverage_candidates(candidate_agg)
    detail: list[dict[str, Any]] = []
    for key, g in candidate_agg.groupby(strata, sort=True):
        oracle = g.loc[g["true_future_mspe"].idxmin()]
        chosen = g.loc[g["estimated_mspe"].idxmin()]
        sel = selected_rows
        mask = np.ones(len(sel), dtype=bool)
        for col, val in zip(strata, key if isinstance(key, tuple) else (key,)):
            mask &= sel[col].to_numpy() == val
        adaptive_mspe = float(sel.loc[mask, "sq_error"].mean()) if np.any(mask) else np.nan
        detail.append({
            **{col: val for col, val in zip(strata, key if isinstance(key, tuple) else (key,))},
            "oracle_mspe": float(oracle["true_future_mspe"]),
            "oracle_K": int(oracle["K"]),
            "oracle_q_h": float(oracle["q_h"]),
            "oracle_eta": float(oracle["eta"]),
            "criterion_selected_mspe": float(chosen["true_future_mspe"]),
            "criterion_selected_K": int(chosen["K"]),
            "criterion_selected_q_h": float(chosen["q_h"]),
            "criterion_selected_h_mean": float(chosen["h_mean"]),
            "criterion_selected_eta": float(chosen["eta"]),
            "regret": float(chosen["true_future_mspe"] - oracle["true_future_mspe"]),
            "adaptive_algorithm_mspe": adaptive_mspe,
        })
    detail_df = pd.DataFrame(detail)
    summary = (
        detail_df.groupby(["T", "N"], as_index=False)
        .agg(
            oracle_mspe=("oracle_mspe", "mean"),
            criterion_selected_mspe=("criterion_selected_mspe", "mean"),
            regret=("regret", "mean"),
            adaptive_algorithm_mspe=("adaptive_algorithm_mspe", "mean"),
            selected_K_mean=("criterion_selected_K", "mean"),
            selected_h_mean=("criterion_selected_h_mean", "mean"),
            selected_eta_mean=("criterion_selected_eta", "mean"),
        )
    )
    return detail_df, summary


def summarize_k_bias_variance(k_rows: pd.DataFrame) -> pd.DataFrame:
    """Empirical bias^2, variance and MSPE for K-profiled estimators.

    h and eta are reselected by the same GDF criterion conditional on each fixed
    K. Bias and variance are computed within fixed target-point/regime strata,
    where the true target mean is constant, and then weighted by stratum sample
    size. ``variance`` uses denominator n so that bias^2 + variance = MSPE up to
    floating-point rounding within each stratum.
    """
    stratum_keys = ["T", "N", "K", "point_id", "x1_star", "x2_star", "target_regime"]
    strata_rows: list[dict[str, Any]] = []
    for key, g in k_rows.groupby(stratum_keys, sort=True):
        pred = g["pred"].to_numpy(float)
        truth = float(g["truth"].iloc[0])
        mean_pred = float(np.mean(pred))
        bias2 = float((mean_pred - truth) ** 2)
        variance = float(np.mean((pred - mean_pred) ** 2))
        mspe = float(np.mean((pred - truth) ** 2))
        vals = key if isinstance(key, tuple) else (key,)
        strata_rows.append({
            **{col: val for col, val in zip(stratum_keys, vals)},
            "bias2": bias2,
            "variance": variance,
            "true_future_mspe": mspe,
            "identity_error": float(mspe - bias2 - variance),
            "n": int(len(g)),
            "estimated_mspe": float(g["criterion"].mean()),
            "uncorrected_risk": float(g["R"].mean()),
            "q_h_mean": float(g["q_h"].mean()),
            "h_mean": float(g["h"].mean()),
            "eta_mean": float(g["eta"].mean()),
        })
    strata = pd.DataFrame(strata_rows)

    out_rows: list[dict[str, Any]] = []
    for (T, N, K), g in strata.groupby(["T", "N", "K"], sort=True):
        w = g["n"].to_numpy(float)
        w /= np.sum(w)
        out_rows.append({
            "T": int(T),
            "N": int(N),
            "K": int(K),
            "bias2": float(np.sum(w * g["bias2"].to_numpy(float))),
            "variance": float(np.sum(w * g["variance"].to_numpy(float))),
            "true_future_mspe": float(np.sum(w * g["true_future_mspe"].to_numpy(float))),
            "estimated_mspe": float(np.sum(w * g["estimated_mspe"].to_numpy(float))),
            "uncorrected_risk": float(np.sum(w * g["uncorrected_risk"].to_numpy(float))),
            "q_h_mean": float(np.sum(w * g["q_h_mean"].to_numpy(float))),
            "h_mean": float(np.sum(w * g["h_mean"].to_numpy(float))),
            "eta_mean": float(np.sum(w * g["eta_mean"].to_numpy(float))),
            "max_abs_identity_error": float(np.max(np.abs(g["identity_error"].to_numpy(float)))),
            "n": int(np.sum(g["n"])),
        })
    return pd.DataFrame(out_rows)

