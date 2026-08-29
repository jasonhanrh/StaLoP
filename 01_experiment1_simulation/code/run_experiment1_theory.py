from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import scipy

import simulation_core as sim
import experiment1_core as exp1

TARGET_POINTS = ((-0.5, -0.5), (0.0, 0.0), (0.5, 0.5))


def _worker(task: dict):
    return exp1.run_theory_diagnostic_replication(**task)


def _cp_paths(cp: Path, T: int, N: int, rep: int):
    stem = f"T{T}_N{N}_rep{rep:04d}"
    return cp / f"{stem}_candidate.csv", cp / f"{stem}_k.csv", cp / f"{stem}_selected.csv", cp / f"{stem}.done.json"


def _save_cp(cp: Path, task: dict, candidate: pd.DataFrame, kdf: pd.DataFrame, selected: pd.DataFrame, p_hash: str):
    cp.mkdir(parents=True, exist_ok=True)
    ca, kp, sp, dp = _cp_paths(cp, task["T"], task["N"], task["replicate_id"])
    sim._atomic_write_csv(candidate, ca)
    sim._atomic_write_csv(kdf, kp)
    sim._atomic_write_csv(selected, sp)
    done = {
        "protocol_hash": p_hash,
        "seed": int(task["seed"]), "T": int(task["T"]), "N": int(task["N"]), "replicate_id": int(task["replicate_id"]),
        "candidate_sha256": sim.sha256_file(ca), "k_sha256": sim.sha256_file(kp), "selected_sha256": sim.sha256_file(sp),
        "candidate_rows": int(len(candidate)), "k_rows": int(len(kdf)), "selected_rows": int(len(selected)),
    }
    sim._atomic_write_text(dp, json.dumps(done, indent=2, sort_keys=True))


def _load_cp(cp: Path, task: dict, p_hash: str):
    ca, kp, sp, dp = _cp_paths(cp, task["T"], task["N"], task["replicate_id"])
    if not (ca.exists() and kp.exists() and sp.exists() and dp.exists()):
        return None
    try:
        done = json.loads(dp.read_text(encoding="utf-8"))
        if done.get("protocol_hash") != p_hash or int(done.get("seed")) != int(task["seed"]):
            return None
        if sim.sha256_file(ca) != done.get("candidate_sha256") or sim.sha256_file(kp) != done.get("k_sha256") or sim.sha256_file(sp) != done.get("selected_sha256"):
            return None
        c, k, s = pd.read_csv(ca), pd.read_csv(kp), pd.read_csv(sp)
        if len(c) != int(done["candidate_rows"]) or len(k) != int(done["k_rows"]) or len(s) != int(done["selected_rows"]):
            return None
        return c, k, s
    except Exception:
        return None


def _update_candidate_acc(acc: dict, df: pd.DataFrame):
    cols = ["T", "N", "point_id", "x1_star", "x2_star", "target_regime", "K", "q_h", "eta"]
    for row in df.itertuples(index=False):
        d = row._asdict()
        key = (
            int(d["T"]), int(d["N"]), int(d["point_id"]), float(d["x1_star"]), float(d["x2_star"]),
            int(d["target_regime"]), int(d["K"]), float(d["q_h"]), float(d["eta"]),
        )
        vals = acc.get(key)
        if vals is None:
            vals = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            acc[key] = vals
        vals[0] += 1
        vals[1] += float(d["sq_error"])
        vals[2] += float(d["criterion"])
        vals[3] += float(d["R"])
        vals[4] += float(d["h"])
        vals[5] += float(d["GDF"])
        vals[6] += float(d["ESS"])


def _candidate_acc_to_df(acc: dict) -> pd.DataFrame:
    rows = []
    for key, v in acc.items():
        T, N, point_id, x1s, x2s, regime, K, qh, eta = key
        n = int(v[0])
        rows.append({
            "T": T, "N": N, "point_id": point_id, "x1_star": x1s, "x2_star": x2s,
            "target_regime": regime, "K": K, "q_h": qh, "eta": eta,
            "true_future_mspe": v[1] / n,
            "estimated_mspe": v[2] / n,
            "uncorrected_risk": v[3] / n,
            "h_mean": v[4] / n,
            "gdf_mean": v[5] / n,
            "ess_mean": v[6] / n,
            "n": n,
        })
    return pd.DataFrame(rows).sort_values(["T", "N", "point_id", "target_regime", "K", "q_h", "eta"], kind="mergesort").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1 theory/risk diagnostics.")
    parser.add_argument("--formal", action="store_true", help="lock theory diagnostics to R=300 and untouched formal seed0=20260815")
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--seed0", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--settings", choices=["representative", "all"], default="representative")
    parser.add_argument("--outdir", type=str, default="experiment1_theory_pilot")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    THEORY_PILOT_SEED0 = 2026081501
    THEORY_FORMAL_SEED0 = 20260815
    if args.formal:
        if args.R not in (None, 300):
            parser.error("--formal fixes R=300")
        if args.seed0 not in (None, THEORY_FORMAL_SEED0):
            parser.error("--formal uses seed0=20260815")
        R = 300
        seed0 = THEORY_FORMAL_SEED0
    else:
        R = 20 if args.R is None else int(args.R)
        seed0 = THEORY_PILOT_SEED0 if args.seed0 is None else int(args.seed0)

    settings = [(100, 30), (200, 120)] if args.settings == "representative" else [(100, 30), (200, 30), (100, 120), (200, 120)]
    dgp = sim.DGPConfig(sigma_a=0.0, regime_mode="predictable_cycle")
    method = sim.MethodConfig(K_state=1)
    grid = sim.TuningGrid()
    sim.validate_configuration(dgp, method, grid)

    protocol = {
        "seed0": int(seed0), "R": int(R), "settings": [[N, T] for N, T in settings],
        "target_points": [list(p) for p in TARGET_POINTS],
        "dgp_config": dgp.__dict__, "method_config": method.__dict__, "grid": grid.__dict__,
        "simulation_core_sha256": sim.sha256_file(sim.__file__), "experiment1_core_sha256": sim.sha256_file(exp1.__file__),
        "runner_sha256": sim.sha256_file(__file__), "experiment": "Experiment1-theory-v1",
    }
    p_hash = sim.protocol_hash(protocol)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cp = outdir / "checkpoints"
    manifest = {**protocol, "protocol_hash": p_hash, "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "argv": list(sys.argv)}
    sim._atomic_write_text(outdir / "launch.json", json.dumps(manifest, indent=2, sort_keys=True))
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    tasks = []
    for N, T in settings:
        for rep in range(int(R)):
            tasks.append({
                "N": N, "T": T, "seed": sim.deterministic_task_seed(seed0, T, N, rep), "replicate_id": rep,
                "target_points": TARGET_POINTS, "dgp_config": dgp, "method_config": method, "grid": grid,
            })

    cand_acc: dict = {}
    k_parts: list[pd.DataFrame] = []
    selected_parts: list[pd.DataFrame] = []
    pending = []
    for task in tasks:
        cached = _load_cp(cp, task, p_hash) if (not args.no_resume and cp.exists()) else None
        if cached is None:
            pending.append(task)
        else:
            c, k, s = cached
            _update_candidate_acc(cand_acc, c)
            k_parts.append(k)
            selected_parts.append(s)

    tic = perf_counter()
    if pending:
        n_eff = max(1, min(int(args.n_jobs), len(pending), os.cpu_count() or 1))
        if n_eff == 1:
            for j, task in enumerate(pending, 1):
                c, k, s = _worker(task)
                _save_cp(cp, task, c, k, s, p_hash)
                _update_candidate_acc(cand_acc, c)
                k_parts.append(k); selected_parts.append(s)
                print(f"completed {j}/{len(pending)} T={task['T']} N={task['N']} rep={task['replicate_id']}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=n_eff) as ex:
                futs = {ex.submit(_worker, task): task for task in pending}
                for j, fut in enumerate(as_completed(futs), 1):
                    task = futs[fut]
                    try:
                        c, k, s = fut.result()
                    except Exception as exc:
                        raise RuntimeError(f"theory diagnostic failed T={task['T']} N={task['N']} rep={task['replicate_id']}") from exc
                    _save_cp(cp, task, c, k, s, p_hash)
                    _update_candidate_acc(cand_acc, c)
                    k_parts.append(k); selected_parts.append(s)
                    print(f"completed {j}/{len(pending)} T={task['T']} N={task['N']} rep={task['replicate_id']}", flush=True)

    candidate_agg = _candidate_acc_to_df(cand_acc)
    k_rows = pd.concat(k_parts, ignore_index=True) if k_parts else pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()

    risk_tracking = exp1.summarize_risk_tracking(candidate_agg)
    oracle_detail, oracle_summary = exp1.summarize_oracle_configuration(candidate_agg, selected)
    k_summary = exp1.summarize_k_bias_variance(k_rows)

    sim._atomic_write_csv(candidate_agg, outdir / "candidate_risk_tracking.csv")
    sim._atomic_write_csv(risk_tracking, outdir / "risk_tracking_summary.csv")
    sim._atomic_write_csv(k_summary, outdir / "k_bias_variance.csv")
    sim._atomic_write_csv(oracle_detail, outdir / "oracle_configuration_detail.csv")
    sim._atomic_write_csv(oracle_summary, outdir / "oracle_configuration_summary.csv")
    sim._atomic_write_csv(selected, outdir / "adaptive_selected_rows.csv")

    elapsed = perf_counter() - tic
    manifest["elapsed_sec"] = float(elapsed)
    sim._atomic_write_text(outdir / "launch.json", json.dumps(manifest, indent=2, sort_keys=True))

    print("\nRisk tracking summary:")
    print(risk_tracking.to_string(index=False))
    print("\nOracle configuration summary:")
    print(oracle_summary.to_string(index=False))
    print("\nK bias-variance summary:")
    print(k_summary.to_string(index=False))
    print(f"elapsed_sec={elapsed:.3f}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
