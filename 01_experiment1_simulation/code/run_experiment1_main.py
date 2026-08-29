from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
from time import perf_counter
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import scipy

import simulation_core as sim
import experiment1_core as exp1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final Experiment 1 main comparison.")
    parser.add_argument("--formal", action="store_true", help="lock R=300 and untouched formal seed family")
    parser.add_argument("--R", type=int, default=None)
    parser.add_argument("--seed0", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--part", choices=["all", "t30", "t120"], default="all")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    PILOT_SEED0 = 2026081401
    FORMAL_SEED0 = 20260814
    if args.formal:
        if args.R not in (None, 300):
            parser.error("--formal fixes R=300")
        if args.seed0 not in (None, FORMAL_SEED0):
            parser.error("--formal uses seed0=20260814")
        R, seed0, mode = 300, FORMAL_SEED0, "formal"
        outdir_arg = args.outdir or "experiment1_main_R300"
    else:
        R = 5 if args.R is None else int(args.R)
        seed0 = PILOT_SEED0 if args.seed0 is None else int(args.seed0)
        mode = "pilot"
        outdir_arg = args.outdir or "experiment1_main_pilot"

    settings: list[tuple[int, int]] = []
    if args.part in {"all", "t30"}:
        settings.extend([(100, 30), (200, 30)])
    if args.part in {"all", "t120"}:
        settings.extend([(100, 120), (200, 120)])

    outdir = Path(outdir_arg)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(outdir / f"experiment1_main_{args.part}")
    cp_dir = outdir / "checkpoints"

    launch = {
        "runner_sha256": sim.sha256_file(__file__),
        "simulation_core_sha256": sim.sha256_file(sim.__file__),
        "experiment1_core_sha256": sim.sha256_file(exp1.__file__),
        "argv": list(sys.argv),
        "R": int(R),
        "run_mode": mode,
        "seed0": int(seed0),
        "n_jobs_requested": int(args.n_jobs),
        "part": args.part,
        "settings": [[int(N), int(T)] for N, T in settings],
        "resume": bool(not args.no_resume),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": sim.np.__version__,
        "pandas": sim.pd.__version__,
        "scipy": scipy.__version__,
    }
    sim._atomic_write_text(outdir / "launch.json", json.dumps(launch, indent=2, sort_keys=True))
    if args.dry_run:
        print(json.dumps(launch, indent=2, sort_keys=True))
        return

    tic = perf_counter()
    results, tuning, summary = exp1.run_experiment1_main_parallel(
        settings=settings,
        R=R,
        seed0=seed0,
        dgp_config=sim.DGPConfig(sigma_a=0.0, regime_mode="predictable_cycle"),
        method_config=sim.MethodConfig(K_state=1),
        grid=sim.TuningGrid(),
        n_jobs=args.n_jobs,
        save_prefix=prefix,
        checkpoint_dir=cp_dir,
        resume=not args.no_resume,
    )
    print(summary.to_string(index=False))
    print("\nStaLoP tuning / compatibility diagnostics:")
    diag = (
        tuning.groupby(["T", "N"], as_index=False)
        .agg(
            targets=("K", "size"),
            compatible_retained_pct=("compatible_retained_pct", "mean"),
            compatible_recall_pct=("compatible_recall_pct", "mean"),
            compatible_enrichment=("compatible_enrichment", "mean"),
            theta_compatible_mean=("theta_compatible_mean", "mean"),
            theta_incompatible_mean=("theta_incompatible_mean", "mean"),
            K_mean=("K", "mean"),
            q_h_mean=("q_h", "mean"),
            eta_mean=("eta", "mean"),
            ESS_mean=("ESS", "mean"),
            sigma2_mean=("sigma2_hat", "mean"),
            q_h_boundary=("q_h_boundary", "mean"),
            eta_boundary=("eta_boundary", "mean"),
        )
    )
    print(diag.to_string(index=False))
    sim._atomic_write_csv(diag, outdir / "experiment1_main_compatibility_summary.csv")

    elapsed = perf_counter() - tic
    print(f"elapsed_sec={elapsed:.3f}")
    launch["elapsed_sec"] = float(elapsed)
    sim._atomic_write_text(outdir / "launch.json", json.dumps(launch, indent=2, sort_keys=True))


if __name__ == "__main__":
    mp.freeze_support()
    main()
