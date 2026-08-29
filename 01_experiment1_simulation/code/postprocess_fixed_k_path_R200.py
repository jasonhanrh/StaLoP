"""Read-only fixed-(q_h, eta) K-path diagnostic for Experiment 1.

This script consumes existing theory-run candidate checkpoints.  It neither
runs a simulation nor changes any production source or output.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import simulation_core as sim


Q_H_REF = 0.10
ETA_REF = 1.00
REQUIRED = {
    "T", "N", "point_id", "x1_star", "x2_star", "target_regime", "K",
    "q_h", "h", "eta", "pred", "truth", "criterion", "R", "GDF", "ESS",
    "error", "sq_error",
}
SOURCE_FILES = (
    "simulation_core.py",
    "experiment1_core.py",
    "run_experiment1_theory.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_verified_candidates(checkpoint_dir: Path) -> tuple[pd.DataFrame, int]:
    done_paths = sorted(checkpoint_dir.glob("*.done.json"))
    candidate_paths = sorted(checkpoint_dir.glob("*_candidate.csv"))
    if not done_paths or len(done_paths) != len(candidate_paths):
        raise RuntimeError("candidate/done checkpoint files are missing or incomplete")

    frames: list[pd.DataFrame] = []
    for done_path in done_paths:
        done = json.loads(done_path.read_text(encoding="utf-8"))
        candidate_path = checkpoint_dir / done_path.name.replace(".done.json", "_candidate.csv")
        if not candidate_path.is_file():
            raise RuntimeError(f"missing candidate checkpoint: {candidate_path.name}")
        expected = done.get("candidate_sha256")
        actual = sha256_file(candidate_path)
        if expected != actual:
            raise RuntimeError(f"candidate checkpoint hash mismatch: {candidate_path.name}")
        frame = pd.read_csv(candidate_path)
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise RuntimeError(f"{candidate_path.name} lacks required columns: {sorted(missing)}")
        frames.append(frame)

    if {p.name for p in candidate_paths} != {
        p.name.replace(".done.json", "_candidate.csv") for p in done_paths
    }:
        raise RuntimeError("unpaired candidate checkpoint files detected")
    return pd.concat(frames, ignore_index=True), len(candidate_paths)


def main() -> None:
    root = Path(__file__).resolve().parent
    theory_dir = root / "experiment1_theory_R200"
    checkpoint_dir = theory_dir / "checkpoints"
    outdir = root / "experiment1_fixed_k_path_R200"
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"missing theory checkpoint directory: {checkpoint_dir}")
    if outdir.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {outdir}")

    source_hashes_before = {name: sha256_file(root / name) for name in SOURCE_FILES}
    grid = sim.TuningGrid()
    q_values = np.asarray(tuple(float(v) for v in grid.q_h_grid))
    eta_values = np.asarray(tuple(float(v) for v in grid.eta_grid))
    if not np.any(q_values == Q_H_REF):
        raise RuntimeError("q_h_ref=0.10 does not exist exactly in the frozen TuningGrid")
    if not np.any(eta_values == ETA_REF):
        raise RuntimeError("eta_ref=1.00 does not exist exactly in the frozen TuningGrid")

    candidates, checkpoint_count = read_verified_candidates(checkpoint_dir)
    fixed = candidates.loc[
        np.isclose(candidates["q_h"], Q_H_REF) & np.isclose(candidates["eta"], ETA_REF)
    ].copy()
    if fixed.empty:
        raise RuntimeError("no exact fixed-reference candidate rows were found")
    if not ((fixed["q_h"] == Q_H_REF).all() and (fixed["eta"] == ETA_REF).all()):
        raise RuntimeError("np.isclose selected non-exact q_h or eta rows; refusing nearest-grid matching")
    if set(fixed[["T", "N"]].drop_duplicates().itertuples(index=False, name=None)) != {(30, 100), (120, 200)}:
        raise RuntimeError("fixed-reference candidates do not cover exactly the representative settings")

    stratum_cols = ["T", "N", "point_id", "x1_star", "x2_star", "target_regime", "K"]
    base_cols = stratum_cols[:-1]
    rows: list[dict[str, float | int]] = []
    for key, group in fixed.groupby(stratum_cols, sort=True, dropna=False):
        pred = group["pred"].to_numpy(dtype=float)
        truth = group["truth"].to_numpy(dtype=float)
        if not np.allclose(truth, truth[0], rtol=0.0, atol=1e-14):
            raise RuntimeError(f"truth is not fixed within stratum {key}")
        mean_pred = float(np.mean(pred))
        bias2 = float((mean_pred - truth[0]) ** 2)
        variance = float(np.mean((pred - mean_pred) ** 2))
        true_mspe = float(np.mean((pred - truth) ** 2))
        identity_error = float(true_mspe - bias2 - variance)
        row = dict(zip(stratum_cols, key))
        row.update({
            "q_h_ref": Q_H_REF, "eta_ref": ETA_REF,
            "h_mean": float(group["h"].mean()),
            "bias2": bias2, "variance": variance,
            "true_future_mspe": true_mspe,
            "estimated_mspe": float(group["criterion"].mean()),
            "uncorrected_risk": float(group["R"].mean()),
            "GDF_mean": float(group["GDF"].mean()),
            "ESS_mean": float(group["ESS"].mean()),
            "n": int(len(group)), "identity_error": identity_error,
        })
        rows.append(row)
    detail = pd.DataFrame(rows).sort_values(stratum_cols, kind="mergesort").reset_index(drop=True)
    if not np.isfinite(detail.select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("non-finite value in fixed-K stratum output")
    if not (detail["identity_error"].abs() < 1e-12).all():
        raise RuntimeError("bias-variance identity failed at the required 1e-12 tolerance")

    max_coverage = detail.groupby(base_cols, dropna=False)["n"].transform("max")
    detail["max_n"] = max_coverage.astype(int)
    detail["coverage_fraction"] = detail["n"] / detail["max_n"]
    retained = detail.loc[detail["n"] == detail["max_n"]].copy()
    excluded = detail.loc[detail["n"] != detail["max_n"]].copy()
    excluded["exclusion_reason"] = "n_less_than_maximum_fixed_reference_coverage"

    weighted_fields = [
        "h_mean", "bias2", "variance", "true_future_mspe", "estimated_mspe",
        "uncorrected_risk", "GDF_mean", "ESS_mean", "coverage_fraction",
    ]
    summary_rows: list[dict[str, float | int]] = []
    for (t_value, n_value, k_value), group in retained.groupby(["T", "N", "K"], sort=True):
        weights = group["n"].to_numpy(dtype=float)
        row: dict[str, float | int] = {
            "T": int(t_value), "N": int(n_value), "K": int(k_value),
            "q_h_ref": Q_H_REF, "eta_ref": ETA_REF,
        }
        for field in weighted_fields:
            row[field] = float(np.average(group[field].to_numpy(dtype=float), weights=weights))
        row["n"] = int(weights.sum())
        row["max_abs_identity_error"] = float(group["identity_error"].abs().max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["T", "N", "K"], kind="mergesort").reset_index(drop=True)
    if not np.isfinite(summary.select_dtypes(include="number").to_numpy()).all():
        raise RuntimeError("non-finite value in fixed-K summary output")

    source_hashes_after = {name: sha256_file(root / name) for name in SOURCE_FILES}
    if source_hashes_before != source_hashes_after:
        raise RuntimeError("an existing production source changed while post-processing")

    outdir.mkdir()
    atomic_write_csv(detail, outdir / "fixed_k_path_stratum_detail.csv")
    atomic_write_csv(summary, outdir / "fixed_k_path_summary.csv")
    atomic_write_csv(excluded, outdir / "fixed_k_path_excluded_coverage.csv")
    manifest = {
        "q_h_ref": Q_H_REF,
        "eta_ref": ETA_REF,
        "source_theory_directory": str(theory_dir),
        "checkpoint_files_read": checkpoint_count,
        "sha256": {
            "postprocess_fixed_k_path.py": sha256_file(Path(__file__)),
            **source_hashes_after,
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": {
            "fixed_k_path_stratum_detail.csv": int(len(detail)),
            "fixed_k_path_summary.csv": int(len(summary)),
            "fixed_k_path_excluded_coverage.csv": int(len(excluded)),
        },
        "max_abs_identity_error": float(detail["identity_error"].abs().max()),
    }
    (outdir / "fixed_k_path_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary.to_csv(index=False), end="")
    print(f"max_abs_identity_error={manifest['max_abs_identity_error']:.17g}")
    print(f"excluded_incomplete_coverage_rows={len(excluded)}")


if __name__ == "__main__":
    main()
