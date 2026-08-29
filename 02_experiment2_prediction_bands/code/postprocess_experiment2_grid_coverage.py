"""Read frozen Experiment-2 NPZ outputs and summarize 15-point coverage."""

from pathlib import Path
import csv
import argparse
import numpy as np


CRITICAL_90 = 1.6448536269514722
INPUTS = (
    "experiment2_formal_R500",
    "experiment2_extension_N2000_R500",
    "experiment2_extension_N5000_R500",
    "experiment2_N10000_R500",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="experiment2_grid_coverage_final")
    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=False)
    point_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for directory in INPUTS:
        z = np.load(Path(directory) / "experiment2_grid_results.npz")
        grid = z["grid"]
        for n in np.unique(z["N"]):
            use = z["N"] == n
            for method, key in (("full", "z_full"), ("naive", "z_naive")):
                values = z[key][use]
                coverage = np.mean(np.abs(values) <= CRITICAL_90, axis=0)
                for j, value in enumerate(coverage, start=1):
                    point_rows.append(
                        {
                            "N": int(n), "R": int(use.sum()), "method": method,
                            "grid_index": j, "u": float(grid[j - 1, 0]),
                            "v": float(grid[j - 1, 1]), "coverage_90": float(value),
                        }
                    )
                summary_rows.append(
                    {
                        "N": int(n), "R": int(use.sum()), "method": method,
                        "pointwise_coverage_mean": float(coverage.mean()),
                        "pointwise_coverage_sd_grid_ddof1": float(coverage.std(ddof=1)),
                        "pointwise_coverage_min": float(coverage.min()),
                        "pointwise_coverage_max": float(coverage.max()),
                        "pooled_coverage_check": float(np.mean(np.abs(values) <= CRITICAL_90)),
                        "mean_minus_pooled": float(coverage.mean() - np.mean(np.abs(values) <= CRITICAL_90)),
                    }
                )
    for filename, rows in (("experiment2_pointwise_by_grid.csv", point_rows),
                           ("experiment2_pointwise_grid_summary.csv", summary_rows)):
        with (outdir / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
