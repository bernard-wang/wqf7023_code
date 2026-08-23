from .io_utils import read_las_points, nearest_tide, cell_indices, load_tide

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.stats import binned_statistic_2d


def depth_sanity_filter(z, ceiling=None, margin=5.0):
    if ceiling is None:
        ceiling = float(np.median(z)) + margin
    return z <= ceiling


def filter_outliers_mad(x, y, z, k=3.0, res=4.0, thresh=0.6):
    ix = (x // res).astype(np.int64)
    iy = (y // res).astype(np.int64)
    df = pd.DataFrame({"ix": ix, "iy": iy, "z": z})
    med = df.groupby(["ix", "iy"])["z"].transform("median")
    mad = df.groupby(["ix", "iy"])["z"].transform(
        lambda s: (s - s.median()).abs().median()
    )
    keep = np.abs(z - med) <= np.maximum(k * 1.4826 * mad.values, thresh)
    return keep


def across_track(x, y):
    """PCA on horizontal coords: minor axis is the across-track direction."""
    c = np.column_stack([x - x.mean(), y - y.mean()])
    _, evecs = np.linalg.eigh(np.cov(c.T))
    return c @ evecs[:, 0]


# def outlier_stats(x, y, z, cell=CELL_M, thresh=THRESH_M, shallow=SHALLOW_M):
#     ix = (x // cell).astype(np.int64)
#     iy = (y // cell).astype(np.int64)
#     df = pd.DataFrame({"ix": ix, "iy": iy, "z": z})
#     med = df.groupby(["ix", "iy"])["z"].transform("median").values
#     dev = z - med

#     return {
#         "n_points": len(z),
#         "outlier_pct": float((np.abs(dev) > thresh).mean() * 100),
#         "shallow_pct": float((dev > shallow).mean() * 100),
#         "deep_pct": float((dev < -shallow).mean() * 100),
#         "dev_std_cm": float(dev.std() * 100),
#         "dev_p99_cm": float(np.percentile(np.abs(dev), 99) * 100),
#         "max_shallow_dev": float(dev.max()),
#         "z_span_m": float(z.max() - z.min()),
#     }


def grid_one_file(x, y, z, x_bins, y_bins):
    H, W = len(y_bins) - 1, len(x_bins) - 1
    ix = np.clip(np.digitize(x, x_bins) - 1, 0, W - 1)
    iy = np.clip(np.digitize(y, y_bins) - 1, 0, H - 1)
    s = np.zeros((H, W))
    c = np.zeros((H, W), dtype=np.int64)
    np.add.at(s, (iy, ix), z)
    np.add.at(c, (iy, ix), 1)

    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / c, np.nan)


def merge_grids(per_file, method="median"):
    stack = np.stack(list(per_file.values()))
    if method == "median":
        return np.nanmedian(stack, axis=0)
    s = np.nansum(np.where(np.isnan(stack), 0, stack), axis=0)
    c = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(invalid="ignore"):
        return np.where(c > 0, s / c, np.nan)


def file_coverage_stats(las_paths, cfg, tide_df=None, subsample=4000):
    """Per-file coverage and quality statistics for pre-merge inspection.

    Returns a DataFrame of statistics and a dict of subsampled point
    coordinates for the coverage plot. Density (points per occupied cell) is
    the primary anomaly signal: a line logging water-column returns records
    several times more points per unit area than its neighbours.
    """
    W = cfg.W
    rows, samples = [], {}

    for p in las_paths:
        name = os.path.basename(p)
        x, y, z = read_las_points(p)
        n_raw = len(z)

        if tide_df is not None:
            z = z + nearest_tide(p, tide_df)[0]

        keep_s = depth_sanity_filter(z)
        keep_o = np.zeros_like(keep_s)
        if keep_s.any():
            sub = filter_outliers_mad(x[keep_s], y[keep_s], z[keep_s])
            keep_o[np.where(keep_s)[0]] = sub

        iy, ix = cell_indices(x, y, cfg.x_bins, cfg.y_bins)
        n_cells = np.unique(iy.astype(np.int64) * W + ix).size

        rows.append(
            {
                "file": name,
                "n_points": n_raw,
                "n_cells": n_cells,
                "density": n_raw / max(n_cells, 1),
                "pct_shallow": float((~keep_s).mean() * 100),
                "pct_outlier": float((keep_s & ~keep_o).sum() / max(n_raw, 1) * 100),
            }
        )

        step = max(1, n_raw // subsample)
        samples[name] = (x[::step], y[::step])

    df = pd.DataFrame(rows)
    med = df["density"].median()
    df["density_ratio"] = df["density"] / med
    return df, samples


def exclusion_impact(samples, exclude, cfg):
    """Coverage that would be lost by excluding the given files."""
    W = cfg.W

    def cells(names):
        acc = set()
        for n in names:
            x, y = samples[n]
            iy, ix = cell_indices(x, y, cfg.x_bins, cfg.y_bins)
            acc.update((iy.astype(np.int64) * W + ix).tolist())
        return acc

    total = cells(samples.keys())
    kept = cells([n for n in samples if n not in exclude])
    lost = len(total - kept)
    return dict(
        total_cells=len(total),
        lost_cells=lost,
        lost_pct=lost / max(len(total), 1) * 100,
    )


def flag_anomalous(stats_df, density_ratio=2.5, shallow_pct=2.0):
    """Flag files that are likely dominated by water-column returns."""
    m = (stats_df["density_ratio"] > density_ratio) | (
        stats_df["pct_shallow"] > shallow_pct
    )
    return stats_df.loc[m, "file"].tolist()


def flag_anomalous_files(per_file_stats, ratio=2.5):
    """
    Flag files whose point density is far above the survey norm.
    A line that records several times more points per unit area than its
    neighbours is usually logging water-column returns rather than seabed.
    """
    dens = np.array([s["n_points"] / max(s["n_cells"], 1) for s in per_file_stats])
    med = np.median(dens)
    flagged = [s["name"] for s, d in zip(per_file_stats, dens) if d > ratio * med]
    return flagged, dens, med


def plot_file_coverage(samples, flagged, path, title=None):
    """Coverage of each file. Flagged files are drawn in red beneath the
    others, so any area still showing red is covered by a flagged file alone
    and would be lost if that file were removed.
    """
    fig, ax = plt.subplots(figsize=(11, 10))

    for name in flagged:  # 底层：异常文件
        if name in samples:
            x, y = samples[name]
            ax.scatter(x, y, s=3, c="red", alpha=0.9, zorder=1)

    normal = [n for n in samples if n not in flagged]
    cmap = plt.cm.viridis(np.linspace(0, 1, max(len(normal), 1)))
    for c, name in zip(cmap, normal):  # 上层：正常文件
        x, y = samples[name]
        ax.scatter(x, y, s=2, color=c, alpha=0.55, zorder=2, label=name[-10:])

    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(
        title
        or f"File coverage ({len(flagged)} flagged, shown in red beneath the others)"
    )
    if len(normal) <= 30:
        ax.legend(
            fontsize=6,
            markerscale=3,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
    ax.ticklabel_format(style="plain")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()


def micro_hole_fill(grid, max_dist=3.0):
    """Fill isolated empty cells from their nearest valid neighbour.

    Beam geometry and occasional detection failures leave scattered single
    cells with no return, which would otherwise fragment every patch that
    contains one. Only cells within `max_dist` of real data are filled, so
    genuine coverage gaps at the survey edge stay empty rather than being
    invented.

    Parameters
    ----------
    grid : (H, W) float array with NaN where there is no data
    max_dist : fill radius in cells

    Returns
    -------
    (H, W) float array, a copy with small holes filled
    """
    empty = np.isnan(grid)
    if not empty.any():
        return grid.copy()
    dist, idx = distance_transform_edt(
        empty, return_distances=True, return_indices=True
    )
    return np.where(dist <= max_dist, grid[tuple(idx)], np.nan)


def build_footprint(grid):
    """Boolean mask of the surveyed area.

    Interior voids are closed so that the footprint is the region the survey
    covered, not the region where every cell happens to hold a value. Gap
    filling later writes into those interior cells; the ragged outer boundary
    is left as it is, since extending it would fabricate coverage.

    This mask defines the extent of the deployment output and is derived from
    the raw data alone, with no reference surface involved.
    """
    return binary_fill_holes(~np.isnan(grid))


def build_gt_grid(gt_df, stack):
    """Interpolate a reference point cloud onto the survey grid.

    The reference is delivered as scattered points on its own grid, so it has
    to be resampled before any cell-by-cell comparison. Nearest-neighbour
    interpolation is used rather than a smooth interpolant, because the
    reference is the target the model is trained against and should not be
    altered on the way in.

    Note that this returns a value for every cell, including cells the
    reference survey never covered, since NearestNDInterpolator extrapolates
    instead of returning NaN. Always pair it with `build_gt_valid_mask`.

    Parameters
    ----------
    gt_df : DataFrame with X, Y, Z columns, Z as negative-down elevation
    stack : SurveyGridStack providing the target geometry

    Returns
    -------
    (H, W) float32 array
    """
    GX, GY = stack.meshgrid()
    interp = NearestNDInterpolator(
        np.column_stack([gt_df["X"].values, gt_df["Y"].values]), gt_df["Z"].values
    )
    return interp(GX, GY).astype(np.float32)


def build_gt_valid_mask(gt_df, stack, tol=3):
    """Cells where the reference surface is real rather than extrapolated.

    `build_gt_grid` fills the whole array by nearest-neighbour lookup, so
    outside the reference survey's own coverage it repeats the value of a
    distant point. Those cells look like a flat plateau with abrupt steps and
    are not data. Including them corrupts metrics and produces a fabricated
    reference line in profile plots.

    Cells are marked valid when they lie within `tol` cells of a position
    that actually contains a reference point.

    Parameters
    ----------
    gt_df : DataFrame with X, Y columns
    stack : SurveyGridStack providing the target geometry
    tol : tolerance in cells, matching the micro-hole fill radius

    Returns
    -------
    (H, W) bool array
    """
    r = stack.resolution / 2
    counts, _, _, _ = binned_statistic_2d(
        gt_df["X"].values,
        gt_df["Y"].values,
        None,
        statistic="count",
        bins=[
            np.append(stack.gx - r, stack.gx[-1] + r),
            np.append(stack.gy - r, stack.gy[-1] + r),
        ],
    )
    occupied = counts.T > 0
    return distance_transform_edt(~occupied) <= tol


if __name__ == "__main__":
    import argparse
    from config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("las_dir", help="Directory of LAS files to inspect")
    parser.add_argument("--tide", help="Tide CSV file for depth correction")
    parser.add_argument(
        "--out", default="coverage.png", help="Output coverage plot path"
    )
    args = parser.parse_args()

    cfg = Config(args.las_dir)
    tide_df = None
    if args.tide:
        tide_df = load_tide(args.tide)

    las_paths = sorted(cfg.las_paths)
    stats_df, samples = file_coverage_stats(las_paths, cfg, tide_df=tide_df)
    flagged = flag_anomalous(stats_df)
    plot_file_coverage(samples, flagged, args.out)
