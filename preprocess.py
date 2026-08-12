from config import FILTER_CELL, FILTER_THRESH

import numpy as np
import pandas as pd


def depth_sanity_filter(z, ceiling=None, margin=5.0):
    if ceiling is None:
        ceiling = float(np.median(z)) + margin
    return z <= ceiling


def filter_outliers(x, y, z, k=3.0, res=FILTER_CELL, thresh=FILTER_THRESH):
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


def outlier_stats(x, y, z, cell=CELL_M, thresh=THRESH_M, shallow=SHALLOW_M):
    ix = (x // cell).astype(np.int64)
    iy = (y // cell).astype(np.int64)
    df = pd.DataFrame({"ix": ix, "iy": iy, "z": z})
    med = df.groupby(["ix", "iy"])["z"].transform("median").values
    dev = z - med

    return {
        "n_points": len(z),
        "outlier_pct": float((np.abs(dev) > thresh).mean() * 100),
        "shallow_pct": float((dev > shallow).mean() * 100),
        "deep_pct": float((dev < -shallow).mean() * 100),
        "dev_std_cm": float(dev.std() * 100),
        "dev_p99_cm": float(np.percentile(np.abs(dev), 99) * 100),
        "max_shallow_dev": float(dev.max()),
        "z_span_m": float(z.max() - z.min()),
    }


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
