from config import RESOLUTION

import os, re, glob

import numpy as np
import pandas as pd

import laspy


def load_tide(tide_file):
    df = pd.read_csv(tide_file, sep=r"\s+", header=None, names=["Date", "Time", "Tide"])
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", dayfirst=True
    )
    print(
        f"Tide record: {len(df)} readings, {df['Tide'].min():.3f} to {df['Tide'].max():.3f} m"
    )
    return df


def nearest_tide_for_file(las_path, tide_df):
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(las_path))
    if not m:
        raise ValueError(f"No timestamp in filename: {las_path}")
    t = pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
    idx = (tide_df["Timestamp"] - t).abs().idxmin()
    return float(tide_df.loc[idx, "Tide"]), t


def load_xyz(path):
    df = pd.read_csv(
        path, sep=r"[,\s]+", engine="python", header=None, names=["X", "Y", "Z"]
    )
    for c in df:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    if df["Z"].mean() > 0:
        df["Z"] = -df["Z"]
    return df


def export_xyz(grid, footprint, gx, gy, path):
    # meta = json.load(open(os.path.join(path, "grid_meta.json")))
    # res  = meta["resolution"]
    # pred = np.load(os.path.join(path, "pred_grid.npy")).astype(np.float64)
    # foot = np.load(os.path.join(path, "footprint_mask.npy"))
    # H, W = pred.shape
    # gx = meta["x_origin"] + (np.arange(W) + 0.5) * res
    # gy = meta["y_origin"] + (np.arange(H) + 0.5) * res
    GX, GY = np.meshgrid(gx, gy)
    m = footprint & ~np.isnan(grid)
    pd.DataFrame({"X": GX[m], "Y": GY[m], "Z": grid[m]}).to_csv(
        path, sep=",", index=False, header=False, float_format="%.3f"
    )
    return int(m.sum())


def read_las_points(path):
    las = laspy.read(path)
    return (
        np.asarray(las.x, dtype=np.float64),
        np.asarray(las.y, dtype=np.float64),
        -np.abs(np.asarray(las.z, dtype=np.float64)),
    )


def survey_bounds(las_dir_path):
    las_files = sorted(
        glob.glob(os.path.join(las_dir_path, "*.las"))
        + glob.glob(os.path.join(las_dir_path, "*.laz"))
    )
    if not las_files:
        raise FileNotFoundError(f"No LAS files in {las_dir_path!r}")

    xmin = ymin = float("inf")
    xmax = ymax = -float("inf")
    for f in las_files:
        with laspy.open(f) as lf:
            h = lf.header
            xmin, xmax = min(xmin, h.x_min), max(xmax, h.x_max)
            ymin, ymax = min(ymin, h.y_min), max(ymax, h.y_max)
    return (xmin, xmax, ymin, ymax)


def make_grid_axes(bounds, res=RESOLUTION):
    xmin, xmax, ymin, ymax = bounds
    x_bins = np.arange(np.floor(xmin), np.ceil(xmax) + res, res)
    y_bins = np.arange(np.floor(ymin), np.ceil(ymax) + res, res)
    H, W = len(y_bins) - 1, len(x_bins) - 1
    gx = x_bins[:-1] + res / 2
    gy = y_bins[:-1] + res / 2
    return x_bins, y_bins, gx, gy, H, W
