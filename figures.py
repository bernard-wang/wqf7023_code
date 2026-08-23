"""
figures.py
==========
Plotting for the dataset, training and evaluation stages.

Every function writes to a path and returns it, so callers can collect the
filenames without holding figures open. Matplotlib runs on the Agg backend,
so this works headless.

Colour conventions kept consistent across the report:
    depth surfaces      jet, per-panel range
    differences         RdBu_r, symmetric about zero
    reference           red dashed
    model output        blue dashed
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────────────────
def _depth_axes(ax, extent, title=None):
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (m)" if extent else "Column (cells)")
    ax.set_ylabel("Northing (m)" if extent else "Row (cells)")
    if extent:
        ax.ticklabel_format(style="plain")
    if title:
        ax.set_title(title, fontsize=12, pad=8)


def _metre_colorbar(im, ax, label=None):
    cb = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label=label)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f} m"))
    cb.ax.tick_params(labelsize=8)
    return cb


def _robust_range(g, lo=0.5, hi=99.5):
    v = g[~np.isnan(g)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


# ─────────────────────────────────────────────────────────────────────────────
# Dataset stage
# ─────────────────────────────────────────────────────────────────────────────
def plot_file_coverage(samples, flagged, path, title=None):
    """Point coverage per file, with flagged files drawn underneath.

    Flagged files are plotted first and in red, so any red still visible
    after the others are drawn is ground that only a flagged file covers and
    would be lost by excluding it. That is the check to make before removing
    anything.
    """
    fig, ax = plt.subplots(figsize=(11, 10))
    for name in flagged:
        if name in samples:
            x, y = samples[name]
            ax.scatter(x, y, s=3, c="red", alpha=0.9, zorder=1)

    normal = [n for n in samples if n not in flagged]
    colours = plt.cm.viridis(np.linspace(0, 1, max(len(normal), 1)))
    for c, name in zip(colours, normal):
        x, y = samples[name]
        ax.scatter(x, y, s=2, color=c, alpha=0.55, zorder=2, label=name[-10:])

    _depth_axes(
        ax,
        True,
        title or f"File coverage: {len(flagged)} flagged, shown in red beneath",
    )
    if 0 < len(normal) <= 30:
        ax.legend(
            fontsize=6,
            markerscale=3,
            ncol=2,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
        )
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def plot_coverage_density(coverage, extent, path, overlap_pct=None):
    """Number of survey lines contributing to each cell."""
    fig, ax = plt.subplots(figsize=(9, 8))
    show = np.where(coverage > 0, coverage, np.nan)
    im = ax.imshow(
        show,
        cmap="turbo",
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
    )
    plt.colorbar(
        im, ax=ax, fraction=0.04, pad=0.02, label="Lines contributing per cell"
    )
    t = "Track coverage density"
    if overlap_pct is not None:
        t += f"  ({overlap_pct:.1f}% of cells covered more than once)"
    _depth_axes(ax, extent, t)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def plot_split_map(raw, coords, masks, patch, extent, path):
    """Where each split sits on the survey.

    The gaps between the coloured regions are the exclusion buffer. They are
    genuinely unused: no patch may straddle a boundary, or patches on either
    side would share cells and the test set would no longer be independent.
    """
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(raw, cmap="gray", origin="lower", extent=extent, aspect="equal")

    half = patch / 2
    res = (extent[1] - extent[0]) / raw.shape[1] if extent else 1.0
    x0 = extent[0] if extent else 0
    y0 = extent[2] if extent else 0
    colours = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    for name, colour in colours.items():
        cc = coords[masks[name]]
        if len(cc) == 0:
            continue
        ax.scatter(
            x0 + (cc[:, 1] + half) * res,
            y0 + (cc[:, 0] + half) * res,
            s=3,
            c=colour,
            alpha=0.6,
            label=f"{name} ({masks[name].sum()})",
        )
    ax.legend(markerscale=4, fontsize=9)
    _depth_axes(ax, extent, "Spatial split: gaps are the exclusion buffer")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Training stage
# ─────────────────────────────────────────────────────────────────────────────
def plot_learning_curve(history_json, path):
    """Training and validation loss, with the selected epoch marked.

    A validation curve sitting persistently above the training curve is the
    signal to watch with a training set this small.
    """
    with open(history_json) as f:
        d = json.load(f)
    h = pd.DataFrame(d["history"])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.plot(h.epoch, h.train_loss * 100, label="train", lw=1.6)
    ax.plot(h.epoch, h.val_loss * 100, label="validation", lw=1.6)
    ax.axvline(
        d["best_epoch"],
        color="k",
        ls=":",
        lw=1,
        label=f"selected (epoch {d['best_epoch']})",
    )
    ax.set_ylabel("L1 loss (cm)")
    ax.set_yscale("log")
    ax.set_title(f"{d['model']}, {d['n_parameters']:,} parameters")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax2.plot(h.epoch, h.lr, color="tab:green", lw=1.4)
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("Epoch")
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation stage
# ─────────────────────────────────────────────────────────────────────────────
def plot_triple_panel(
    raw, pred, gt, masks, extent, path, labels=("Raw", "U-Net processed", "Reference")
):
    """Raw, model output and reference side by side.

    Each panel takes its own colour range, as the survey contractor's own
    figures do, because the three surfaces do not share a depth span.
    """
    fp, em = masks
    panels = [
        np.where(fp, raw, np.nan),
        np.where(fp, pred, np.nan),
        np.where(em, gt, np.nan),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, g, lbl in zip(axes, panels, labels):
        vmin, vmax = _robust_range(g)
        im = ax.imshow(
            g,
            cmap="jet",
            vmin=vmin,
            vmax=vmax,
            origin="lower",
            extent=extent,
            aspect="equal",
        )
        _metre_colorbar(im, ax)
        _depth_axes(ax, extent, lbl)
    fig.suptitle("Grid data (1 m grid spacing)", fontsize=17, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_diff_map(pred, gt, mask, extent, path, vlim=None, title=None):
    """Signed difference between the model output and the reference.

    The colour range is set from the 98th percentile of the absolute
    difference rather than the maximum, so a handful of outlying cells cannot
    flatten the whole map to one colour.
    """
    diff = np.where(mask, pred - gt, np.nan)
    if vlim is None:
        v = np.abs(diff[~np.isnan(diff)])
        vlim = float(np.percentile(v, 98)) if v.size else 1.0
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(
        diff,
        cmap="RdBu_r",
        vmin=-vlim,
        vmax=vlim,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Model - reference (m)")
    _depth_axes(ax, extent, title or "Difference map")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def plot_profiles(pred, gt, mask, extent, path, row=None, col=None):
    """North-south and east-west depth sections through the survey.

    Both series are masked to cells where the reference is real. Without
    that, the reference line continues across areas the reference survey
    never covered, showing interpolated values as though they were measured.
    """
    H, W = gt.shape
    r = H // 2 if row is None else row
    c = W // 2 if col is None else col
    g = np.where(mask, gt, np.nan)
    p = np.where(mask, pred, np.nan)
    res = (extent[1] - extent[0]) / W if extent else 1.0

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    for ax, gl, pl, n, tag in (
        (axes[0], g[:, c], p[:, c], H, "A-A' (N-S)"),
        (axes[1], g[r, :], p[r, :], W, "B-B' (E-W)"),
    ):
        d = np.arange(n) * res
        ax.plot(d, gl, "r--", lw=1.5, label="Reference")
        ax.plot(d, pl, "b--", lw=1.2, label="Model")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"Path profile {tag}")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def plot_profile_planview(gt, mask, extent, path, row=None, col=None):
    """Where the two profile sections are taken."""
    H, W = gt.shape
    r = H // 2 if row is None else row
    c = W // 2 if col is None else col
    base = np.where(mask, gt, np.nan)
    vmin, vmax = _robust_range(base)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(
        base,
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    res = (extent[1] - extent[0]) / W if extent else 1.0
    x0 = extent[0] if extent else 0
    y0 = extent[2] if extent else 0
    xc, yc = x0 + c * res, y0 + r * res
    ax.axvline(xc, color="red", lw=2)
    ax.axhline(yc, color="red", lw=2)

    pad = 12 * res
    box = dict(fc="white", ec="black", boxstyle="square,pad=0.25")
    for lbl, xy, ha, va in (
        ("A", (xc, extent[3] - pad), "center", "top"),
        ("A'", (xc, extent[2] + pad), "center", "bottom"),
        ("B", (extent[0] + pad, yc), "left", "center"),
        ("B'", (extent[1] - pad, yc), "right", "center"),
    ):
        ax.annotate(lbl, xy, ha=ha, va=va, fontsize=12, fontweight="bold", bbox=box)
    _depth_axes(ax, extent, "Profile locations (A-A': N-S, B-B': E-W)")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def plot_method_comparison(metrics_csv, path, metric="MAE_cm"):
    """Bar chart of one metric across methods."""
    df = pd.read_csv(metrics_csv)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = ["0.6" if m in ("raw",) else "tab:blue" for m in df.method]
    ax.bar(df.method, df[metric], color=colours, alpha=0.85)
    for i, v in enumerate(df[metric]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_cm', '')} by method")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def plot_error_distribution(pred_npz, path):
    """Absolute error distribution per method, on a log count axis.

    Mean error can hide a long tail. This is where the two network variants
    separate: their averages are close but their worst cases are not.
    """
    z = np.load(pred_npz)
    Y = z["Y"]
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in z.files:
        if name in ("Y", "mu"):
            continue
        err = np.abs(z[name] - Y).ravel() * 100
        ax.hist(
            err,
            bins=120,
            range=(0, 40),
            histtype="step",
            lw=1.6,
            label=f"{name} (p99 {np.percentile(err, 99):.1f} cm)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Absolute error (cm)")
    ax.set_ylabel("Cell count")
    ax.set_title("Test-set error distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def plot_batch_quality(stats_csvs, path, labels=None, thresh_m=1.0):
    """Per-file outlier rate for one or more surveys, on a shared axis.

    A shared y-axis matters here: plotted separately, two surveys an order of
    magnitude apart in quality look similar.
    """
    dfs = [pd.read_csv(p) for p in stats_csvs]
    labels = labels or [os.path.basename(p) for p in stats_csvs]
    ymax = max(d["outlier_pct"].max() for d in dfs) * 1.1

    fig, axes = plt.subplots(1, len(dfs), figsize=(9 * len(dfs), 5), squeeze=False)
    for ax, df, lbl in zip(axes[0], dfs, labels):
        df = df.sort_values("file")
        ax.bar(
            range(len(df)),
            df.outlier_pct,
            color="tab:red",
            alpha=0.75,
            label=f"|deviation| > {thresh_m} m",
        )
        if "shallow_pct" in df:
            ax.bar(
                range(len(df)),
                df.shallow_pct,
                color="tab:orange",
                alpha=0.9,
                label="shallow returns",
            )
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df.file, rotation=90, fontsize=6)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("Outlier rate (%)")
        ax.set_title(f"{lbl}  (mean {df.outlier_pct.mean():.2f}%)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight")
    plt.close()
    return path


def make_3d_html(
    grid,
    gx,
    gy,
    path,
    title="",
    z_exaggeration=8,
    downsample=2,
    zrange=None,
    camera=None,
):
    """Rotatable 3D surface as a standalone HTML file.

    Written to file rather than embedded, so many views can be produced
    without accumulating in a notebook. Pass the same zrange and camera to
    every call for comparable screenshots.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping 3D view")
        return None

    s = downsample
    z = grid[::s, ::s]
    if zrange is None:
        v = z[~np.isnan(z)]
        zrange = (
            float(np.percentile(v, 0.5)) - 0.5,
            float(np.percentile(v, 99.5)) + 0.5,
        )
    dx = gx[-1] - gx[0]
    dy = gy[-1] - gy[0]

    fig = go.Figure(
        go.Surface(
            z=z,
            x=gx[::s],
            y=gy[::s],
            colorscale="Viridis",
            cmin=zrange[0],
            cmax=zrange[1],
            connectgaps=False,
            colorbar=dict(title="Elev (m)", len=0.7),
            lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            zaxis=dict(range=list(zrange)),
            aspectmode="manual",
            aspectratio=dict(
                x=1, y=dy / dx, z=(zrange[1] - zrange[0]) / dx * z_exaggeration
            ),
            camera=camera or dict(eye=dict(x=1.4, y=-1.4, z=0.9)),
            xaxis_title="Easting (m)",
            yaxis_title="Northing (m)",
            zaxis_title="Elevation (m)",
        ),
        width=1100,
        height=750,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    fig.write_html(path, include_plotlyjs="cdn")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def make_evaluation_figures(results_dir, cfg, model="lightunet"):
    """Produce the standard figure set from what evaluate.py wrote."""
    apath = os.path.join(results_dir, "area_grids.npz")
    if not os.path.exists(apath):
        print("No area grids; skipping evaluation figures")
        return []

    z = np.load(apath)
    gt = z["gt"].astype(np.float64)
    fp, em = z["footprint"], z["eval_mask"]
    raw = z["raw"].astype(np.float64)
    pred = z[model].astype(np.float64)

    extent = None
    jpath = os.path.join(cfg.patch_dir, "patches.json")
    if os.path.exists(jpath):
        with open(jpath) as f:
            g = json.load(f).get("grid")
        if g:
            r = g["resolution"]
            extent = [
                g["x_origin"],
                g["x_origin"] + g["W"] * r,
                g["y_origin"],
                g["y_origin"] + g["H"] * r,
            ]

    fdir = os.path.join(results_dir, "figures")
    os.makedirs(fdir, exist_ok=True)
    made = [
        plot_triple_panel(
            raw, pred, gt, (fp, em), extent, os.path.join(fdir, "grid_triple.png")
        ),
        plot_diff_map(pred, gt, em, extent, os.path.join(fdir, "diff_map.png")),
        plot_profiles(pred, gt, em, extent, os.path.join(fdir, "profiles.png")),
        plot_profile_planview(
            gt, em, extent, os.path.join(fdir, "profile_planview.png")
        ),
    ]
    for csv, metric in (
        ("area_metrics.csv", "MAE_cm"),
        ("patch_metrics.csv", "MAE_cm"),
    ):
        p = os.path.join(results_dir, csv)
        if os.path.exists(p):
            made.append(
                plot_method_comparison(
                    p, os.path.join(fdir, csv.replace(".csv", "_bar.png")), metric
                )
            )

    ppath = os.path.join(results_dir, "patch_predictions.npz")
    if os.path.exists(ppath):
        made.append(
            plot_error_distribution(ppath, os.path.join(fdir, "error_distribution.png"))
        )

    if extent:
        r = (extent[1] - extent[0]) / gt.shape[1]
        gx = np.arange(gt.shape[1]) * r + extent[0] + r / 2
        gy = np.arange(gt.shape[0]) * r + extent[2] + r / 2
        v = gt[em]
        zr = (float(np.percentile(v, 0.5)) - 0.5, float(np.percentile(v, 99.5)) + 0.5)
        for name, grid in (
            ("raw", np.where(fp, raw, np.nan)),
            ("model", np.where(fp, pred, np.nan)),
            ("reference", np.where(em, gt, np.nan)),
        ):
            made.append(
                make_3d_html(
                    grid,
                    gx,
                    gy,
                    os.path.join(fdir, f"view3d_{name}.html"),
                    title=name,
                    zrange=zr,
                )
            )

    print(f"Figures written to {fdir}")
    return [m for m in made if m]
