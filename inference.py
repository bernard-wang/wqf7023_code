"""
inference.py
============
Full-area inference and post-processing.

Imported by both the evaluation script and the deployment app, so that the
numbers reported in evaluation come from exactly the code that runs in
production.

No reference data is used anywhere here. The normalisation baseline of each
tile comes from that tile's own raw values, and the vertical datum is set by
the tidal correction applied earlier.
"""

import numpy as np
import torch
from scipy.interpolate import griddata as scipy_griddata
from scipy.ndimage import distance_transform_edt, gaussian_filter


def hann_window(size):
    """Separable Hann window, zero-free at the edges.

    np.hanning ends in exact zeros, which would make the corner cells of
    every tile contribute nothing and leave gaps where tiles only just
    overlap. Taking an interior slice of a slightly longer window avoids
    that while keeping the taper.
    """
    w = np.hanning(size + 2)[1:-1]
    return np.outer(w, w)


def tile_inference(
    model, raw, patch=32, stride=8, max_nan=0.30, device=None, progress=None
):
    """Run the model over a full grid with Hann-weighted overlapping tiles.

    Predictions are blended with a Hann window rather than averaged
    uniformly. Under uniform averaging the number of tiles covering a cell
    steps between values across the grid, and each tile contributes equally
    whether the cell sits at its centre or its edge, which leaves a faint
    grid-aligned texture in the output. Weighting by distance from the tile
    centre removes it.

    Each tile is normalised by its own raw mean, matching how the training
    patches were prepared, and missing cells are set to that mean so they
    carry no deviation.

    Parameters
    ----------
    model : trained network in eval mode
    raw : (H, W) float array, tide-corrected, NaN where there is no data
    patch, stride : tile size and step in cells
    max_nan : skip tiles with worse coverage than this
    progress : optional callable(fraction, desc) for a progress display

    Returns
    -------
    (H, W) float array, NaN where no tile contributed
    """
    device = device or next(model.parameters()).device
    H, W = raw.shape
    acc = np.zeros((H, W))
    wgt = np.zeros((H, W))
    win = hann_window(patch)

    rows = list(range(0, H - patch + 1, stride))
    cols = list(range(0, W - patch + 1, stride))

    model.eval()
    with torch.no_grad():
        for n, r in enumerate(rows):
            if progress and n % 10 == 0:
                progress(n / len(rows), desc="Running model")
            for c in cols:
                sl = (slice(r, r + patch), slice(c, c + patch))
                tile = raw[sl]
                nan_m = np.isnan(tile)
                if nan_m.mean() > max_nan:
                    continue

                mu = float(np.nanmean(tile))
                x = tile - mu
                x[nan_m] = 0.0

                t = torch.from_numpy(x.astype(np.float32))
                t = t.unsqueeze(0).unsqueeze(0).to(device)
                pred = model(t).squeeze().cpu().numpy() + mu

                acc[sl] += pred * win
                wgt[sl] += win

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wgt > 1e-6, acc / wgt, np.nan)


def gap_fill(pred, footprint):
    """Interpolate cells inside the footprint that no tile could reach.

    Tiles are skipped where raw coverage is poor, so small areas inside the
    surveyed region can come back empty. Linear interpolation is used within
    the convex hull of valid predictions, with nearest-neighbour as a
    fallback outside it.
    """
    out = pred.copy()
    H, W = out.shape
    gaps = np.isnan(out) & footprint
    if not gaps.any():
        return out

    yy, xx = np.mgrid[0:H, 0:W]
    src = ~np.isnan(out) & footprint
    if src.sum() >= 4:
        out[gaps] = scipy_griddata(
            np.column_stack([yy[src], xx[src]]),
            out[src],
            np.column_stack([yy[gaps], xx[gaps]]),
            method="linear",
        )

    still = np.isnan(out) & footprint
    if still.any() and src.sum() >= 1:
        out[still] = scipy_griddata(
            np.column_stack([yy[src], xx[src]]),
            out[src],
            np.column_stack([yy[still], xx[still]]),
            method="nearest",
        )
    return out


def smooth(grid, footprint, sigma=0.8):
    """NaN-aware Gaussian smoothing, weighted so gaps do not bleed in."""
    if sigma <= 0:
        return np.where(footprint, grid, np.nan)
    valid = ~np.isnan(grid)
    num = gaussian_filter(np.where(valid, grid, 0.0), sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return np.where(footprint, out, np.nan)


def restore_sharpness(pred, raw, alpha, sigma=2.0):
    """Add back a fraction of the raw high-frequency content.

    The network suppresses detail at the same spatial scale as the noise it
    was trained to remove, so fine seabed features are partly lost with it.
    This returns some of that content, at the cost of returning some noise
    with it. Measured on this data the trade was not worth making and alpha
    is 0 by default; the control is kept because the balance is a matter of
    what the output is for.
    """
    if alpha <= 0:
        return pred.copy()
    valid = ~np.isnan(raw)
    num = gaussian_filter(np.where(valid, raw, 0.0), sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        low = np.where(den > 1e-6, num / den, np.nan)
    high = np.nan_to_num(raw - low, nan=0.0)
    return pred + alpha * high


def coherence_sharpen(pred, raw, alpha=0.5, sigma=1.5, tensor_sigma=3.0, power=2.0):
    """Restore detail only where the surface has a consistent direction.

    Plain sharpening returns noise along with the features it recovers,
    because at this scale both occupy the same spatial frequencies. What
    separates them is not frequency but orientation: an anchor scar or a
    bedform crest is a linear feature whose gradient points the same way over
    several cells, whereas fish-shoal returns and residual sounding noise
    have no preferred direction.

    The structure tensor measures that. Its two eigenvalues describe how much
    the surface varies along and across the dominant local direction, and
    their normalised difference, the coherence, approaches one for a clean
    edge and zero for isotropic noise. Using it to weight the amount of
    high-frequency content added back sharpens features without amplifying
    noise as strongly.

    This is a perceptual adjustment. It is unlikely to improve error metrics
    much, and `power` above 2 begins to look artificial, so check the maximum
    error before adopting a setting.

    Parameters
    ----------
    alpha : peak fraction of high-frequency content restored, at coherence 1
    sigma : scale of the detail being restored
    tensor_sigma : scale over which orientation is judged; larger demands
        that a feature persist further before it counts as one
    power : sharpens the distinction between coherent and isotropic areas
    """
    if alpha <= 0:
        return pred.copy()

    valid = ~np.isnan(raw)
    filled = np.where(valid, raw, 0.0)

    gy, gx = np.gradient(gaussian_filter(filled, sigma))
    jxx = gaussian_filter(gx * gx, tensor_sigma)
    jyy = gaussian_filter(gy * gy, tensor_sigma)
    jxy = gaussian_filter(gx * gy, tensor_sigma)

    # Eigenvalue difference over sum, which is the standard coherence measure
    diff = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy**2, 0.0))
    total = jxx + jyy
    with np.errstate(invalid="ignore", divide="ignore"):
        coherence = np.where(total > 1e-12, diff / total, 0.0)
    weight = np.clip(coherence, 0.0, 1.0) ** power

    num = gaussian_filter(filled, sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        low = np.where(den > 1e-6, num / den, np.nan)
    high = np.nan_to_num(raw - low, nan=0.0)

    return pred + alpha * weight * high


def run_inference(model, raw, footprint, cfg, device=None, progress=None):
    """Tile inference followed by gap filling, smoothing and sharpening."""
    pred = tile_inference(
        model,
        raw,
        patch=cfg.patch_size,
        stride=cfg.stride_infer,
        max_nan=cfg.max_nan_ratio,
        device=device,
        progress=progress,
    )
    pred = gap_fill(pred, footprint)
    pred = smooth(pred, footprint, cfg.smooth_sigma)
    if cfg.sharpen_alpha > 0:
        pred = np.where(
            footprint, coherence_sharpen(pred, raw, cfg.sharpen_alpha), np.nan
        )
    return pred


# ─────────────────────────────────────────────────────────────────────────────
# Classical baselines, for comparison only
# ─────────────────────────────────────────────────────────────────────────────
def nan_gaussian(grid, sigma):
    """Gaussian filter that ignores missing cells instead of spreading them."""
    valid = ~np.isnan(grid)
    num = gaussian_filter(np.where(valid, grid, 0.0), sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    return np.where(valid, out, np.nan)


def nan_median(grid, size):
    """Median filter applied after nearest-neighbour filling, then re-masked.

    An exact NaN-aware median is far slower and the difference is confined to
    cells adjacent to gaps, which the footprint excludes anyway.
    """
    from scipy.ndimage import median_filter

    valid = ~np.isnan(grid)
    if valid.all():
        return median_filter(grid, size=size)
    _, idx = distance_transform_edt(~valid, return_indices=True)
    filled = grid[tuple(idx)]
    return np.where(valid, median_filter(filled, size=size), np.nan)
