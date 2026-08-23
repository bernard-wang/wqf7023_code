"""
dataset.py
==========
Patch extraction, spatial splitting, and persistence of training data.

Normalisation uses the mean of the raw patch, not of the reference patch.
At inference there is no reference, so the baseline has to come from the raw
data; using the reference mean during training would leave the network seeing
a differently centred input in the two settings.

The same mean is subtracted from the target, so that adding it back after
inference recovers absolute depth:

    training    X = raw - mean(raw),  Y = reference - mean(raw)
    inference   output = model(raw - mean(raw)) + mean(raw)

X is therefore exactly zero-mean and the network never observes absolute
depth. Y is not zero-mean: its mean is the local difference between raw and
reference, which the network learns to correct along with the noise.
"""

import os
import json
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_patches(raw, gt, valid_mask, patch=32, stride=16, max_nan=0.30):
    """Extract normalised (input, target) patch pairs from aligned grids.

    A patch is kept only where the reference is genuinely valid across the
    whole window, since the interpolated reference grid extrapolates beyond
    the area the reference survey actually covered.

    Missing raw cells are set to the patch mean, which is zero after
    normalisation, so they carry no deviation. They are not filled from the
    reference: that would put reference values into the input, which cannot
    be reproduced at inference time.

    Parameters
    ----------
    raw, gt : (H, W) float arrays on the same grid
    valid_mask : (H, W) bool, cells where the reference is real
    patch : window size in cells
    stride : step between window origins
    max_nan : reject a patch whose raw coverage is worse than this

    Returns
    -------
    X : (N, patch, patch) float32, exactly zero-mean
    Y : (N, patch, patch) float32, mean equals the local raw/reference bias
    coords : (N, 2) int32, (row, col) of each patch origin
    mu : (N, 2) float32, columns are mean(raw) and mean(reference).
         The first is needed to restore absolute depth; the second is kept
         so the bias distribution can be inspected afterwards.
    info : dict, extraction statistics
    """
    H, W = raw.shape
    X, Y, C, M = [], [], [], []
    n_bad_gt = n_bad_cov = 0

    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            sl = (slice(r, r + patch), slice(c, c + patch))
            p_gt = gt[sl]

            if not valid_mask[sl].all() or np.isnan(p_gt).any():
                n_bad_gt += 1
                continue

            p_raw = raw[sl]
            nan_m = np.isnan(p_raw)
            if nan_m.mean() > max_nan:
                n_bad_cov += 1
                continue

            mu_raw = float(np.nanmean(p_raw))
            x = p_raw - mu_raw
            x[nan_m] = 0.0

            X.append(x.astype(np.float32))
            Y.append((p_gt - mu_raw).astype(np.float32))
            C.append((r, c))
            M.append((mu_raw, float(np.mean(p_gt))))

    if not X:
        raise RuntimeError(
            "No patches extracted. Check that the raw and reference grids "
            "overlap and that valid_mask is not empty."
        )

    mu = np.array(M, dtype=np.float32)
    bias = mu[:, 1] - mu[:, 0]  # reference minus raw, per patch

    info = {
        "n_extracted": len(X),
        "n_skipped_reference": n_bad_gt,
        "n_skipped_coverage": n_bad_cov,
        "patch": patch,
        "stride": stride,
        "max_nan_ratio": max_nan,
        "normalisation": "raw patch mean subtracted from input and target",
        "local_bias_cm": {
            "mean": float(bias.mean() * 100),
            "std": float(bias.std() * 100),
            "p05": float(np.percentile(bias, 5) * 100),
            "p95": float(np.percentile(bias, 95) * 100),
        },
    }

    return (np.stack(X), np.stack(Y), np.array(C, dtype=np.int32), mu, info)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial split
# ─────────────────────────────────────────────────────────────────────────────
def spatial_split(
    coords, patch=32, axis="col", test_frac=0.20, val_frac=0.15, buffer=32
):
    """Assign patches to train/val/test by position, not at random.

    Neighbouring patches share half their content at stride 16, so a random
    split would place near-duplicates on both sides of it. A file-level split
    is also insufficient because adjacent survey lines cover the same ground.

    Patches are assigned by which band of the split axis they fall in, and a
    buffer is excluded between bands so that no patch in one split shares a
    single cell with a patch in another.
    """
    ax = 1 if axis == "col" else 0
    start = coords[:, ax]
    end = start + patch
    span = int(end.max())

    b_test = span * (1.0 - test_frac)
    b_val = span * (1.0 - test_frac - val_frac)
    half = buffer / 2

    # A band must be at least one patch wide once the buffer is taken off
    # both of its edges, otherwise it silently receives nothing.
    need = patch + buffer
    bands = {"train": b_val, "val": b_test - b_val, "test": span - b_test}
    narrow = {k: w for k, w in bands.items() if w < need}
    if narrow:
        raise ValueError(
            f"Split bands too narrow for patch {patch} with buffer {buffer} "
            f"(each band needs >= {need} px): "
            + ", ".join(f"{k}={w:.0f} px" for k, w in narrow.items())
            + ". Reduce the buffer, widen the fractions, or split on the "
            "longer axis."
        )

    train = end <= (b_val - half)
    val = (start >= b_val + half) & (end <= b_test - half)
    test = start >= b_test + half
    dropped = ~(train | val | test)

    info = {
        "axis": axis,
        "span_px": span,
        "boundary_val_px": float(b_val),
        "boundary_test_px": float(b_test),
        "buffer_px": buffer,
        "band_widths_px": {k: float(v) for k, v in bands.items()},
        "counts": {
            "train": int(train.sum()),
            "val": int(val.sum()),
            "test": int(test.sum()),
            "dropped": int(dropped.sum()),
        },
    }

    return {"train": train, "val": val, "test": test, "dropped": dropped}, info


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────
def save_split(path, X, Y, coords, mu, masks, info):
    """Write the split arrays and their provenance to `path`.

    Coordinates and means are kept for every split, so predictions can be
    placed back on the survey grid and restored to absolute depth without
    re-running extraction.
    """
    os.makedirs(path, exist_ok=True)
    arrays = {}
    for name in ("train", "val", "test"):
        m = masks[name]
        arrays[f"X_{name}"] = X[m]
        arrays[f"Y_{name}"] = Y[m]
        arrays[f"coords_{name}"] = coords[m]
        arrays[f"mu_{name}"] = mu[m]
    arrays["coords_all"] = coords
    arrays["mu_all"] = mu
    arrays["mask_dropped"] = masks["dropped"]

    np.savez_compressed(os.path.join(path, "patches.npz"), **arrays)
    with open(os.path.join(path, "patches.json"), "w") as f:
        json.dump(info, f, indent=2)


def load_split(path, splits=("train", "val", "test")):
    """Read arrays written by `save_split`.

    Returns
    -------
    data : dict mapping split name to (X, Y, coords, mu)
    info : dict written alongside the arrays
    """
    z = np.load(os.path.join(path, "patches.npz"))
    data = {
        s: (z[f"X_{s}"], z[f"Y_{s}"], z[f"coords_{s}"], z[f"mu_{s}"]) for s in splits
    }
    with open(os.path.join(path, "patches.json")) as f:
        info = json.load(f)
    return data, info
