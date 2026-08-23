"""
evaluate.py
===========
Evaluate the trained network against classical filters and the ablation.

    python evaluate.py
    python evaluate.py --models lightunet encoderdecoder

Two levels are reported and they answer different questions.

Test evaluation uses the spatially held-out patches. Those cells were never
seen in training and are separated from the training region by a buffer, so
these figures are the evidence of generalisation.

Full-area comparison covers the whole survey, including the region the model
was trained on. It is not evidence of generalisation and is not presented as
such; it shows what the delivered surface looks like and is the form the
industry partner works with.

Filter parameters are chosen on the validation split, never on the test
split.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter, median_filter

from .config import Config
from .dataset import load_split
from .model import LightUNet, EncoderDecoder
from .inference import run_inference, nan_gaussian, nan_median

MODELS = {"lightunet": LightUNet, "encoderdecoder": EncoderDecoder}


# ─────────────────────────────────────────────────────────────────────────────
def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


def metrics(pred, target, mask=None):
    """Error statistics in centimetres, plus R2 and the 10 cm compliance rate.

    within_10cm is the fraction of cells agreeing with the reference to
    within 0.1 m, which is the tolerance the survey contractor works to and
    is more directly comparable to their acceptance criteria than a mean.
    """
    p = pred[mask] if mask is not None else pred.ravel()
    t = target[mask] if mask is not None else target.ravel()
    ok = ~np.isnan(p) & ~np.isnan(t)
    p, t = p[ok], t[ok]
    if p.size == 0:
        return {
            k: float("nan")
            for k in (
                "MAE_cm",
                "RMSE_cm",
                "bias_cm",
                "p95_cm",
                "max_cm",
                "R2",
                "within_10cm",
            )
        }
    d = p - t
    ss_tot = np.sum((t - t.mean()) ** 2)
    return {
        "MAE_cm": float(np.mean(np.abs(d)) * 100),
        "RMSE_cm": float(np.sqrt(np.mean(d**2)) * 100),
        "bias_cm": float(np.mean(d) * 100),
        "p95_cm": float(np.percentile(np.abs(d), 95) * 100),
        "max_cm": float(np.max(np.abs(d)) * 100),
        "R2": float(1 - np.sum(d**2) / ss_tot) if ss_tot > 0 else float("nan"),
        "within_10cm": float(np.mean(np.abs(d) <= 0.10)),
        "n_cells": int(p.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
def tune_filter(fn, params, X_val, Y_val):
    """Pick the filter parameter minimising validation MAE.

    Tuning on validation rather than test keeps the classical baselines on
    the same footing as the network, which also never sees test data during
    development.
    """
    best, best_p = np.inf, params[0]
    for p in params:
        pred = np.stack([fn(x, p) for x in X_val])
        mae = float(np.mean(np.abs(pred - Y_val)))
        if mae < best:
            best, best_p = mae, p
    return best_p, best * 100


def predict_patches(model, X, device, batch=64):
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            t = torch.from_numpy(X[i : i + batch]).float().unsqueeze(1).to(device)
            out.append(model(t).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def evaluate_patches(cfg, models, device, out_dir):
    """Test-set metrics for every method."""
    data, info = load_split(cfg.patch_dir)
    X_val, Y_val, _, _ = data["val"]
    X_te, Y_te, _, mu_te = data["test"]
    print(
        f"Test patches: {len(X_te)}  (validation {len(X_val)} used only "
        f"for tuning the classical filters)"
    )

    sigma, v_mae = tune_filter(
        lambda x, s: gaussian_filter(x, sigma=s),
        [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
        X_val,
        Y_val,
    )
    ksize, v_mae_m = tune_filter(
        lambda x, k: median_filter(x, size=k), [3, 5, 7], X_val, Y_val
    )
    print(
        f"  Gaussian sigma {sigma} (val MAE {v_mae:.2f} cm), "
        f"median size {ksize} (val MAE {v_mae_m:.2f} cm)"
    )

    preds = {
        "raw": X_te,
        f"gaussian (s={sigma})": np.stack(
            [gaussian_filter(x, sigma=sigma) for x in X_te]
        ),
        f"median (k={ksize})": np.stack([median_filter(x, size=ksize) for x in X_te]),
    }
    for name, model in models.items():
        preds[name] = predict_patches(model, X_te, device)

    rows = []
    for name, p in preds.items():
        m = metrics(p, Y_te)
        m["method"] = name
        rows.append(m)

    df = pd.DataFrame(rows)[["method", "MAE_cm", "RMSE_cm", "p95_cm", "max_cm", "R2"]]
    df.to_csv(os.path.join(out_dir, "patch_metrics.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "patch_predictions.npz"),
        Y=Y_te,
        mu=mu_te,
        **{n.split()[0]: p for n, p in preds.items()},
    )

    print("\nTEST-SET PATCH METRICS")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    return df, {
        "gaussian_sigma": sigma,
        "median_size": ksize,
        "filter_tuning": "validation split",
    }


# ─────────────────────────────────────────────────────────────────────────────
def evaluate_area(cfg, models, device, out_dir):
    """Whole-survey comparison against the reference surface."""
    gpath = os.path.join(cfg.patch_dir, "grids", "grids.npz")
    if not os.path.exists(gpath):
        print(f"\nNo grids at {gpath}; skipping the full-area comparison.")
        return None, {}

    g = np.load(gpath)
    raw, gt = g["raw"].astype(np.float64), g["gt"].astype(np.float64)
    footprint, gt_valid = g["footprint"], g["gt_valid"]
    eval_mask = footprint & gt_valid
    print(
        f"\nFull area: {int(footprint.sum()):,} cells in footprint, "
        f"{int(eval_mask.sum()):,} comparable to the reference"
    )

    grids = {
        "raw": np.where(footprint, raw, np.nan),
        "gaussian": np.where(footprint, nan_gaussian(raw, 0.8), np.nan),
        "median": np.where(footprint, nan_median(raw, 3), np.nan),
    }
    for name, model in models.items():
        print(f"  inference: {name}")
        grids[name] = run_inference(model, raw, footprint, cfg, device)

    rows = []
    for name, grid in grids.items():
        m = metrics(grid, gt, eval_mask)
        m["method"] = name
        rows.append(m)

    df = pd.DataFrame(rows)[
        [
            "method",
            "MAE_cm",
            "RMSE_cm",
            "bias_cm",
            "p95_cm",
            "max_cm",
            "R2",
            "within_10cm",
        ]
    ]
    df.to_csv(os.path.join(out_dir, "area_metrics.csv"), index=False)

    np.savez_compressed(
        os.path.join(out_dir, "area_grids.npz"),
        gt=gt.astype(np.float32),
        footprint=footprint,
        eval_mask=eval_mask,
        **{n: v.astype(np.float32) for n, v in grids.items()},
    )

    print(
        "\nFULL-AREA METRICS  (includes the training region, "
        "not a generalisation measure)"
    )
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))
    return df, grids


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["light_unet"], choices=list(MODELS))
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--patch-dir", default=None)
    ap.add_argument("--out", default="./results/evaluation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-area", action="store_true", help="patch metrics only")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    over = {
        k: v
        for k, v in {"patch_dir": args.patch_dir, "model_dir": args.model_dir}.items()
        if v is not None
    }
    cfg = Config().override(**over)
    device = pick_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    models = {}
    for name in args.models:
        ckpt = os.path.join(cfg.model_dir, f"best_{name}.pth")
        if not os.path.exists(ckpt):
            print(f"Skipping {name}: no checkpoint at {ckpt}")
            continue
        m = MODELS[name]().to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        models[name] = m
    if not models:
        raise SystemExit("No model checkpoints found.")
    print(f"Loaded {len(models)} model(s) on {device}")

    pdf, tuning = evaluate_patches(cfg, models, device, args.out)
    adf, grids = (
        (None, {}) if args.no_area else evaluate_area(cfg, models, device, args.out)
    )

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(
            {
                "models": list(models),
                "tuning": tuning,
                "patch_metrics": pdf.to_dict("records"),
                "area_metrics": adf.to_dict("records") if adf is not None else None,
                "config": cfg.to_dict(),
            },
            f,
            indent=2,
        )

    if not args.no_figures and grids:
        try:
            from figures import make_evaluation_figures

            make_evaluation_figures(args.out, cfg)
        except ImportError:
            print("figures.py not available; skipping plots")

    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
