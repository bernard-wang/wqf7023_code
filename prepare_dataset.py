"""
prepare_dataset.py
==================
Build training patches from raw LAS survey files and a reference surface.

    LAS files ──► per-file grids (cached) ──► exclude anomalous files
              ──► merge ──► align to reference ──► extract patches
              ──► spatial split ──► save

The per-file grid stack is cached, so re-running with different patch or
split settings does not re-read any LAS data. Delete the cache directory or
pass --rebuild to force a fresh read.

This script requires a reference surface and is for training data only.
Inference on new surveys uses no reference data and is handled elsewhere.

Usage
-----
    python prepare_dataset.py
    python prepare_dataset.py --rebuild --stride 8
    python prepare_dataset.py --keep-all          # no anomaly exclusion
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd

from .config import Config
from .io_utils import load_tide, load_xyz
from .preprocess import (
    micro_hole_fill,
    build_footprint,
    build_gt_grid,
    build_gt_valid_mask,
)
from .dataset import extract_patches, spatial_split, save_split
from .grid_survey_stack import SurveyGridStack


# ─────────────────────────────────────────────────────────────────────────────
def build_or_load_stack(cfg, cache_dir, rebuild=False):
    """Return the per-file grid stack, reading from cache when available."""
    has_cache = os.path.exists(os.path.join(cache_dir, "stack.npz")) and os.path.exists(
        os.path.join(cache_dir, "stack.json")
    )

    if has_cache and not rebuild:
        stack = SurveyGridStack.load(cache_dir)
        print(
            f"Loaded cached stack: {len(stack.names)} files, "
            f"{stack.H} x {stack.W} cells at {stack.resolution} m"
        )
        return stack

    las_paths = sorted(
        glob.glob(os.path.join(cfg.las_dir, "*.las"))
        + glob.glob(os.path.join(cfg.las_dir, "*.laz"))
    )
    if not las_paths:
        sys.exit(f"No LAS files found in {cfg.las_dir}")

    tide_df = load_tide(cfg.tide_file) if os.path.exists(cfg.tide_file) else None
    if tide_df is None:
        print(
            f"No tide record at {cfg.tide_file}; "
            f"using the constant {cfg.mean_tide:+.4f} m"
        )

    print(f"Reading {len(las_paths)} LAS files ...")
    stack = SurveyGridStack.from_las(las_paths, tide_df, cfg)
    stack.save(cache_dir)
    print(f"Built stack: {stack.H} x {stack.W} cells, cached to {cache_dir}")
    return stack


def report_and_exclude(stack: SurveyGridStack, keep_all=False):
    """Flag files dominated by water-column returns and exclude them.

    The coverage each candidate uniquely provides is reported first, since a
    file is only safe to remove if neighbouring lines already cover the same
    ground.
    """
    stats = pd.DataFrame(stack.meta.get("stats", []))
    if not stats.empty:
        stats = stats.sort_values("density", ascending=False)
        print("\nPer-file statistics (worst density first):")
        print(
            stats[["file", "n_points", "density", "pct_shallow", "pct_outlier"]]
            .head(8)
            .to_string(index=False)
        )

    flagged = stack.flag_anomalous()
    if not flagged:
        print("\nNo files flagged as anomalous.")
        return []

    impact = stack.exclusion_impact(flagged)
    print(f"\n{len(flagged)} file(s) flagged: {', '.join(flagged)}")
    print(
        f"  excluding them loses {impact['lost_cells']:,} of "
        f"{impact['total_cells']:,} cells ({impact['lost_pct']:.2f}%)"
    )

    if keep_all:
        print("  --keep-all given, keeping them anyway")
        return []

    if impact["lost_pct"] > 5.0:
        print(
            "  WARNING: this removes a substantial part of the survey. "
            "Inspect the coverage plot before trusting this result."
        )

    stack.exclude(flagged)
    return flagged


# def align_to_reference(raw, gt, footprint, gt_valid):
#     """Remove any residual constant offset between the raw and reference grids.

#     Per-file tidal correction handles the dominant vertical variation, but a
#     constant difference can remain from vessel draft or a differing datum
#     definition. Without removing it, that offset would enter every training
#     patch as a uniform shift, and the network would learn to reproduce it.

#     Returns the shifted raw grid and the offset applied.
#     """
#     both = footprint & gt_valid & ~np.isnan(raw) & ~np.isnan(gt)
#     if both.sum() < 100:
#         sys.exit("Raw and reference grids barely overlap; check the inputs.")
#     delta = float(np.mean(gt[both]) - np.mean(raw[both]))
#     return raw + delta, delta


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="./cache/stack")
    ap.add_argument("--out", default="./data/patches")
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="re-read the LAS files instead of using the cache",
    )
    ap.add_argument(
        "--keep-all",
        action="store_true",
        help="do not exclude files flagged as anomalous",
    )
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    patch = args.patch or cfg.patch_size
    stride = args.stride or cfg.stride_train

    # ── 1. per-file grids ────────────────────────────────────────────────
    stack = build_or_load_stack(cfg, args.cache, args.rebuild)
    excluded = report_and_exclude(stack, args.keep_all)

    # ── 2. merge ─────────────────────────────────────────────────────────
    raw = micro_hole_fill(stack.merged(method=cfg.merge_method))
    footprint = build_footprint(raw)
    coverage = stack.coverage_count()
    print(
        f"\nMerged grid: {int(footprint.sum()):,} cells in footprint, "
        f"mean coverage {coverage[coverage > 0].mean():.2f} lines/cell"
    )

    # ── 3. reference surface ─────────────────────────────────────────────
    gt_df = load_xyz(cfg.reference_file)
    gt = build_gt_grid(gt_df, stack)
    gt_valid = build_gt_valid_mask(gt_df, stack, tol=cfg.gt_valid_tol)
    print(
        f"Reference: {len(gt_df):,} points, "
        f"{int(gt_valid.sum()):,} cells with genuine coverage"
    )

    # ── 4. datum alignment ───────────────────────────────────────────────
    # raw, delta = align_to_reference(raw, gt, footprint, gt_valid)
    # print(f"Residual datum offset applied: {delta:+.4f} m")
    # if abs(delta) > 0.5:
    #     print(
    #         "  WARNING: larger than expected after tidal correction. "
    #         "Check the tide record timezone and sign convention."
    # )

    # ── 5. patches ───────────────────────────────────────────────────────
    valid = footprint & gt_valid
    X, Y, coords, pinfo = extract_patches(
        raw, gt, valid, patch=patch, stride=stride, max_nan=cfg.max_nan_ratio
    )
    print(
        f"\nExtracted {pinfo['n_extracted']:,} patches "
        f"({pinfo['n_skipped_gt']:,} rejected for reference coverage, "
        f"{pinfo['n_skipped_coverage']:,} for raw coverage)"
    )

    # ── 6. spatial split ─────────────────────────────────────────────────
    masks, sinfo = spatial_split(
        coords,
        patch=patch,
        axis=cfg.split_axis,
        test_frac=cfg.test_frac,
        val_frac=cfg.val_frac,
        buffer=cfg.split_buffer,
    )
    c = sinfo["counts"]
    print(
        f"Split on '{sinfo['axis']}': train {c['train']}, val {c['val']}, "
        f"test {c['test']}, buffer-dropped {c['dropped']}"
    )
    if c["val"] < 40:
        print(
            "  NOTE: the validation set is small; early stopping will be "
            "noisy. Consider raising val_frac or reducing split_buffer."
        )

    # ── 7. save ──────────────────────────────────────────────────────────
    info = {
        "patches": pinfo,
        "split": sinfo,
        # "datum_offset_m": delta,
        "excluded_files": excluded,
        "n_files_used": len(stack.active_idx),
        "grid": {
            "H": stack.H,
            "W": stack.W,
            "resolution": stack.resolution,
            "x_origin": float(stack.x_bins[0]),
            "y_origin": float(stack.y_bins[0]),
        },
        "config": cfg.to_dict(),
    }
    save_split(args.out, X, Y, coords, masks, info)

    grid_dir = os.path.join(args.out, "grids")
    os.makedirs(grid_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(grid_dir, "grids.npz"),
        raw=raw.astype(np.float32),
        gt=gt.astype(np.float32),
        footprint=footprint,
        gt_valid=gt_valid,
        coverage=coverage.astype(np.int16),
    )
    print(f"\nSaved patches to {args.out}")
    print(f"Saved aligned grids to {grid_dir} for evaluation")

    # ── 8. figures ───────────────────────────────────────────────────────
    if not args.no_figures:
        try:
            from figures import plot_split_map, plot_coverage_density

            fig_dir = os.path.join(args.out, "figures")
            os.makedirs(fig_dir, exist_ok=True)
            plot_split_map(
                raw, coords, masks, patch, stack, os.path.join(fig_dir, "split_map.png")
            )
            plot_coverage_density(
                coverage, stack, os.path.join(fig_dir, "coverage.png")
            )
            print(f"Saved figures to {fig_dir}")
        except ImportError:
            print("figures.py not available; skipping plots")


if __name__ == "__main__":
    main()
