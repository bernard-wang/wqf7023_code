from .io_utils import read_las_points, nearest_tide
from .preprocess import depth_sanity_filter, filter_outliers_mad

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.ndimage import distance_transform_edt
from scipy.stats import trim_mean


@dataclass
class SurveyGridStack:
    """Per-file grids on a shared geometry, kept as separate layers.

    Layers are (n_files, H, W) float32 with NaN where a file has no data.
    Merging is derived on demand rather than stored, so files can be excluded
    and the result re-evaluated without re-reading any LAS data.

    Geometry (x_bins, y_bins) is carried here rather than in a separate
    object, since nothing else needs it independently.
    """

    x_bins: np.ndarray
    y_bins: np.ndarray
    resolution: float
    layers: np.ndarray  # (n, H, W) float32
    names: list
    meta: dict = field(default_factory=dict)
    excluded: set = field(default_factory=set)

    # ── geometry ────────────────────────────────────────────────────────
    @property
    def H(self):
        return len(self.y_bins) - 1

    @property
    def W(self):
        return len(self.x_bins) - 1

    @property
    def gx(self):
        """Cell-centre eastings."""
        return self.x_bins[:-1] + self.resolution / 2

    @property
    def gy(self):
        """Cell-centre northings."""
        return self.y_bins[:-1] + self.resolution / 2

    @property
    def extent(self):
        """(left, right, bottom, top) for matplotlib imshow."""
        r = self.resolution / 2
        return [
            self.x_bins[0] - r,
            self.x_bins[-1] + r,
            self.y_bins[0] - r,
            self.y_bins[-1] + r,
        ]

    def cell_indices(self, x, y):
        ix = np.clip(np.digitize(x, self.x_bins) - 1, 0, self.W - 1)
        iy = np.clip(np.digitize(y, self.y_bins) - 1, 0, self.H - 1)
        return iy, ix

    def meshgrid(self):
        return np.meshgrid(self.gx, self.gy)

    # ── construction ────────────────────────────────────────────────────
    @staticmethod
    def _bins_from_las(las_paths, res):
        import laspy

        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for p in las_paths:
            with laspy.open(p) as lf:
                h = lf.header
                xmin, xmax = min(xmin, h.x_min), max(xmax, h.x_max)
                ymin, ymax = min(ymin, h.y_min), max(ymax, h.y_max)
        return (
            np.arange(np.floor(xmin), np.ceil(xmax) + res, res),
            np.arange(np.floor(ymin), np.ceil(ymax) + res, res),
        )

    @classmethod
    def from_las(cls, las_paths, tide_df, cfg, progress=None):
        """Read, tide-correct, filter and grid every file into its own layer."""
        x_bins, y_bins = cls._bins_from_las(las_paths, cfg.resolution)
        obj = cls(
            x_bins=x_bins,
            y_bins=y_bins,
            resolution=cfg.resolution,
            layers=np.empty((0, 0, 0), np.float32),
            names=[],
        )

        layers, names, stats = [], [], []
        for i, p in enumerate(las_paths):
            if progress:
                progress(i / len(las_paths), desc=os.path.basename(p))

            x, y, z = read_las_points(p)
            n_raw = len(z)

            tide = nearest_tide(p, tide_df)[0] if tide_df is not None else cfg.mean_tide
            z = z + tide

            keep_s = depth_sanity_filter(z, cfg.ceiling)
            x, y, z = x[keep_s], y[keep_s], z[keep_s]
            # keep_o = filter_outliers_mad(
            #     x, y, z, res=cfg.cell, k=cfg.k, thresh=cfg.floor
            # )

            iy, ix = obj.cell_indices(x, y)
            n_cells = np.unique(iy.astype(np.int64) * obj.W + ix).size

            layers.append(obj._grid_points(x, y, z))
            names.append(os.path.basename(p))
            stats.append(
                {
                    "file": names[-1],
                    "n_points": n_raw,
                    "n_cells": n_cells,
                    # "density": n_raw / max(n_cells, 1),
                    "tide": tide,
                    # "pct_shallow": float((~keep_s).mean() * 100),
                    # "pct_outlier": float((~keep_o).mean() * 100),
                }
            )

        obj.layers = np.stack(layers).astype(np.float32)
        obj.names = names
        obj.meta = {"stats": stats, "config": cfg.to_dict()}
        return obj

    def _grid_points(self, x, y, z, statistic="trimmed_mean"):
        """Rasterise one line's soundings onto the survey grid.

        Parameters
        ----------
        x, y, z : array-like
            Sounding coordinates and depth/elevation values.
        statistic : str
            Aggregation method:

            - "mean":
                Mean of all soundings in each cell.
            - "median":
                Median sounding in each cell.
            - "trimmed_mean":
                Mean after removing the lowest and highest values.
            - "median_centered_mean":
                Mean of the soundings closest to the cell median.

        fraction : float
            Fraction of points retained by "median_centered_mean".
            For example, 0.5 keeps the 50% of points closest to
            the median.

        Returns
        -------
        np.ndarray
            Gridded Z values with shape (H, W).
        """

        H, W = self.H, self.W

        iy, ix = self.cell_indices(x, y)
        flat = iy.astype(np.int64) * W + ix

        if statistic == "mean":
            acc = np.bincount(flat, weights=z, minlength=H * W)
            cnt = np.bincount(flat, minlength=H * W)

            with np.errstate(invalid="ignore"):
                out = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        elif statistic == "median":
            g = pd.DataFrame({"k": flat, "z": z}).groupby("k")["z"].median()
            out = np.full(H * W, np.nan)
            out[g.index.values] = g.values
        elif statistic == "trimmed_mean":
            g = (
                pd.DataFrame({"k": flat, "z": z})
                .groupby("k")["z"]
                .apply(lambda x: trim_mean(x, proportiontocut=0.4))
            )
            out = np.full(H * W, np.nan)
            out[g.index.values] = g.values

        else:
            raise ValueError(
                "statistic must be one of: "
                "'mean', 'median', 'trimmed_mean', "
                "'median_centered_mean'"
            )
        return out.reshape(H, W).astype(np.float32)

    # ── selection ───────────────────────────────────────────────────────
    def exclude(self, names):
        self.excluded |= set(names)
        return self

    def include_all(self):
        self.excluded.clear()
        return self

    @property
    def active_idx(self):
        return [i for i, n in enumerate(self.names) if n not in self.excluded]

    # def flag_anomalous(self, density_ratio=2.5, shallow_pct=2.0):
    #     """Files likely dominated by water-column returns."""
    #     st = self.meta.get("stats", [])
    #     if not st:
    #         return []
    #     med = float(np.median([s["density"] for s in st]))
    #     return [
    #         s["file"]
    #         for s in st
    #         if s["density"] > density_ratio * med or s["pct_shallow"] > shallow_pct
    #     ]

    # ── derived products ────────────────────────────────────────────────
    def merged(self, method="median"):
        """Merge active layers with per-file equal weighting."""
        sub = self.layers[self.active_idx]
        if method == "median":
            with np.errstate(invalid="ignore"):
                return np.nanmedian(sub, axis=0)
        s = np.nansum(np.where(np.isnan(sub), 0, sub), axis=0)
        c = np.sum(~np.isnan(sub), axis=0)
        with np.errstate(invalid="ignore"):
            return np.where(c > 0, s / c, np.nan)

    def coverage_count(self):
        return np.sum(~np.isnan(self.layers[self.active_idx]), axis=0)

    def leave_one_out(self, i):
        """Merge of every active layer except layer i."""
        idx = [j for j in self.active_idx if j != i]
        sub = self.layers[idx]
        s = np.nansum(np.where(np.isnan(sub), 0, sub), axis=0)
        c = np.sum(~np.isnan(sub), axis=0)
        with np.errstate(invalid="ignore"):
            return np.where(c > 0, s / c, np.nan)

    def exclusion_impact(self, names):
        """Cells that would lose all coverage if these files were removed."""
        drop = set(names)
        keep = [
            i
            for i, n in enumerate(self.names)
            if n not in self.excluded and n not in drop
        ]
        now = self.coverage_count() > 0
        after = (
            np.sum(~np.isnan(self.layers[keep]), axis=0) > 0
            if keep
            else np.zeros_like(now)
        )
        lost = int((now & ~after).sum())
        return {
            "total_cells": int(now.sum()),
            "lost_cells": lost,
            "lost_pct": lost / max(int(now.sum()), 1) * 100,
        }

    def apply_offsets(self, offsets):
        """Subtract a constant per-layer offset (per-line adjustment)."""
        self.layers = self.layers - np.asarray(offsets, np.float32)[:, None, None]
        return self

    def coverage_contribution(self, fill_radius=3.0):
        """What each line uniquely contributes to the survey.

        Two figures are reported. `unique_pct` counts every cell a line alone
        reaches; `lost_pct` counts only those that would still be empty after
        hole filling, which recovers anything within `fill_radius` of remaining
        coverage. Isolated cells inside otherwise covered ground inflate the
        first without representing a real loss, so the second is what an
        exclusion should be judged on.
        """
        stack = self.layers
        present = ~np.isnan(stack)
        count = present.sum(axis=0)
        covered = count > 0
        total = max(int(covered.sum()), 1)

        rows = []
        for i, name in enumerate(self.names):
            sole = present[i] & (count == 1)
            others = covered & ~present[i]
            reachable = distance_transform_edt(~others) <= fill_radius
            lost = sole & ~reachable
            rows.append(
                {
                    "file": name,
                    "unique_pct": int(sole.sum()) / total * 100,
                    "lost_pct": int(lost.sum()) / total * 100,
                }
            )
        return pd.DataFrame(rows)

    # def file_rejection_report(self, shallow_pct=1.0, lost_pct=0.5, fill_radius=3.0):
    #     """Identify lines that can be dropped, and say why.

    #     A line is a candidate when it carries an unusual share of water-column
    #     returns. It is only safe to drop when the ground it reaches is also
    #     reached by other lines, since a noisy line covering ground nothing else
    #     covers still holds the only measurements there.

    #     Neither test is sufficient alone, so both are reported and a line is
    #     marked for exclusion only when it fails the first and passes the second.

    #     Returns a table with the two measures, a boolean recommendation, and a
    #     short reason for each line.
    #     """
    #     stats = pd.DataFrame(self.meta.get("stats", []))
    #     if stats.empty:
    #         raise RuntimeError("No per-file statistics recorded on this stack.")
    #     df = stats.merge(self.coverage_contribution(fill_radius), on="file")

    #     noisy = df["pct_shallow"] > shallow_pct
    #     redundant = df["lost_pct"] <= lost_pct
    #     df["exclude"] = noisy & redundant

    #     def reason(r):
    #         if r.exclude:
    #             return (
    #                 f"{r.pct_shallow:.1f}% water-column returns, covered by other lines"
    #             )
    #         if noisy[r.name] and not redundant[r.name]:
    #             return (
    #                 f"{r.pct_shallow:.1f}% water-column returns, but "
    #                 f"{r.lost_pct:.2f}% of the survey depends on it"
    #             )
    #         return "within normal range"

    #     df["reason"] = df.apply(reason, axis=1)
    #     return df.sort_values("pct_shallow", ascending=False)

    # ── persistence ─────────────────────────────────────────────────────
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(
            os.path.join(path, "stack.npz"),
            layers=self.layers,
            x_bins=self.x_bins,
            y_bins=self.y_bins,
        )
        json.dump(
            {
                "resolution": self.resolution,
                "names": self.names,
                "excluded": sorted(self.excluded),
                "meta": self.meta,
                "H": self.H,
                "W": self.W,
            },
            open(os.path.join(path, "stack.json"), "w"),
            indent=2,
        )

    @classmethod
    def load(cls, path):
        d = json.load(open(os.path.join(path, "stack.json")))
        z = np.load(os.path.join(path, "stack.npz"))
        return cls(
            x_bins=z["x_bins"],
            y_bins=z["y_bins"],
            resolution=d["resolution"],
            layers=z["layers"],
            names=d["names"],
            meta=d["meta"],
            excluded=set(d["excluded"]),
        )
