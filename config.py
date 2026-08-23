"""
config.py
=========
Every tunable parameter in one place.

The defaults are the configuration validated on the second survey. Values
marked as validated were selected by measurement; see PROJECT_CONTEXT.md for
what was tried and rejected.
"""

import json
from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # ── paths ────────────────────────────────────────────────────────────
    las_dir: str = "./data/raw_las"
    tide_file: str = "./data/tide.txt"
    reference_file: str = "./data/processed.xyz"
    cache_dir: str = "./cache/stack"
    patch_dir: str = "./data/patches"
    model_dir: str = "./models"

    # ── grid ─────────────────────────────────────────────────────────────
    resolution: float = 1.0  # m per cell
    hole_fill_dist: float = 3.0  # cells; isolated gaps filled within this

    # ── tide ─────────────────────────────────────────────────────────────
    mean_tide: float = 0.0  # fallback when no record or no timestamp

    # ── point filtering ──────────────────────────────────────────────────
    # Depth ceiling is applied AFTER tide correction, so it is referenced to
    # chart datum: no seabed in the area is shallower than this.
    ceiling: float = -5.0  # m

    # MAD-based outlier rejection. The threshold is
    #   max(k * 1.4826 * MAD, floor)
    # so it tightens on flat ground and relaxes on slopes. `floor` matters
    # more than `k`: it is what governs behaviour in flat areas, where MAD is
    # small. Validated at 0.4; raising it toward 0.6 keeps more fine detail
    # at the cost of more residual noise.
    filter_cell: float = 4.0  # m, neighbourhood for the local median
    filter_k: float = 3.0  # multiples of the robust sigma estimate
    filter_floor: float = 0.4  # m, lower bound on the threshold

    # ── file exclusion ───────────────────────────────────────────────────
    # A line logging water-column returns records several times more points
    # per occupied cell than its neighbours.
    flag_density_ratio: float = 2.5  # times the survey median
    flag_shallow_pct: float = 2.0  # percent of points above the ceiling

    # ── merging ──────────────────────────────────────────────────────────
    # Per-file equal weighting, never per-point: one line with many times the
    # point count would otherwise dominate every overlapping cell.
    merge_method: str = "mean"  # "mean" or "median"

    # ── reference surface ────────────────────────────────────────────────
    gt_valid_tol: int = 3  # cells; beyond this the reference is
    # extrapolated and must not be used

    # ── patches ──────────────────────────────────────────────────────────
    patch_size: int = 32  # cells; spans the 5-30 m washboarding
    # wavelength at 1 m resolution
    stride_train: int = 16  # 50% overlap during extraction
    max_nan_ratio: float = 0.30  # reject patches with worse raw coverage

    # ── spatial split ────────────────────────────────────────────────────
    # Region-based. Random splitting would place near-duplicate patches on
    # both sides; file-level splitting is also insufficient because adjacent
    # lines cover the same ground.
    split_axis: str = "col"  # "col" splits on easting
    test_frac: float = 0.20
    val_frac: float = 0.15
    split_buffer: int = 32  # cells excluded between bands, so no
    # patch shares a cell across a boundary

    # ── training ─────────────────────────────────────────────────────────
    model: str = "light_unet"  # "light_unet" or "encoderdecoder"
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-3
    epochs: int = 200
    patience: int = 15  # epochs without improvement before stop
    min_delta: float = 1e-5  # improvement below this does not count
    lr_patience: int = 7  # epochs before the scheduler steps down
    lr_factor: float = 0.5
    augment: bool = True
    seed: int = 42

    # ── inference ────────────────────────────────────────────────────────
    stride_infer: int = 8  # denser than extraction; Hann blending
    # needs the overlap to remove tile seams
    smooth_sigma: float = 0.8  # light smoothing after gap filling
    sharpen_alpha: float = 0.35  # high-frequency restoration; 0 after
    # measurement showed no benefit

    # ── methods ──────────────────────────────────────────────────────────
    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))

    def override(self, **kwargs):
        """Return a copy with the given fields replaced.

        Unknown field names raise rather than being silently ignored, so a
        typo in a script argument does not pass unnoticed.
        """
        unknown = set(kwargs) - set(self.to_dict())
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        d = self.to_dict()
        d.update(kwargs)
        return Config(**d)
