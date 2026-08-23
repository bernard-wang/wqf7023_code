# Project Context

Automated denoising of multibeam echosounder (MBES) bathymetric survey data
using a compact U-Net. Master's thesis project with an industry partner
(Petroseis Sdn. Bhd.).

This document records decisions already made and validated experimentally.
**Read this before modifying the pipeline.** Several changes that look like
obvious improvements were tested and rejected for measured reasons; they are
listed in "Tested and rejected" below.

---

## Core idea

The network learns **noise patterns only**, never absolute depth.

Before inference the local mean is subtracted from each patch, so the model
sees zero-mean residuals. After inference the mean is added back and the
depth baseline is restored from physical measurements (tidal height). The
model is therefore structurally unable to shift the vertical datum, rather
than merely discouraged from doing so by the loss function.

This is the distinction from DnCNN-style residual learning: there the
residual is `raw - clean`, which still carries the local depth baseline. Here
the baseline is removed *before* the residual is formed.

---

## Repository state

Five modules exist, all partially written:

| File | Purpose |
|---|---|
| `config.py` | constants |
| `model.py` | `SmallUNet`, `EncoderDecoder` |
| `io_utils.py` | LAS/XYZ reading, tide lookup, grid axes, export |
| `preprocess.py` | filtering, gridding, merging |
| `dataset.py` | patch extraction, spatial split |

Still to be written: `train.py`, `inference.py`, `evaluate.py`,
`diagnostics.py`, `figures.py`, `app.py` (deployment).

These were consolidated from a set of ad-hoc scripts. Where the old scripts
and this document disagree, **this document is authoritative**.

---

## Final configuration

Validated on batch 2. Do not change these without measuring the effect.

```
read_las_points          sign normalised with -abs(z)
tide correction          per-file, nearest reading in a 5-minute record,
                         matched by YYYYMMDD_HHMMSS in the filename
depth_sanity_filter      ceiling = -5.0 m, applied AFTER tide correction
filter_outliers_mad      cell = 4.0 m, k = 3.0, floor = 0.4
anomalous file exclusion 20260328_120355.las removed manually
grid_one_file            per-cell mean
merge_grids              per-file equal weighting (NOT per-point)
hann_tile_inference      patch 32, stride 8, mu from the tile's own raw mean
gap fill + smooth        sigma = 0.8
```

### Ordering constraints

Tide correction must come **before** `depth_sanity_filter`. The ceiling is
referenced to chart datum, so it only has a stable meaning post-correction. A
water-surface return at -0.6 m raw becomes +0.6 m after a +1.2 m tide
correction, which is what the ceiling catches.

`filter_outliers_mad` must come **before** gridding. It is a point-level
operation; once points are averaged into cells the neighbourhood information
it needs is gone.

### Sign normalisation

`read_las_points` uses `-abs(z)`, not a sign test.

A test like `if z.min() > 0: z = -z` fails on files containing
water-surface returns: a single point at -0.6 in a positive-depth file makes
`z.min()` negative, the flip is skipped, and the entire file ends up
inverted. This is not hypothetical — batch 2 has such files.

`-abs(z)` is correct under both conventions and additionally maps false
returns to the shallow end, where the depth ceiling catches them.

Assumes no genuine returns above the vertical datum. True for seabed
bathymetry, false for intertidal or above-water features.

### Merge weighting

`merge_grids` averages **per-file grids**, not pooled points. Each file is
gridded independently first.

This matters: file 120355 has 4-8x the point count of its neighbours. Under
per-point pooling it would dominate every overlapping cell in proportion to
its point count. Under per-file weighting it contributes 1/N like any other
line.

Do not "simplify" this by pooling all points before gridding.

---

## Tested and rejected

Each of these was implemented and measured. None improved the final result.
Keep the code (the thesis reports them), but leave them disabled.

### Per-line vertical alignment

Weighted least-squares network adjustment of constant per-line offsets, with
a zero-mean constraint so the overall datum is preserved.

Improved **line-to-line consistency** from 6.98 cm to 2.74 cm mean absolute
pairwise difference (61%), across 100 overlapping line pairs. But accuracy
against the reference surface did not improve.

The likely reason is that internal consistency and external agreement are
different quantities. Forcing the lines to agree with each other moves them
away from the reference in places, because the reference was produced by a
different per-line treatment.

Worth keeping behind a flag: the client's stated requirement (0.1 m) is an
internal-consistency criterion, so this may be wanted later.

### Grid-level despiking

7x7 median-filter comparison, cells deviating more than 0.8 m removed.
Removes around 55 cells. Largely redundant with the point-level MAD filter,
which already catches these.

### Sharpness restoration

Adds back a fraction `alpha` of the raw high-frequency component after
inference.

On own-merged data: alpha 0 to 0.5 improved MAE marginally (6.73 to 6.46 cm)
but RMSE rose (10.67 to 11.19) and max error grew from 247 to 379 cm.

On company pre-merged raw: almost no effect (10.31 to 10.26 cm), because
their export has already removed the fine-scale content there is to restore.

Final setting: alpha = 0.

### Across-track refraction correction

Residual depth varies systematically with beam angle within each swath, the
classic sound-velocity signature.

Diagnosis is sound: using a leave-one-out reference (each line compared
against the merge of all *other* lines), the median residual traces a clean
symmetric arc, amplitude 8.81 cm, +0.71 / -5.00 / +0.71 cm at left edge /
centre / right edge.

A naive self-referencing version underestimates this at 5.2 cm and forces the
outer beams to zero, because in single-coverage areas the reference equals
the line itself. Use leave-one-out if revisiting this.

Applying the correction did not change line-to-line consistency
(2.74 to 2.87 cm). This is expected and does not mean the correction is
wrong: the consistency metric compares two lines in their overlap zone, where
their across-track coordinates are roughly symmetric about zero. A symmetric
correction subtracts nearly the same amount from both, and cancels in the
difference. **The metric is blind to symmetric across-track error.**

The error is real and shows up in the merged product as banding. It was not
adopted because no available metric confirmed an improvement.

### Fixed-threshold outlier filter

`|z - local median| <= 0.8` as an alternative to the MAD version. The MAD
version performed better and is the one in use.

Note these are different strategies, not the same parameter:

| Region | local MAD | MAD threshold | fixed |
|---|---|---|---|
| flat | ~0.05 | max(0.22, 0.4) = 0.4 | 0.8 |
| rough | ~0.30 | max(1.33, 0.4) = 1.33 | 0.8 |

The MAD version is stricter on flat ground and looser on slopes. `floor` is
the parameter that matters most; `k` has less influence.

---

## Data

### Batch 1 (20250408), training

29 LAS files, roughly 600 x 600 m, depths 6 to 15 m. Used for training,
validation and the spatially held-out test set.

Reference surfaces come in two forms:
- **Processed**: 3 decimal places. Used as the training target.
- **Final**: 1 decimal place, quantised to 0.1 m. Used for the whole-area
  comparison, because this is what the client compares against.

Training on Final would teach the model to reproduce the quantisation
staircase, since that quantisation *is* the label.

### Batch 2 (20260328), generalisation test

27 LAS files. Never used for training. Reserved for measuring
generalisation, so **do not tune the model on it**. Pre- and post-processing
may be adjusted; the network may not be retrained.

Tide record covers 10:55 to 15:05, range 1.048 to 1.528 m, mean 1.179549 m.
The 0.48 m variation over four hours is why per-file tide lookup matters
here.

### Input quality differs by an order of magnitude

Per-file outlier rate (deviation > 1.0 m from the local 4 m median):

| | batch 1 | batch 2 |
|---|---|---|
| mean | 0.04% | 1.65% |
| character | 4 of 28 files affected | every file affected |

The distributions do not overlap: batch 1's worst file (0.37%) is still below
batch 2's best (~0.58%).

The two surveys are a year apart and were probably acquired under different
conditions. This is a property of the delivered data, not a controlled
experiment.

### File 20260328_120355.las

Excluded manually. Point count 4-8x its neighbours; a water-surface return
band spans the entire 12-second record rather than appearing intermittently.
Neighbouring files cover the same ground, so removing it leaves no gap.

Should be automated by flagging files whose points-per-occupied-cell exceeds
about 2.5x the survey median.

### Company-provided raw XYZ

Petroseis also supplied a pre-merged raw XYZ for batch 2. It has already been
beam-filtered: point density 1-4 per cell, and the swath-edge lines visible
in own-merged data are absent.

Describe it as "raw data as delivered", not "unprocessed".

Processing it gave MAE 10.32 cm, worse than own-merged (5.31 cm), mainly
because a single mean tide value must be used — line timestamps are lost in
their merge.

---

## Results

### Batch 1

Patch-level test set:

| method | MAE cm | RMSE cm | p95 | max | R2 |
|---|---|---|---|---|---|
| raw | 12.74 | | | | |
| gaussian | 12.71 | | | | |
| median | 12.71 | | | | |
| encdec | **3.03** | 4.73 | 7.92 | 322 | 0.9993 |
| unet | 3.20 | **4.54** | **7.38** | **183** | **0.9994** |

EncoderDecoder has slightly better MAE but U-Net wins on RMSE, p95 and
especially max error. Skip connections suppress extreme failures rather than
improving average accuracy. Report this openly; it looks like a
counter-result but it is the ablation's actual finding.

Whole area vs Final reference: MAE 3.66, RMSE 4.72, bias +1.31, R2 0.9993.

### Batch 2 progression

| configuration | MAE | RMSE | bias | p95 | max | R2 |
|---|---|---|---|---|---|---|
| initial own merge | 14.44 | 22.45 | -9.81 | 38.23 | 340 | 0.909 |
| company pre-merged raw | 10.32 | 13.36 | -3.93 | 28.28 | 109 | 0.968 |
| + point filter | 6.21 | 14.10 | +2.64 | 15.60 | 354 | 0.964 |
| + align + despike | 6.73 | 10.67 | +2.61 | 17.27 | 247 | 0.979 |
| **final (MAD filter)** | **5.31** | **8.68** | **+1.82** | **14.18** | **240** | **0.986** |

**87.3% of cells agree with the reference within 10 cm.** This is the number
that answers the client's stated 0.1 m requirement most directly.

Note RMSE and MAE do not move together between rows. RMSE falling while MAE
rises means the error tail was compressed while the bulk widened slightly.

---

## Evaluation protocol

Two masks, different purposes:

- `footprint` — derived from raw coverage. This is the **deployment output
  range**, valid without any reference data.
- `gt_valid` — cells within 3 px of a genuine reference point.
- `eval_mask = footprint & gt_valid` — used for metrics and comparison
  figures only.

The reference grid is built with `NearestNDInterpolator`, which extrapolates
rather than returning NaN outside its input hull. Without `gt_valid`, profile
plots show a fabricated reference in areas the reference survey never
covered, and metrics are computed against invented values.

**Deployment mode uses no reference data anywhere in the processing path.**
Normalisation takes `mu` from each tile's own raw mean; the datum comes from
the tidal correction alone. `hann_tile_inference` should expose this as a
`mu_source` parameter ("raw" for deployment, "gt" for training-time
evaluation) so one function serves both.

---

## Spatial split

Region-based, not random, not file-level.

Patches are 32x32 px at 1 m, extracted with stride 16, so neighbouring
patches share 50% of their content. A random split would put near-identical
patches on both sides. A file-level split is also insufficient because
adjacent files overlap the same ground (34% of cells in batch 1 are covered
by more than one line).

Assignment is by easting band on the merged grid, with a 32 m buffer excluded
between regions so no patch shares a pixel across a split boundary.

Counts: train 646, val 64, test 134, buffer-dropped 41, total 885.

Augmentation is applied stochastically per batch during training, so these
counts are not multiplied.

Validation is deliberately small: it only monitors convergence and tunes
baseline filter parameters.

---

## Model

`SmallUNet`: 16-32-64-128 channels, Dropout2d(0.5) at the bottleneck, skip
connections. About 430K parameters.

`EncoderDecoder`: identical capacity ladder, no skip connections. Ablation
only. **Keep the two structurally identical apart from the skips**, or the
comparison stops being controlled.

The original U-Net is about 31M parameters, which would be heavily
over-parameterised for 646 training patches. Note in mitigation that each
patch supplies 1024 pixel-level targets, not one label.

Training: L1 loss, AdamW, lr 3e-4, weight decay 1e-3, batch 32, early
stopping patience 15.

---

## Known limitations

**Patch-scale receptive field.** 32 m patches cannot correct errors varying
over hundreds of metres. Per-file tide correction handles the dominant part,
but residual line-to-line offset survives as banding.

**Isolated feature attenuation.** Sharp isolated structures resemble acoustic
spikes at patch scale and are partly suppressed. In batch 1 a known feature
in the central area is reconstructed well, but it falls inside the training
region — this may reflect training coverage rather than a general capability.
Do not present it as evidence of feature preservation.

**Residual water-column returns.** Where false returns are dense enough to
fill a 4 m cell, the local median itself is contaminated and no
neighbourhood-based criterion can separate them from seabed. This is the
source of the remaining 240 cm max error.

**Time saving is not measured.** The pipeline plausibly saves manual effort
but no formal timing comparison against expert editing was made. State it as
an implication, not a result.

---

## Confidentiality

Refer to the structure in the survey centre as "known feature in the central
area". Do not name the facility type in code, figures, or documentation.

---

## Deployment

Target is a Hugging Face Space (Gradio). Constraints encountered:

- ZeroGPU pins torch versions and requires at least one `@spaces.GPU`
  function to exist at startup, even for CPU-only work. Free accounts may not
  be able to select CPU Basic.
- gradio 4.44 has a schema-generation bug with recent `gradio_client`
  (`TypeError: argument of type 'bool' is not iterable`) that returns 500 on
  every route. Use gradio 5.x, set the version in the README `sdk_version`
  and **not** in `requirements.txt`.
- Multi-file upload of 29 files is fragile over unreliable networks. Accept
  a zip (and optionally rar) archive as an alternative.

Planned but not yet built: a two-stage interface where files are inspected
first (coverage plot with flagged files drawn in red beneath the others,
per-file statistics, quantified coverage loss for any exclusion) and only
then merged and processed.

---

## Open tasks

- `train.py`, `inference.py`, `evaluate.py`, `diagnostics.py`, `figures.py`
- Automate anomalous-file flagging by point density
- Two-stage inspect-then-process interface
- Results chapter: needs a side-by-side outlier-rate figure for the two
  batches with a shared y-axis, and the batch 2 progression table above
