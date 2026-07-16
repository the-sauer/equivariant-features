# Patch scale budget, detector error, and jitter

Why the blob-scale settings are what they are. Everything here is measured on real
boards (seeds 40/60/80) with `blob_descriptor_base.yaml` params, **after** the scale
normalization fixes in [data pipeline](data_pipeline.md#shape-normalization--and-three-bugs)
— those fixes change every number, so pre-fix measurements do not transfer.

## How small a blob gets

A blob's scale reaches the patch extractor through three multipliers:

```
σ_warped  =  σ_intrinsic  ×  s_place  ×  sqrt(det J)
             (board raster)  (placement)  (per-view homography)
```

- **`σ_intrinsic`** floors at **2.362 px** — `min_scale: 0.2` mm at 300 dpi, which lands
  right on BlobBoards' hard `min_scale >= 2.0 px` assertion. The ladder is 17 levels,
  2.362 … 95.238 px, and it is bottom-heavy: **26.6%** of all blobs sit on the lowest
  rung, 61% at or below 3.75 px.
- **`s_place`** is `board_scale_range`, currently `[0.4, 0.7]`.
- **`sqrt(det J)`** measured over `sample_homography(fit_to_frame=True)`: p5 **0.329**,
  p50 **0.727**, min **0.118**.

Net (percentiles over all keypoint × warped-view pairs):

| stage | p1 | p5 | p50 | p95 |
|---|--:|--:|--:|--:|
| intrinsic (board raster) | 2.36 | 2.36 | 3.75 | 18.90 |
| × placement (composite) | 0.97 | 1.08 | 1.95 | 11.36 |
| **× warp** (what patches sample) | **0.34** | **0.54** | **1.40** | **8.02** |

The absolute floor is `2.36 × 0.4 × 0.118` ≈ **0.11 px σ** — a rendered disc of radius
0.22 px. 27% of warped views land below σ=1 px; 55% below σ=1.5.

Below about `s_place · sqrt(det J)` = 0.15 the smallest blob **collapses to exactly one
black pixel** regardless of sub-pixel placement (integrated ink 1.00, CV 0%). It does
not vanish — it becomes a scale-less impulse.

## Small blobs are not the problem

The intuition that these need a lower bound is wrong, and it is worth being precise
about why. With the normalization correct, each anchor was compared to its own positive
and to the hardest *other* same-scale blob on the same board (348 anchors, best NCC over
rotations):

| σ_w | n | pos NCC | hard-neg | margin | separable |
|---|--:|--:|--:|--:|--:|
| [0, 0.8) | 12 | 0.978 | 0.647 | 0.334 | **100%** |
| [0.8, 1.0) | 15 | 0.987 | 0.662 | 0.310 | **100%** |
| [1.0, 1.5) | 48 | 0.993 | 0.664 | 0.322 | **100%** |
| [2.0, 3.0) | 49 | 0.989 | 0.612 | 0.371 | **100%** |
| [5, 10) | 70 | 0.985 | 0.410 | 0.476 | **100%** |
| [10, 20) | 62 | 0.827 | 0.251 | 0.582 | **100%** |
| [20, ∞) | 21 | 0.654 | 0.154 | 0.478 | **100%** |

Even the sub-pixel blobs are separable with a healthy margin, and their patches carry
real contrast (std ≈ 0.22, not featureless). The reason is structural: with
`logpolar_inner_factor: 2` the annulus **starts at the blob's own cutoff radius** and
reads out to `16σ`. The anchor blob is deliberately excluded — the signal is the
*surround*, which contains the (larger) neighbouring blobs. A 2-px anchor dot is fine
because nothing reads it.

If anything the **large** end is the weak one (pos NCC 0.827 → 0.654): a 16σ annulus
around a σ=95 blob runs off the board and samples frame/background instead.

**So: no lower bound on the training set.** The real constraint is on the *deployment*
side — see [detector error](#detector-error-measured) below.

## Detector error (measured)

Raw VulkanSift on the warped views, matched to exact GT by highest response over 3573
in-frame blobs (90% overall recall):

| σ_w | n | recall | pos err p50 (×σ) | scale ratio det/GT | within-bin spread |
|---|--:|--:|--:|--:|--:|
| <1.0 | 898 | **59%** | 0.880 | 2.125 | ±10% |
| 1.0–1.5 | 1196 | 100% | 0.673 | 1.645 | ±12% |
| 1.5–2.0 | 428 | 100% | 0.492 | 1.071 | ±3% |
| 2.0–3.0 | 466 | 100% | 0.284 | 0.836 | ±5% |
| 3.0–5.0 | 268 | 100% | 0.200 | 0.792 | ±3% |
| 5–10 | 187 | 100% | 0.120 | 0.766 | ±3% |
| >10 | 130 | 100% | 0.056 | 0.735 | ±3% |

Three findings, each of which changed a decision:

**1. Position error is ~1 px absolute, not a fraction of σ.** It looks scale-dependent
in σ units (0.88σ → 0.056σ) but in pixels it is flat at 0.7–1.1 px in every band. Hence
`keypoint_jitter` is **in pixels**. A σ-relative jitter tuned to the small blobs would
put ~10 px of error on a σ=20 blob against a real 1.1 px — and large blobs are the most
jitter-sensitive of all.

**2. The scale error is systematic, not random.** The headline spread (log-std 0.376,
"±45.6%") is almost entirely a *scale-dependent bias*: within any band the spread is
only ±3–5%. The random component that jitter should model is therefore **±5%**. The bias
is a calibration problem, not a jitter problem.

**3. The scale estimate is not invertible below σ_w ≈ 2.5.** A detected σ of ~2.0 is
what you get for a true σ of 1.25, 1.75 **and** 2.5 — the detector saturates at its
scale-space floor. Combined with 59% recall below σ_w=1, the deployed system cannot
recover the scale that a small blob's training patch was normalized by. Since ~70% of
warped views sit below σ_w=2, this — not patch quality — is the argument for making
blobs bigger relative to the camera.

> **Caveat: this is the raw detector.** The deployed chain is
> `detect_blobs → adapt → filter_blobs`, and `adapt`'s whole job is refining the scale
> fit — exactly the step that might undo the saturation in (3). It could not be run:
> `blobboards.detect_blobs` is broken from Python (the wrapper passes
> `overlap_threshold`, the Julia ext wants `nms_mode`, and `detect.jl` under that wants
> a `config` object), and the CUDA `_adapt` returns 0 frames even on a clean synthetic
> 3-disc image. **Position error should carry over; treat finding (3) and the bias
> numbers as provisional until `adapt` runs.**

## Jitter

Train-time coords and scale come from exact Julia GT; at inference they come from the
detector. Without jitter the descriptor never sees that error. `keypoint_jitter` (px)
and `scale_jitter` (log-normal relative) simulate it on the **warped views only** —
the identity view is the clean reference and stays on exact GT.

Sensitivity, clean anchor vs jittered positive (122 blobs):

| jitter | pos NCC | margin | separable |
|---|--:|--:|--:|
| none | 0.939 | 0.259 | 98% |
| coords 0.5σ | 0.807 | 0.181 | 87% |
| coords 1.0σ | 0.623 | 0.049 | 70% |
| scale ×1.10 | 0.809 | 0.170 | 96% |
| **scale ×1.25** | 0.546 | **−0.039** | **27%** |
| scale ×1.50 | 0.307 | −0.212 | 1% |

**Scale error dominates, and the reason is geometric.** The log-polar radial axis is
log-spaced, so a scale error of factor *s* is a **rigid horizontal shift** of
`log(s) · 30.8` px (with `inner=2, outer=16, patch_size=64`: `64 / log(8)` = 30.8 px per
e-fold). A ×1.25 error shifts the patch 6.9 of 64 px and the margin goes *negative* —
the true positive resembles the anchor **less** than some other blob does. Position
error degrades gracefully by comparison. Large blobs are far more jitter-sensitive than
small ones (NCC 0.114 at σ_w>15 vs 0.761 at σ_w<1 under 0.5σ + ×1.25).

**Do not over-jitter.** Jitter is only worth its cost if it matches what the detector
actually does — jittering ±25% when the detector achieves ±5% buys robustness you do not
need by discarding real discriminative information.

Settings the measurement supports:

| param | value | why |
|---|--:|---|
| `keypoint_jitter` | **1.0** | px; measured 0.7–1.1 px in every scale band |
| `scale_jitter` | **0.05** | measured within-band spread; revisit once `adapt` runs |

The draw is keyed on the flat dataset index, so it is **fixed per (keypoint, view)**:
with `in_memory: true` patches are extracted once, so this perturbs the training pair
rather than resampling per epoch. With thousands of keypoints the *distribution* is
still well sampled. `jitter_seed` varies the draw.

## `resolution` picks the board, not just its sampling

**Changing `resolution` gives a different board, not a finer rendering of the same one.**
`blob_board` asserts `raster_aligned=true`, and under it `_sample_center` quantizes every
blob centre to a pixel centre:

```julia
ix = lo_x + floor(Int, rand(rng) * (hi_x - lo_x))
return (Float64(ix) + 0.5, Float64(iy) + 0.5)
```

Collisions are then tested against grid-snapped boxes (`bbox(...; snap_to_grid=true)`).
So accept/reject depends on the pixel grid; one decision differing desynchronizes the
RNG stream and everything after it. Measured, seed 40: **1189 blobs at 300 dpi vs 1202
at 600 dpi**, and the layouts are *unrelated* rather than perturbed (agreement at
chance level in pattern-relative coordinates).

What is preserved is the **distribution**: the σ ladder is physically identical (17
levels either way; the 600 dpi ladder is exactly 2× the 300 dpi one, 4.724…190.476 vs
2.362…95.238). So a different `resolution` is a different *draw* from the same
distribution.

Consequences:

- Two datasets at different `resolution` are **not comparable board-for-board**. Blob *i*
  is a different blob. Any A/B across resolution is confounded by the layout change and
  needs enough boards to average it out.
- A board cannot be re-rendered at a second resolution and matched against itself. If a
  fine and a coarse raster of the *same* board are ever needed, they must come from one
  render plus an image downsample — resampling pixels preserves geometry, re-rendering
  does not.
- `resolution` is in the cache key, so this is handled automatically.

## Why the camera stays at 300 dpi

Raising `resolution` scales the whole frame (1772 → 3543 px), so it costs 4× memory and
compute everywhere — and per above, hands you a different board. Measured at the small
end (different boards, so read this as a distribution-level comparison):

| | pos NCC | hard-neg | margin |
|---|--:|--:|--:|
| 300 dpi | 0.978 | 0.647 | **0.334** |
| 600 dpi | 0.994 | 0.765 | **0.229** |

Positives get cleaner but negatives get *more confusable*, and the margin drops. The
honest reading is that the 300 dpi margin is partly an artifact — resampling noise
decorrelates patches and flatters the metric, while at 600 dpi you see the real
difficulty. Either way there is no separability to buy: 300 dpi is already at 100%.

Note the interaction with backgrounds: `load_background` resizes every scene to the
canvas, so a 3543 px frame would upsample even 2000 px source photos. Raising the camera
dpi degrades the negatives unless the backgrounds grow too.

## Levers, if blobs need to be bigger

In rough order of cost:

1. **`board_scale_range`** (`[0.4, 0.7]`) — free. The board currently throws away up to
   2.5× before the homography touches it. Costs visible background for garbage keypoints.
2. **`min_fit_scale`** in `transform_params` — floors `fit_to_frame`'s shrink, truncating
   the worst tail (it re-introduces minor clipping). Measured effect on `sqrt(det J)`:

   | `min_fit_scale` | p0 | p1 | p5 | p50 |
   |---|--:|--:|--:|--:|
   | 0.0 (default) | 0.118 | 0.221 | 0.329 | 0.727 |
   | 0.5 | 0.220 | 0.281 | 0.362 | 0.732 |
   | 0.7 | 0.309 | 0.387 | 0.467 | 0.767 |

   It only floors the *fit* component, so it trims the tail rather than shifting the
   median.
3. **`min_scale`** in `blobboard_params` — moves the whole ladder up. 0.2 mm is already
   at BlobBoards' 2.0 px floor at 300 dpi, so it cannot go *down* without raising dpi.
4. **`resolution`** — 4×; see above.
