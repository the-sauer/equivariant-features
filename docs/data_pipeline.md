# Blob-board data pipeline

The descriptor is trained/validated on synthetic blob boards rendered by
`BlobBoards.jl`, wrapped by `BlobBoardHomographyData` (`src/aef/data/blobboards.py`),
which subclasses `HomographyData` (`src/aef/data/homography.py`).

## Views, keypoints, patches

Each board contributes `transforms_per_image` **warped views** plus **one identity
(un-warped) view** (homography `= I`). Blob keypoints are the boards' ground-truth
blob centres/scales. For every keypoint × view a shape-normalized patch is
pre-extracted once (`in_memory: true`) and cached, so per-item cost at train time is a
tensor index + collate. The FPR95/contrastive label is the keypoint's feature id
(`keypoints[..., 1]`), shared across a keypoint's views → those form the positive
pairs.

## `in_memory` is task-specific, not a free toggle

`in_memory` decides *what a batch contains*, and the two `process_batch` functions
expect different things — so its value is dictated by the task, not a performance dial:

| | `in_memory: true` | `in_memory: false` |
|---|---|---|
| patches | pre-extracted once, cached in RAM (`patches_available=True`) | not extracted |
| batch carries | `"patches"` | `"images"` (collate warps each view on the fly) |
| works with | `process_batch_blobs` (reads `data["patches"]`) | `process_batch_canonicalize` (reads `data["images"]`) |

Only `blob_descriptor_base.yaml` sets `in_memory: true`; `blob_descriptor_canonicalization.yaml`
omits the key and `blobboards.py` applies `kwargs.setdefault("in_memory", False)`, so it
runs the on-the-fly path. **Consequences:** the blob-descriptor configs must keep
`in_memory: true` (with `false`, `process_batch_blobs` raises `KeyError: 'patches'`),
and the `in_memory=False` collate fallback must not be deleted as "dead" — canonicalization
depends on it.

Note the flag no longer selects how *images* are loaded: `HomographyData` takes an
in-memory tensor only (the on-disk image path went away with `KaggleHomographyData`).

## Background compositing

Boards can be placed onto real background scenes so warped views look like a board
photographed in a scene (rather than on plain white). Controlled by dataset params:

| param | meaning |
|---|---|
| `background` | local folder of scene images; `null` disables all of the below |
| `background_kaggle_slug` | fallback dataset (default `arnaud58/landscape-pictures`) when the local folder is empty/missing |
| `board_scale_range` | board extent as a fraction of the frame (min, max); the placement's uniform scale |
| `background_lighting` | brightness-match the board to the local scene + low-frequency shading gradient |
| `shading_strength` | weight of the multiplicative shading field |
| `background_seed` | per-board RNG seed (per-board = seed + board index) → reproducible pinned sets |

**Placement is a strict similarity — scale + translation, no rotation.** Rotation is
supplied by the per-view homography, so adding it at placement would double it; the
axis-aligned box also keeps the identity view an upright canonical reference. A
similarity preserves circles, so blobs stay isotropic and the identity-view shape
normalization (which assumes isotropy) stays valid — a general affine would silently
break it. Placement + coord/scale transform + lighting live in
`sample_placement_similarity` / `composite_board` (`homography.py`).

## The identity (reference) view stays clean

The un-warped identity view is the canonical *reference* patch: it gets **no
background compositing and no augmentation**. Only warped views carry the scene +
augmentation. This is implemented with parallel `*_clean` arrays
(`keypoint_coords_clean`, `keypoint_scales_clean`, `images_clean`) used only by the
identity branch of `__getitem__`/the collate, and an augmentation gate in
`compute_patches` (`img_id % views == views-1` → no aug). When `background` is unset
the `_clean` arrays alias the mains, so only the (always-on) augmentation gate differs
from plain-white boards.

## Background "garbage" keypoints

`garbage_fraction > 0` adds background distractor keypoints — **untracked
singletons**: each gets a globally-unique, always-negative label (`-(1 + index)` in
`__getitem__`), so it never forms a positive pair with a blob or another garbage point;
it is a pure negative that hardens FPR95. Points come from SIFT on the composited
background (preferred) topped up with random background points to hit an exact target
(`garbage_source = "sift" | "random"`). They are added **after** the scale filter and
the `max_keypoints` cap, and trimmed to `round(garbage_fraction · n_capped_blobs)`, so
every split gets the same constant garbage set and stays the same size.

## Scale bands and equal-sized splits (validation)

The validation config reports FPR95 per blob-scale band. Splits share one pinned set of
boards; each keeps a third of the **raw intrinsic** blob-scale distribution via
`scale_quantile_range` (`small` `[0,0.33]`, `medium` `[0.33,0.66]`, `large`
`[0.66,1.0]`; `overall` keeps all). Filtering on *raw* scales (not the placement-scaled
ones) keeps the bands clean even with a randomized placement scale.

- `max_keypoints` caps each split to a fixed count (deterministic subsample) **after**
  the band filter, so all four splits are exactly the same size (`overall` becomes a
  subsample of the full pool, not the literal union of the bands).
- The cap must be ≤ the smallest band. For the current pinned 8-board set (9449 blobs;
  bands ≈ small 2403 / medium 3365 / large 3681) that ceiling is ~2403; the config uses
  a value below it (a multiple of the val batch size is convenient).

Only `overall` drives `best.pth` selection; the bands carry loss `weight: 0` (monitored,
not optimized).

## Dataset cache (disk)

Preparing a dataset is slow: board rendering (Julia), compositing, SIFT, garbage
detection and patch pre-extraction. Setting `cache_dir` writes the **fully prepared**
dataset to disk and reuses it on the next run.

- **What's stored**: transforms, all keypoint arrays (incl. `*_clean` and
  `keypoint_is_garbage`), images/composites, and `precomputed_patches` — everything
  needed to reconstruct the dataset without re-running the pipeline.
- **Key**: a short hash (`dataset_cache_key`) of every param that determines the
  contents, plus `CACHE_VERSION`. It hashes the **effective** params — `effective_params`
  overlays what the caller passed onto the constructor *defaults* of
  `HomographyData.__init__` and `BlobBoardHomographyData.__init__` — so a param the
  config never mentions (`patch_size`, `supersample`, `patch_type`, the log-polar
  factors, …) still keys the cache at the value it actually ran with, and **changing a
  default in code invalidates every key** instead of silently reusing stale data.
  Passing a param explicitly at its default does *not* split the cache.
  `data_dir`, `cache_dir`, `extraction_batch_size` and `sift_batch_size` are excluded —
  they don't change the data.
  The *algorithm* is not hashed: if you change the compositing/garbage code itself,
  bump `CACHE_VERSION` (in `homography.py`) by hand.
- **Checked before board generation**: `BlobBoardHomographyData` derives the key from
  its params and, on a hit, skips rendering entirely (the slowest step) — so the check
  cannot live in `HomographyData` alone.
- **Measured**: cold build 54 s → warm load ~0 s, with bit-identical tensors.
- **Sharing**: the key covers dataset params only, *not the model* — so every model at
  the same `scale` (e.g. `steerable`, `efficient8`, `efficient4`) shares one entry. A
  12-job model sweep builds 4 datasets instead of 12. Each validation split gets its
  own entry (they differ by `scale_quantile_range`).
- **Concurrency**: written via temp file + atomic rename, so parallel sweep jobs can
  never read a half-written cache. Two cold jobs may both build it (last wins) —
  wasteful but correct. Saving is best-effort: an unwritable `cache_dir` warns and the
  run continues.

**Caveat — a hit reuses the exact same random draw.** Homographies are sampled with an
unseeded RNG, and board seeds (when `seeds` is unset) and `polarity: random` are random
per run; the cache freezes whichever draw was written first. That is *desirable* for
pinned validation sets and for controlled model comparisons, but it means repeats
sharing a key see identical data. Clear the entry (or vary a keyed param) to force a
fresh draw. Since patches are pre-extracted once per run and never recomputed
(there is no re-sampling of homographies mid-run), caching is semantically identical *within* a run.

## Reproducing example composites

See scripts under the scratchpad during development; the essentials: construct
`BlobBoardHomographyData(num_boards=…, seeds=[…], image_size=[150,150],
background=<dir or kaggle path>, board_scale_range=[0.4,0.7], garbage_fraction=0.3,
background_seed=…)`, then save `self.images[i]` (composites) / `self.images_clean[i]`
(clean references) with `keypoint_coords` overlaid (blobs vs `keypoint_is_garbage`).
