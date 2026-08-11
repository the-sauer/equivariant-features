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

## Shape normalization — and three bugs

`blob_normalizations` (`homography.py`) SVD-whitens the local Jacobian of the view's
homography and **drops the rotational factor `U`**, so an elliptical blob maps to an
isotropic one and the residual difference between an anchor and its positive is a pure
rotation (which the descriptor is supposed to handle). Both extractors use it.

That reduction to a pure rotation is **affine-tight**: `N` is linear, so it removes the
Jacobian and nothing beyond it. With `perspective_amplitude` non-zero the residual carries
a second-order term the whitening cannot see —
[perspective_and_rotation.md](perspective_and_rotation.md) derives it and sizes it for
these configs.

This one function has now produced three separate bugs — the mirror below, plus a
double scale correction and a mis-evaluated Jacobian. All three shared a signature:
patches that *looked* plausible in isolation but did not match their own anchor.

### The mirror bug

**The bug (fixed):** an SVD is only unique up to a *reflection pair*
(`det(U) = det(Vh) = -1`), and with `perspective: false` the sampled warps are
similarities — **100% of them have singular-value ratio 1.0000**, i.e. degenerate
singular values, so the factorization is *fully* ambiguous. torch was returning
`N = Σ@Vh ∝ diag(-1,1)` (a **reflection**, with no angle dependence) for warped views
while the identity view got `N = I`. The positive patch therefore came out **mirrored**
relative to its own anchor.

Measured on log-polar pairs (mean residual after the best angular alignment):

| | best roll | best roll of **flipped** anchor | verdict |
|---|--:|--:|:--|
| before | 0.880 | **0.711** | mirrored |
| after (`det>0` forced) | **0.711** | 0.880 | not mirrored |

The two numbers swap exactly. **This hurt log-polar most**: a mirror is an angular
*flip*, and no roll — hence no angular max-pool — can undo it, so it silently defeated
that architecture's only invariance mechanism (see
[log-polar descriptor](logpolar_descriptor.md)). The steerable path barely moved
(anchor↔positive descriptor distance 0.092 → 0.089), consistent with `NoStride` working
well — though that comparison used untrained nets and is weak evidence either way.

The fix negates a row of `Vh` when `det(Vh) < 0`: that flips both dets and leaves
`U@Σ@Vh` unchanged, so whatever survives dropping `U` is a proper rotation.
`CACHE_VERSION` was bumped to 3 — the cache key covers constructor params, **not** this
code, so pre-fix caches would otherwise be reused silently.

**The residual rotation this leaves is deliberate — keep it.** Downstream the detector
yields blobs at arbitrary orientation, so the training pairs *should* exercise the
rotation the descriptor has to absorb; that is what makes the task realistic. A polar
decomposition (`P = Vhᵀ Σ Vh`) would be rotation-free by construction and would strip it,
training the descriptor on a distribution the deployed detector never produces. Only the
**reflection** was wrong: a real homography has `det > 0` and never mirrors, so the
mirror was a pure SVD-sign artifact injecting a deformation the downstream task cannot
generate — and one no rotation-invariant descriptor can absorb.

### The double scale correction

**The bug (fixed):** `blob_normalizations` returned `Σ@Vh = Uᵀ inv(J)`, which removes
the Jacobian's **size** as well as its shape. The extractors then divide by the
keypoint's `scales`, which *already* carries the warp's `sqrt(det J)` — so the size came
out twice. The identity view (`J = I`) was unaffected, so every warped view disagreed
with its own anchor by a factor `1/sqrt(det J)`.

Measured on a pure-zoom homography (apparent radius of the anchor blob inside a
cartesian patch; the model's own centre-mask formula expects a constant 2.0 px):

| zoom | 1.0 | 0.8 | 0.6 | 0.5 | 0.4 | 0.3 |
|---|--:|--:|--:|--:|--:|--:|
| before | 2.03 | 2.52 | 3.24 | 3.99 | 5.01 | 6.63 |
| after | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 | 2.03 |

**This hurt log-polar worst — again.** A radial zoom is a *rigid horizontal shift* on a
log-spaced radial axis: `log(1/sqrt(det J)) · 30.8` px ≈ **18 of 64 px** at a typical
warp. The fix normalizes the factor to `det == 1` — shape and orientation only, size
left to `scales`, which is the only place it belongs.

### The Jacobian evaluated at the wrong point

**The bug (fixed):** `linearize_homography(H, coords)` differentiates `H` at a point of
its **domain**. Both `blob_normalizations` and `__getitem__`'s `scale_factor` passed the
*warped* coords — and `__getitem__` passed the un-normalized homogeneous vector, whose
overall scale the formula is not even invariant to. Measured error on a sampled
homography: up to **2.25×** in `sqrt(det J)`. It is invisible for affine warps and only
bites because `perspective_amplitude` is non-zero.

`blob_normalizations` wants `inv(J_H(p))` at the source point `p`, which by the chain
rule *is* the Jacobian of the inverse homography at the warped point — so it now
linearizes `inv(homographies)` at `coords`, fixing the evaluation point and removing a
`torch.linalg.inv` at once. `__getitem__` linearizes at the source coords.

### What the two fixes were worth

Each anchor vs its own positive and vs the hardest *other* same-scale blob on the same
board (348 anchors, best NCC over rotations; margin = pos − hard-neg):

| σ_w | before: pos / margin / separable | after: pos / margin / separable |
|---|--:|--:|
| [0.8, 1.0) | 0.166 / **−0.013** / **47%** | 0.987 / 0.310 / **100%** |
| [1.0, 1.5) | 0.875 / 0.138 / 70% | 0.993 / 0.322 / **100%** |
| [2.0, 3.0) | 0.822 / 0.189 / 73% | 0.989 / 0.371 / **100%** |
| [5, 10) | 0.673 / 0.173 / 71% | 0.985 / 0.476 / **100%** |
| [20, ∞) | 0.213 / 0.075 / 79% | 0.654 / 0.478 / **100%** |

At σ_w ≈ 0.9 the pre-fix pipeline was **worse than chance**: a blob's true positive
resembled it *less* than a random other blob did. Those were the pairs SupCon was being
asked to pull together. On a single min-scale log-polar pair the anchor↔positive NCC
went **0.179 → 0.782** (det-1 alone) → **0.989** (both fixes).

`CACHE_VERSION` was bumped to **4**. The invariant is pinned by
`tests/unit/test_blob_normalizations.py`: `N @ J_H(p)` must be exactly `sqrt(det J)`
times a rotation — one assertion that catches both bugs (verified to fail on the old
code, not assumed to).

See [scale budget and jitter](scale_budget_and_jitter.md) for what this implies about
blob size, and note that **any patch measurement taken before these fixes is void**.

### Measuring patches — two traps

Both of these produced confidently wrong conclusions before being caught:

- **Normalize before comparing patches.** Raw L2 between an anchor (clean board, white
  surround) and its positive (warped, grey surround) is dominated by the global
  intensity offset, making matched pairs look no better than random ones. The networks
  never see it (`HardNet.input_norm`), so neither should the metric.
- **Set `homography_seed` when comparing builds.** By default `sample_homography` draws
  from numpy's *global* RNG, so every dataset construction gets different warps and two
  builds differ by a random rotation on top of whatever you changed. `homography_seed`
  threads one `default_rng` through every draw, making the whole sequence a function of
  the seed — with it fixed, two builds are bit-identical (verified: same transforms,
  same extracted patches).

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

## Board-validity masks (`precompute_masks`)

With `precompute_masks: true` the extractors co-sample a per-patch validity mask
(1 = on the board) on the *same* warp as the patch, cached alongside it and delivered
as `"masks"` in the batch, next to the `"is_pdf"` flag (which the collate emits
regardless of this setting — it is also a loss-side label, see `ProxyAnchoredSupCon`). It is the GT for the learned-mask descriptor
heads (`model.params.learned_mask`, see
[log-polar](logpolar_descriptor.md#learned_masktrue--mask-aware-pooling-without-a-target-mask)
and [steerable](steerable_descriptors.md#learned-board-validity-masking-learned_mask)).

Both patch types support it. For the identity view (`is_pdf`) the source is a full-frame
ones image (the clean board fills the frame, so validity == in-frame); for warped views
it is the composited board mask, warped by the same homography — a warped view's
off-board region is real background *inside* the frame, so `oob` alone would be wrong.

**Cartesian masks are single-channel**, and each `patch_scale_factors` entry samples a
different physical extent onto the same pixel grid, so one mask cannot describe several:
`precompute_masks` with `patch_type: cartesian` requires exactly one scale factor and
raises otherwise. (All the steerable configs use a single `["${scale}"]` entry.)

Enabling it changes the dataset cache key, so masked and mask-less runs of the same
network do not share a prepared dataset — which is why the launcher gives the `*_mask`
nets their own `cartesian_mask` dataset group.

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

**Backgrounds are resized to cover + centre-crop, not stretched.** `load_background`
used to pass a 2-element size to `torchvision.resize`, which honours it exactly and so
squashed a landscape photo onto the square canvas. Backgrounds are what the garbage
keypoints — the dataset's pure negatives — are detected on, so distorting their image
statistics makes those negatives unrepresentative. Note the scene is always resized to
the canvas: source photos **below** the frame resolution get upsampled, making the
negatives artificially easy. At 300 dpi (1772 px canvas) a ≥2000 px source downsamples
cleanly; this is one more reason not to raise the camera dpi.

## Lens distortion (`distortion_params`)

A warped view is no longer `H` alone but **`D ∘ H`**: the plane-to-image homography
followed by a radial lens. `src/aef/transforms/distortion.py` holds the model;
`distortion_params` (`None` → pinhole, i.e. every pre-lens run is unchanged) turns it
on per dataset:

```yaml
training:
  dataset:
    params:
      distortion_params:
        lambda1: [0.05, 0.25]   # fixed number, or [lo, hi) drawn uniformly per view
        lambda2: 0.02
```

The **undistortion** — observed view pixel `q` back to the pinhole point it would have
had — is the closed form

```
U(q) = c + (q − c)·(1 + λ₁ρ + λ₂ρ²),   ρ = |q − c| / R
```

with `c` the frame centre and `R` half the frame diagonal, so `ρ = 1` at the corners and
the λ are dimensionless (portable across resolutions: λ₁ = 0.1 moves the corner by ~10%
of the half-diagonal). **λ ≥ 0 is barrel/fisheye** — undistortion expands the periphery,
so the lens compressed it. Negative λ gives pincushion, and is checked at construction:
a radial map that stops increasing folds the image onto itself, which makes the view map
non-injective and `distort` root-free, so the draw is rejected rather than silently used.

**Why the closed form sits in the undistortion direction.** Rendering a view fills each
output pixel `q` from the source point `H⁻¹(U(q))` — millions of evaluations per frame,
so that direction has to be closed-form. The forward map `D = U⁻¹` is only ever needed
for keypoints (a few hundred per board) and is Newton on the radius. That also means
`warp_perspective` cannot render a lensed view at all (the view-pixel → source map is
not projective); `render_view` builds the grid explicitly instead. It agrees with
`warp_perspective` to ~1e-5 when the lens is off, and the pinhole path still takes the
kornia call, so nothing about a lens-free dataset changes.

**What the lens does to the shape normalization** is the part that is easy to get
wrong — and it fails the same way the [three earlier bugs](#shape-normalization--and-three-bugs)
did, with patches that look fine but stop matching their anchor. The Jacobian to whiten
is `J_D(u)·J_H(p)`, so `blob_normalizations` builds its inverse as `J_H(p)⁻¹·J_U(q)`.
Two things follow: `coords` are now the **observed** point `q`, so the homography must be
linearized at the **undistorted** `u = U(q)` and not at `q` (exactly the "Jacobian at the
wrong point" trap again), and the lens' own `|det J|` has to reach `scales` — `__getitem__`
divides the warp's `sqrt(det J_H)` by `sqrt(det J_U(q))`. A blob whose patch lattice is not
divided by the *composed* size sits at a view-dependent scale in its own patch.

The extractors themselves are unchanged: a patch has always been a local *linear*
approximation of the view map around the keypoint (that is what `blob_normalizations`
produces), and a lens only changes which linear map that is.

Two deliberate scoping choices: the **identity view never gets a lens** (it is the clean
reference — see below), and the λ are drawn from their own generator seeded off
`homography_seed`, so switching a lens on does **not** shift the warp stream of a pinned
split. `process_batch_canonicalize` still reads `data["homographies"][..., :2, :2]` as its
target shape and therefore ignores the lens; canonicalization on lensed data would need
that to consume the composed Jacobian instead.

## The identity (reference) view stays clean

The un-warped identity view is the canonical *reference* patch: it gets **no
background compositing and no augmentation**. Only warped views carry the scene +
augmentation. This is implemented with parallel `*_clean` arrays
(`keypoint_coords_clean`, `keypoint_scales_clean`, `images_clean`) used only by the
identity branch of `__getitem__`/the collate, and an augmentation gate in
`compute_patches` (`img_id % views == views-1` → no aug). When `background` is unset
the `_clean` arrays alias the mains, so only the (always-on) augmentation gate differs
from plain-white boards.

It also gets **no jitter** (below). Both views read rasters rendered at the same
`resolution` — note that `resolution` selects *which board you get*, not just how
finely it is sampled (see
[scale budget and jitter](scale_budget_and_jitter.md#resolution-picks-the-board-not-just-its-sampling)).

## Simulated detector error (jitter)

Coords and scales come from exact Julia GT, but at inference they come from a detector
that misses them. `keypoint_jitter` (std, **in px** of the warped frame) and
`scale_jitter` (log-normal relative std) perturb the **warped views only**;
`jitter_seed` varies the draw, which is keyed on the flat index and therefore fixed per
(keypoint, view) — with `in_memory: true` the patch is extracted once, so this perturbs
the training pair rather than resampling per epoch.

Both default to `0.0` (off). The measured detector error is ~1.0 px and ~±5%; scale
error is *far* more damaging than position error, and over-jittering costs real
discriminative power — see
[scale budget and jitter](scale_budget_and_jitter.md#jitter) before raising either.
`tests/unit/test_jitter.py` pins that the identity view is never jittered.

## Background "garbage" keypoints

`garbage_fraction > 0` adds background distractor keypoints — **untracked
singletons**: each gets a globally-unique, always-negative label (`-(1 + index)` in
`__getitem__`), so it never forms a positive pair with a blob or another garbage point;
it is a pure negative that hardens FPR95. Points come from SIFT on the composited
background (preferred) topped up with random background points to hit an exact target
(`garbage_source = "sift" | "random"`). They are added **after** the scale filter and
the `max_keypoints` cap, and trimmed to `round(garbage_fraction · n_capped_blobs)`, so
every split gets the same constant garbage set and stays the same size.

## Keypoint order, and what FPR95 actually counts

Two coupled details decide whether the reported FPR95 is meaningful. Both were bugs
that made the metric read **better than reality**; expect reported FPR to rise now.

**Keypoint order is shuffled once** (`shuffle_keypoints=True`, `shuffle_seed`). Garbage
is appended to the end of the keypoint arrays and the validation loader runs
`shuffle=False`, so without this the *trailing batches contained only garbage*. Garbage
labels are all unique singletons, so such a batch has **no real positive pair** — its
only positives were the diagonal self-pairs at distance 0, which trivially satisfy 95%
recall ⇒ a meaningless `FPR = 0` that was then averaged in. The shuffle is a single
deterministic `randperm` over the keypoint arrays, applied after the garbage block and
**before** `compute_patches` (so the extracted patches match the new order). A
keypoint's views stay contiguous (`index = keypoint_i * views + j`), so its views still
land in the same batch and positives are preserved.

**The distance matrix diagonal is excluded** (`fpr_from_features`, `aef/evaluate`).
Pairing a patch with itself is a trivial positive at distance 0 that always ranks
first. With `views=2` that made *half* of all "positives" free self-pairs, so the
threshold was effectively set at ~90% real recall instead of 95%. `fpr_from_distances`
now returns **NaN** when a batch has no positive pair at all (FPR@recall is undefined
there) and the validation loop skips non-finite batches rather than averaging them in.

Neither fix works alone: excluding the diagonal without shuffling leaves the tail
batches with *zero* positives (still degenerate); shuffling without excluding the
diagonal leaves the low bias in every batch.

## Scale bands and equal-sized splits (validation)

The validation config reports FPR95 per blob-scale band. Splits share one pinned set of
boards; each keeps a third of the **raw intrinsic** blob-scale distribution via
`scale_quantile_range` (`small` `[0,0.33]`, `medium` `[0.33,0.66]`, `large`
`[0.66,1.0]`; `overall` keeps all). Filtering on *raw* scales (not the placement-scaled
ones) keeps the bands clean even with a randomized placement scale.

**The bands also share their warps** (`homography_seed: 40` in `shared_params`). Pinning
`seeds` alone was not enough: each band builds its own dataset and so drew its *own*
warps from the global RNG, meaning the bands differed by a random rotation on top of the
scale filter they exist to isolate. With the seed fixed, every band sees the same warp of
each board, so a gap between `small` and `large` is attributable to blob scale. Verified
by construction: without the seed the bands' `transforms` differ; with it they are equal.

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

- **What's stored**: transforms, the per-view lens parameters (`distortions`), all
  keypoint arrays (incl. `*_clean` and `keypoint_is_garbage`), images/composites, and
  `precomputed_patches` — everything needed to reconstruct the dataset without
  re-running the pipeline. A cache written before lens distortion existed has no
  `distortions` entry; `_load_cache` fills in zeros (it can only have been pinhole).
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
  bump `CACHE_VERSION` (in `homography.py`) by hand. Currently **5** (3 → 4: the scale
  normalization fixes, which change the patch footprint of every warped view; 4 → 5:
  the optional per-patch validity mask joined the cached state).
- **`_CACHE_KEY_OPTIONAL` is the escape hatch for a *newly added* optional param.**
  Because defaults are hashed, adding one would otherwise change every existing key and
  force a full rebuild of caches whose contents it provably cannot have changed — the
  feature did not exist when they were written and is off now. A param listed there is
  dropped from the key while it holds its stated "off" value (`distortion_params`:
  `None` or `{}`). It only qualifies if unset is bit-for-bit the old behaviour; anything
  that changes the data even when nominally disabled must not be listed.
- **A new named param must be added to the key by hand.** `effective_params` sees the
  constructor defaults, but `BlobBoardHomographyData` builds its key dict from an
  explicit list plus `**kwargs` — so a param declared in its *signature* is not in
  `kwargs`, and would silently key at its default and reuse a stale cache. Add it next
  to `resolution`. (Params that ride through `**kwargs`, like the jitter settings, are
  keyed automatically.)
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

**Caveat — a hit reuses the exact same random draw.** Unless `homography_seed` is set
the warps are drawn from numpy's global RNG, and board seeds (when `seeds` is unset) and
`polarity: random` are random per run; the cache freezes whichever draw was written
first. That is *desirable* for pinned validation sets and for controlled model
comparisons, but it means repeats sharing a key see identical data — and, more
awkwardly, that the reproducibility is a property of the *cache file* rather than the
config. Setting `homography_seed` moves it into the config, where a cache miss
reproduces the same warps instead of silently drawing new ones. It is part of the key,
so two seeds get separate entries. Clear the entry (or vary a keyed param) to force a
fresh draw. Since patches are pre-extracted once per run and never recomputed
(there is no re-sampling of homographies mid-run), caching is semantically identical *within* a run.

## Reproducing example composites

See scripts under the scratchpad during development; the essentials: construct
`BlobBoardHomographyData(num_boards=…, seeds=[…], image_size=[150,150],
background=<dir or kaggle path>, board_scale_range=[0.4,0.7], garbage_fraction=0.3,
background_seed=…)`, then save `self.images[i]` (composites) / `self.images_clean[i]`
(clean references) with `keypoint_coords` overlaid (blobs vs `keypoint_is_garbage`).
