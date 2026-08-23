# Blob-board data pipeline

Two datasets feed the same training loop and the same `process_batch_blobs`:

- **`BlobBoardHomographyData`** (`src/aef/data/blobboards.py`, subclassing
  `HomographyData` in `homography.py`) — synthetic blob boards rendered by
  `BlobBoards.jl`, warped by random homographies. Used by the two
  `bootstrap_synthetic*` configs. Needs the pixi env (Julia bridge).
- **`BlobTrackData`** (`src/aef/data/track.py`) — already-canonicalized patches read out
  of a `.tracks` HDF5 file produced from real footage. Used by the two `bootstrap_real*`
  configs. Needs only h5py.

Both `get_collate_func()`s emit the same batch keys, which is what lets one
`process_batch` serve both.

## Views, keypoints, patches (synthetic)

Each board contributes `transforms_per_image` **warped views** plus **one identity
(un-warped) view** (homography `= I`). Blob keypoints are the boards' ground-truth
blob centres/scales. For every keypoint × view a shape-normalized patch is
pre-extracted once (`in_memory: true`) and cached, so per-item cost at train time is a
tensor index + collate. The FPR95/contrastive label is the keypoint's feature id
(`keypoints[..., 1]`), shared across a keypoint's views → those form the positive
pairs. The identity view is flagged `is_pdf`, which is what the proxy-anchored loss and
metric key on.

## Shape normalization

`blob_normalizations` (`homography.py`) SVD-whitens the local Jacobian of the view's
homography and **drops the rotational factor `U`**, so an elliptical blob maps to an
isotropic one and the residual difference between a reference patch and its positive is
a pure rotation (which the descriptor is supposed to handle).

Three invariants make it work, all of which have been violated at some point and each
of which fails the same silent way — patches that look plausible in isolation but stop
matching their own reference:

1. **The factor is proper (`det > 0`).** An SVD is unique only up to a reflection pair
   (`det(U) = det(Vh) = -1`), and for a near-similarity warp the singular values are
   degenerate, so the factorization is fully ambiguous. A reflection is an angular
   *flip*, which no rotation — hence no angular pooling — can undo, so it defeats the
   log-polar architecture's only invariance mechanism outright. Negating a row of `Vh`
   flips both dets and leaves `U@Σ@Vh` unchanged.
2. **The factor carries shape only (`det == 1`).** `scales` already carries the warp's
   `sqrt(det J)` and the extractor divides by it, so a factor that also removed the size
   would remove it twice. On a log-spaced radial axis that surplus is a *rigid shift*:
   ~18 of 64 px at a typical warp.
3. **Every Jacobian is evaluated at the source point.** `linearize_homography(H, coords)`
   differentiates `H` at a point of its **domain**. What is wanted, `inv(J_H(p))` at the
   source point `p`, is by the chain rule the Jacobian of the *inverse* homography at the
   warped point — which is what the code builds. Invisible for affine warps; up to 2.25×
   wrong in `sqrt(det J)` once `perspective_amplitude` is non-zero.

The **residual rotation this leaves is deliberate — keep it.** Downstream the detector
yields blobs at arbitrary orientation, so the training pairs *should* exercise the
rotation the descriptor has to absorb. A polar decomposition (`P = Vhᵀ Σ Vh`) would be
rotation-free by construction and would train the descriptor on a distribution the
deployed detector never produces. Only the *reflection* was ever wrong.

Invariants 2 and 3 are pinned by `tests/unit/test_blob_normalizations.py`: `N @ J_H(p)`
must be exactly `sqrt(det J)` times a rotation. Note that **any patch measurement taken
before `CACHE_VERSION` 4 is void.**

### Measuring patches — two traps

- **Normalize before comparing patches.** Raw L2 between a reference patch (clean board,
  white surround) and its positive (warped, grey surround) is dominated by the global
  intensity offset, making matched pairs look no better than random ones. The network
  never sees it (`input_norm`), so neither should the metric.
- **Set `homography_seed` when comparing builds.** By default `sample_homography` draws
  from numpy's *global* RNG, so every dataset construction gets different warps and two
  builds differ by a random rotation on top of whatever you changed. `homography_seed`
  threads one `default_rng` through every draw, making the whole sequence a function of
  the seed — with it fixed, two builds are bit-identical.

## Log-polar patch extraction

`extract_logpolar_patches` samples an annulus around each keypoint with inner radius
`logpolar_inner_factor * scale` and outer radius `logpolar_outer_factor * scale`
(measured in the shape-normalized frame). The radial axis (dim `-1`) is log-spaced; the
angular axis (dim `-2`) spans a full turn. One channel out — the radial axis already
encodes scale.

Sampling runs at `supersample` sub-taps per output pixel per axis, area-averaged back
down. Because the sub-taps live in log-polar coordinates they automatically span each
output pixel's true source footprint, which grows with radius, so this approximates area
interpolation and suppresses the aliasing a single bilinear tap produces in the outer
(condensed) radii. `supersample=1` recovers the plain single-tap behaviour.

## `in_memory`

`in_memory: true` (what all four configs use) pre-extracts every patch once and batches
carry `"patches"`; `process_batch_blobs` reads that key and raises `KeyError: 'patches'`
without it. With `false` the collate warps each view on the fly and batches carry
`"images"` instead. Note the flag does not select how *images* are loaded:
`HomographyData` takes an in-memory tensor only.

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

**Backgrounds are resized to cover + centre-crop, not stretched.** Passing a 2-element
size to `torchvision.resize` honours it exactly and so squashes a landscape photo onto a
square canvas. Backgrounds are what the garbage keypoints — the dataset's pure negatives
— are detected on, so distorting their image statistics makes those negatives
unrepresentative. Note the scene is always resized to the canvas: source photos **below**
the frame resolution get upsampled, making the negatives artificially easy. At 300 dpi
(1772 px canvas) a ≥2000 px source downsamples cleanly; this is one more reason not to
raise the camera dpi.

## The identity (reference) view stays clean

The un-warped identity view is the canonical *reference* patch: it gets **no background
compositing and no augmentation**. Only warped views carry the scene + augmentation.
This is implemented with parallel `*_clean` arrays (`keypoint_coords_clean`,
`keypoint_scales_clean`, `images_clean`) used only by the identity branch of
`__getitem__`/the collate, and an augmentation gate in `compute_patches`
(`img_id % views == views-1` → no aug). When `background` is unset the `_clean` arrays
alias the mains, so only the (always-on) augmentation gate differs from plain-white
boards.

Both views read rasters rendered at the same `resolution` — and note `resolution`
selects *which board you get*, not just how finely it is sampled: the packer snaps blob
centres to pixel centres and collides against grid-snapped boxes, so the same seed at
300 vs 600 dpi yields a different board (1189 vs 1202 blobs, unrelated layouts).

## Background "garbage" keypoints

`garbage_fraction > 0` adds background distractor keypoints — **untracked singletons**:
each gets a globally-unique, always-negative label (`-(1 + index)` in `__getitem__`), so
it never forms a positive pair with a blob or another garbage point; it is a pure
negative that hardens FPR95. Points come from SIFT on the composited background
(preferred) topped up with random background points to hit an exact target
(`garbage_source = "sift" | "random"`), then trimmed to
`round(garbage_fraction · n_blobs)`.

## Keypoint order, and what FPR95 actually counts

Two coupled details decide whether the reported FPR95 is meaningful. Both were bugs that
made the metric read **better than reality**.

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
Pairing a patch with itself is a trivial positive at distance 0 that always ranks first.
With `views=2` that made *half* of all "positives" free self-pairs, so the threshold was
effectively set at ~90% real recall instead of 95%. `fpr_from_distances` returns **NaN**
when a batch has no positive pair at all (FPR@recall is undefined there) and the
validation loop skips non-finite batches rather than averaging them in.

Neither fix works alone: excluding the diagonal without shuffling leaves the tail batches
with *zero* positives (still degenerate); shuffling without excluding the diagonal leaves
the low bias in every batch.

## Track data: viewing-angle balance

`.tracks` files record one viewing obliquity per detected frame (`view_angles`, radians,
0 = fronto-parallel → 90° edge-on, derived intrinsics-free from the local affine tilt of
the world→image homography at the board centroid). `_load_all_sequences` joins that
per-frame table to each observation's `frame_id`, so every patch carries the angle it
was seen at (NaN for PDF patches, which have no pose, and for confusers, which carry no
frame id).

Real footage is heavily fronto-parallel-biased, so the default packing would let the
near-frontal band own every batch and leave the oblique tail — the regime an
affine-equivariant descriptor exists for — contributing a handful of pairs per epoch.
With `balance_view_angles: true` (the default in `track_descriptor_base.yaml`)
`get_sampler` switches to `_pack_band_balanced`, which asks **every populated band for
the same number of in-band patches in every batch**. Per-batch matters because SupCon
only ever sees one batch at a time — a per-epoch quota would not help.

Positives are drawn as **disjoint pairs**: an in-band observation plus its GT (PDF)
patch, or two in-band observations once that one is spent. Never a repeated index — a
duplicate inside a batch is a distance-0 fake positive. A band that runs out of disjoint
pairs under-fills rather than duplicating; `_band_capacity` predicts that ceiling and the
sampler prints it per band on startup.

| param | meaning |
|---|---|
| `view_angle_band_edges` | band edges in degrees; `null` → 0–90 in 10° steps |
| `view_angle_band_target` | epoch *length* only (`min`/`median`/`mean`/`max` over the populated bands' sizes, or an int) |
| `view_angle_band_bias` | tilts the per-batch quotas: `0.0` = plain balance, `b > 0` gives the most oblique band `exp(b)×` the most frontal one's quota |

`log_stats: true` prints the per-sequence breakdown and the 10° histogram the bands are
drawn against.

## Dataset cache (disk)

Preparing a synthetic dataset is slow: board rendering (Julia), compositing, SIFT,
garbage detection and patch pre-extraction. Setting `cache_dir` writes the **fully
prepared** dataset to disk and reuses it on the next run.

- **What's stored**: transforms, all keypoint arrays (incl. `*_clean` and
  `keypoint_is_garbage`), images/composites, and `precomputed_patches` — everything
  needed to reconstruct the dataset without re-running the pipeline. `_load_cache`
  restores only the keys the current class declares, so a cache written by an older
  revision (which carries entries for features since removed) still loads.
- **Key**: a short hash (`dataset_cache_key`) of every param that determines the
  contents, plus `CACHE_VERSION`. It hashes the **effective** params —
  `effective_params` overlays what the caller passed onto the constructor *defaults* of
  `HomographyData.__init__` and `BlobBoardHomographyData.__init__` — so a param the
  config never mentions (`patch_size`, `supersample`, the log-polar factors, …) still
  keys the cache at the value it actually ran with, and **changing a default in code
  invalidates every key** instead of silently reusing stale data. `data_dir`,
  `cache_dir`, `extraction_batch_size` and `sift_batch_size` are excluded — they don't
  change the data. The *algorithm* is not hashed: if you change the compositing/garbage
  code itself, bump `CACHE_VERSION` by hand.
- **`_CACHE_KEY_REMOVED` pins params that no longer exist.** Because defaults are hashed,
  deleting a constructor param would change every existing key and force a full rebuild
  of caches whose contents it provably cannot have altered. The dict holds those names at
  the values the shipped configs ran with, so already-prepared datasets stay addressable;
  a caller that still passes one of the names overrides the pin. Dropping an entry — and
  bumping `CACHE_VERSION` — is how you *do* invalidate them.
- **A new named param must be added to the key by hand.** `effective_params` sees the
  constructor defaults, but `BlobBoardHomographyData` builds its key dict from an
  explicit list plus `**kwargs` — so a param declared in its *signature* is not in
  `kwargs`, and would silently key at its default and reuse a stale cache. Add it next to
  `resolution`. (Params that ride through `**kwargs` are keyed automatically.)
- **Checked before board generation**: `BlobBoardHomographyData` derives the key from its
  params and, on a hit, skips rendering entirely (the slowest step) — so the check cannot
  live in `HomographyData` alone.
- **Measured**: cold build 54 s → warm load ~0 s, with bit-identical tensors.
- **Sharing**: the key covers dataset params only, *not the model* — so every model at
  the same `scale` shares one entry. Each validation split gets its own.
- **Concurrency**: written via temp file + atomic rename, so parallel jobs can never read
  a half-written cache. Two cold jobs may both build it (last wins) — wasteful but
  correct. Saving is best-effort: an unwritable `cache_dir` warns and the run continues.

**Caveat — a hit reuses the exact same random draw.** Unless `homography_seed` is set the
warps are drawn from numpy's global RNG, and board seeds (when `seeds` is unset) and
`polarity: random` are random per run; the cache freezes whichever draw was written
first. That is *desirable* for pinned validation sets and for controlled model
comparisons, but it means repeats sharing a key see identical data — and, more
awkwardly, that the reproducibility is a property of the *cache file* rather than the
config. Setting `homography_seed` moves it into the config, where a cache miss reproduces
the same warps instead of silently drawing new ones. It is part of the key, so two seeds
get separate entries.
