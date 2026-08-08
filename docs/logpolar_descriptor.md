# Log-polar descriptor

`blob_descriptor_logpolar.yaml` describes blobs from **log-polar** patches
(`patch_type: logpolar`, extracted by `extract_logpolar_patches` in
`src/aef/data/homography.py`) with a HardNet-style CNN (`src/aef/models/hardnet.py`).

## The geometry it relies on

In a log-polar patch (angular axis = dim −2, radial axis = dim −1):

- an image **rotation** becomes a **cyclic shift along the angular axis** — the axis is
  periodic, row 0 and row H−1 are neighbours;
- a **scale change** becomes a shift along the **radial** axis, which is *not* periodic
  (inner radius ≠ outer radius).

So a plain CNN can be made rotation-invariant by pooling over the whole angular axis,
while staying scale-sensitive by reading the radial axis densely. That is exactly what
`HardNet`'s head does:

```python
nn.MaxPool2d(kernel_size=(pool, 1)),         # max over the ANGULAR axis -> rotation invariance
nn.Conv2d(depths[2], 128, (1, pool)),        # dense over the RADIAL axis -> keeps scale structure
```

## …but `HardNet` is not actually rotation-invariant

The angular max-pool only delivers invariance if the feature map really is a *cyclic
shift* of itself. Two things break that:

1. **Zero padding on a periodic axis** — every conv uses `padding=2` with zeros,
   injecting fake content at the 0/2π seam. (`nn.Conv2d(padding_mode="circular")` is no
   help: it would wrap *both* axes, and the radial axis must not wrap.)
2. **Stride-2 aliasing** — two stride-2 convs downsample 64→16, so only shifts that are
   multiples of 4 map to integer shifts of the final map; finer rotations land between
   samples.

Measured (mean relative L2 drift of the descriptor under an angular roll of an untrained
net — see the ablation below for the method), `HardNet` drifts **0.16–0.25** for
rotations of 22.5°–180°. For comparison a *scale* change (radial roll of 4) moves it
0.31, and the steerable `BlobDescriptorNoStride` under an exact rot90 drifts **0.0001**.
So rotation perturbs the descriptor about as much as a scale change does: rotation is
barely being factored out at all, which is the likely reason the log-polar track
underperforms.

## `HardNetLogPolar`

Same conv count and parameter count (1.06 M), two geometry fixes:

- **`LogPolarPad`** — wraps the angular axis (`F.pad(..., mode="circular")` on dim −2)
  and zero-pads the radial axis.
- **`LogPolarBlurPool`** — antialiased ×2 downsample (binomial 1-2-1 blur then
  subsample) replacing the stride-2 convs; the blur itself wraps angularly, otherwise it
  would reintroduce the seam.

Both are toggleable (`circular_pad`, `antialias`); turning both off reproduces the plain
`HardNet` structure, which is what makes the ablation exact.

## Ablation

All variants built from the same seed, so they share **identical conv weights** — only
padding/stride differ. Row (a) reproduces the real `HardNet` exactly (0.2191 vs 0.2191),
which validates the harness. Values are mean relative L2 drift under an angular roll
(= rotation); lower is more invariant.

| variant | 5.6° | 11.2° | 16.9° | 22.5° | 45° | 90° | 180° | **mean** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| (a) neither *(= HardNet)* | .218 | .252 | .245 | .157 | .199 | .233 | .228 | **0.219** |
| (b) circular-pad only | .218 | .240 | .215 | .0003 | .0003 | .0003 | .0003 | **0.096** |
| (c) antialias only | .073 | .110 | .123 | .131 | .186 | .212 | .212 | **0.150** |
| (d) both *(`HardNetLogPolar`)* | .066 | .084 | .064 | .0002 | .0002 | .0002 | .0002 | **0.031** |

Radial roll (scale, shift 4) — must stay sensitive: (a) 0.311 · (b) 0.310 · (c) 0.247 ·
(d) 0.249.

**The two fixes are complementary — they repair disjoint failure modes**, so neither
alone is close to sufficient:

- **Circular padding** exactly fixes the rotations the grid can represent: restricted to
  multiples of 4 (22.5°/45°/90°/180°) the drift goes **0.2045 → 0.0003**, a ~700×
  improvement, for free (no params, no compute). It does nothing below 22.5°.
- **Antialiasing** is the opposite: it is the only thing that helps sub-22.5° rotations
  (0.218 → 0.073 at 5.6°, ~3×) but leaves the coarse ones at 0.185–0.212 because the
  seam is still there. On its own it only gets the multiples-of-4 case to 0.1852.
- Together: mean **0.219 → 0.031**. Coarse rotations exact, fine ones ~3.3× better.

Caveats:

- **Circular padding is the load-bearing fix.** If only one is taken, take that.
- **The residual ~0.07 at fine rotations is inherent, not a bug.** With 64 angular bins a
  5.6° rotation is a *one-bin* shift; no amount of blurring makes a sub-bin shift exact.
  Only more angular resolution would.
- Antialiasing slightly reduces radial (scale) sensitivity (0.311 → 0.249) — expected,
  since blurring costs some high-frequency radial detail. Still far from the 0.0002
  rotation drift, so the rotation/scale separation holds.
- **Method:** these are *untrained* nets, which is the right test for an *architectural*
  invariance (the steerable reference reads 0.0001 untrained). Training cannot make the
  old architecture invariant; it can only spend capacity compensating for the seam.

## The angular pooling: independent vs "locked"

`MaxPool2d((pool, 1))` pools **each column independently** — every one of the 128
channels × `pool` radial columns picks its own angle. A natural idea is to *lock* the
pooling: choose one angular offset and read every column at it.

Worth being precise about what that would buy, because it is **not** invariance:
independent max-pooling is *already* exactly rotation-invariant (max over a cyclically
shifted vector is unchanged — measured 0.0002). What locking buys is
**discriminativeness**: independent pooling discards *which* angle each column peaked
at, destroying the angular relationship between radii, so structurally different blobs
can collapse to the same descriptor. The cost is rigidity — locking assumes one global
rotation explains the whole patch, which fails under deformations that are not a pure
rotation (perspective, anisotropy).

**It was premature while the patches were broken.** Positives were reaching the head
*mirrored*, and then — after that was fixed — still shifted along the radial axis by a
double scale correction (see
[data pipeline → shape normalization](data_pipeline.md#shape-normalization--and-three-bugs)).
Log-polar was the worst-hit path in **all three** bugs, which is not a coincidence: its
two axes are exactly where those errors land. A mirror is an angular *flip* no angular
pool can undo; a scale error is a rigid radial shift of ~18 of 64 px. With all three
fixed and `perspective: false` (warps are pure similarities), the residual really is a
clean rotation — the regime where locking, or dropping the angular pool entirely, should
pay off. Re-measure before investing.

## The DFT-magnitude head and mask-aware pooling

Two **opt-in** heads on `HardNetLogPolar` act on exactly the two weaknesses above — the
max-pool's lost angular structure, and the off-board junk. Both default off, so the
plain `HardNetLogPolar` (angular max-pool, no mask) — including its `state_dict` keys —
is byte-for-byte unchanged unless a config asks otherwise.

### `head="fft"` — keep the whole angular spectrum, not one peak

A rotation is a *cyclic shift* of the angular axis (dim −2). The magnitude of the
angular DFT is invariant to that shift — it lives entirely in the phase, and `|X_k|`
drops it — exactly for integer-bin shifts, the same regime the max-pool is exact in.
`AngularRFFTMag` replaces `MaxPool2d((pool, 1))` with `torch.fft.rfft(...).abs()`,
keeping the lowest `n_harmonics` frequency magnitudes:

```python
nn.MaxPool2d(kernel_size=(pool, 1))    # keeps ONE peak per (channel, radius)
AngularRFFTMag(n_harmonics=F)          # keeps |X_0..F-1| == the angular autocorrelation
```

Where the max-pool collapses each `(channel, radius)` angular profile to a single scalar
(its peak), the magnitude spectrum is — by Wiener–Khinchin — the profile's full circular
**autocorrelation**: the entire angular *shape* minus its orientation. A one-bump ring
and a two-bump ring have the same peak but different spectra, so this directly repairs
the "structurally different blobs collapse to the same descriptor" failure named above.
It is not a strict superset (the peak is a nonlinear order-statistic the power spectrum
does not determine), and it still discards the *relative* phase between radii — that gap
is the bispectrum's to fill. Measured on an untrained net, the fft head's angular-roll
drift is ≈1e-7, on par with the max-pool. The theory (Fourier–Mellin, the shift theorem,
Wiener–Khinchin, the bispectral upgrade) and references are collected in
[fft_theory.md](fft_theory.md).

### `head="relphase"` and `head="bispectrum"` — keep the phase relations too

`|X_k|` is invariant *because* it throws the phase away — and with it where each ripple
sits relative to the others, which is genuine shape: a 1-cycle and a 2-cycle bump have
identical magnitude spectra whether they are aligned or offset by 90°, so two
structurally different angular profiles still collide. Both heads append an invariant
*phase* feature to the same magnitudes, so they are exactly as rotation-invariant as the
max-pool and the fft head (a function of invariants is invariant):

| head | invariant added | rows for `F` harmonics (`F = 5`) | trade-off |
|------|-----------------|----------------------------------|-----------|
| `fft` | — | `F` (5) | phase-blind |
| `relphase` | `c_k = X_k (conj(X_1)/\|X_1\|)^k` | `F + 2(F−2)` (11) | cheap; the `X_1` reference is noise where `\|X_1\| ≈ 0` |
| `bispectrum` | `B(k1,k2) = X_{k1} X_{k2} conj(X_{k1+k2})` | `F + 2·#pairs` (13) | no reference, complete (Kakarala 2012); a triple product, so noisier |

`relphase` cancels the unknown rotation by referencing every harmonic to the first
(`Δ_k = φ_k − k·φ_1`); `bispectrum` cancels it with triple products whose indices sum to
zero, needing no reference at all. `bispectrum_normalize=True` (default) divides each
triple by `|X_{k1}||X_{k2}||X_{k1+k2}|` so only the phase coupling survives — the
magnitudes are already in the leading rows, and the raw product is cubic in scale.
Derivation and references: [fft_theory.md → keeping phase](fft_theory.md#keeping-phase-while-staying-rotation-invariant);
figures: `docs/figures/logpolar_fft_head.tex` (the head in detail) and
`docs/figures/relative_phase_information_loss.tex` (what magnitude-only discards).

Both are drop-in replacements for the angular reduction, so they compose with
`learned_mask` and inherit the fft head's junk-fragility (the DFT is global over angle —
watch the junk-sensitivity probe, not only FPR95). The extra rows widen the head's final
conv, so **a checkpoint does not transfer that layer across heads**: warm-start each head
from its own synthetic run (the loader drops the mismatched tensor and says so). Run the
comparison with

```sh
./launch_training_matrix.sh -n lpphase logpolar_fft logpolar_relphase logpolar_bispectrum
```

(both launchers carry `logpolar_relphase` / `logpolar_bispectrum`, and both are in the
default sweep). What is verified so far is invariance and discriminative *capacity*
(`tests/unit/test_angular_phase_heads.py`: exact roll-invariance, and separation of two
profiles the magnitude head cannot tell apart) — **not** FPR95, which is the open
measurement.

### `learned_mask=True` — mask-aware pooling without a target mask

Part of every patch is off-board junk. The two views are treated **asymmetrically**:

- **Reference patch** (`is_pdf`: identity view, clean board): its board mask is *given*
  to the network and used directly — the learned predictor is not used here. (There the
  validity is just the out-of-bounds `oob` region, since the clean board fills the frame.)
- **Target** (warped view, composited on a real background): the predictor emits a
  per-cell estimate `m_pred ∈ [0, 1]` from the trunk features, which is *both* used to
  weight the target's descriptor *and* trained by a standalone loss.

Concretely the head:

- downweights the pre-head feature map by the **given GT mask on `is_pdf` patches** and
  by the **predicted `m_pred` on targets** (masked `input_norm` likewise uses the given
  mask there);
- returns `(descriptor, m_pred)` so `process_batch_blobs` can add a BCE that supervises
  `m_pred` **on the targets** against their *true* board coverage (weight
  `training.mask_loss_weight`).

The GT the target is supervised against is the real warped board coverage, co-sampled
from `_board_masks` on the same log-polar lattice when `precompute_masks=true` — **not**
`oob`, because a warped view's off-board region is real background *inside* the frame.
This is the standard train-supervise / test-predict pattern: the true target mask exists
in synthetic training but not at test, so the predictor learns to supply it. No relative
orientation is ever needed, because the mask co-rotates with the patch.

The **steerable** descriptors (`BlobDescriptorEfficient`, `BlobDescriptorNoStride`)
implement the same contract — same `learned_mask` flag, same `(descriptor, m_pred)`
return, same BCE — on cartesian patches, which `precompute_masks` now covers too. See
[steerable_masking.md](steerable_masking.md) (why weighting by a scalar field keeps the
escnn models equivariant, and where each architecture applies it).
Only a mask-aware model is ever *called* with the mask kwargs (`process_batch_blobs`
checks `learned_mask`); `training.ignore_mask=true` disables the whole path — inputs and
supervision — for the ablation.

Two honest caveats, both to **measure**, not assume:

- **Reference-white vs target-real-background.** The reference patch's off-board is
  bland/white; a
  real target's is textured scene. The predictor only generalizes to test-time targets
  if it has *seen* varied junk — so this pairs with strong off-board augmentation
  (`garbage_fraction` / background variety), otherwise it overfits the training scenes.
- **The fft head is junk-fragile.** The DFT is global over angle, so off-board content
  smears into *every* coefficient — unlike the max-pool, which lets each channel dodge a
  low-junk angle. That is *why* the mask path pairs with it. When you evaluate, track a
  junk-sensitivity probe (descriptor change when the off-board region is scrambled), not
  only FPR95 and the angular-roll drift.

### Toggling and ablating

The knobs ride through the usual config dicts (`model.params.head`,
`model.params.n_harmonics`, `model.params.learned_mask`,
`training.dataset.params.precompute_masks`, `training.mask_loss_weight`). A ready config
bundles them:

```sh
pixi run --manifest-path deps/BlobBoards.jl/pixi.toml \
  python src/run_training.py --config-name blob_descriptor_logpolar_fftmask
```

The launcher exposes each as a first-class variant — `logpolar_fft` (fft head only),
`logpolar_relphase` / `logpolar_bispectrum` (the phase-keeping heads above) and
`logpolar_fftmask` (fft head + learned mask). All four sit in the same `logpolar_mask`
dataset group, so one prebuilt mask-carrying dataset serves them and the mask-less ones
simply ignore the extra channel. To run the head/mask ablation against the max-pool
baseline in one warm-cache matrix:

```sh
./launch_training_matrix.sh -n lpfft logpolar logpolar_fft logpolar_fftmask
```

## `outer_factor` interacts with the blob scale

The patch reaches `logpolar_outer_factor × σ`, in units of the blob's own scale. With
`max_diameter_fraction: 0.4` a large blob at 16σ already extends several board-widths
out (mostly fill), while a small blob at 16σ sees a sensible neighbourhood — so **no
single `outer_factor` suits every blob**, and the scale bands span exactly that range.
The sweep drives `scale` → `logpolar_outer_factor` ∈ {32, 64, 96, 128}, all well above
the original default of 16; with transforms held fixed the matched-vs-random separation
degrades gently as it grows (best ≈ 8–16). Treat large values with suspicion.

**That sweep predates the scale-normalization fixes and is void** — it was measured when
every warped patch was radially shifted by `log(1/sqrt(det J))·30.8` px, which is
precisely a change of effective `outer_factor`, so the sweep was partly measuring the
bug. Re-run it. What *is* measured post-fix (see
[scale budget and jitter](scale_budget_and_jitter.md#small-blobs-are-not-the-problem)):
at `outer=16` the large blobs are the weak end — anchor↔positive NCC falls to 0.827 at
σ_w>10 and 0.654 at σ_w>20, while sub-pixel blobs sit at 0.98. That is the "extends
several board-widths out" failure above, now with a number on it, and it argues for
*smaller* `outer_factor`, or one that adapts to the blob's scale.

## Comparing them

The baseline config keeps `name: HardNet`. The launcher adds a `logpolar_circ` entry that
reuses `blob_descriptor_logpolar` and overrides only the model
(`NET_EXTRA[logpolar_circ]="model.name=HardNetLogPolar"`), so both land in the same
dataset group and share prebuilt datasets (verified: the prebuild stays at 60 tasks, not
120):

```sh
./launch_training_matrix.sh -n lpcmp logpolar logpolar_circ
```

The individual fixes can be carried through to FPR95 the same way, e.g.
`model.name=HardNetLogPolar model.params.antialias=false` for variant (b).

## Architecture directions to try

Ranked by value-for-effort, tied to the failure modes above. "(arch)" is a model change;
"(data)" is a pipeline change that tends to matter more than most model tweaks.

1. ~~**Put some phase back in the head (arch).**~~ **Implemented** as `head="relphase"`
   and `head="bispectrum"` — see [the phase-keeping heads](#headrelphase-and-headbispectrum--keep-the-phase-relations-too)
   below. Still the biggest discriminative lever to *measure* after `head="fft"`; only the
   FPR95 comparison is outstanding.
2. **Union the peak with the spectrum (arch, ~free).** Max-pool keeps a nonlinear
   order-statistic (the peak) the power spectrum does not determine; the spectrum keeps
   the autocorrelation the peak does not. Concatenate both angular reductions
   (`[AngularRFFTMag(x) ; max_angular(x)]`) before the final layer — strictly richer than
   either, no invariance risk.
3. **Modernize the trunk (arch).** The trunk is a plain 2019-HardNet conv stack. Residual
   connections and a HyNet-style normalization (filter-response norm + a learnable
   component) in place of `BatchNorm(affine=False)` gave measurable FPR95 gains over
   HardNet in the descriptor literature. Diffuse "probably helps"; ablate behind a flag.
4. **Orientation head from `∠X_1` (arch, near-free).** The phase of the angular
   fundamental is a continuous, differentiable orientation estimate the `fft` head
   already computes. Expose it to (a) supervise against the known relative rotation
   (an equivariance-teaching loss) or (b) canonicalize (the "locked pooling" idea from
   *independent vs locked* above).
5. **Attend/gate the pooling with the mask (arch).** Make the angular reduction
   mask-weighted (weight the DFT input / soft-attention pool gated by `m_pred`) instead of
   a plain multiplicative downweight — the structural counter to the fft head's
   junk-fragility. Only if the junk-sensitivity probe shows the input-weight is not
   enough.
6. **Higher angular resolution (data).** The ~0.07 fine-rotation residual is *inherent* to
   64 angular bins (5.6° = one bin, see the ablation); more bins is the only thing that
   shrinks it, at compute cost.
7. **Adaptive / multi-band `outer_factor` (data).** "No single `outer_factor` suits every
   blob" (above) is architectural in spirit: read several radial annuli (a radial pyramid)
   or scale `outer_factor` to the blob, rather than one fixed band. Likely beats any head
   tweak for the large-blob regime the scale-budget doc quantifies.

First two to run: **#1 (relative-phase/bispectrum — now a launcher entry, so this is a
measurement, not an implementation)** for discriminativeness and **#7 (adaptive
`outer_factor`)** for the large-blob regime — both target *measured* failures.
