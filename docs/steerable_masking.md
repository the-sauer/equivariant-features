# Learned board-validity masking on the steerable (escnn) descriptors

How `learned_mask=True` works on `BlobDescriptorEfficient` and `BlobDescriptorNoStride`,
why it leaves rotation equivariance intact, and what it costs. The log-polar
counterpart (`HardNetLogPolar`) implements the same contract — see
[logpolar_descriptor.md](logpolar_descriptor.md#learned_masktrue--mask-aware-pooling-without-a-target-mask)
— so the two families are configured, supervised and compared identically. Two
extensions, [`cascade`](#cascade-the-gate-in-the-loop) and
[`mask_resolution`](#predictor-resolution-mask_resolution), exist on the log-polar model
**only**; they are documented here because this is the note that carries the mechanism,
and marked as log-polar-only where they appear. Every *measurement* below is log-polar
too — `src/mask_ablation.py` hooks that family alone.

## Three different things called "mask"

`models/blob_descriptor.py` now carries three, and they are unrelated:

| name | what it is | when |
| --- | --- | --- |
| `escnn.nn.MaskModule` | fixed **corner/border** mask — kills the square grid's corners, which a rotation cannot map into themselves | always, per block |
| `inner_mask` | fixed **centre (donut)** mask — kills the anchor blob, identical for every keypoint after normalization, hence non-discriminative | `inner_mask=True` |
| `learned_mask` | **data-dependent board validity** — kills the off-board part of *this* patch, which is real background, not a fixed geometric region | `learned_mask=True` |

The first two are constant buffers. Only the third depends on the input, which is what
makes it interesting and what the rest of this note is about.

## The problem: the target's mask does not exist at test time

Part of every patch is off-board junk: background the warp dragged in, or out-of-frame
fill. It is not shared between a reference patch and its target, so it is pure noise for the
metric loss. Removing it needs a per-pixel validity map — and that map is available for
exactly one of the two views:

- **Reference patch** (`is_pdf`: identity view, clean board): validity is *given*. In
  the synthetic pipeline it is the un-warped board raster; on `.tracks` data it ships
  with the patch. (In-tree the flag is `is_pdf` — "anchor" is SupCon's word for the rows
  of its logit matrix; the `.tracks` files still spell the attribute `is_anchor`.)
- **Target** (warped view, composited on a real background): validity exists in
  training (the warped board mask) but **not at test time**, where you only have the
  image.

So the model is trained with the answer and must learn to produce it: GT on `is_pdf`, a
prediction on targets, and a standalone loss that trains the prediction against the
target's true coverage. Train-supervise / test-predict.

## Setup and notation

Let `G = C_N` act on `R^2` (escnn's `rot2dOnR2(N)`). A feature field `f: R^2 -> R^c`
carrying representation `ρ` transforms as

```
(π(g) f)(x) = ρ(g) · f(g⁻¹ x)
```

A **scalar field** is the special case `ρ = ρ_triv` (`ρ(g) = 1` for all `g`), i.e. it
only moves with the grid:

```
(π₀(g) m)(x) = m(g⁻¹ x)
```

The GT mask is a scalar field by construction — it is co-sampled on the *same* warp as
the patch, so it rotates exactly with it. That single fact is what the rest rests on.

## Why weighting by a mask stays equivariant

Pointwise multiplication of a feature field by a scalar field is equivariant:

```
π(g)(m · f)(x) = ρ(g) · m(g⁻¹x) f(g⁻¹x)
               = (π₀(g) m)(x) · (π(g) f)(x)
```

i.e. `(m, f) ↦ m·f` intertwines `π₀ ⊗ π` with `π`. Rotating the input and then
weighting is the same as weighting and then rotating. This is the same argument that
licenses the fixed radial `inner_mask`; the only new requirement is that the weight be
a genuine scalar *field*, not an arbitrary tensor.

The **predicted** mask satisfies that too. `m_pred` comes from a 1×1 steerable conv
mapping the backbone's regular-representation field to a single trivial representation,
followed by a sigmoid:

```
m_pred = σ(W ⋆ f),   W: ρ_reg^c -> ρ_triv
```

`R2Conv` is equivariant by construction, so `W ⋆ (π(g)f) = π₀(g)(W ⋆ f)`, and `σ` is
pointwise and acts on a trivial representation, so it commutes with `π₀`. Hence
`m_pred(π(g) f) = π₀(g) m_pred(f)`: the predicted validity co-rotates with the patch,
exactly like the GT one. No relative orientation is ever needed, and nothing in the
pipeline has to know which way the patch is turned.

Routing between the two is a per-sample convex combination and does not touch the
spatial index at all:

```
w = a · pool(m_gt) + (1 - a) · m_pred,    a = is_pdf ∈ {0, 1}
```

`pool` is `adaptive_avg_pool2d` from patch resolution to feature resolution; averaging a
0/1 mask gives *partial* coverage in boundary cells, which is the right target scale for
a sigmoid output. (Average pooling is itself equivariant only up to the grid — see the
caveat at the end.)

## Where the weight enters

There are **two** positions, and they are not equally valuable:

| position | what it multiplies | where | families |
| --- | --- | --- | --- |
| **late weight** | the feature field, at the reduction | `_validity_weight` | all three descriptors |
| **input gate** | the patch, before the trunk | `input_norm(x, mask=…)` / `_fill_invalid_with_mean` | all three on the reference view; targets only under [`cascade`](#cascade-the-gate-in-the-loop) (log-polar) |

This section is the late one; [the input gate](#the-input-gate) and
[`cascade`](#cascade-the-gate-in-the-loop) follow. Measurement first, since it decides
how to read the rest: **a perfect gate is worth 2.7x and a perfect late weight 1.7x**, and
the shipped predictor cashes in about two thirds of the latter and *none* of the former.

The late weight goes at the point where the spatial map collapses, because that is where
an off-board cell would otherwise contribute to the descriptor.

**`BlobDescriptorEfficient`, `head="attention"`** — fold it into the pooling weights:

```
d = Σ_x a(x) w(x) f_inv(x) / Σ_x a(x) w(x)
```

with `a = σ(att(f))` the learned saliency and `f_inv` the group-pooled (invariant)
field. Two things happen at once: an off-board cell can no longer win the attention,
and — because the same weight divides the normalizer — the descriptor becomes the
average over the **valid** region rather than a downscaled average over everything. The
latter matters: without it, a patch that is half off-board would produce a descriptor of
roughly half the magnitude, and the metric loss would read that as a difference in
content.

**`BlobDescriptorEfficient`, `head="dense"`** — weight the field before the readout
conv, `f ← w · f`, then the usual dense `R2Conv` to 1×1 and group pooling.

**`BlobDescriptorNoStride`** — same, on the 24×24 field just before the dense readout.
The weighting is spliced *into* `self.net`'s forward (the layer list is iterated and the
product inserted before the last two layers) rather than splitting the module into
`trunk`/`readout` submodules. That is deliberate: the layer names, and hence the
`state_dict` keys, must stay identical to the unmasked model, or an
`init_from_checkpoint` warm-start from a synthetic run would silently transfer nothing.
Verified — with the weight forced to 1, the spliced forward reproduces `self.net(x)` to
0.0 absolute difference, and `mask_head.*` is the only added key.

## The input gate

On `is_pdf` patches the known off-board region is also neutralized at the input:

```
x ← x · m + mean_m(x) · (1 - m),    mean_m(x) = Σ m x / Σ m
```

The extractors fill off-board/out-of-frame area with **white**, which is strong, fake
structure. Zeroing it instead would only trade it for a fake *black* edge; substituting
the mean of the valid pixels removes the contrast entirely. This is the un-normalized
analogue of `hardnet.input_norm(x, mask)`, which sets invalid pixels to the normalized
mean (i.e. 0). `mean_m(x)` is a per-sample scalar, so the gate is still equivariant.

By default targets get no input gate — their mask is exactly what is unavailable at test
time. [`cascade`](#cascade-the-gate-in-the-loop) lifts that restriction by gating them
with `m_pred` instead, and the measurements below say that is the position that matters.

## `cascade`: the gate in the loop

`HardNetLogPolar` only (`cascade=True`, needs `learned_mask`; the steerable descriptors
have no equivalent yet). It moves the masking from the pooling to the **input**: the
trunk runs twice, once to predict the validity and once on the patch gated by that
prediction.

```
m_pred      = mask_head(trunk(input_norm(x)))        # pass 1, prediction only
gate        = a · m_gt + (1 - a) · upsample(m_pred)  # a = is_pdf; GT where it exists
d           = head(trunk(input_norm(x, mask=gate)))  # pass 2, on the gated patch
```

The motivation is the **receptive field**. At 64×64 the trunk's nominal RF is 51 px, so a
weight applied to the 16×16 feature grid can only drop contaminated cells, never clean
them — by the time the field exists, every cell has already mixed in off-board content.
A gate at the input removes that content before it spreads.

Three consequences, all of them load-bearing:

- **No parameters, and no `state_dict` change.** The same `trunk` and `mask_head` serve
  both passes, so a cascade model warm-starts exactly from a non-cascade checkpoint and
  vice versa. The cost is the second trunk pass.
- **The predictor is trained for the job it does.** Gradients reach `mask_head` through
  *both* the gate and the BCE, which is what "in the loop" buys — bolting a predicted
  gate onto a late-weighted checkpoint after the fact does not work (see below).
- **`cascade_late_weight`** additionally keeps the late weighting on top of the gate. It
  is **off** by default: with a correct gate it costs the entire gain (0.0030 → 0.0088),
  plausibly because multiplying the field by a spatially varying weight convolves the
  angular spectrum the `fft`-family heads read out.

Measured on `iteration_4` track data with a frozen trunk and the GT mask as the gate,
FPR95 0.0108 → **0.0030**, against 0.0087 for a *perfect* mask used as a late weight —
i.e. the input gate has ~3.6x the headroom. The trained-checkpoint re-scoring in
[mask_ceiling_experiment.md](mask_ceiling_experiment.md#4-the-gate-is-the-position-that-matters-and-only-for-a-trunk-that-saw-one)
reproduces the ordering: `gate_gt` beats `late_gt` by 1.8x on the shipped masked model
and by 7.9x on the oracle-gate arm.

The catch is error tolerance, and it runs the other way. Gating with **another patch's**
mask scores 0.094 — worse than not masking at all — and on a trained `fftmask` checkpoint
`gate_shuffled` costs 9.2x against the late weight's 1.28x. The more a trunk exploits the
gate, the less mask error it tolerates, which is why the shipped predictor collects none
of the gate's prize even though it collects two thirds of the late weight's.

Ready-made: `track_descriptor_logpolar_cascade.yaml`, launcher entries `logpolar_cascade`
and `logpolar_cascade_late`.

## Predictor resolution (`mask_resolution`)

`HardNetLogPolar` only. By default `mask_head` reads the **trunk output** — at
`patch_size=64` a 16×16 grid whose cells each see ~51 px of the patch, so the predicted
boundary is 2–4 cells wide where the true one is 1.

How badly that hurts is **not settled**, and the two measurements on record disagree:
the `HardNetLogPolar` docstring reports IoU@0.5 0.81 at a mean coverage of 0.81 against
a true 0.67 (i.e. systematic under-masking), while the trained checkpoints re-scored
below and in [mask_ceiling_experiment.md](mask_ceiling_experiment.md) reach IoU@0.5
0.885–0.888 at a mean of 0.784–0.792 against a true 0.777–0.778 (i.e. well calibrated in
the mean). Different runs and different data; nobody has reconciled them. What both agree
on is that the boundary is the error, and that is what a finer tap targets.

`mask_resolution` taps the mask head *earlier*, where the grid is still fine:

| value | tap | grid at `patch_size=64` |
| --- | --- | --- |
| `None` (default) | trunk output | 16×16 (`patch_size // 4`) |
| `patch_size // 2` | mid-trunk | 32×32 |
| `patch_size` | after the first two blocks | 64×64 |

The trade is receptive field for grid sharpness — a shallower tap has less context to
tell board from background. Two mechanical notes: the late weighting averages a finer
`m_pred` down to the field it multiplies (so the weight always means "coverage of this
cell"), while the BCE supervises it at its **own** resolution; and because the tap
changes `mask_head`'s input channels, that one tensor does not transfer on a warm start
(the loader reports it) — the trunk and the head still do.

This is the knob behind the "higher-resolution or better-tapped mask head" that
[mask_ceiling_experiment.md](mask_ceiling_experiment.md#verdict) names as a next step; it
is implemented but has not been swept.

## Supervision

`forward` returns `(descriptor, m_pred)`; `process_batch_blobs` adds

```
L_mask = BCE( m_pred[t] , pool(m_gt)[t] ),   t = targets ∧ in-bounds
```

weighted by `training.mask_loss_weight`. Three restrictions, each with a reason:

- **targets only** — the reference patch's mask is *given* to the model, so training
  the predictor on it would fit the predictor to the one case where it is never used;
- **in-bounds only** — out-of-frame patches are meaningless white fill;
- **not during validation** — validation withholds the GT mask entirely (test time has
  none), so the metric stays reproducible; `m_pred` alone drives the weighting there.

`training.ignore_mask=true` disables the whole path — input gate, weighting *and*
supervision — for the ablation. A model that never receives the mask must not be scored
against it either.

Only a model that advertises `learned_mask` is ever *called* with the mask kwargs;
every other descriptor has a plain `forward(patches)` and would raise on them.

## Cost

Measured on untrained C4 nets (batch 4, random 64×64 input), mean relative L2 drift of
the descriptor under a 90° rotation of patch **and** mask:

| head | unmasked | `learned_mask=True` |
| --- | --- | --- |
| attention | 0.0112 | 0.0081 |
| dense | 0.0889 | 0.0821 |

The mask path does not degrade equivariance — the residual is the C4 discretization and
the padded convs (see [steerable_descriptors.md](steerable_descriptors.md#equivariance-trade-off-important)),
not the weighting. Parameter cost is one 1×1 steerable conv (`mask_head`); compute cost
is one extra conv over the backbone grid plus the pooling.

`m_pred` resolution is the backbone grid: 8×8 for `Efficient` at `patch_size=64`
(three stride-2 pools), 24×24 for `NoStride`.

**Caveat, honestly:** exact equivariance of the *weighted* pooling holds on the
continuous plane and for the 90° rotations that permute grid cells exactly. For other
angles, `adaptive_avg_pool2d` of the GT mask and the sum over a square grid are
approximations, just like everywhere else in these architectures. Whether the mask buys
FPR95 is an empirical question — run the ablation.

## What the mask is worth, and what the predictor collects

`src/mask_ablation.py` answers this without training anything: one fixed pass over the
validation split (`BlobTrackData`'s seeded eval sampler, so the numbers sit on the same
scale as the loop's `FPR95@all`), with the two mask entry points — the input **gate** and
the late **weight** on the feature field — driven from a chosen source. `none` is a
literal 1 at both, which the model's own `forward` cannot express: called with
`mask=None` it falls back to the *prediction*, not to no masking. Because it bypasses
`forward` and applies both entry points itself, it scores an oracle model and a plain one
identically under identical conditions.

Ten conditions — `none`, `native`, `native_gt`, `late_{gt,pred,shuffled}`,
`gate_{gt,pred,shuffled}`, `both_gt` — plus four probes:

| flag | what it changes | what it answers |
| --- | --- | --- |
| `--occlusion F` | overwrites an angular wedge (fraction `F`, all radii) of every target patch with **another patch's** content | is the mask worth more when the off-board region is not just a bland rim? |
| `--blind-mask` | the "true" mask covers the off-board rim only, occluder included in the valid region | how much of the occlusion gain survives a *board-coverage* supervision target? |
| `--content-blind` | replaces every patch by **one fixed random texture**, so the mask is the only per-sample variable | does the mask by itself leak blob identity? |
| `--tails` | prints the positive/negative distance quantiles that set FPR95, split by pair type | why do `FPR95@all` and the per-band curves disagree? |

`--content-blind` uses a fixed random *field* rather than a constant on purpose:
`input_norm` divides a flat patch by ~0 std, and a gate that fills the invalid region
with the valid region's mean leaves a constant patch constant, so nothing — not even the
mask boundary — would reach the trunk. Read its numbers against `gate_shuffled` in the
same run, which is that mode's chance level.

**`HardNetLogPolar` only.** The harness hooks the log-polar family; the steerable models
apply their weight inside `self.net` and would need their own hook, so every number in
this section is log-polar even where the mechanism is shared.

Measured on `iteration_4` track data (5-board val split, batch 1024, 20 batches),
`2026_08_10_RealFourthIteration/track_logpolar_fftmask_s96` (trained WITH the learned
mask) against `track_logpolar_fft_s96` (identical net and head, no mask at all):

| condition | fftmask | fft (no mask) |
| --- | --- | --- |
| `none` — no mask anywhere | 0.0513 | **0.0332** |
| `native` — GT on the PDF view, `m_pred` on the target (what it ships) | 0.0383 | — |
| `late_gt` — perfect mask as the late weight | 0.0268 | 0.0200 |
| `gate_gt` — perfect mask as the input gate | **0.0079** | 0.0280 |
| `both_gt` | 0.0262 | 0.0196 |
| `late_shuffled` — another patch's mask (control) | 0.0655 | 0.0486 |
| `gate_shuffled` | 0.2339 | 0.0704 |

Three things fall out, and together they explain why the masked arms never beat their
unmasked twins in training (best `FPR95@all` 0.036 vs 0.035 for `fft`, 0.034 vs 0.030 for
`bispectrum`, 0.373 vs 0.363 for `efficient8`):

1. **The mask is worth something.** A perfect one is worth 1.7x as a late weight and 6.5x
   as an input gate — the receptive-field argument in
   [`cascade`](#cascade-the-gate-in-the-loop), reproduced on a trained checkpoint.
2. **The predictor collects a third of it.** 0.0513 -> 0.0383 against a ceiling of 0.0268.
   And only in the *late* position: `gate_pred` scores 0.0498, barely better than no mask
   at all, because the gate is far more sensitive to mask error (`gate_shuffled` 0.2339 —
   4.6x WORSE than not masking). The predictor is a 1x1 conv on a 16x16 grid whose cells
   each see ~51 px of a 64 px patch; measured against held-out GT it reaches IoU@0.5 0.885
   with mean coverage 0.784 against a true 0.778.
3. **The mask path currently costs rather than pays.** At its own shipped operating
   point the masked model scores 0.0383 against 0.0332 for never having masked at all —
   ~15% worse, and consistent with the training logs. It has also become *dependent* on
   the mask: stripped of it entirely it falls to 0.0513 (though that condition is
   off-manifold for a model trained with a gated PDF input, so read it as dependence, not
   as a fair ablation). Note the routing is asymmetric — the PDF view is weighted by the
   truth, the target by a guess — so the two halves of every positive pair are processed
   with differently reliable masks, and `late_shuffled` (1.28x) / `gate_shuffled` (4.56x)
   show how little mask error the descriptor tolerates.

The one number that is not worse is `gate_gt` = 0.0079, the best result anywhere in the
table and 2.5x better than the unmasked model's own ceiling (0.0196). Training with a
gated input puts gated inputs on-manifold; the mask-free trunk gains almost nothing from
a gate (0.84x) because it has never seen one.

### Occlusion: the test background masking does not perform

Off-board background is a weak probe — on canonicalized track patches it is a thin rim,
and often bland. `--occlusion F` overwrites an angular wedge of every target patch (F of
the angular axis, all radii — what an occluder crossing the neighbourhood in the image
plane maps to) with **another patch's content**, so the occluder carries structure the
trunk cannot dismiss as flat fill, and tells the "true" mask about it. FPR95, `fftmask` /
`fft`:

| occluded | `none` | `native` | `late_gt` | `gate_gt` | `both_gt` |
| --- | --- | --- | --- | --- | --- |
| 0% | 0.051 / 0.033 | 0.038 / – | 0.027 / 0.020 | 0.008 / 0.028 | 0.026 / 0.020 |
| 10% | 0.155 / 0.125 | 0.145 / – | 0.101 / 0.060 | **0.015** / 0.047 | 0.055 / 0.034 |
| 20% | 0.407 / 0.365 | 0.426 / – | 0.314 / 0.179 | **0.043** / 0.104 | 0.180 / 0.099 |
| 40% | 0.804 / 0.807 | 0.828 / – | 0.664 / 0.554 | **0.315** / 0.485 | 0.600 / 0.479 |

A perfect input gate turns a 20%-occluded patch (FPR95 0.41) back into a nearly clean one
(0.043) — a 9.4x recovery, where the late weight only manages 1.3x. That is the strongest
case for masking in this codebase, and it is entirely unrealised: the shipped predictor is
*worse than no mask* at 20% and 40% occlusion.

The diagnostic line says why. Mean predicted coverage against mean true coverage:

| occluded | `m_pred` mean | true mean | corr | IoU@0.5 |
| --- | --- | --- | --- | --- |
| 0% | 0.784 | 0.778 | 0.818 | 0.885 |
| 10% | 0.802 | 0.705 | 0.648 | 0.771 |
| 20% | 0.810 | 0.620 | 0.512 | 0.671 |
| 40% | 0.816 | 0.461 | 0.354 | 0.497 |

The predictor does not move. It is supervised against the board-in-frame mask and nothing
else, so it has learned "board vs. the rim outside the board" and is structurally blind to
anything occluding the board itself — which is the case where the mask would be worth 10x.
Whatever else is done, **the predictor cannot be better than its target**: as long as the
BCE's label is the board-coverage mask, no amount of capacity or resolution buys the
occlusion gain.

`--blind-mask` puts a number on that ceiling: the "true" mask covers the off-board rim
only, which is what a board-in-frame mask really gives you, occluder included in the valid
region. `fftmask`, `gate_gt`:

| occluded | occlusion-aware mask | board mask only | no mask |
| --- | --- | --- | --- |
| 10% | 0.015 | 0.040 | 0.155 |
| 20% | 0.043 | 0.118 | 0.407 |

So even a perfect *board* mask is worth 3.9x resp. 3.4x on occluded patches — the
supervision target is not the binding constraint yet, the predictor's accuracy is. But it
becomes the constraint at ~2.6x remaining, which is roughly where an occlusion-aware
target (occlusion augmentation with a truthful mask) would have to take over.

## Oracle masks

`oracle_mask=True` (needs `learned_mask`) is the ceiling arm made trainable: the true mask
reaches the model on **every** view, target included, at both entry points. It is not a
deployable model — the target's mask does not exist at test time — it exists to separate
the two readings of a masked run that fails to beat its unmasked twin: a mask that buys
nothing, versus a predictor that cannot cash it in. The table above already separates them
on a *trained-with-a-guess* trunk; the oracle arm asks what the trunk becomes when the
information is actually there, which is the number the predictor's error must be judged
against.

`m_pred` is still emitted and still supervised (`mask_loss_weight: 0` in the leaf, so it
is a diagnostic curve and does not pull on the shared trunk — otherwise the arm would
differ from its control by two things at once). `process_batch_blobs` stops withholding
the mask at validation for such a model: an oracle model validated without its mask is a
different model. `state_dict` keys are unchanged, so `training.init_from_checkpoint`
transfers between an oracle arm and a normal one in both directions — which is what makes
the interesting cross-eval cheap: train with the oracle, then score the same weights with
`m_pred` (`src/mask_ablation.py`'s `late_pred` / `gate_pred`) and read off how much of the
gain survives contact with a real predictor.

Ready-made: `track_descriptor_logpolar_oracle.yaml`, launcher entries `logpolar_oracle`
(the non-cascade path, which with the oracle in place masks at **both** entry points —
`input_norm(patches, mask=mask)` *and* the late weight, i.e. `both_gt`) and
`logpolar_oracle_gate` (`cascade: true`, the input gate alone). Submit them **with**
`logpolar_fftmask` (predicted mask) and `logpolar_fft` (no mask) — three arms or the
comparison says nothing:

```sh
./launch_track_matrix.sh --track FILE -n oracle \
  logpolar_fft logpolar_fftmask logpolar_oracle logpolar_oracle_gate
```

### What the oracle arms measured (`2026_08_19_MaskCeiling`)

Four arms — no mask, predicted mask, perfect mask at both entry points, perfect mask at
the gate — 50 epochs each, then every checkpoint re-scored under every condition. The
full record, with the per-epoch curves, the per-band tables, the three hypotheses tried
against the `FPR95@all`-vs-bands discrepancy, and the leakage control, is
[mask_ceiling_experiment.md](mask_ceiling_experiment.md). The result:

| arm | best `FPR95@all` | at epoch | epoch 49 | `gate_gt` at `best` | `none` at `best` |
| --- | --- | --- | --- | --- | --- |
| `logpolar_fft` (no mask) | **0.0349** | 32 | 0.0483 | 0.0311 | 0.0353 |
| `logpolar_fftmask` (predicted) | 0.0397 | 41 | 0.0539 | 0.0174 | 0.0546 |
| `logpolar_oracle` (perfect, both) | 0.0247 | 2 | 0.1299 | 0.0190 | **0.8333** |
| `logpolar_oracle_gate` (perfect, gate) | **0.0128** | 0 | 0.0533 | **0.0129** | **0.5343** |

1. **A model trained under a perfect mask is not a descriptor.** The `none` column: strip
   the mask off the oracle weights and they score 0.83 / 0.53, where chance is 0.95 and
   the unmasked control is 0.035. They learned a function of *(patch, mask)* and spent
   their capacity on the half that will not exist at test time. Both arms also peak in
   their first two epochs and degrade for the remaining 48 while their training loss keeps
   falling below the controls' — the shape of a shortcut, not of a ceiling.
2. **The ceiling is real and small.** A perfect gate is worth 2.7x over the best
   deployable model (0.0129 vs 0.0353). Training under the oracle buys only 1.35x of that
   over handing the ordinarily-trained `fftmask` trunk a perfect gate at test time
   (0.0174) — and it is the unstable 1.35x, since `oracle_gate` ends at 0.0537 while
   `fftmask`'s `gate_gt` barely moves (0.0198 at `latest`).
3. **So the trunk is not compensating for its imperfect mask** in any way that costs
   measurable accuracy. There is no reservoir of performance being spent on robustness to
   `m_pred`'s errors. The whole gap between the shipped 0.0399 and the 0.0174 ceiling is
   the predictor at *test* time, not the trunk's training-time diet.

`oracle_mask` is implemented for `HardNetLogPolar` and for both steerable descriptors
(`BlobDescriptorEfficient`, `BlobDescriptorNoStride`); `src/mask_ablation.py` currently
hooks the log-polar family only (the steerable ones apply their weight inside `self.net`).

## Configuration

```yaml
model:
  params:
    learned_mask: true
    # oracle_mask: true       # ceiling ablation: the TRUE mask on every view
    # cascade: true           # log-polar only: mask at the INPUT, trunk runs twice
    # cascade_late_weight: false   # keep the late weight on top of the gate (costs the gain)
    # mask_resolution: 32     # log-polar only: tap the mask head earlier (None = 16x16 at 64)
training:
  ignore_mask: false          # true -> whole mask path off (ablation)
  mask_loss_weight: 1.0       # 0.0 for an oracle arm (diagnostic only)
  dataset:
    params:
      precompute_masks: true  # synthetic: co-extract + cache the GT mask
      # with_mask: true       # track data: the .tracks file already carries it
```

Which flag exists where:

| flag | `HardNetLogPolar` | `BlobDescriptorEfficient` | `BlobDescriptorNoStride` |
| --- | --- | --- | --- |
| `learned_mask` | yes | yes | yes |
| `oracle_mask` | yes | yes | yes |
| `cascade`, `cascade_late_weight` | yes | — | — |
| `mask_resolution` | yes | — | — |
| `m_pred` grid at `patch_size=64` | 16×16 | 8×8 | 24×24 |

Ready-made leaves: `blob_descriptor_{efficient,steerable}_mask` and
`blob_descriptor_logpolar_fftmask` (synthetic); `track_descriptor_{efficient,steerable}_mask`
and `track_descriptor_logpolar_{fftmask,cascade,oracle}` (track data). All train **from
scratch** — `launch_track_matrix.sh` no longer discovers a synthetic checkpoint to
warm-start from; pass `training.init_from_checkpoint` deliberately (e.g. via the
launcher's `EXTRA_OVERRIDES`) if you want one. Launcher entries `efficient8_mask`,
`efficient4_mask`, `steerable_mask` exist in both `launch_training_matrix.sh` and
`launch_track_matrix.sh`; `logpolar_fftmask`, `logpolar_cascade`, `logpolar_cascade_late`,
`logpolar_oracle` and `logpolar_oracle_gate` are track-only.

`precompute_masks` covers cartesian patches as well as log-polar ones. Cartesian masks
are single-channel and each `patch_scale_factors` entry samples a different physical
extent onto the same pixel grid, so exactly one scale factor is required (it raises
otherwise). Enabling it changes the dataset cache key, which is why the masked nets form
their own `cartesian_mask` prebuild group — the existing mask-less caches stay valid.

To run the ablation:

```sh
./launch_training_matrix.sh -n maskabl steerable steerable_mask efficient8 efficient8_mask
```
