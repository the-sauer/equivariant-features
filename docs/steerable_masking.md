# Learned board-validity masking on the steerable (escnn) descriptors

How `learned_mask=True` works on `BlobDescriptorEfficient` and `BlobDescriptorNoStride`,
why it leaves rotation equivariance intact, and what it costs. The log-polar
counterpart (`HardNetLogPolar`) implements the same contract — see
[logpolar_descriptor.md](logpolar_descriptor.md#learned_masktrue--mask-aware-pooling-without-a-target-mask)
— so the two families are configured, supervised and compared identically.

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

Always at the point where the spatial map collapses, because that is where an off-board
cell would otherwise contribute to the descriptor.

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

Targets get no input gate — their mask is exactly what is unavailable at test time.

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

## Configuration

```yaml
model:
  params:
    learned_mask: true
training:
  ignore_mask: false          # true -> whole mask path off (ablation)
  mask_loss_weight: 1.0
  dataset:
    params:
      precompute_masks: true  # synthetic: co-extract + cache the GT mask
      # with_mask: true       # track data: the .tracks file already carries it
```

Ready-made leaves: `blob_descriptor_{efficient,steerable}_mask` (synthetic) and
`track_descriptor_{efficient,steerable}_mask` (track data, warm-started from the
former). Launcher entries `efficient8_mask`, `efficient4_mask`, `steerable_mask` exist in
both `launch_training_matrix.sh` and `launch_track_matrix.sh`.

`precompute_masks` covers cartesian patches as well as log-polar ones. Cartesian masks
are single-channel and each `patch_scale_factors` entry samples a different physical
extent onto the same pixel grid, so exactly one scale factor is required (it raises
otherwise). Enabling it changes the dataset cache key, which is why the masked nets form
their own `cartesian_mask` prebuild group — the existing mask-less caches stay valid.

To run the ablation:

```sh
./launch_training_matrix.sh -n maskabl steerable steerable_mask efficient8 efficient8_mask
```
