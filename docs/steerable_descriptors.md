# Steerable blob descriptors

Descriptor models live in `src/aef/models/blob_descriptor.py` (built on `escnn`,
C_N-steerable CNNs). Variants: `BlobDescriptorHardNet`, `…Deep`, `…NoStride`,
`…Robust`, `…Hierarchical`, `…Efficient`. `NoStride` is accurate but heavy;
`Efficient` targets the same accuracy at a fraction of the cost.

## Why NoStride is heavy (measured, `n_rotations=8`, 1×64×64 patches)

- It **never downsamples** — all ~10 conv layers run at 24–62 px, and `regular_repr`
  on C8 multiplies every feature map ×8, so the deep layers (64/128 regular = 512/1024
  effective channels) are held at high resolution and stored for backprop. This is the
  dominant memory/compute driver.
- The final `R2Conv(128·regular → 128·regular, kernel=24)` readout: its escnn
  basis-expanded filter alone is **~2.25 GB** (1024×1024×24×24), a batch-independent
  spike — for a layer whose output is 1×1 (a glorified FC).
- No frequency/ring bandlimiting → square kernels carry high-freq corner components
  that cost params and hurt rotation equivariance.

Measured (batch 32, 12 GB GPU): **42.3 M params, 1831 ms/step, 9.44 GB peak**; OOMs at
batch 256. Dropping C8→C4 alone (same architecture): ×0.5 params, ×0.24 time, ×0.34
memory.

## BlobDescriptorEfficient

Keeps the equivariant inductive bias but fixes the cost drivers:

1. **Antialiased downsampling** — `PointwiseAvgPoolAntialiased2D` (Gaussian blur +
   subsample, equivariance-preserving) halves the resolution after each stage
   (64 → 32 → 16 → 8), so the wide deep layers run cheaply. This is the principled
   version of HardNet's stride (which strides too hard and drops to trivial reps).
2. **Head** (`head=`):
   - `attention` (**default**): a `1×1` conv produces a spatial saliency map; the
     descriptor is a *learned weighted* global pool of the group-pooled features.
     Preserves spatial information (non-uniform weighting) and is robust to the
     backbone's residual equivariance error.
   - `dense`: a single dense `R2Conv` over the whole `8×8` grid (equivariant analogue
     of flatten+FC, full spatial layout). **Not recommended with downsampling** — see
     the equivariance note below.
3. **Per-block boundary masking** — padded convs leak zero-padding artifacts at the
   border that break equivariance (`NoStride` avoids this with valid convs); re-masking
   each block (as `Robust` does) keeps the field circularly supported and tightens
   equivariance.
4. **Centre "donut" mask** (`inner_mask`) — the anchor blob within
   `sigma_cutoff · (patch_size/2) / scale_factor` px of the centre is the same
   normalized structure for every keypoint, so it carries no discriminative signal.
   A radially symmetric mask commutes with the rotation group, so zeroing it keeps
   equivariance intact and forces the descriptor onto the informative surround.
5. `n_rotations` selects the group (C8 vs C4); `frequencies_cutoff` can bandlimit the
   basis; `l2_normalize` defaults `False` to match `NoStride`.

### Benchmarks (batch 32, attention head)

| model | params | step | peak mem | 90° rot-error |
|---|--:|--:|--:|--:|
| NoStride (N=8) | 42.3 M | 1831 ms | 9.44 GB | 0.0001 |
| Efficient (N=8) | 2.8 M (15× ↓) | 222 ms (8× ↓) | 3.05 GB (3× ↓) | 0.014 |
| Efficient (N=4) | 1.4 M (30× ↓) | 68 ms (27× ↓) | 1.25 GB (7.5× ↓) | 0.014 |

### Equivariance trade-off (important)

`NoStride` is near-exactly rotation-invariant (0.0001) because its valid convs avoid
padding. `Efficient`'s downsampling + padded convs introduce ~1.4% residual equivariance
error (rot-error 0.014 with the attention head). The **dense** head *amplifies* this to
~0.14 — a per-cell spatial readout is very sensitive to the pooling's residual — which
is why the attention head (which globally pools and tolerates it) is the default. 1.4%
is small and the model trains on rotated views, but whether it costs any FPR95 is an
empirical question — run the sweep.

## Learned board-validity masking (`learned_mask`)

Both steerable descriptors take `learned_mask=True`: part of every patch is off-board
junk, and it is not shared between an anchor and its target, so it is pure noise for the
metric loss. The mask that would remove it exists for the **anchor** (given) but not for
the **target** at test time — so a 1×1 steerable conv predicts one (`m_pred ∈ [0, 1]`)
from the backbone features, `forward` returns `(descriptor, m_pred)`, and a BCE trains
that prediction on the targets against their true coverage.

Where the weight is applied differs by architecture, always at the point where the
spatial map collapses:

| model | weighting point |
| --- | --- |
| `BlobDescriptorEfficient` (`head="attention"`) | multiplied into the saliency, so it divides the normalizer too — the descriptor is the average over the *valid* region, not a downscaled average over everything |
| `BlobDescriptorEfficient` (`head="dense"`) | the feature field, before the readout conv |
| `BlobDescriptorNoStride` | the 24×24 feature field, before the dense readout conv |

Equivariance is untouched — the mask, GT or predicted, is a *scalar field*, and a
pointwise product with one commutes with the group action (measured rot90 drift is the
same with and without the mask path). `mask_head.*` is the only added `state_dict` key,
so a masked run still warm-starts from a plain checkpoint.

**→ [steerable_masking.md](steerable_masking.md)** for the derivation, the input gate,
the supervision rules, the measured cost, and the config/launcher surface. Same contract
as the log-polar [`learned_mask`](logpolar_descriptor.md#learned_masktrue--mask-aware-pooling-without-a-target-mask),
so the two families are configured and compared identically.

## Using / comparing

Config: `src/conf/blob_descriptor_efficient.yaml` (mirrors `blob_descriptor_steerable`,
swapping only the model). Head-to-head FPR95 vs `NoStride`:

```sh
./launch_training_matrix.sh -n modelcmp steerable efficient8 efficient4
```

submits `NoStride` vs `Efficient` (C8) vs `Efficient` (C4) across the scale sweep, all
under one dated folder, with comparable titled loss curves. Add `steerable_mask
efficient8_mask` to the same command to put the masked variants beside them.

### Further ideas (not yet done)

- Gradient checkpointing to fit larger batches without changing the architecture.
- `model.export()` to a plain `torch.nn.Conv2d` for faster/lighter validation &
  inference (no per-forward basis expansion).
- `RestrictionModule`: keep C8 in the first layers, restrict to C4 deeper.
