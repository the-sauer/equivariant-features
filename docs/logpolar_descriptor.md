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
