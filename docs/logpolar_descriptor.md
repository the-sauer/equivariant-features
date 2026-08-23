# Log-polar descriptor

`HardNetLogPolar` (`src/aef/models/hardnet.py`) describes blobs from **log-polar**
patches, extracted by `extract_logpolar_patches` (`src/aef/data/homography.py`) on the
synthetic path and read straight out of the `.tracks` file on the real one. It is the
only model in the repo; all four bootstrap configs use it, differing in the angular head
(`maxpool` vs `fft`).

## The geometry it relies on

In a log-polar patch (angular axis = dim −2, radial axis = dim −1):

- an image **rotation** becomes a **cyclic shift along the angular axis** — the axis is
  periodic, row 0 and row H−1 are neighbours;
- a **scale change** becomes a shift along the **radial** axis, which is *not* periodic
  (inner radius ≠ outer radius).

So a plain CNN can be made rotation-invariant by pooling over the whole angular axis,
while staying scale-sensitive by reading the radial axis densely:

```python
nn.MaxPool2d(kernel_size=(pool, 1)),         # max over the ANGULAR axis -> rotation invariance
nn.Conv2d(depths[2], 128, (1, pool)),        # dense over the RADIAL axis -> keeps scale structure
```

Both hold only to the extent the view map is **affine** over the patch. A perspective
view leaves a residual that is neither an angular nor a radial shift but a *shear between
the two axes* — one no angular operation can absorb, and which grows linearly with
`logpolar_outer_factor`.

## A plain HardNet trunk is not actually rotation-invariant

The angular max-pool only delivers invariance if the feature map really is a *cyclic
shift* of itself. Two things break that:

1. **Zero padding on a periodic axis** — every conv pads with zeros, injecting fake
   content at the 0/2π seam. (`nn.Conv2d(padding_mode="circular")` is no help: it would
   wrap *both* axes, and the radial axis must not wrap.)
2. **Stride-2 aliasing** — two stride-2 convs downsample 64→16, so only shifts that are
   multiples of 4 map to integer shifts of the final map; finer rotations land between
   samples.

Measured (mean relative L2 drift of the descriptor under an angular roll of an untrained
net), the unwrapped trunk drifts **0.16–0.25** for rotations of 22.5°–180°. For
comparison a *scale* change (radial roll of 4) moves it 0.31. So rotation perturbs the
descriptor about as much as a scale change does: it is barely being factored out at all.

## The two geometry fixes

Same conv count and parameter count (1.06 M):

- **`LogPolarPad`** — wraps the angular axis (dim −2) and zero-pads the radial axis.
- **`LogPolarBlurPool`** — antialiased ×2 downsample (binomial 1-2-1 blur then
  subsample) replacing the stride-2 convs; the blur itself wraps angularly, otherwise it
  would reintroduce the seam.

Both are toggleable (`circular_pad`, `antialias`); turning both off reproduces the plain
HardNet structure, which is what makes the ablation below exact.

Both spell the wrap as a slice-and-concat rather than `F.pad(..., mode="circular")`,
which is what it reads as. `F.pad` exports to ONNX `Pad(mode="wrap")`, and onnxruntime
has no CUDA kernel for that — the ten of them in this trunk would each drag a GPU session
back through host memory. `Slice`/`Concat` are accelerated everywhere and give bitwise
identical results (`tests/unit/test_onnx_friendly_ops.py` pins that).

### Ablation

All variants built from the same seed, so they share **identical conv weights** — only
padding/stride differ. Values are mean relative L2 drift under an angular roll
(= rotation); lower is more invariant.

| variant | 5.6° | 11.2° | 16.9° | 22.5° | 45° | 90° | 180° | **mean** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| (a) neither | .218 | .252 | .245 | .157 | .199 | .233 | .228 | **0.219** |
| (b) circular-pad only | .218 | .240 | .215 | .0003 | .0003 | .0003 | .0003 | **0.096** |
| (c) antialias only | .073 | .110 | .123 | .131 | .186 | .212 | .212 | **0.150** |
| (d) both *(the default)* | .066 | .084 | .064 | .0002 | .0002 | .0002 | .0002 | **0.031** |

Radial roll (scale, shift 4) — must stay sensitive: (a) 0.311 · (b) 0.310 · (c) 0.247 ·
(d) 0.249.

**The two fixes repair disjoint failure modes**, so neither alone is close to sufficient:

- **Circular padding** exactly fixes the rotations the grid can represent: restricted to
  multiples of 4 (22.5°/45°/90°/180°) the drift goes **0.2045 → 0.0003**, a ~700×
  improvement, for free (no params, no compute). It does nothing below 22.5°.
- **Antialiasing** is the opposite: it is the only thing that helps sub-22.5° rotations
  (0.218 → 0.073 at 5.6°, ~3×) but leaves the coarse ones at 0.185–0.212 because the seam
  is still there.

Caveats:

- **Circular padding is the load-bearing fix.** If only one is taken, take that.
- **The residual ~0.07 at fine rotations is inherent, not a bug.** With 64 angular bins a
  5.6° rotation is a *one-bin* shift; no amount of blurring makes a sub-bin shift exact.
  Only more angular resolution would.
- Antialiasing slightly reduces radial (scale) sensitivity (0.311 → 0.249) — expected,
  since blurring costs some high-frequency radial detail. Still far from the 0.0002
  rotation drift, so the rotation/scale separation holds.
- **Method:** these are *untrained* nets, which is the right test for an *architectural*
  invariance. Training cannot make the old architecture invariant; it can only spend
  capacity compensating for the seam.

## The angular pooling: independent vs "locked"

`MaxPool2d((pool, 1))` pools **each column independently** — every one of the 128
channels × `pool` radial columns picks its own angle. A natural idea is to *lock* the
pooling: choose one angular offset and read every column at it.

Worth being precise about what that would buy, because it is **not** invariance:
independent max-pooling is *already* exactly rotation-invariant (max over a cyclically
shifted vector is unchanged — measured 0.0002). What locking buys is
**discriminativeness**: independent pooling discards *which* angle each column peaked at,
destroying the angular relationship between radii, so structurally different blobs can
collapse to the same descriptor. The cost is rigidity — locking assumes one global
rotation explains the whole patch, which fails under deformations that are not a pure
rotation (perspective, anisotropy).

## `head="fft"` — keep the whole angular spectrum, not one peak

The magnitude of the angular DFT is invariant to a cyclic shift — the shift lives
entirely in the phase, and `|X_k|` drops it — exactly for integer-bin shifts, the same
regime the max-pool is exact in. `AngularRFFTMag` replaces `MaxPool2d((pool, 1))` with
`|rfft(...)|`, keeping the lowest `n_harmonics` frequency magnitudes:

```python
nn.MaxPool2d(kernel_size=(pool, 1))    # keeps ONE peak per (channel, radius)
AngularRFFTMag(n_harmonics=F)          # keeps |X_0..F-1| == the angular autocorrelation
```

Where the max-pool collapses each `(channel, radius)` angular profile to a single scalar
(its peak), the magnitude spectrum is — by Wiener–Khinchin — the profile's full circular
**autocorrelation**: the entire angular *shape* minus its orientation. A one-bump ring
and a two-bump ring have the same peak but different spectra, so this directly repairs
the "structurally different blobs collapse to the same descriptor" failure above. It is
not a strict superset (the peak is a nonlinear order-statistic the power spectrum does
not determine), and it still discards the *relative* phase between radii. Measured on an
untrained net, the fft head's angular-roll drift is ≈1e-7, on par with the max-pool.

The DFT is spelled as a matmul (`angular_rdft`) rather than `torch.fft.rfft`: the latter
exports to the ONNX `DFT` op, which onnxruntime implements on the CPU execution provider
only, so in a GPU session the head and the host round trips around it would dominate
inference. ONNX has no complex dtype either, so the complex arithmetic would be
decomposed anyway. The angular axis is tiny (`A = patch_size // 4`, and only the `F`
lowest bins survive), so an `(A, F)` matmul beats the FFT even in eager torch. The basis
is evaluated in float64 and rounded once — it costs nothing and keeps the round-trip
error ~6× below a float32 basis.

The head changes the width of the final conv (`F` rows instead of 1), so a checkpoint
does **not** transfer that layer between `maxpool` and `fft`. The two heads also differ
in `state_dict` layout: `maxpool` keeps the historical flat `features` module, `fft`
splits `trunk`/`head`. Each is load-compatible with the runs that produced it.

## `outer_factor` interacts with the blob scale

The patch reaches `logpolar_outer_factor × σ`, in units of the blob's own scale. With
`max_diameter_fraction: 0.4` a large blob at 16σ already extends several board-widths out
(mostly fill), while a small blob at 16σ sees a sensible neighbourhood — so **no single
`outer_factor` suits every blob**. The configs run at 96; treat large values with
suspicion, and note that at `outer=16` the *large* blobs are the weak end
(reference↔positive NCC 0.827 at σ_w>10 and 0.654 at σ_w>20, against 0.98 for sub-pixel
blobs). That is the "extends several board-widths out" failure with a number on it, and
it argues for a *smaller* `outer_factor`, or one that adapts to the blob's scale.

## Architecture directions to try

Ranked by value-for-effort, tied to the failure modes above.

1. **Union the peak with the spectrum (~free).** Max-pool keeps a nonlinear
   order-statistic (the peak) the power spectrum does not determine; the spectrum keeps
   the autocorrelation the peak does not. Concatenate both angular reductions
   (`[AngularRFFTMag(x) ; max_angular(x)]`) before the final layer — strictly richer than
   either, no invariance risk.
2. **Put phase back in the head.** `|X_k|` is invariant *because* it throws the phase
   away — and with it where each ripple sits relative to the others, which is genuine
   shape: a 1-cycle and a 2-cycle bump have identical magnitude spectra whether aligned or
   offset by 90°. A relative-phase feature (`c_k = X_k (conj(X_1)/|X_1|)^k`) or a low-order
   bispectrum (`B(k1,k2) = X_{k1} X_{k2} conj(X_{k1+k2})`, complete per Kakarala 2012)
   appends an invariant phase term to the same magnitudes, so it stays exactly as
   rotation-invariant while recovering that structure.
3. **Adaptive / multi-band `outer_factor`.** "No single `outer_factor` suits every blob"
   (above) is architectural in spirit: read several radial annuli (a radial pyramid) or
   scale `outer_factor` to the blob, rather than one fixed band. Likely beats any head
   tweak for the large-blob regime.
4. **Orientation head from `∠X_1` (near-free).** The phase of the angular fundamental is a
   continuous, differentiable orientation estimate the `fft` head already computes. Expose
   it to (a) supervise against the known relative rotation (an equivariance-teaching loss)
   or (b) canonicalize (the "locked pooling" idea above).
5. **Modernize the trunk.** The trunk is a plain 2019-HardNet conv stack. Residual
   connections and a HyNet-style normalization (filter-response norm + a learnable
   component) in place of `BatchNorm(affine=False)` gave measurable FPR95 gains over
   HardNet in the descriptor literature. Diffuse "probably helps"; ablate behind a flag.
6. **Higher angular resolution.** The ~0.07 fine-rotation residual is *inherent* to 64
   angular bins (5.6° = one bin, see the ablation); more bins is the only thing that
   shrinks it, at compute cost.
