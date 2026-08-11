# Perspective vs. rotation equivariance

Why a perspective view map breaks the descriptor's rotation invariance, in what form, and
how large the effect is for this pipeline's settings.

**Status: derivation only.** Everything below is analytic. Unlike
[logpolar_descriptor.md](logpolar_descriptor.md)'s ablation or
[data_pipeline.md](data_pipeline.md)'s bug tables, none of it has been measured yet —
§6 states the predictions that a measurement would have to confirm or kill.

The short version: `blob_normalizations` removes the local *affine* part of the view map
exactly, leaving a residual the descriptor is supposed to absorb as a pure rotation. That
reduction is affine-tight and only affine-tight. What perspective leaves behind is a
2-parameter residual that, on a log-polar patch, is a **k=1 angular harmonic modulating
the radial axis** — and in the angular spectrum it is an **off-diagonal coupling between
neighbouring harmonics `k <-> k±1`**. Rotation acts diagonally in `k`; that is the entire
reason `|X_k|` and the angular max-pool work. Perspective lives precisely where they
don't.

## 1. The exact local form of a homography at a blob

Write the view map as `H(p) = (A p + b) / (c'p + d)`, and put `w0 = c'p + d`, `q = H(p)`
at the blob's source point `p`. In blob-centred coordinates (`x` in the source, `u` in the
view):

```
u(x) = H(p + x) - q = J x / (1 + e'x)

with   J = (A - q c') / w0        (the Jacobian; what linearize_homography computes)
       e = c / w0                 (the perspective gradient)
```

This is **exact**, not a truncation — the algebra is just collecting `w0` out of the
denominator. So *locally, at a point, a homography is an affine map `J` followed by a
direction-dependent radial rescaling*, and the entire non-affine content is the single
vector `e ∈ R²`.

`e` has a direct geometric reading:

- `e'x = 0` is the direction of the **vanishing line** in the local frame;
- `|e|⁻¹` is the **distance from `p` to the horizon**, in source pixels;
- `e = 0` ⟺ the view map is locally affine (`c = 0` ⟺ globally affine).

This is also why the "Jacobian evaluated at the wrong point" bug in
[data_pipeline.md](data_pipeline.md#the-jacobian-evaluated-at-the-wrong-point) was
invisible for affine warps and "only bites because `perspective_amplitude` is non-zero":
the discrepancy it introduced is `O(e)`.

## 2. What the normalization removes — and the one thing it cannot

`blob_normalizations` builds `N` with `N J = sqrt(det J) · R`, `R ∈ SO(2)` (shape only,
`det == 1`; size is `scales`' job — see
[data_pipeline.md](data_pipeline.md#the-double-scale-correction)). Dividing the size out
too, the normalized anchor→positive map is

```
y(x) = R x / (1 + e'x)
```

The design premise is the `e = 0` case: `y = R x`, a **pure rotation**, which a
rotation-invariant descriptor closes exactly. The premise holds to the extent the view map
is affine over the patch, and no further. `N` is a *linear* map and consumes the 4 DoF of
`J`; the 2 DoF of `e` are untouched, because they are not in the linear part to begin
with. Nothing downstream of `blob_normalizations` has removed them either.

Define the dimensionless **deformation number**

```
    η = |e| · r  =  (patch radius) / (distance to the horizon)
```

with `r` the patch radius in source pixels. `η` is the only parameter that matters; every
number in §5 is a function of it.

## 3. The group-theoretic break

The true residual nuisance group is not `SO(2)`. It is the stabilizer of the blob in
`PGL(3)`, modulo the affine part the whitening already ate:

```
    SO(2) ⋉ R²,        P_e ∘ T_θ  =  T_θ ∘ P_{R_θ⁻¹ e}
```

where `T_θ` is the patch rotation and `P_e` the perspective residual. Two consequences,
and they are the formal reason this is not cheap to patch:

**`e` co-rotates.** There is no "rotation first, then a fixed perspective term"
decomposition — rotating the camera about its optical axis rotates the perturbation's
azimuth `φ_e`. For a rotation-invariant `f`:

```
f(P_e T_θ I) = f(T_θ P_{R_-θ e} I) = f(P_{R_-θ e} I)
```

The descriptor's residual dependence on `θ` is mediated **entirely** by `φ_e`. Useful
corollary: *restoring rotation equivariance does not require perspective invariance — only
invariance to the azimuth of `e`.* That is a 1-DoF ask instead of a 2-DoF one, and §6
turns it into a concrete architectural suggestion.

**The group is non-compact.** The angular max-pool and `|X_k|` work because `SO(2)` is
compact: you can average or max over the whole orbit. `R²` has no normalizable Haar
measure, so "make the head invariant to that too, the same way" is *structurally*
unavailable, not merely expensive. The remaining options are: bound `η`, estimate-and-
rectify (the canonicalization shape), or train the tolerance in from samples.

## 4. The log-polar picture

Substitute `x = e^ρ (cos φ, sin φ)`, so `e'x = |e| e^ρ cos(φ − φ_e)`. Because
`1/(1 + e'x)` is a positive **scalar**, it does not move the direction of `x` at all. To
first order in `η`:

```
φ' = φ + θ                                   (angle: pure rotation, untouched by e)
ρ' = ρ − η(ρ) cos(φ − φ_e),   η(ρ) = |e| e^ρ (radius: a k=1 angular modulation)
```

So on a log-polar patch, perspective is **exactly an angle-dependent radial shift**: one
angular harmonic (`k = 1`), phase `φ_e`, amplitude growing linearly with radius.

Two facts about that are special to homographies and worth stating, because a *generic*
smooth view map does not behave this way. A general quadratic displacement field produces
`k = 1` **and** `k = 3` radial terms plus a tangential component; for the projective form
`d(x) = −(e'x) x` the `k = 3` terms cancel and the tangential component is identically
zero. (`RadialDistortion` in `src/aef/transforms/distortion.py` is the generic case: at a
blob off the distortion centre it contributes `k = 1` *and* `k = 3`, along the radial
direction from the centre.)

Set this against the two motions the architecture is built around:

| | acts on the log-polar map as |
|---|---|
| rotation | rigid cyclic shift **along φ** |
| scale | rigid shift **along ρ** |
| **perspective** | **shear between the axes**: a ρ-shift modulated by cos φ |

The angular max-pool and `|X_k|` are invariant to shifts *along* `φ`. A `φ`-dependent
`ρ`-shift is not one, and no operator acting on the angular axis alone can undo it — after
the shear, the pool is combining rows that are radially misaligned with one another. This
is the same class of failure as the mirror bug
([data_pipeline.md](data_pipeline.md#the-mirror-bug)): a deformation that no angular
operation can absorb, silently defeating the architecture's only invariance mechanism.
The difference is that the mirror was an SVD-sign artifact and could be deleted; this one
is in the data.

### The spectral statement

Take the angular DFT `X_k(ρ)` of each radial profile, and let `D_k` be the angular
spectrum of `∂_ρ I`. Expanding `I(ρ − η cos(φ − φ_e), φ + θ)` to first order in `η`:

```
rotation:      X_k  ->  e^{-ikθ} · X_k                                   DIAGONAL in k
perspective:   X_k  ->  X_k − (η/2)·( e^{-iφ_e}·D_{k−1} + e^{+iφ_e}·D_{k+1} )
                                                                     OFF-DIAGONAL in k
```

This is the whole story in two lines. Rotation acts **diagonally in the irreps of
`SO(2)`** — which is exactly why `|X_k|` is invariant to it and why the max-pool works.
Perspective is a **rank-one coupling between neighbouring irreps**, strength `η/2`, phase
`φ_e`.

The dividing line this draws is sharp and useful: anything acting *diagonally* in `k`
leaves rotation equivariance perfectly intact — blur, contrast change, radial reweighting,
a scale change, an isotropic lens centred on the blob. Perspective does not, because it is
the off-diagonal. And `|X_k|` is not rescued by discarding phase: the perturbation *adds*
to `X_k` carrying a `φ_e`-dependent phase, so `|X_k|` moves at first order in `η` too.

This is also precisely the phase-coupling structure the bispectrum notes in
[fft_theory.md](fft_theory.md#the-upgrade-bispectrum-invariant-and-phase-preserving) are
about: `B(k1, k2)` is built to read coupling between `k1`, `k2`, `k1+k2`, and perspective
manufactures exactly that between adjacent harmonics.

## 5. Magnitudes for this pipeline

`e` scales like the corner-displacement amplitude per image width, so

```
η ≈ perspective_amplitude · r / W,     r = logpolar_outer_factor · σ
```

with `σ` the blob scale in source pixels and `W` the source image width. Converting to
pixels on the radial axis, whose scaling is `R_pix / ln(outer/inner)` — the 30.8 px per
e-fold quoted in [data_pipeline.md](data_pipeline.md#the-double-scale-correction), which
is the default `scale: 16.0` with `logpolar_inner_factor: 2.0` on a 64-px axis:

```
radial shear (px) ≈ η · 30.8
```

Plugging in `blob_descriptor_base.yaml` (`perspective_amplitude_{x,y}: 0.3`,
`image_size: [150, 150]`) and `blob_descriptor_logpolar.yaml` (`scale: 16.0`):

```
η ≈ 0.3 · 16σ / 150 = 0.032 · σ        ⇒  radial shear ≈ 0.99 · σ  px
```

**About one pixel of radial shear per unit of blob σ**, half the patch pushed out and half
pulled in. Calibrating against the ablation table in
[logpolar_descriptor.md](logpolar_descriptor.md#ablation), where a radial roll of 4 px
moves the descriptor by 0.25: a blob at `σ ≈ 4` px suffers a perspective shear of
comparable amplitude. Two caveats on the number — `perspective_amplitude` is the *cap* of
a truncated normal with std `amplitude/2`, so typical draws are roughly half this; and
`fit_to_frame` clamps the amplitude further for large warps.

Three things fall out:

**η is linear in blob scale.** Large blobs are perspective-wrecked; small ones are nearly
safe. This is a testable prediction and it lines up with the `σ_w ≥ 20` row being by far
the worst in the post-fix table in
[data_pipeline.md](data_pipeline.md#what-the-two-fixes-were-worth) (pos 0.654 where every
other band is ≥ 0.985). Whether that residual is `k = 1` structured or generic noise is
the measurement §6 asks for.

**`logpolar_outer_factor` is the tension knob, quantitatively.** Scale sensitivity wants a
large radial span (more e-folds), and `η ∝ outer_factor` exactly. Going from `scale: 8` to
`scale: 32` buys 2 e-folds of scale range and costs 4× the perspective shear. That is a
real, one-dimensional design trade-off in the log-polar track, and it is not present in
the cartesian/steerable track in the same form (see
[steerable_descriptors.md](steerable_descriptors.md)), whose patches are smaller multiples
of `σ`.

**Keypoint re-centring only half-helps, and confounds with detector error.** The mean of
the displacement field over the disc is `−e r²/4`, so the detector absorbs the DC part of
the `k = 1` mode as a centre shift. The residual still has `k = 1` angular dependence with
an `r² − r̄²` radial profile. Meanwhile a *centre error* `d` produces its own `k = 1`
modulation `∝ |d| / r` — dominant at **small** radii, where perspective (`∝ r`) is
weakest. The two share a harmonic and are confounded, which means the detector-error
budget in [scale_budget_and_jitter.md](scale_budget_and_jitter.md) and this effect cannot
be separated by amplitude alone; only by their opposite radial profiles.

## 6. Predictions, and what a measurement would look like

The derivation makes falsifiable claims. The natural harness is the one already used for
the rotation/scale ablation — untrained nets, mean relative L2 drift — since this is an
*architectural* claim, not a trained-performance one:

1. **Drift is linear in `η` at small `η`**, with `η` swept two independent ways
   (`perspective_amplitude` at fixed `σ`, and `σ` at fixed amplitude). If both sweeps
   collapse onto one line in `η`, §2's reduction to a single parameter is confirmed.
2. **The residual is `k = 1` in the angular spectrum**, with no `k = 3` and no tangential
   component. Take the anchor↔positive residual displacement field on a log-polar pair and
   look at its angular spectrum per radius. This is the sharpest test, because a generic
   second-order warp would show `k = 3` as well.
3. **The residual's radial profile grows like `r²`** (like `r` after re-centring removes
   the DC term), distinguishing it from detector centre error, which falls like `1/r`.
4. **`|X_k|` moves at first order in `η`, not second.** If the `fft` head's drift were
   `O(η²)` the off-diagonal analysis in §4 would be wrong.
5. **`HardNetLogPolar`'s two fixes should not help here.** Circular padding and
   antialiasing repair seam and aliasing artifacts, both of which are about representing
   *shifts along φ*. Neither touches an off-diagonal `k` coupling. If the (a)/(d) gap in
   the ablation table closes under perspective, something in this analysis is wrong.

Set `homography_seed` for any of these — see
[data_pipeline.md](data_pipeline.md#measuring-patches--two-traps).

## 7. What the structure suggests

In rough order of cost, reading off §3 and §4:

- **Bound `η`.** The only free fix, and the one already available: a smaller
  `logpolar_outer_factor`, or a scale-dependent radial span so large blobs do not reach as
  far. Costs scale range, per §5.
- **Kill the azimuth, not the perspective.** Per §3 only `φ_e`-invariance is needed to
  restore rotation equivariance, and per §4 the corruption is a `k = 1` modulation *of the
  radial axis* whose phase is `φ_e`. A head invariant to the phase of that mode — rather
  than to `e` itself — is a substantially smaller ask than perspective invariance.
- **Estimate and rectify.** `e` is 2 DoF and is exactly computable from the GT homography
  at training time (`c / w0` at the blob's source point), so it is directly supervisable.
  This is the canonicalization-shaped answer, and it is the only one that removes rather
  than tolerates the effect. Note `process_batch_canonicalize` currently predates even the
  lens model and ignores it.
- **Bispectral / relative-phase features.** They read adjacent-harmonic phase coupling,
  which is precisely what perspective creates, so at minimum they make it *measurable*
  instead of silently absorbed into the descriptor. See
  [fft_theory.md](fft_theory.md#keeping-phase-while-staying-rotation-invariant).
