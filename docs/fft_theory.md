# FFT theory behind the log-polar head

Theoretical background for the DFT-magnitude angular head (`AngularRFFTMag` /
`head="fft"` in `src/aef/models/hardnet.py`; see [logpolar_descriptor.md](logpolar_descriptor.md)
for the implementation and ablations). Each section ties a reference to a claim the
head actually relies on.

## The exact classical framework: Fourier–Mellin

A log-polar patch followed by an FFT-magnitude along the angular axis **is** the angular
restriction of the **Fourier–Mellin transform**. In log-polar coordinates a rotation is
a shift of the angular axis and a scale change is a shift of the log-radial axis; taking
the magnitude of the angular DFT makes the descriptor invariant to the former while the
radial axis stays scale-sensitive. This is the single most on-point theory for the head.

- **Reddy & Chatterji (1996).** "An FFT-Based Technique for Translation, Rotation, and
  Scale-Invariant Image Registration." *IEEE Transactions on Image Processing* 5(8),
  1266–1271. The canonical log-polar + Fourier-magnitude construction.
- **Casasent & Psaltis (1976).** "Position, rotation, and scale invariant optical
  correlation." *Applied Optics* 15(7), 1795–1799. The original Fourier–Mellin
  invariance argument.
- **Derrode & Ghorbel (2001).** "Robust and efficient Fourier–Mellin transform
  approximations for gray-level image reconstruction and complete invariant
  description." *Computer Vision and Image Understanding* 83(1), 57–78. Analytic
  Fourier–Mellin descriptors and how many harmonics to keep — the closed-form version of
  the `n_harmonics` knob.

## The core theorems the head leans on

- **Shift theorem** — a circular shift maps to a linear phase (`X'_k = e^{-i2πks/N} X_k`),
  so `|X_k|` is exactly shift-invariant. Any DSP text: **Oppenheim & Schafer,
  *Discrete-Time Signal Processing*** (Prentice Hall); or **Bracewell, *The Fourier
  Transform and Its Applications*** (McGraw-Hill).
- **Wiener–Khinchin** — the power spectrum is the DFT of the (circular) autocorrelation.
  This is *why* `{|X_k|}` is equivalent to keeping the angular autocorrelation of each
  `(channel, radius)` profile — the "full angular shape minus its orientation." See
  **Papoulis, *Probability, Random Variables, and Stochastic Processes*** (McGraw-Hill).
- **Why magnitude alone loses structure** — the caveat that the head discards the
  relative phase between radii/channels. **Oppenheim & Lim (1981).** "The Importance of
  Phase in Signals." *Proceedings of the IEEE* 69(5), 529–541. The classic magnitude-only
  vs phase-only reconstruction demonstration.

## The upgrade: bispectrum (invariant *and* phase-preserving)

The principled fix for the relative-phase the magnitude throws away: the bispectrum
`B(k1,k2) = X_{k1} X_{k2} conj(X_{k1+k2})` is shift-invariant yet preserves phase
couplings, and is *complete* (invertible up to translation).

- **Kakarala (2012).** "The Bispectrum as a Source of Phase-Sensitive Invariants for
  Fourier Descriptors: A Group-Theoretic Approach." *Journal of Mathematical Imaging and
  Vision* 44(3), 341–353. Proves completeness of the bispectral invariants.
- **Kakarala (1992).** *Triple Correlation on Groups.* PhD thesis, University of
  California, Irvine. The deep group-theoretic version.
- **Nikias & Raghuveer (1987).** "Bispectrum Estimation: A Digital Signal Processing
  Framework." *Proceedings of the IEEE* 75(7), 869–891. The estimation/practical side.
- **Sanborn, Shewmake, Olshausen & Hillar (2023).** "Bispectral Neural Networks." *ICLR*.
  The modern *learned* group-invariant version — "learn the invariant instead of
  hand-coding `|X|`."

## Keeping phase while staying rotation-invariant

The `fft` head keeps `|X_k|` and discards the phase `φ_k = ∠X_k`. That is invariant but
also drops the *relative* position between ripples, which is genuine shape. Two shapes
can share every `|X_k|` and differ only in relative phase — e.g. a 1-cycle bump and a
2-cycle bump *aligned* vs *offset by 90°* have identical magnitudes and differ only in
`φ_2 − φ_1`. Below are two ways to recover that while staying invariant. Both are
computed from the same `rfft` output and concatenated with `|X_k|` before the final
linear layer; a function of invariants is invariant, so neither changes the head's
rotation invariance.

Setup (one channel, one radius): under a rotation of `s` bins,

```
|X_k| -> |X_k|             (unchanged)
φ_k   -> φ_k − k·θ,   θ = 2π s / N  (the unknown rotation)
```

Every phase shifts by `k·θ` — proportional to the ripple index `k` and the unknown `θ`.

### Relative-phase features

Reference every phase to the first ripple so the unknown `θ` cancels:

```
Δ_k = φ_k − k·φ_1
```

Under rotation `φ_k → φ_k − kθ` and `k·φ_1 → k·φ_1 − kθ`, so `Δ_k → Δ_k` — invariant. It
encodes where ripple `k`'s crest sits *relative to* ripple 1, independent of the overall
rotation. Compute it with complex numbers to keep the magnitude and avoid `2π` wrap:

```
c_k = X_k · ( conj(X_1) / |X_1| )^k        # |c_k| = |X_k|,  ∠c_k = Δ_k
```

Feed `Re(c_k), Im(c_k)` for a few `k` as extra invariant channels. **Weakness:** it
references everything to ripple 1, so if `|X_1| ≈ 0` the reference is undefined/noisy —
which is why the bispectrum, with no single reference, exists.

### Low-order bispectrum

Invariant phase combinations with *no* reference, via triple products:

```
B(k1, k2) = X_{k1} · X_{k2} · conj(X_{k1+k2})
```

Its phase is `φ_{k1} + φ_{k2} − φ_{k1+k2}`; under rotation the shift terms are
`−k1·θ − k2·θ + (k1+k2)·θ = 0` — they cancel **because the indices sum to zero**. A pair
`X_{k1} conj(X_{k2})` only cancels when `k1 = k2` (giving just magnitude), so a *triple*
is the minimal combination that cancels the rotation while keeping phase. Each `B(k1,k2)`
is a complex invariant encoding how frequencies `k1`, `k2`, `k1+k2` are phase-locked.

**"Low-order"** = only small indices, e.g. `k1, k2 ∈ {1, 2, 3}` — a handful of the
lowest-frequency triples rather than the full `O(N²)` set. The coarse shape and the
rotation-relevant phase coupling live in the low frequencies, so a few low-order terms
capture most of it cheaply; the full bispectrum is large and highly redundant. Versus
relative-phase: no fragile reference, and it is *complete* (recovers the signal up to
rotation, Kakarala 2012) — but it is a product of three coefficients, so more
noise/scale-sensitive; normalize (divide by `|X_{k1}||X_{k2}||X_{k1+k2}|`) if only the
phase-coupling is wanted.

### How it plugs into the head — implemented

Both come from the `rfft` the head already computes — a few extra invariant scalars per
`(channel, radius)`, concatenated with `|X_k|` before the final linear layer. No trunk
change, no invariance change. They ship as two more angular heads in
`src/aef/models/hardnet.py`:

| head | class | rows per `(channel, radius)` for `F` harmonics |
|------|-------|-----------------------------------------------|
| `head="fft"` | `AngularRFFTMag` | `F` — `\|X_k\|` |
| `head="relphase"` | `AngularRelPhase` | `F + 2(F−2)` — `\|X_k\|` + `Re/Im(c_k)`, `k ≥ 2` |
| `head="bispectrum"` | `AngularBispectrum` | `F + 2·#{(k1,k2)}` — `\|X_k\|` + `Re/Im B(k1,k2)` |

```python
X = torch.fft.rfft(feat, dim=-2)              # already computed by AngularRFFTMag
mag   = X.abs()                               # current head
relph = X * (X[:, :, 1:2, :].conj() / X[:, :, 1:2, :].abs()).pow(k_index)   # relative-phase
bisp  = X[:, :, k1] * X[:, :, k2] * X[:, :, k1 + k2].conj()                 # low-order bispectrum
# feed [mag ; Re/Im(relph)]  or  [mag ; Re/Im(bisp)]  into the final Conv/Linear
```

`c_0` and `c_1` are real by construction, so `relphase` only adds rows for `k ≥ 2`; the
bispectrum keeps the low-order triples `1 ≤ k1 ≤ k2, k1 + k2 ≤ F−1` (`F = 5` → four
pairs). `bispectrum_normalize=True` (the default) divides each triple by
`|X_k1||X_k2||X_k1+k2|`, leaving pure phase coupling on the unit circle — the magnitudes
are already in the first rows, and the raw triple product is cubic in scale.

The extra rows widen the head's final conv (`(F, radial)` → `(rows, radial)`), so
**that layer's weights do not transfer between heads**; warm-starting across heads drops
it and reports the drop (`training.init_from_checkpoint`).

Start with **relative-phase** (cheapest); move to the **low-order bispectrum** if the
`|X_1|`-reference fragility bites or the completeness guarantee is wanted. Both attack the
one thing magnitude-only discards — the relative "where" between ripples. Both matrix
launchers expose them as `logpolar_relphase` / `logpolar_bispectrum`:

```sh
./launch_training_matrix.sh -n lpphase logpolar_fft logpolar_relphase logpolar_bispectrum
```

`tests/unit/test_angular_phase_heads.py` pins what they are for: exact invariance to an
angular roll, *and* separation of two profiles with identical magnitude spectra that
differ only in relative phase (a head that only passes the first is a costlier
`AngularRFFTMag`).

## Why a modulus gives *stable* invariance

For the theory of why taking a modulus (Fourier- or wavelet-magnitude) yields invariants
that are also stable to deformation — relevant because the patches are not pure rotations
(perspective, anisotropy) even after shape normalization:

- **Mallat (2012).** "Group Invariant Scattering." *Communications on Pure and Applied
  Mathematics* 65(10), 1331–1398.
- **Bruna & Mallat (2013).** "Invariant Scattering Convolution Networks." *IEEE
  Transactions on Pattern Analysis and Machine Intelligence* 35(8), 1872–1886. Same
  `|·|`-for-invariance principle with Lipschitz-stability guarantees.

## The equivariant framing

How the log-polar/FFT head connects to the steerable (`asel`/escnn) descriptor track:

- **Worrall, Garbin, Turmukhambetov & Brostow (2017).** "Harmonic Networks: Deep
  Translation and Rotation Equivariance." *CVPR*. Circular-harmonic (Fourier-on-rotation)
  equivariant CNNs — the bridge between this head and the steerable descriptors.
- **Cohen & Welling (2016).** "Group Equivariant Convolutional Networks." *ICML*. The
  group-CNN foundation.

## Start here

Two to read first for the thesis:

1. **Reddy & Chatterji (1996)** — the exact log-polar + Fourier-magnitude construction the
   `fft` head implements.
2. **Kakarala (2012)** — the precise statement of what `|X|` throws away and what the
   bispectrum recovers; frames both the head and its known limitation.
