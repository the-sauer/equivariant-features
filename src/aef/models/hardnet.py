# Copyright 2019 EPFL, Google LLC
# Copyright 2025-2026 Hendrik Sauer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import L2Norm


def input_norm(x, mask=None):
    """Per-patch mean/std normalization (shift-invariant).

    ``mask`` (broadcastable to ``x``, 1 = valid, 0 = invalid) restricts the
    statistics to valid pixels and sets invalid pixels to 0 (== the normalized
    mean), so off-board fill does not skew the normalization. ``mask=None``
    reproduces the original all-pixel behaviour bit-for-bit.
    """
    flat = x.view(x.size(0), -1)
    if mask is None:
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
                sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))
    m = mask.expand_as(x).reshape(x.size(0), -1)
    n = m.sum(dim=1).clamp(min=1.0)
    mp = (flat * m).sum(dim=1) / n
    var = (((flat - mp.unsqueeze(1)) ** 2) * m).sum(dim=1) / n
    sp = torch.sqrt(var) + 1e-7
    xn = (x - mp.detach().view(-1, 1, 1, 1)) / sp.detach().view(-1, 1, 1, 1)
    return xn * mask.expand_as(x)


class HardNet(nn.Module):
    def __init__(self, in_channels=1, patch_size=32, shallow=False, slim=False):
        super(HardNet, self).__init__()

        # model processing patches of size [32 x 32] and giving description vectors of length 2**7
        if patch_size == 32:
            kernel_size = 3
            padding = 1
            pool = 8
        elif patch_size == 64:
            kernel_size = 5
            padding = 2
            pool = 16
        elif patch_size == 128:
            kernel_size = 9
            padding = 4
            pool = 32
        else:
            raise ValueError(f"Unsupported patch size {patch_size}")
        self.patch_size = patch_size

        depths = [16, 32, 64] if slim else [32, 64, 128]
        if shallow:
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, depths[0], kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(depths[0], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[0], depths[0], kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(depths[0], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[0], depths[1], kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(depths[1], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[1], depths[2], kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(depths[2], affine=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.MaxPool2d(kernel_size=(pool, 1)),
                nn.Conv2d(depths[2], 128, (1, pool), bias=False),
                nn.BatchNorm2d(128, affine=False),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, depths[0], kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(depths[0], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[0], depths[0], kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(depths[0], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[0], depths[1], kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(depths[1], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[1], depths[1], kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(depths[1], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[1], depths[2], kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(depths[2], affine=False),
                nn.ReLU(),
                nn.Conv2d(depths[2], depths[2], kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(depths[2], affine=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.MaxPool2d(kernel_size=(pool, 1)),
                nn.Conv2d(depths[2], 128, (1, pool), bias=False),
                nn.BatchNorm2d(128, affine=False),
            )

        # initialize weights
        self.features.apply(weights_init)
        return

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
                sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))

    # function to forward-propagate inputs through the network
    def forward(self, patches):
        x_features = self.features(self.input_norm(patches))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class LogPolarPad(nn.Module):
    """Pad a log-polar map: wrap the angular axis, zero-pad the radial axis.

    In a log-polar patch the angular axis (dim -2) is periodic — row 0 and row H-1 are
    neighbours — while the radial axis (dim -1) is not (inner radius != outer radius).
    ``nn.Conv2d(padding_mode="circular")`` is no use here because it would wrap *both*.
    """

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if self.pad == 0:
            return x
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="circular")            # angular wraps
        return F.pad(x, (self.pad, self.pad, 0, 0), mode="constant", value=0)  # radial does not


class LogPolarBlurPool(nn.Module):
    """Antialiased stride-2 downsample that wraps on the angular axis.

    A plain ``stride=2`` conv aliases: a cyclic shift that isn't a multiple of the total
    downsampling factor does not map to an integer shift of the subsampled map, so the
    angular max-pool downstream no longer sees a shifted copy. Blurring (binomial 1-2-1)
    before subsampling suppresses that — with the blur itself wrapping angularly, or it
    would reintroduce the seam the padding above removes.
    """

    def __init__(self, channels):
        super().__init__()
        k = torch.tensor([1.0, 2.0, 1.0])
        kernel = k[:, None] * k[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.expand(channels, 1, 3, 3).clone())
        self.channels = channels

    def forward(self, x):
        x = F.pad(x, (0, 0, 1, 1), mode="circular")
        x = F.pad(x, (1, 1, 0, 0), mode="replicate")
        x = F.conv2d(x, self.kernel, groups=self.channels)
        return x[..., ::2, ::2]


class AngularRFFTMag(nn.Module):
    """Cyclic-shift-invariant angular embedding — a drop-in for the angular max-pool.

    A rotation is a circular shift of the angular axis (dim -2). The magnitude of the
    angular DFT is invariant to that shift (the shift lives entirely in the phase, and
    ``|X_k|`` drops it), exactly for integer-bin shifts. Unlike the max-pool — which
    keeps a single peak per (channel, radius) — this keeps the strength of every
    angular frequency, i.e. the whole angular *shape* minus its orientation, which is
    the structure the max-pool discards. ``n_harmonics`` keeps only the lowest ``F``
    frequencies (rotation/shape info concentrates there) to bound the descriptor size.
    """

    def __init__(self, n_harmonics=None):
        super().__init__()
        self.n_harmonics = n_harmonics

    def forward(self, x):                       # (B, C, A, R)
        mag = torch.fft.rfft(x, dim=-2).abs()   # (B, C, A // 2 + 1, R) — shift-invariant
        if self.n_harmonics is not None:
            mag = mag[:, :, : self.n_harmonics, :]
        return mag


class AngularRelPhase(nn.Module):
    """Magnitude **plus relative phase** — invariant, but not phase-blind.

    ``AngularRFFTMag`` keeps ``|X_k|`` and throws the phase away, which also throws away
    where each ripple sits *relative to* the others: a 1-cycle and a 2-cycle bump have the
    same magnitudes whether they are aligned or offset by 90 deg, so structurally
    different blobs collide. Referencing every phase to the first harmonic cancels the
    unknown rotation ``theta`` while keeping that relation (under a roll,
    ``phi_k -> phi_k - k*theta``)::

        Delta_k = phi_k - k*phi_1                       (invariant)
        c_k     = X_k * (conj(X_1) / |X_1|)**k          (|c_k| = |X_k|, angle(c_k) = Delta_k)

    Output rows are ``[ |X_0..F-1| ; Re(c_2..F-1) ; Im(c_2..F-1) ]`` — ``c_0`` and ``c_1``
    are real by construction (``c_0 = X_0``, ``c_1 = |X_1|``), so they carry nothing the
    magnitude rows do not. Costs one extra complex multiply over the fft head.

    **Weakness** (the reason :class:`AngularBispectrum` exists): everything is referenced
    to ripple 1, so where ``|X_1| ~ 0`` the reference is noise. ``eps`` only keeps that
    finite, it cannot make it informative.

    See ``docs/fft_theory.md`` -> "Keeping phase while staying rotation-invariant".
    """

    def __init__(self, n_harmonics=None, eps=1e-6):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.eps = eps

    @staticmethod
    def n_rows(n_harmonics):
        """Rows this head emits for ``n_harmonics`` kept frequencies."""
        return n_harmonics + 2 * max(0, n_harmonics - 2)

    def forward(self, x):                            # (B, C, A, R)
        spec = torch.fft.rfft(x, dim=-2)             # (B, C, A // 2 + 1, R), complex
        if self.n_harmonics is not None:
            spec = spec[:, :, : self.n_harmonics, :]
        mag = spec.abs()
        if spec.shape[-2] < 3:                       # no k >= 2 to reference
            return mag
        ref = spec[:, :, 1:2, :]                     # X_1, the phase reference
        ref = ref.conj() / (ref.abs() + self.eps)    # unit modulus (eps: |X_1| ~ 0)
        k = torch.arange(2, spec.shape[-2], device=x.device).view(1, 1, -1, 1)
        c = spec[:, :, 2:, :] * ref.pow(k)           # rotation cancels in the phase
        return torch.cat([mag, c.real, c.imag], dim=-2)


class AngularBispectrum(nn.Module):
    """Magnitude **plus the low-order bispectrum** — invariant phase, no reference.

    The bispectrum keeps phase relations without singling out a reference harmonic::

        B(k1, k2) = X_k1 * X_k2 * conj(X_{k1+k2})

    Under a roll the shift terms are ``-k1*theta - k2*theta + (k1+k2)*theta = 0``: they
    cancel *because the indices sum to zero*, which is why a triple product is the
    smallest phase-keeping invariant (a pair only cancels when ``k1 == k2``, leaving mere
    magnitude). Unlike :class:`AngularRelPhase` it has no fragile ``|X_1|`` reference and
    is complete (recovers the signal up to rotation, Kakarala 2012), at the price of being
    a product of three coefficients — hence noisier and cubic in scale.

    Only the *low-order* triples are kept: ``1 <= k1 <= k2`` with ``k1 + k2 <= F - 1``,
    where the coarse shape and the rotation-relevant phase coupling live. Output rows are
    ``[ |X_0..F-1| ; Re(B_p) ; Im(B_p) ]`` over those pairs.

    Keep ``n_harmonics`` small: the pair count grows ~``F**2 / 4``, and every pair costs
    two rows of the final conv's kernel (``F = 5`` -> 13 rows; the default
    ``F = patch_size // 8 + 1`` at ``patch_size=128`` -> 145 rows, i.e. a ~76M-parameter
    layer). The launchers pin 4-5.

    ``normalize=True`` (default) divides by ``|X_k1||X_k2||X_k1+k2|``, leaving the pure
    phase coupling on the unit circle: the magnitudes are already in the first rows, and
    the raw triple product's cubic dynamic range is what makes this head hard to train.
    Set it False to let the amplitude coupling through.

    See ``docs/fft_theory.md`` -> "Low-order bispectrum".
    """

    def __init__(self, n_harmonics=None, normalize=True, eps=1e-6):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.normalize = normalize
        self.eps = eps

    @staticmethod
    def pairs(n_harmonics):
        """The ``(k1, k2)`` triples kept for ``n_harmonics``: ``k1 <= k2, k1 + k2 <= F-1``."""
        return [(k1, k2)
                for k1 in range(1, n_harmonics)
                for k2 in range(k1, n_harmonics)
                if k1 + k2 <= n_harmonics - 1]

    @classmethod
    def n_rows(cls, n_harmonics):
        """Rows this head emits for ``n_harmonics`` kept frequencies."""
        return n_harmonics + 2 * len(cls.pairs(n_harmonics))

    def forward(self, x):                            # (B, C, A, R)
        spec = torch.fft.rfft(x, dim=-2)             # (B, C, A // 2 + 1, R), complex
        if self.n_harmonics is not None:
            spec = spec[:, :, : self.n_harmonics, :]
        mag = spec.abs()
        pairs = self.pairs(spec.shape[-2])
        if not pairs:
            return mag
        k1 = torch.tensor([p[0] for p in pairs], device=x.device)
        k2 = torch.tensor([p[1] for p in pairs], device=x.device)
        x1, x2 = spec.index_select(-2, k1), spec.index_select(-2, k2)
        x12 = spec.index_select(-2, k1 + k2)
        bisp = x1 * x2 * x12.conj()
        if self.normalize:
            bisp = bisp / (x1.abs() * x2.abs() * x12.abs() + self.eps)
        return torch.cat([mag, bisp.real, bisp.imag], dim=-2)


class HardNetLogPolar(nn.Module):
    """HardNet variant that respects log-polar geometry.

    Log-polar maps an image rotation to a *cyclic* shift along the angular axis
    (dim -2) and a scale change to a shift along the radial axis (dim -1). Rotation
    invariance is meant to come from ``MaxPool2d((pool, 1))`` — a max over the whole
    angular axis — but that only holds if the feature map really is a cyclic shift of
    itself, which plain ``HardNet`` breaks two ways: every conv zero-pads the periodic
    angular axis (fake content at the 0/2pi seam), and the stride-2 convs alias shifts
    that aren't multiples of the total downsampling.

    Measured on untrained nets (mean relative L2 drift of the descriptor under an
    angular roll, i.e. a pure rotation): plain ``HardNet`` drifts 0.13-0.22 for
    rotations of 22.5-180 deg — about as much as a scale change, so rotation is barely
    factored out. Wrapping the angular padding takes those to 0.0003 (on par with the
    steerable descriptor); the blur-pool targets the sub-4-pixel shifts that remain.

    The two fixes are separately toggleable (`circular_pad`, `antialias`) so they can be
    ablated; turning both off reproduces the plain `HardNet` structure.

    Two further **opt-in** heads (default off, so the module is unchanged unless asked):

    - ``head="fft"`` replaces the angular max-pool with :class:`AngularRFFTMag` — same
      rotation invariance, but it keeps the full angular spectrum (``n_harmonics`` low
      bins) instead of a single peak per (channel, radius). ``head="relphase"``
      (:class:`AngularRelPhase`) and ``head="bispectrum"`` (:class:`AngularBispectrum`)
      go one step further and append an invariant *phase* feature to those magnitudes —
      the relation between ripples that ``|X_k|`` alone discards. All three are exactly
      as rotation-invariant as the max-pool; they differ in how much shape survives.
    - ``learned_mask=True`` makes the head *mask-aware*. The pre-head feature map is
      downweighted by the **GT mask on anchors** (identity view, where the off-board
      region is given) and by a **predicted** ``m_pred`` on targets (warped view, where
      it is not) — a small 1x1 predictor emits ``m_pred`` in [0, 1] from the trunk
      features. ``forward`` returns ``(descriptor, m_pred)`` so the caller can add a
      standalone loss supervising ``m_pred`` on the **targets** against their true board
      coverage (the anchor's mask is given, not predicted). The predictor thus learns to
      supply, at test time, the target mask that is no longer available.

    Both leave the default ``head="maxpool"``/``learned_mask=False`` path — including
    its ``state_dict`` keys — byte-for-byte identical to the original.
    """

    def __init__(self, in_channels=1, patch_size=64, slim=False,
                 circular_pad=True, antialias=True,
                 head="maxpool", n_harmonics=None, bispectrum_normalize=True,
                 learned_mask=False, **_):
        super().__init__()
        if patch_size == 32:
            kernel_size, padding = 3, 1
        elif patch_size == 64:
            kernel_size, padding = 5, 2
        elif patch_size == 128:
            kernel_size, padding = 9, 4
        else:
            raise ValueError(f"Unsupported patch size {patch_size}")
        self.patch_size = patch_size
        self.head_type = head
        self.learned_mask = learned_mask
        pool = patch_size // 4          # spatial size after the two downsamples
        depths = [16, 32, 64] if slim else [32, 64, 128]

        def block(c_in, c_out, stride=1):
            pad = LogPolarPad(padding) if circular_pad else nn.ZeroPad2d(padding)
            return [
                pad,
                nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride,
                          padding=0, bias=False),
                nn.BatchNorm2d(c_out, affine=False),
                nn.ReLU(),
            ]

        def down(c_in, c_out):
            """Halve the resolution: blur-then-subsample, or a plain stride-2 conv."""
            if antialias:
                return [*block(c_in, c_out, stride=1), LogPolarBlurPool(c_out)]
            return block(c_in, c_out, stride=2)

        trunk_layers = [
            *block(in_channels, depths[0]),
            *block(depths[0], depths[0]),
            *down(depths[0], depths[1]),                      # patch -> patch/2
            *block(depths[1], depths[1]),
            *down(depths[1], depths[2]),                      # patch/2 -> patch/4
            *block(depths[2], depths[2]),
            nn.Dropout(0.1),
        ]

        # Angular reduction head: max-pool (one peak), DFT-magnitude (full spectrum), or
        # magnitude + a phase invariant (relative phase / low-order bispectrum).
        n_harm = n_harmonics if n_harmonics is not None else (pool // 2 + 1)
        if head == "maxpool":
            angular_reduce = nn.MaxPool2d(kernel_size=(pool, 1))  # max over angular
            final_angular = 1
        elif head == "fft":
            angular_reduce = AngularRFFTMag(n_harmonics=n_harm)
            final_angular = n_harm
        elif head == "relphase":
            angular_reduce = AngularRelPhase(n_harmonics=n_harm)
            final_angular = AngularRelPhase.n_rows(n_harm)
        elif head == "bispectrum":
            angular_reduce = AngularBispectrum(n_harmonics=n_harm,
                                               normalize=bispectrum_normalize)
            final_angular = AngularBispectrum.n_rows(n_harm)
        else:
            raise ValueError(
                f"Unsupported head {head!r} (expected 'maxpool', 'fft', 'relphase' "
                "or 'bispectrum')"
            )
        head_layers = [
            angular_reduce,
            nn.Conv2d(depths[2], 128, (final_angular, pool), bias=False),  # dense over radial
            nn.BatchNorm2d(128, affine=False),
        ]

        if head == "maxpool" and not learned_mask:
            # Exact original structure and state_dict keys — full backward compatibility.
            self.features = nn.Sequential(*trunk_layers, *head_layers)
        else:
            self.trunk = nn.Sequential(*trunk_layers)
            self.head = nn.Sequential(*head_layers)
            if learned_mask:
                # 1x1 predictor: per-cell validity in [0, 1] from the trunk features.
                self.mask_head = nn.Sequential(nn.Conv2d(depths[2], 1, 1), nn.Sigmoid())
        self.apply(weights_init)

    def _validity_weight(self, feat, mask, m_pred, is_anchor):
        """Per-cell validity weight at trunk resolution: GT on anchors, ``m_pred`` else."""
        _, _, A, R = feat.shape
        if mask is None or is_anchor is None:
            return m_pred                                     # no GT routing available
        gt = F.adaptive_avg_pool2d(mask, (A, R))              # (B,1,A,R), board coverage
        a = is_anchor.view(-1, 1, 1, 1).to(feat.dtype)
        return a * gt + (1.0 - a) * m_pred

    def forward(self, patches, mask=None, is_anchor=None):
        # Default path: identical to the original (self.features only exists then).
        if hasattr(self, "features"):
            x = self.features(input_norm(patches))
            return L2Norm()(x.view(x.size(0), -1))

        # Masked input-norm on anchors (known off-board fill); plain on targets/unknown.
        if self.learned_mask and mask is not None and is_anchor is not None:
            a = is_anchor.view(-1, 1, 1, 1).to(patches.dtype)
            innorm_mask = a * mask + (1.0 - a) * torch.ones_like(mask)
            x = input_norm(patches, mask=innorm_mask)
        else:
            x = input_norm(patches)

        feat = self.trunk(x)                                  # (B, C, A, R)
        m_pred = None
        if self.learned_mask:
            m_pred = self.mask_head(feat)                     # (B, 1, A, R) in [0, 1]
            feat = feat * self._validity_weight(feat, mask, m_pred, is_anchor)
        d = L2Norm()(self.head(feat).view(patches.size(0), -1))
        return (d, m_pred) if self.learned_mask else d


def weights_init(m):
    '''
    Conv2d module weight initialization method
    '''

    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant_(m.bias.data, 0.01)
        except:
            pass
    return
