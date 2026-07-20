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
      bins) instead of a single peak per (channel, radius).
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
                 head="maxpool", n_harmonics=None, learned_mask=False, **_):
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

        # Angular reduction head: max-pool (one peak) or DFT-magnitude (full spectrum).
        if head == "maxpool":
            angular_reduce = nn.MaxPool2d(kernel_size=(pool, 1))  # max over angular
            final_angular = 1
        elif head == "fft":
            n_harm = n_harmonics if n_harmonics is not None else (pool // 2 + 1)
            angular_reduce = AngularRFFTMag(n_harmonics=n_harm)
            final_angular = n_harm
        else:
            raise ValueError(f"Unsupported head {head!r} (expected 'maxpool' or 'fft')")
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
