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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import L2Norm


def input_norm(x):
    """Per-patch mean/std normalization (shift-invariant)."""
    flat = x.view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + 1e-7
    return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
            sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))


class LogPolarPad(nn.Module):
    """Pad a log-polar map: wrap the angular axis, zero-pad the radial axis.

    In a log-polar patch the angular axis (dim -2) is periodic — row 0 and row H-1 are
    neighbours — while the radial axis (dim -1) is not (inner radius != outer radius).
    ``nn.Conv2d(padding_mode="circular")`` is no use here because it would wrap *both*.

    The wrap is spelled as a slice-and-concat rather than ``F.pad(mode="circular")``
    because the latter exports to ONNX ``Pad(mode="wrap")``, which onnxruntime's CUDA
    EP does not implement — every one of these (ten in the log-polar trunk) would drop
    the graph back to the CPU. ``Slice``/``Concat`` are accelerated everywhere and the
    result is identical.
    """

    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        if self.pad == 0:
            return x
        p = self.pad
        x = torch.cat((x[..., -p:, :], x, x[..., :p, :]), dim=-2)             # angular wraps
        return F.pad(x, (p, p, 0, 0), mode="constant", value=0)               # radial does not


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
        # Slice-and-concat instead of F.pad(mode="circular"/"replicate"): see LogPolarPad.
        x = torch.cat((x[..., -1:, :], x, x[..., :1, :]), dim=-2)   # angular wraps
        x = torch.cat((x[..., :1], x, x[..., -1:]), dim=-1)         # radial replicates
        x = F.conv2d(x, self.kernel, groups=self.channels)
        return x[..., ::2, ::2]


def angular_rdft(x, n_harmonics=None):
    """``torch.fft.rfft(x, dim=-2)``, spelled as a real matmul; returns ``(re, im)``.

    Deliberately not ``torch.fft``: the exporter turns ``rfft`` into the ONNX ``DFT``
    op, which onnxruntime only implements on the **CPU** execution provider — in a GPU
    session the head (and the copies in and out of host memory around it) dominate
    inference. ONNX has no complex dtype either, so the complex arithmetic downstream
    would be decomposed anyway.

    The angular axis here is tiny — ``A = patch_size // 4``, and only the ``F`` lowest
    bins survive — so the transform is just an ``(A, F)`` matmul, which at this size
    beats the FFT even in eager torch and stays on the accelerator. The basis is rebuilt
    on every call: it is a few hundred elements (and a compile-time constant once
    exported), and keeping it out of the state dict means checkpoints stay compatible.

    The basis is evaluated in float64 and rounded once, which costs nothing and keeps
    the round-trip error ~6x below a float32 basis — the phase heads below multiply
    three coefficients together, so the basis is the one place worth being exact.
    """
    a = x.shape[-2]
    n = a if n_harmonics is None else min(n_harmonics, a)
    freq = torch.arange(n, device=x.device, dtype=torch.float64).view(1, n)
    step = torch.arange(a, device=x.device, dtype=torch.float64).view(a, 1)
    ang = (-2.0 * math.pi / a) * (step * freq)          # (A, F)
    xt = x.transpose(-1, -2)                            # (B, C, R, A)
    re = torch.matmul(xt, torch.cos(ang).to(x.dtype)).transpose(-1, -2)   # (B, C, F, R)
    im = torch.matmul(xt, torch.sin(ang).to(x.dtype)).transpose(-1, -2)
    return re, im


def _magnitude(re, im, eps=1e-12):
    """``|X_k|``. The ``eps`` only keeps the gradient finite where the bin vanishes
    exactly (``torch.abs`` on a complex tensor used to absorb that via ``sgn``)."""
    return torch.sqrt(re * re + im * im + eps)


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

    def forward(self, x):                                 # (B, C, A, R)
        return _magnitude(*angular_rdft(x, self.n_harmonics))   # (B, C, F, R)


class HardNetLogPolar(nn.Module):
    """HardNet variant that respects log-polar geometry.

    Log-polar maps an image rotation to a *cyclic* shift along the angular axis
    (dim -2) and a scale change to a shift along the radial axis (dim -1). Rotation
    invariance is meant to come from ``MaxPool2d((pool, 1))`` — a max over the whole
    angular axis — but that only holds if the feature map really is a cyclic shift of
    itself, which a plain HardNet trunk breaks two ways: every conv zero-pads the
    periodic angular axis (fake content at the 0/2pi seam), and the stride-2 convs
    alias shifts that aren't multiples of the total downsampling.

    Measured on untrained nets (mean relative L2 drift of the descriptor under an
    angular roll, i.e. a pure rotation): the unwrapped trunk drifts 0.13-0.22 for
    rotations of 22.5-180 deg — about as much as a scale change, so rotation is barely
    factored out. Wrapping the angular padding takes those to 0.0003; the blur-pool
    targets the sub-4-pixel shifts that remain. Both fixes are separately toggleable
    (``circular_pad``, ``antialias``) so they can be ablated.

    ``head="fft"`` replaces the angular max-pool with :class:`AngularRFFTMag` — same
    rotation invariance, but it keeps the ``n_harmonics`` lowest angular frequencies
    instead of a single peak per (channel, radius). See ``docs/logpolar_descriptor.md``.
    """

    def __init__(self, in_channels=1, patch_size=64, slim=False,
                 circular_pad=True, antialias=True,
                 head="maxpool", n_harmonics=None, **_):
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

        trunk_layers = []
        trunk_layers += block(in_channels, depths[0])
        trunk_layers += block(depths[0], depths[0])
        trunk_layers += down(depths[0], depths[1])                   # patch -> patch/2
        trunk_layers += block(depths[1], depths[1])
        trunk_layers += down(depths[1], depths[2])                   # patch/2 -> patch/4
        trunk_layers += block(depths[2], depths[2])
        trunk_layers += [nn.Dropout(0.1)]

        # Angular reduction head: max-pool (one peak) or DFT-magnitude (full spectrum).
        n_harm = n_harmonics if n_harmonics is not None else (pool // 2 + 1)
        if head == "maxpool":
            angular_reduce = nn.MaxPool2d(kernel_size=(pool, 1))  # max over angular
            final_angular = 1
        elif head == "fft":
            angular_reduce = AngularRFFTMag(n_harmonics=n_harm)
            final_angular = n_harm
        else:
            raise ValueError(f"Unsupported head {head!r} (expected 'maxpool' or 'fft')")
        head_layers = [
            angular_reduce,
            nn.Conv2d(depths[2], 128, (final_angular, pool), bias=False),  # dense over radial
            nn.BatchNorm2d(128, affine=False),
        ]

        # `maxpool` keeps the historical flat `features` module (and hence its
        # state_dict keys); `fft` splits trunk/head, which is how its checkpoints were
        # written. Both layouts are load-compatible with the runs that produced them.
        if head == "maxpool":
            self.features = nn.Sequential(*trunk_layers, *head_layers)
        else:
            self.trunk = nn.Sequential(*trunk_layers)
            self.head = nn.Sequential(*head_layers)
        self.apply(weights_init)

    def forward(self, patches):
        if hasattr(self, "features"):
            x = self.features(input_norm(patches))
        else:
            x = self.head(self.trunk(input_norm(patches)))
        return L2Norm()(x.view(x.size(0), -1))


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
