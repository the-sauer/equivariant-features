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


def input_norm(x):
    """Per-patch mean/std normalization (shift-invariant)."""
    flat = x.view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + 1e-7
    return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
            sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))


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
    """

    def __init__(self, in_channels=1, patch_size=64, slim=False,
                 circular_pad=True, antialias=True, **_):
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

        self.features = nn.Sequential(
            *block(in_channels, depths[0]),
            *block(depths[0], depths[0]),
            *down(depths[0], depths[1]),                      # patch -> patch/2
            *block(depths[1], depths[1]),
            *down(depths[1], depths[2]),                      # patch/2 -> patch/4
            *block(depths[2], depths[2]),
            nn.Dropout(0.1),
            nn.MaxPool2d(kernel_size=(pool, 1)),              # max over angular -> rotation invariance
            nn.Conv2d(depths[2], 128, (1, pool), bias=False),  # dense over radial -> keeps scale structure
            nn.BatchNorm2d(128, affine=False),
        )
        self.features.apply(weights_init)

    def forward(self, patches):
        x = self.features(input_norm(patches))
        return L2Norm()(x.view(x.size(0), -1))


def weights_init(m):
    '''
    Conv2d module weight initialization method
    '''

    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=0.6)
        try:
            nn.init.constant(m.bias.data, 0.01)
        except:
            pass
    return
