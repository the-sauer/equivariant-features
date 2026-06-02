# Affine Equivariant Features, the main implementation of my master thesis.
# Copyright (C) 2026 Hendrik Sauer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from math import ceil

import sesn
import torch


def NeuralScaleSpaceSESN(
    in_channels: int = 1,
    factor: float = 2.0,
    num_scales: int = 4,
    min_scale: float = 1.0,
    effective_size: int = 5,
    scale_size: int = 5,
    **_
) -> torch.nn.Module:
    """
    Neural Scale Field based on Scale-Equivariant Steerable Networks (SESN).
    """
    if min_scale < 1:
        raise ValueError("min_scale must be at least 1.")
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    kernel_size = ceil(effective_size * max(scales))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    layer_kwargs = {"effective_size": effective_size, "kernel_size": kernel_size, "scales": scales, "padding": padding}
    return torch.nn.Sequential(
        sesn.SESConv_Z2_H(in_channels, 8, **layer_kwargs),
        sesn.SESConv_H_H(8, 16, scale_size, **layer_kwargs),
        sesn.SESConv_H_H(16, 32, scale_size, **layer_kwargs),
        sesn.SESConv_H_H_1x1(32, 1, num_scales=num_scales),
        sesn.SESArgMaxProjection(scales)
    )


class ConstantScaleSpace(torch.nn.Module):
    """
    Identity Scale Space that simply returns the input as the scale field.
    """
    def __init__(self, **_):
        super().__init__()

    def forward(self, x):
        size = list(x.size())
        size[1] = 1
        return torch.ones(size, dtype=torch.float32, device=x.device)
