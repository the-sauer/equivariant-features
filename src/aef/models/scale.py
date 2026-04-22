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

import sesn
import torch


def NeuralScaleSpace(in_channels=1, factor=2.0, num_scales=4, min_scale=1.0) -> torch.nn.Module:
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    return torch.nn.Sequential(
        sesn.SESConv_Z2_H(in_channels, 8, 11, 7, scales, padding=5),
        sesn.SESConv_H_H(8, 16, 5, 11, 7, scales, padding=5),
        sesn.SESConv_H_H(16, 32, 5, 11, 7, scales, padding=5),
        sesn.SESConv_H_H_1x1(32, 1, num_scales=len(scales)),
        sesn.SESArgMaxProjection(scales)
    )
