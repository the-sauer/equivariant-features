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

import juliacall

import asel
import torch
from torch import nn

from .scale import NeuralScaleSpace


class AffineFeatureNetOne(torch.nn.Module):
    def __init__(self, in_channels=1, feature_size=128):
        super(AffineFeatureNetOne, self).__init__()
        self.scale_space = NeuralScaleSpace(in_channels)
        self.feature_net = torch.nn.Sequential(
            asel.affine.BasicBlock(in_channels+1, 8, option="B"),
            asel.affine.BasicBlock(8, 16, option="B"),
            asel.affine.BasicBlock(16, feature_size, option="B"),
        )
        self.feature_size = feature_size

    def forward(self, x):
        channel_dim = 1
        scale_field = self.scale_space(x)
        x = torch.concat((x, scale_field), dim=channel_dim)
        x = nn.functional.normalize(x, dim=channel_dim)
        return self.feature_net(x)
