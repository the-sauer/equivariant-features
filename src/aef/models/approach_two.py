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

import torch

from .asel.affine import BasicBlock, EquivarLayer, LearnedSaliencyLayer


class AffineFeatureNetCanonicalTwo(torch.nn.Module):
    def __init__(
        self,
        scales: list[float | int] | dict[str, float | int],
        in_channels,
        conv_depths,
        basic_block_params=None
    ):
        super(AffineFeatureNetCanonicalTwo, self).__init__()
        scale_list: list[float]
        if isinstance(scales, list):
            scale_list = scales
        elif isinstance(scales, dict) and "min_scale" in scales and "factor" in scales and "num_scales" in scales:
            q = scales["factor"] ** (1 / (scales["num_scales"] - 1))
            scale_list = [scales["min_scale"] * q**i for i in range(int(scales["num_scales"]))]
        else:
            raise ValueError("scales must be either a list of scales or a dictionary with keys 'min_scale', 'factor'\
                              and 'num_scales'.")

        if basic_block_params is None:
            basic_block_params = {}

        self.feature_net = torch.nn.Sequential(
            BasicBlock(in_channels, conv_depths[0], scales=scale_list, **basic_block_params),
            *(BasicBlock(conv_depths[i], conv_depths[i+1], scales=scale_list, **basic_block_params)
              for i in range(len(conv_depths)-1)),
            EquivarLayer(conv_depths[-1], 4, scales=scale_list, type=["0", "c"]),
            LearnedSaliencyLayer(scales=scale_list),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        assert x.size(1) == 1

        return self.feature_net(x)
