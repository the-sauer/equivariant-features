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


class NonSingular(torch.nn.Module):
    def __init__(self, sigma=1e-4):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        pred = x["pred"]
        ε = 1e-6
        return torch.mean(-torch.log(torch.min(torch.linalg.svdvals(pred[0]), dim=-1)[0] + ε))


class DeterminantLoss(torch.nn.Module):
    def forward(self, x):
        pred = x["pred"][0]
        if "scale" in x:
            scale = x["scale"][0]
            scale = scale.view(-1, 1, 1)
        else:
            scale = 1.0
        return torch.mean((torch.linalg.det(pred) - scale) ** 2)
