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
        if "detections_unfiltered" in x and x["detections_unfiltered"] is not None:
            pred = x["detections_unfiltered"]
        elif "pred" in x and x["pred"] is not None:
            pred = x["pred"]
        else:
            return torch.Tensor([0]).to(x["device"])
        ε = 1e-6
        return torch.mean(-torch.log(torch.min(torch.linalg.svdvals(pred[0]), dim=-1)[0] + ε)).reshape(1)


class Determinant(torch.nn.Module):
    def forward(self, x):
        pred = x["detections_unfiltered"] if "detections_unfiltered" in x else x["pred"][0]
        ε = 1e-6
        pred = pred[..., :2, :2].view(-1, 2, 2)
        return torch.mean(torch.maximum(-pred + ε, torch.zeros_like(pred)))
        return torch.mean(-torch.log(torch.minimum(torch.linalg.det(pred), torch.full_like(torch.linalg.det(pred), 1)) + ε))
