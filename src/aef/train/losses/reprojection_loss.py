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


class HomographyReprojectionLoss(torch.nn.Module):
    def __init__(self, reduction="mean", distance_metric="euclidean"):
        super().__init__()
        self.reduction = reduction
        self.distance_metric = distance_metric

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        grid = torch.stack((
            *torch.meshgrid(
                torch.arange(10, device=pred.device),
                torch.arange(10, device=pred.device), indexing="ij"
            ),
            torch.ones((1, 1), device=pred.device).expand(10, 10)
        ),
            dim=-1
        ).float().reshape(1, -1, 3).expand(pred.shape[0], -1, -1)

        pred_proj = torch.bmm(pred.view(pred.shape[0], 3, 3), grid.permute(0, 2, 1)).permute(0, 2, 1)
        target_proj = torch.bmm(target.view(target.shape[0], 3, 3), grid.permute(0, 2, 1)).permute(0, 2, 1)

        if self.distance_metric == "euclidean":
            loss = torch.norm(pred_proj - target_proj, dim=-1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
