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

import logging

import torch

from ..detector import homogenize, linearize_homography


class HomographyReprojectionLoss(torch.nn.Module):
    def __init__(self, reduction="mean", distance_metric="euclidean", stride=4):
        super().__init__()
        self.reduction = reduction
        self.distance_metric = distance_metric
        self.stride = stride

    def forward(self, x) -> torch.Tensor:
        pred = x["pred"]
        target = x["target"]
        H = x["H"]
        if isinstance(pred, tuple):
            pred = pred[0]
        if isinstance(target, tuple):
            target = target[0]
        gt = homogenize(linearize_homography(H, pred.shape[1:-2], stride=self.stride).unsqueeze(1).reshape(-1, 2, 2))
        pred = pred[:, ::self.stride, ::self.stride].reshape(-1, 3, 3)
        target = target[:, ::self.stride, ::self.stride].reshape(-1, 3, 3)

        non_singular_mask = torch.linalg.det(pred) > 1e-6
        if non_singular_mask.int().sum() < 0.01 * pred.size(0):
            logging.warning("Warning: More than 99%% of predicted transforms are degenerate.")
            return 1e9

        rel_t = target[non_singular_mask] @ torch.linalg.inv(pred[non_singular_mask])   # use linalg.solve

        gt = gt[non_singular_mask]

        grid = torch.stack((
            *torch.meshgrid(
                torch.arange(10, device=pred.device),
                torch.arange(10, device=pred.device), indexing="ij"
            ),
            torch.ones((1, 1), device=pred.device).expand(10, 10)
        ),
            dim=-1
        ).float().reshape(1, -1, 3)
        pred_proj = (rel_t.view(-1, 3, 3) @ grid.permute(0, 2, 1)).permute(0, 2, 1)
        pred_proj = pred_proj[..., :2] / pred_proj[..., 2:3]
        target_proj = (gt.view(-1, 3, 3) @ grid.permute(0, 2, 1)).permute(0, 2, 1)
        target_proj = target_proj[..., :2] / target_proj[..., 2:3]

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
