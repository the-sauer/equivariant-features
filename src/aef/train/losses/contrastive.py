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

import kornia
import pytorch_metric_learning
import torch


class ContrastiveLoss(torch.nn.Module):
    def __init__(
        self,
        contrastive_loss: str = "NPairsLoss",
        contrastive_loss_kwargs: dict = None,
        patch_size: tuple[int, int] = (64, 64),
        **_
    ):
        super().__init__()
        try:
            self.contrastive_loss = getattr(pytorch_metric_learning.losses, contrastive_loss)(**(contrastive_loss_kwargs or {}))
        except AttributeError as e:
            raise ValueError(f"Unsupported distance metric: {contrastive_loss}" + e.msg)
        self.patch_size = patch_size
        self.patch_scale = torch.diag(
            torch.Tensor([patch_size[0], patch_size[1], 1]).to(torch.float32)
        ).unsqueeze(0)

    def forward(
        self,
        x
    ) -> torch.Tensor:
        pred = x["pred"]
        target = x["target"]
        descriptor_model = x["descriptor_model"]

        patch_scale = self.patch_scale.to(pred[0].device)
        # assert pred[0].dim() == 4
        assert pred[1].dim() == 4
        # assert target[0].dim() == 4
        assert target[1].dim() == 4
        if (torch.linalg.det(pred[0]) > 1e-6).int().sum() < 0.01 * pred[0].size(0) * pred[0].size(1) or torch.any(torch.sum((torch.linalg.det(pred[0]) > 1e-6).int(), dim=1) == 0):
            logging.warning("Warning: More than 99%% of predicted transforms are degenerate.")
            # We will try to increase the determinants first
            return 1e9
        pred_transform, pred_image = pred
        pred_transform = pred_transform[:, ::8, ::8].reshape(pred_transform.size(0), -1, 3, 3)
        target_transform, target_image = target
        target_transform = target_transform[:, ::8, ::8].reshape(target_transform.size(0), -1, 3, 3)
        pred_transform_masks = [(torch.linalg.det(pred_transform[i]) > 1e-6)  & (torch.linalg.det(target_transform[i]) > 1e-6) for i in range(pred_transform.size(0))]
        pred_patch = torch.cat([kornia.geometry.transform.warp_perspective(
            pred_image[i].unsqueeze(0).expand(pred_transform_masks[i].int().sum(), -1, -1, -1),
            patch_scale @ torch.linalg.inv(pred_transform[i][pred_transform_masks[i]]),
            dsize=self.patch_size
        ) for i in range(pred[0].size(0))])
        target_patch = torch.cat([kornia.geometry.transform.warp_perspective(
            target_image[i].unsqueeze(0).expand(pred_transform_masks[i].int().sum(), -1, -1, -1),
            patch_scale @ torch.linalg.inv(target_transform[i][pred_transform_masks[i]]),
            dsize=self.patch_size
        ) for i in range(target[0].size(0))])

        features_pred = descriptor_model(pred_patch)
        features_target = descriptor_model(target_patch)
        features = torch.cat([features_pred, features_target], dim=0)

        loss = self.contrastive_loss(
            features,
            torch.cat([torch.arange(features_pred.size(0)), torch.arange(features_target.size(0))], dim=0).to(features.device)
        )
        return loss
