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

import kornia
import torch
import torchvision


class ImageGenerationLoss(torch.nn.Module):
    def __init__(
        self,
        distance_metric: str = "mse",
        patch_size: tuple[int, int] = (64, 64),
        sigma: float = 1,
    ):
        super().__init__()
        if distance_metric == "mse":
            self.image_distance_metric = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported image distance metric: {distance_metric}")
        self.patch_size = patch_size
        self.patch_scale = torch.diag(
            torch.Tensor([patch_size[0], patch_size[1], 1]).to(torch.float32)
        ).unsqueeze(0)
        kernel_size = ceil(sigma * 4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        target: tuple[torch.Tensor, torch.Tensor],
        num_features: torch.Tensor
    ) -> torch.Tensor:
        patch_scale = self.patch_scale.to(pred[0].device)
        assert pred[0].dim() == 4
        assert pred[1].dim() == 4
        assert target[0].dim() == 4
        assert target[1].dim() == 4
        pred_transform, pred_image = pred
        pred_patch = torch.cat([kornia.geometry.transform.warp_perspective(
            torchvision.transforms.functional.gaussian_blur(
                pred_image[i].unsqueeze(0).expand(int(num_features[i]), -1, -1, -1),
                kernel_size=self.kernel_size,
                sigma=self.sigma
            ),
            patch_scale @ torch.linalg.inv(pred_transform[i, :num_features[i]]),
            dsize=self.patch_size
        ) for i in range(pred[0].size(0))])
        target_transform, target_image = target
        target_patch = torch.cat([kornia.geometry.transform.warp_perspective(
            torchvision.transforms.functional.gaussian_blur(
                target_image[i].unsqueeze(0).expand(int(num_features[i]), -1, -1, -1), kernel_size=self.kernel_size, sigma=self.sigma),
            patch_scale @ torch.linalg.inv(target_transform[i, :num_features[i]]),
            dsize=self.patch_size
        ) for i in range(target[0].size(0))])

        image_loss = self.image_distance_metric(pred_patch, target_patch)
        return image_loss
