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

from math import exp, sqrt

import torch
from matplotlib import pyplot as plt


class BlobLoss(torch.nn.Module):
    def __init__(self, image_distance_metric="mse", patch_size=(65, 65), **_):
        super().__init__()
        self.image_distance_metric = torch.nn.MSELoss()
        self.blob = torch.empty((1, patch_size[0], patch_size[1]), dtype=torch.float32)
        self.blob_mask = torch.empty((1, patch_size[0], patch_size[1]), dtype=torch.bool)
        sigma = patch_size[0] / 4
        for i in range(patch_size[0]):
            for j in range(patch_size[1]):
                r = sqrt((i - patch_size[0] / 2) ** 2 + (j - patch_size[1] / 2) ** 2)
                self.blob[0, i, j] = (1 - exp(-r ** 2 / (2 * sigma ** 2))) / (1 - exp(-3 * sigma ** 2 / 2))
                self.blob_mask[0, i, j] = r <= sigma * 2
        plt.imsave("/data/debug_imgs/blob.png", self.blob[0].cpu().numpy())

    def forward(self, x):
        if "patches" in x and x["patches"] is not None:
            return self.image_distance_metric(
                    self.blob.unsqueeze(0).expand(x["patches"].shape[0], -1, -1, -1).to(x["device"]),
                    x["patches"]
            )
        else:
            return torch.tensor([1e9]).to(x["device"])
