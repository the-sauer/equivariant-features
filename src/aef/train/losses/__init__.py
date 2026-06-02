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

import pytorch_metric_learning.losses as pml_losses
import torch

from .contrastive import Contrastive, FPR95
from .determinant import Determinant, NonSingular
from .epipolar import EpipolarLoss
from .geodesic_loss import GeodesicLoss
from .image_generation_loss import ImageGeneration
from .rel_scale_loss import RELScaleLoss, RELScaleLossSquared
from .reprojection_loss import Reprojection
from .sift import SiftScaleLoss
from ...evaluate import fpr


class SupCon(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = pml_losses.SupConLoss(**kwargs)

    def forward(self, x, **_):
        return self.criterion(x["features"], x["indices"])
