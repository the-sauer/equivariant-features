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
import omegaconf
import torch

from ...evaluate import fpr
from .determinant import DeterminantLoss
from .epipolar import EpipolarLoss
from .geodesic_loss import GeodesicLoss
from .image_generation_loss import ImageGenerationLoss
from .rel_scale_loss import RELScaleLoss, RELScaleLossSquared
from .reprojection_loss import HomographyReprojectionLoss


_LOSSES = {
    "triplet_margin": pml_losses.TripletMarginLoss,
    "npairs": pml_losses.NPairsLoss,
    "supcon": pml_losses.SupConLoss,
    "mse": torch.nn.MSELoss,
    "rel": RELScaleLoss,
    "rel_squared": RELScaleLossSquared,
    "geodesic": GeodesicLoss,
    "reprojection": HomographyReprojectionLoss,
    "img_gen": ImageGenerationLoss,
    "fpr": fpr,
    "epipolar": EpipolarLoss,
    "determinant": DeterminantLoss,
}


class Loss(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Loss, self).__init__()
        if len(args) == 1 and isinstance(args[0], str):
            [arg] = args
            self.losses = [(1, _LOSSES[arg](**kwargs.get("params", {})))]
        elif isinstance(args[0], dict | omegaconf.ListConfig):
            self.losses = [
                (loss_cfg.get("weight", 1), _LOSSES[loss_cfg["name"]](**loss_cfg.get("params", {})))
                for loss_cfg in args[0]
            ]
        elif isinstance(args[0], dict | omegaconf.DictConfig):
            [loss_cfg] = args
            self.losses = [loss_cfg.get("weight", 1), _LOSSES[loss_cfg["name"]](**loss_cfg.get("params", {}))]

    def forward(self, *args):
        return sum(w * loss(*args) for w, loss in self.losses)
