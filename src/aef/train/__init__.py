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


from pytorch_metric_learning import losses as pml_losses
import torch

from .losses.geodesic_loss import GeodesicLoss
from .losses.reprojection_loss import HomographyReprojectionLoss


class RELScaleLoss(torch.nn.Module):
    def __init__(self):
        super(RELScaleLoss, self).__init__()

    def forward(self, pred, gt):
        return torch.mean(torch.abs(pred - gt) / (gt + 1e-8))


OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD
}


LOSSES = {
    "triplet_margin": pml_losses.TripletMarginLoss,
    "npairs": pml_losses.NPairsLoss,
    "supcon": pml_losses.SupConLoss,
    "mse": torch.nn.MSELoss,
    "rel": RELScaleLoss,
    "geodesic": GeodesicLoss,
    "reprojection": HomographyReprojectionLoss
}
