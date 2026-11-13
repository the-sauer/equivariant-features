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

"""
This module contains the implementations of the models used in this thesis.
"""

import torch
import lightglue
import gluefactory

from .hardnet import *
try:
    from .steerable import *
except ImportError:
    pass


class LightGlueDescWrapper(torch.nn.Module):
    def __init__(self, lightglue_conf, descriptor_conf):
        super().__init__()
        lightglue_conf["input_dim"] = descriptor_conf.get("descriptor_dim", 128)
        self.graph = gluefactory.models.get_model("lightglue")(lightglue_conf)
        self.descriptor = HardNetLogPolar(**descriptor_conf)

    def forward(self, data):
        desc0 = self.descriptor(data["patches0"])
        desc1 = self.descriptor(data["patches1"])
        return self.graph({
            "descriptors0": desc0,
            "descriptors1": desc1,
            # normalized keypoint coordinates
            "view0": {"image_size": (1, 1)},
            "view1": {"image_size": (1, 1)},
            **data
        })
