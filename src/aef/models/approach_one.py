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

from .asel.affine import BasicBlock, EquivarLayer
from .scale import *
from .sift import SIFTNet
from .. import models


class AffineFeatureNetOne(torch.nn.Module):
    """
    An affine feature network that takes an image as input and outputs a feature map of the same size with a specified
    number of channels.

    The architecture consists of a scale field module based on SESN and a feature extraction network based on Affine
    Steerable CNNs (ASEL). The scale field is concatenated to the input image and fed into the feature extraction
    network together with the original image.
    """
    def __init__(self, conv_depths, in_channels=1, feature_size=128, scale_space=None, basic_block_params=None):
        super(AffineFeatureNetOne, self).__init__()
        if scale_space is None:
            scale_space = {"name": models.scale.NeuralScaleSpaceSESN.__name__, "params": {}}
        if basic_block_params is None:
            basic_block_params = {}
        scale_space["params"]["in_channels"] = 1
        self.scale_space = eval(scale_space["name"])[0](**scale_space["params"])
        if "pretrained" in scale_space:
            self.scale_space.load_state_dict(torch.load(scale_space["pretrained"]))
            if scale_space["freeze"]:
                for param in self.scale_space.parameters():
                    param.requires_grad = False
        self.scale_gain = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.feature_net = torch.nn.Sequential(
            BasicBlock(in_channels, conv_depths[0], **basic_block_params),
            *(BasicBlock(conv_depths[i], conv_depths[i+1], **basic_block_params)
              for i in range(len(conv_depths)-1)),
            BasicBlock(conv_depths[-1], feature_size, **basic_block_params),
        )
        self.feature_size = feature_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_field = self.scale_space(torch.mean(x, dim=1, keepdim=True))
        scale_field = self.scale_gain(scale_field)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x, _ = self.feature_net((x, scale_field))
        x = torch.nn.functional.normalize(x, dim=3)
        return x.squeeze(1)


class AffineFeatureNetCanonicalOne(torch.nn.Module):
    def __init__(self, conv_depths, in_channels=1, scale_space=None, basic_block_params=None):
        super(AffineFeatureNetCanonicalOne, self).__init__()
        if scale_space is None:
            scale_space = {"name": models.scale.NeuralScaleSpaceSESN.__name__, "params": {}}
        if basic_block_params is None:
            basic_block_params = {}
        scale_space["params"]["in_channels"] = 1
        self.scale_space = getattr(models, scale_space["name"])(**scale_space["params"])
        if "pretrained" in scale_space:
            self.scale_space.load_state_dict(torch.load(scale_space["pretrained"])["model_state_dict"])
            if scale_space["freeze"]:
                for param in self.scale_space.parameters():
                    param.requires_grad = False
        self.scale_gain = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.feature_net = torch.nn.Sequential(
            BasicBlock(in_channels, conv_depths[0], **basic_block_params),
            *(BasicBlock(conv_depths[i], conv_depths[i+1], **basic_block_params)
              for i in range(len(conv_depths)-1)),
        )
        self.canonicalization_layer = EquivarLayer(conv_depths[-1], 4, type=["0", "c"])
        self.injected_scale_field = None
        self.descriptor_model = SIFTNet()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.injected_scale_field is None:
            scale_field = self.scale_space(torch.mean(x, dim=1, keepdim=True))
            scale_field = self.scale_gain(scale_field)
        else:
            scale_field = self.injected_scale_field
        if x.dim() == 4:
            x = x.unsqueeze(1)
        assert x.size(1) == 1
        x = self.feature_net(x)
        x = torch.nn.functional.normalize(x, dim=3)
        x = self.canonicalization_layer((x, scale_field))[0]
        return x.squeeze(1)

    def inject_scale_field(self, scale_field):
        self.injected_scale_field = scale_field
