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

import asel
import sesn
import torch

from aef.train.descriptor import train as train_descriptor
from aef.train.detector import train as train_detector
from aef.train.scale import train_scale


def NeuralScaleSpaceSESN(in_channels=1, factor=2.0, num_scales=4, min_scale=1.0) -> torch.nn.Module:
    """
    Neural Scale Field based on Scale-Equivariant Steerable Networks (SESN).
    """
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    return torch.nn.Sequential(
        sesn.SESConv_Z2_H(in_channels, 8, 11, 7, scales, padding=5),
        sesn.SESConv_H_H(8, 16, 5, 11, 7, scales, padding=5),
        sesn.SESConv_H_H(16, 32, 5, 11, 7, scales, padding=5),
        sesn.SESConv_H_H_1x1(32, 1, num_scales=len(scales)),
        sesn.SESArgMaxProjection(scales)
    )


class AffineFeatureNetOne(torch.nn.Module):
    """
    An affine feature network that takes an image as input and outputs a feature map of the same size with a specified
    number of channels.

    The architecture consists of a scale field module based on SESN and a feature extraction network based on Affine
    Steerable CNNs (ASEL). The scale field is concatenated to the input image and fed into the feature extraction
    network together with the original image.
    """
    def __init__(self, in_channels=1, feature_size=128, conv_depths=[8, 16, 32, 64], scale_space=None, basic_block_params=None):
        super(AffineFeatureNetOne, self).__init__()
        if scale_space is None:
            scale_space = {"name": "scale_space_sesn", "params": {}}
        if basic_block_params is None:
            basic_block_params = {}
        scale_space["params"]["in_channels"] = 1
        self.scale_space = MODELS[scale_space["name"]][0](**scale_space["params"])
        if "pretrained" in scale_space:
            self.scale_space.load_state_dict(torch.load(scale_space["pretrained"]))
            if scale_space["freeze"]:
                for param in self.scale_space.parameters():
                    param.requires_grad = False
        self.scale_gain = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.feature_net = torch.nn.Sequential(
            asel.affine.BasicBlock(in_channels + 1, conv_depths[0], **basic_block_params),
            *(asel.affine.BasicBlock(conv_depths[i], conv_depths[i+1], **basic_block_params) for i in range(len(conv_depths)-1)),
            asel.affine.BasicBlock(conv_depths[-1], feature_size, **basic_block_params),
        )
        self.feature_size = feature_size

    def forward(self, x):
        channel_dim = 1
        scale_field = self.scale_space(torch.mean(x, dim=channel_dim, keepdim=True))
        scale_field = self.scale_gain(scale_field)
        x = torch.concat((x, scale_field), dim=channel_dim)
        x = self.feature_net(x)
        x = torch.nn.functional.normalize(x, dim=channel_dim)
        return x


class AffineFeatureNetCanonicalOne(torch.nn.Module):
    def __init__(self, in_channels=1, conv_depths=[8, 16, 32, 64], scale_space=None, basic_block_params=None):
        super(AffineFeatureNetCanonicalOne, self).__init__()
        if scale_space is None:
            scale_space = {"name": "scale_space_sesn", "params": {}}
        if basic_block_params is None:
            basic_block_params = {}
        scale_space["params"]["in_channels"] = 1
        self.scale_space = MODELS[scale_space["name"]][0](**scale_space["params"])
        if "pretrained" in scale_space:
            self.scale_space.load_state_dict(torch.load(scale_space["pretrained"]))
            if scale_space["freeze"]:
                for param in self.scale_space.parameters():
                    param.requires_grad = False
        self.scale_gain = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.feature_net = torch.nn.Sequential(
            asel.affine.BasicBlock(in_channels + 1, conv_depths[0], **basic_block_params),
            *(asel.affine.BasicBlock(conv_depths[i], conv_depths[i+1], **basic_block_params) for i in range(len(conv_depths)-1)),
        )
        self.canonicalization_layer = asel.affine.EquivarLayer(conv_depths[-1] + 1, 4, type=["0", "c"])

    def forward(self, x):
        channel_dim = 1
        scale_field = self.scale_space(torch.mean(x, dim=channel_dim, keepdim=True))
        scale_field = self.scale_gain(scale_field)
        x = torch.concat((x, scale_field), dim=channel_dim)
        x = self.feature_net(x)
        x = torch.nn.functional.normalize(x, dim=channel_dim)
        x = torch.concat((x, scale_field), dim=channel_dim)
        x = self.canonicalization_layer(x)
        return x


""""
All model definition with their respective training functions are collected in this dictionary for easier access.
"""
MODELS = {
    "scale_space_sesn": (NeuralScaleSpaceSESN, train_scale),
    "affine_feature_net_one": (AffineFeatureNetOne, train_descriptor),
    "affine_feature_net_canonical_one": (AffineFeatureNetCanonicalOne, train_detector),
}
