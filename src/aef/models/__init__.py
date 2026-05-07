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

from math import ceil

import asel
import sesn
import torch

from aef.train.descriptor import train as train_descriptor
from aef.train.detector import train as train_detector
from aef.train.scale import train as train_scale


def NeuralScaleSpaceSESN(in_channels:int=1, factor:float=2.0, num_scales:int=4, min_scale:float=1.0, effective_size:int=5, scale_size:int=5) -> torch.nn.Module:
    """
    Neural Scale Field based on Scale-Equivariant Steerable Networks (SESN).
    """
    if min_scale < 1:
        raise ValueError("min_scale must be at least 1.")
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    kernel_size= ceil(effective_size * max(scales))
    if kernel_size % 2 == 0:
        kernel_size += 1
    padding = kernel_size // 2
    layer_kwargs = {"effective_size": effective_size, "kernel_size": kernel_size, "scales": scales, "padding": padding}
    return torch.nn.Sequential(
        sesn.SESConv_Z2_H(in_channels, 8, **layer_kwargs),
        sesn.SESConv_H_H(8, 16, scale_size, **layer_kwargs),
        sesn.SESConv_H_H(16, 32, scale_size, **layer_kwargs),
        sesn.SESConv_H_H_1x1(32, 1, num_scales=num_scales),
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
            asel.affine.BasicBlock(in_channels, conv_depths[0], **basic_block_params),
            *(asel.affine.BasicBlock(conv_depths[i], conv_depths[i+1], **basic_block_params) for i in range(len(conv_depths)-1)),
            asel.affine.BasicBlock(conv_depths[-1], feature_size, **basic_block_params),
        )
        self.feature_size = feature_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_dim = 1
        scale_field = self.scale_space(torch.mean(x, dim=channel_dim, keepdim=True))
        scale_field = self.scale_gain(scale_field)
        x, _ = self.feature_net((x, scale_field))
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
            asel.affine.BasicBlock(in_channels, conv_depths[0], **basic_block_params),
            *(asel.affine.BasicBlock(conv_depths[i], conv_depths[i+1], **basic_block_params) for i in range(len(conv_depths)-1)),
        )
        self.canonicalization_layer = asel.affine.EquivarLayer(conv_depths[-1], 4, type=["0", "c"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_field = self.scale_space(torch.mean(x, dim=1, keepdim=True))
        scale_field = self.scale_gain(scale_field)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        assert x.size(1) == 1
        x, _ = self.feature_net((x, scale_field))
        x = torch.nn.functional.normalize(x, dim=2)
        x = self.canonicalization_layer(x)
        return x.squeeze(1)
    

class AffineFeatureNetCanonicalTwo(torch.nn.Module):
    def __init__(self, scales:list[float | int] | dict[str, float], in_channels=1, conv_depths=[8, 16, 32, 64],  basic_block_params=None):
        super(AffineFeatureNetCanonicalTwo, self).__init__()
        if isinstance(scales, list):
            scale_list: list[float] = scales
        elif isinstance(scales, dict) and "min_scale" in scales and "factor" in scales and "num_scales" in scales:
            q = scales["factor"] ** (1 / (scales["num_scales"] - 1))
            scale_list: list[float] = [scales["min_scale"] * q**i for i in range(scales["num_scales"])]
        else:
            raise ValueError("scales must be either a list of scales or a dictionary with keys 'min_scale', 'factor' and 'num_scales'.")

        if basic_block_params is None:
            basic_block_params = {}

        self.feature_net = torch.nn.Sequential(
            asel.affine.BasicBlock(in_channels, conv_depths[0], scales=scale_list, **basic_block_params),
            *(asel.affine.BasicBlock(conv_depths[i], conv_depths[i+1], scales=scale_list, **basic_block_params) for i in range(len(conv_depths)-1)),
            asel.affine.EquivarLayer(conv_depths[-1], 4, scales=scale_list, type=["0", "c"]),
            asel.affine.ScaleProjectionLayer(scales=scale_list),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        assert x.size(1) == 1

        return self.feature_net(x)


"""
All model definition with their respective training functions are collected in this dictionary for easier access.
"""
MODELS = {
    "scale_space_sesn": (NeuralScaleSpaceSESN, train_scale),
    "affine_feature_net_one": (AffineFeatureNetOne, train_descriptor),
    "affine_feature_net_canonical_one": (AffineFeatureNetCanonicalOne, train_detector),
    "affine_feature_net_canonical_two": (AffineFeatureNetCanonicalTwo, train_detector)
}
