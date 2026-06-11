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

import omegaconf
import torch
import torchvision


from . import scale
from .asel.affine import EquivarLayer
from .hardnet import HardNet
from .sift import SIFTNet


class AffineEquivariantUNet(torch.nn.Module):
    def __init__(self, channels=None, scale_space=None, **_):
        super().__init__()
        if channels is None:
            channels = [10, 20, 40, 80]
        if scale_space is None:
            scale_space = omegaconf.DictConfig({"name": "ConstantScaleSpace", "params": {}})

        self.scale_space = getattr(scale, scale_space["name"])(**scale_space["params"])
        self.scale_gain = torch.nn.Conv2d(1, 1, kernel_size=1, bias=True)  # Learnable gain for the scale space output

        self.conv_down = torch.nn.ModuleList([
            torch.nn.Sequential(
                EquivarLayer(c_in, c_out),
                EquivarLayer(c_out, c_out)
            ) for c_in, c_out in zip([3] + channels[:-1], channels)
        ])

        self.bottleneck = torch.nn.Sequential(
            EquivarLayer(channels[-1], channels[-1] * 2),
            EquivarLayer(channels[-1] * 2, channels[-1] * 2)
        )

        self.conv_up = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    EquivarLayer(c_out * 2, c_out),
                    EquivarLayer(c_out, c_out)
                ),
                EquivarLayer(
                    c_out * 2, c_out,
                    type=["0", "0"],
                    conv_layer=torch.nn.ConvTranspose2d,
                    kernel_size=2,
                    up_stride=2
                )
            ])
            for c_out in reversed(channels[1:])
        ] + [
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    EquivarLayer(channels[0] * 2, channels[0], conv_layer=torch.nn.ConvTranspose2d),
                    EquivarLayer(channels[0], channels[0], conv_layer=torch.nn.ConvTranspose2d)
                ),
                EquivarLayer(
                    channels[0] * 2, channels[0],
                    type=["0", "0"],
                    conv_layer=torch.nn.ConvTranspose2d,
                    kernel_size=2,
                    up_stride=2
                )
            ])
        ])
        self.out_layer = EquivarLayer(channels[0], 4, type=["0", "c"], conv_layer=torch.nn.ConvTranspose2d)
        self.descriptor_model = SIFTNet()
        self.injected_scale_field = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []
        if self.injected_scale_field is not None:
            scale_field = self.injected_scale_field.to(x.device)
            self.injected_scale_field = None
        else:
            scale_field = self.scale_space(x)
        scale_field += 1e-6  # Avoid zero scales
        for i, conv in enumerate(self.conv_down):
            x = conv(x.unsqueeze(1)).squeeze(1)
            if torch.isnan(x).any():
                print(f"\033[93mWarning: Found NaN values in the feature maps after conv_down layer {i}\033[0m")
                print(f"{torch.isnan(x).any()=}, {torch.isinf(x).any()=}, {x.shape=}")
            residuals.append(x)
            x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=3, sigma=1.0)
            x = torch.nn.functional.max_pool2d(x, 2)
        x = self.bottleneck(x.unsqueeze(1)).squeeze(1)
        if torch.isnan(x).any():
            print(f"\033[93mWarning: Found NaN values in the feature maps after bottleneck\033[0m")
            print(f"{torch.isnan(x).any()=}, {torch.isinf(x).any()=}, {x.shape=}")
        for i, (conv, up) in enumerate(self.conv_up):
            x = up(x.unsqueeze(1)).squeeze(1)
            if torch.isnan(x).any():
                print(f"\033[93mWarning: Found NaN values in the feature maps after upsampling layer {i}\033[0m")
                print(f"{torch.isnan(x).any()=}, {torch.isinf(x).any()=}, {x.shape=}")
            res = residuals.pop()
            x = torch.cat([x, res], dim=1)
            x = conv(x.unsqueeze(1)).squeeze(1)
            if torch.isnan(x).any():
                print(f"\033[93mWarning: Found NaN values in the feature maps after conv_up layer {i}\033[0m")
                print(f"{torch.isnan(x).any()=}, {torch.isinf(x).any()=}, {x.shape=}")
        x = self.out_layer((x.unsqueeze(1), scale_field))[0].squeeze(1)
        return x

    def inject_scale_field(self, scale_field):
        self.injected_scale_field = scale_field
