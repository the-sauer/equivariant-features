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

import asel
import torch
import torchvision


class AffineEquivariantUNet(torch.nn.Module):
    def __init__(self, channels=None, **_):
        super().__init__()
        if channels is None:
            channels = [10, 20, 40, 80]
        self.conv_down = torch.nn.ModuleList([
            torch.nn.Sequential(
                asel.affine.EquivarLayer(c_in, c_out),
                asel.affine.EquivarLayer(c_out, c_out)
            ) for c_in, c_out in zip([3] + channels[:-1], channels)
        ])
        self.bottleneck = torch.nn.Sequential(
            asel.affine.EquivarLayer(channels[-1], channels[-1] * 2),
            asel.affine.EquivarLayer(channels[-1] * 2, channels[-1] * 2)
        )
        self.conv_up = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    asel.affine.EquivarLayer(c_out * 2, c_out),
                    asel.affine.EquivarLayer(c_out, c_out)
                ),
                asel.affine.EquivarLayer(c_out * 2, c_out, type=["0", "0"], conv_layer=torch.nn.ConvTranspose2d, kernel_size=2, up_stride=2)
            ])
            for c_out in reversed(channels[1:])
        ] + [
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    asel.affine.EquivarLayer(channels[0] * 2, channels[0], conv_layer=torch.nn.ConvTranspose2d),
                    asel.affine.EquivarLayer(channels[0], channels[0], conv_layer=torch.nn.ConvTranspose2d),
                    asel.affine.EquivarLayer(channels[0], 4, type=["0", "c"], conv_layer=torch.nn.ConvTranspose2d)
                ),
                asel.affine.EquivarLayer(channels[0] * 2, channels[0], type=["0", "0"], conv_layer=torch.nn.ConvTranspose2d, kernel_size=2, up_stride=2)
            ])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = []
        for conv in self.conv_down:
            x = conv(x.unsqueeze(1)).squeeze(1)
            residuals.append(x)
            x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=3, sigma=1.0)
            x = torch.nn.functional.max_pool2d(x, 2)
        x = self.bottleneck(x.unsqueeze(1)).squeeze(1)
        for conv, up in self.conv_up:
            x = up(x.unsqueeze(1)).squeeze(1)
            res = residuals.pop()
            x = torch.cat([x, res], dim=1)
            # x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
            x = conv(x.unsqueeze(1)).squeeze(1)
        return x
