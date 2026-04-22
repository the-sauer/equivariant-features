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


def random_affine(n=1, scale=True, min_scale=0.1, max_scale=1.0, translate=True, rotate=True, size=(128, 128)) -> torch.Tensor:
    # TODO: Implement

    origin_translation = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    origin_translation[..., 0, 2] = size[1] / 2
    origin_translation[..., 1, 2] = size[0] / 2
    origin_translation = origin_translation.expand(n, -1, -1)

    if scale:
        scale = torch.rand(n, dtype=torch.float32) * (max_scale - min_scale) + min_scale
        scale_mat = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
        scale_mat[..., 0, 0] = scale
        scale_mat[..., 1, 1] = scale
    else:
        scale_mat = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1)

    if rotate:
        rotation = torch.rand(n, dtype=torch.float32) * 2 * torch.pi
        rot_mat = torch.stack([
            torch.stack([torch.cos(rotation), -torch.sin(rotation), torch.zeros(1,).expand(n)], dim=-1),
            torch.stack([torch.sin(rotation), torch.cos(rotation), torch.zeros(1,).expand(n)], dim=-1),
            torch.stack([torch.zeros(1,).expand(n), torch.zeros(1,).expand(n), torch.ones(1,).expand(n)], dim=-1)

        ], dim=-1)
    else:
        rot_mat = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(n, -1, -1)

    return origin_translation @ scale_mat @ rot_mat @ (-origin_translation)
