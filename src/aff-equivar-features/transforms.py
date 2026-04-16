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


def random_affine(n=1, scale=True, min_scale=0.1, max_scale=1.0,translate=True, rotate=True) -> torch.tensor:
    # TODO: Implement
    scale = torch.rand(n, dtype=torch.float32) * (max_scale - min_scale) + min_scale
    transform = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(n, 1, 1)
    transform[..., 0, 0] = scale
    transform[..., 1, 1] = scale
    return transform


# def random_scale(n)