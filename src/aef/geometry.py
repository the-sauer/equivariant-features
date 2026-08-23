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

"""Small geometry helpers shared by the data pipeline and the training tasks.

These used to live in ``aef.train.detector`` (whose detector task has been
removed); they sit here so ``aef.data`` does not have to import from ``aef.train``.
"""

import torch


def linearize_homography(H, shape=None, coords=None, stride=1):
    if coords is None:
        if shape is None:
            raise ValueError("Either shape or coords must be provided")
        coords = torch.stack(
            (
                *torch.meshgrid(
                    torch.arange(start=0, end=shape[0], step=stride, device=H.device),
                    torch.arange(start=0, end=shape[1], step=stride, device=H.device), indexing="ij"
                ),
                torch.ones((1, 1), device=H.device).expand(shape[0] // stride, shape[1] // stride)
            ),
            dim=2
        )
        coords = coords.unsqueeze(0)
    coords = coords.unsqueeze(-1)
    proj = H @ coords
    x = proj[..., 0, 0]
    y = proj[..., 1, 0]
    w = proj[..., 2, 0]
    return torch.stack((
        torch.stack(
            ((H[..., 0, 0] * w - H[..., 2, 0] * x) / w ** 2, (H[..., 1, 0] * w - H[..., 2, 0] * y) / w ** 2),
            dim=-1
        ),
        torch.stack(
            ((H[..., 0, 1] * w - H[..., 2, 1] * x) / w ** 2, (H[..., 1, 1] * w - H[..., 2, 1] * y) / w ** 2),
            dim=-1
        )
    ), dim=-1)
