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


class EpipolarLoss(torch.nn.Module):
    def __init__(self, n_samples: int, sampling_distribution: str = "gaussian", reduction="mean", **_):
        super().__init__()
        self.n_samples = n_samples
        if sampling_distribution == "gaussian":
            self.sampling_distribution = torch.distributions.Normal(0, 1)
        else:
            raise ValueError(f"Unsupported sampling distribution: {sampling_distribution}")
        if reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

    def forward(self, A: torch.Tensor, E: torch.Tensor, pt: torch.Tensor) -> torch.Tensor:
        B = A.size(0)
        assert A.size() == (B, 3, 3)
        assert E.size() == (B, 3, 3)
        assert pt.size() == (B, 2)
        pts = pt.unsqueeze(1) + self.sampling_distribution.sample((pt.shape[0], self.n_samples, 2)).to(pt.device)
        pts = torch.cat((pts, torch.ones_like(pts[:, :, :1])), dim=-1).unsqueeze(-1)
        pts_t = pts.transpose(-2, -1)
        epipolar_error = pts_t @ E.unsqueeze(1) @ A.unsqueeze(1) @ pts
        epipolar_error = epipolar_error.squeeze(-1).squeeze(-1)
        return self.reduction(epipolar_error.abs().mean(dim=1))
