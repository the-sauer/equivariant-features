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

import logging

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

    def forward(self, x) -> torch.Tensor:
        if matches := x.get("matches") is None:
            logging.warning("No matches found in input, returning zero loss")
            return torch.tensor(0.0, device=x["indices"].device)
        if "indices" in x:
            assert torch.all(x["indices"][x["matches"][:, 0]] == x["indices"][x["matches"][:, 1]])

        matches = x["matches"]
        assert torch.all(matches[:, 0] < matches[:, 1])
        detections = x["detections"]
        img_ids = x["img_ids"]
        fundamental = x["fundamental"]
        pts = x["pts"]
        A_1 = detections[matches[:, 0]]
        pt_1 = pts[matches[:, 0]]
        A_2 = detections[matches[:, 1]]
        pt_2 = pts[matches[:, 1]]
        F = fundamental[img_ids[matches[:, 0]], img_ids[matches[:, 1]]]
        B = F.size(0)
        assert F.size() == (B, 3, 3)
        assert A_1.size() == (B, 3, 3)
        assert A_2.size() == (B, 3, 3)
        assert pt_1.size() == (B, 2)
        assert pt_2.size() == (B, 2)

        non_nan_mask = ~(torch.isnan(A_1).any(dim=(1, 2)) | torch.isnan(A_2).any(dim=(1, 2)) | torch.isnan(pt_1).any(dim=1) | torch.isnan(pt_2).any(dim=1) | torch.isnan(F).any(dim=(1, 2)))
        if torch.any(~non_nan_mask):
            logging.warning("Found %d samples with NaN values, skipping them in the loss computation", (~non_nan_mask).sum().item())
        A_1 = A_1[non_nan_mask]
        A_2 = A_2[non_nan_mask]
        pt_1 = pt_1[non_nan_mask]
        pt_2 = pt_2[non_nan_mask]
        F = F[non_nan_mask]

        A_rel = A_2 @ torch.linalg.inv(A_1)
        A_rel_inv = torch.linalg.inv(A_rel)

        pt_1 = pt_1.unsqueeze(1) #+ self.sampling_distribution.sample((pt_1.shape[0], self.n_samples, 2)).to(pt_1.device)
        pt_1 = torch.stack((pt_1[..., 0], pt_1[..., 1], torch.ones_like(pt_1[:, :, 1])), dim=-1).unsqueeze(-1)

        pt_2 = pt_2.unsqueeze(1) #+ self.sampling_distribution.sample((pt_2.shape[0], self.n_samples, 2)).to(pt_2.device)
        pt_2 = torch.stack((pt_2[..., 0], pt_2[..., 1], torch.ones_like(pt_2[:, :, 1])), dim=-1).unsqueeze(-1)
        pt_2_t = pt_2.transpose(-2, -1)
        mismatched = ((pt_2_t.squeeze(1) @ F @ pt_1.squeeze(1)) > 2).squeeze()
        if torch.any(mismatched) and mismatched.dim() > 0:
            logging.error("Found (%d/%d) mismatched points with epipolar error > 2", mismatched.int().sum(), len(mismatched))
            print(f"\033[91mFound ({mismatched.int().sum()}/{len(mismatched)}) mismatched points with epipolar error > 2\033[0m")

        pt_1 = pt_1.repeat(1, self.n_samples, 1, 1)
        pt_1[:, :, :2] += self.sampling_distribution.sample((pt_1.shape[0], self.n_samples, 2, 1)).to(pt_1.device)
        pt_2 = pt_2.repeat(1, self.n_samples, 1, 1)
        pt_2[:, :, :2] += self.sampling_distribution.sample((pt_2.shape[0], self.n_samples, 2, 1)).to(pt_2.device)
        pt_2_t = pt_2.transpose(-2, -1)
        epipolar_error = (
            ((A_rel.unsqueeze(1) @ pt_1).transpose(-2, -1) @ F.unsqueeze(1) @ pt_1)[~mismatched]
            + (pt_2_t @ F.unsqueeze(1) @  A_rel_inv.unsqueeze(1) @ pt_2)[~mismatched]
        )
        epipolar_error = epipolar_error.squeeze(-1).squeeze(-1)
        return self.reduction(epipolar_error.abs().mean(dim=1))
