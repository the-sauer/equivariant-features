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

import kornia
import torch

from .detector import homogenize, linearize_homography
from .losses.contrastive import Contrastive


def extract_patches(imgs, _homographies, coords, scales, patch_size=64, scale_factor=64):
    device = imgs.device
    blob_normalizations = torch.stack([torch.diag(torch.tensor([1.0 / s, 1.0 / s], dtype=torch.float32)) for s in scales]).to(device)
    patches = kornia.geometry.transform.warp_perspective(
        imgs.expand(-1, 3, -1, -1),
        (
            torch.diag(torch.tensor([patch_size, patch_size, 1.0], dtype=torch.float32)).to(imgs.device).unsqueeze(0)
            @ homogenize(torch.eye(2), b=torch.tensor([0.5, 0.5])).unsqueeze(0).to(imgs.device)
            @ homogenize(torch.eye(2).to(imgs.device).unsqueeze(0) / scales.view(-1, 1, 1) / scale_factor)
            @ homogenize(blob_normalizations)
            @ homogenize(torch.eye(2).to(imgs.device).unsqueeze(0).expand(coords.size(0), -1, -1), b=-coords)
        ),
        dsize=(patch_size, patch_size),
        padding_mode="fill",
        fill_value=torch.tensor([1.0, 1.0, 1.0], device=device),
    )[:, :1]
    return patches


def process_batch_canonicalize(model, data, criterion, augmentation, device, cfg):
    
    # assert "affine_shape" in data, "canonicalization requires ground truth shape to be available"

    patch = data["patch_orig"].to(device) if "patch_orig" in data else extract_patches(data["imgs"], data["homographies"], data["coords"], data["scales"]).to(device)
    label = data["label"].to(device)
    if "homographies" in data:
        affine_shape = linearize_homography(data["homographies"].to(device))[..., :2, :2]
    else:
        affine_shape = data["affine_shape"].to(device)

    A = model.canonicalization(patch)
    if any(isinstance(c, Contrastive) for c, _, _ in criterion):
        canonicalized_patch = kornia.geometry.transform.warp_perspective(
            patch,
            homogenize(A),
            (cfg.training.patch_size, cfg.training.patch_size)
        )
        features = model.descriptor(canonicalized_patch)
        return {
            n: (criterion({
                "features": features,
                "indices": label,
                "detections": A,
                "affine_shape": affine_shape
            }), weight, report) for n, (criterion, weight, report) in criterion.items()
        }
    else:
        return {
            n: (criterion({
                "detections": A,
                "affine_shape": affine_shape
            }), weight, report) for n, (criterion, weight, report) in criterion.items()
        }
