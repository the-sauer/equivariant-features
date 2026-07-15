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

from ..geometry import homogenize
from .losses.blob import BlobLoss
from .losses.contrastive import Contrastive


def extract_patches(imgs, _homographies, coords, scales, patch_size=64, scale_factor=16):
    device = imgs.device
    patches = kornia.geometry.transform.warp_perspective(
        imgs.expand(-1, 3, -1, -1),
        (
            torch.diag(torch.tensor([patch_size, patch_size, 1.0], dtype=torch.float32)).to(imgs.device).unsqueeze(0)
            @ homogenize(torch.eye(2), b=torch.tensor([0.5, 0.5])).unsqueeze(0).to(imgs.device)
            @ homogenize(torch.eye(2).to(imgs.device).unsqueeze(0) / scales.view(-1, 1, 1) / scale_factor)
            # @ homogenize(blob_normalizations)
            @ homogenize(torch.eye(2).to(imgs.device).unsqueeze(0).expand(coords.size(0), -1, -1), b=-coords)
        ),
        dsize=(patch_size, patch_size),
        padding_mode="fill",
        fill_value=torch.tensor([1.0, 1.0, 1.0], device=device),
    )[:, :1]
    return patches


def process_batch_canonicalize(model, data, criterion, _augmentation, device, cfg, optimizer=None):
    # assert "affine_shape" in data, "canonicalization requires ground truth shape to be available"
    with torch.no_grad():
        img = data["images"]
        keypoints = data["keypoints"].to(device)
        keypoint_coords = data["keypoint_coords"].to(device)
        scales = data["scales"].to(device)

        _, *img_size = next(iter(img.values())).size()
        label = data["keypoints"][..., 1].to(device)
        if "homographies" in data:
            affine_shape = data["homographies"][..., :2, :2].to(device)
        else:
            affine_shape = data["affine_shape"].to(device)

        coordinate_in_bound_mask = (
            (keypoint_coords >= 0).all(dim=1)
            & (keypoint_coords[:, 0] < img_size[1])
            & (keypoint_coords[:, 1] < img_size[0])
        ).cpu()
        if coordinate_in_bound_mask.sum() == 0:
            return {
                n: (criterion({
                    "pred": torch.empty(0, 2, 2, device=device),
                    "target": torch.empty(0, 2, 2, device=device)
                }), weight, report) for n, (criterion, weight, report) in criterion.items()
            }
        keypoints = keypoints[coordinate_in_bound_mask]
        keypoint_coords = keypoint_coords[coordinate_in_bound_mask]
        scales = scales[coordinate_in_bound_mask]
        label = label[coordinate_in_bound_mask]
        affine_shape = affine_shape[coordinate_in_bound_mask]
        if "patch_orig" in data:
            patch = data["patch_orig"].to(device)
        else:
            patch_images = torch.stack(
                [img[img_id.item()].to(device) for img_id in keypoints[..., 0]],
                dim=0,
            )
            patch = extract_patches(
                patch_images,
                data["homographies"],
                keypoint_coords,
                scales,
            ).to(device)
    if optimizer is not None:
        for opt in optimizer.values():
            opt.zero_grad()
    A = model.canonicalizer(patch)
    is_nan_mask = torch.isnan(A).any(dim=(1, 2))
    A = A[~is_nan_mask]
    patch = patch[~is_nan_mask]
    label = label[~is_nan_mask]
    affine_shape = affine_shape[~is_nan_mask]
    canonicalized_patch = None
    needs_canonicalized_patch = any(isinstance(c, (Contrastive, BlobLoss)) for c, _, _ in criterion.values())
    if needs_canonicalized_patch:
        canonicalized_patch = kornia.geometry.transform.warp_perspective(
            patch,
            homogenize(A),
            (cfg.training.patch_size, cfg.training.patch_size)
        )

    if any(isinstance(c, Contrastive) for c, _, _ in criterion.values()):
        features = model.descriptor(canonicalized_patch)

    losses = {}
    for n, (loss_fn, weight, report) in criterion.items():
        if isinstance(loss_fn, BlobLoss):
            losses[n] = (loss_fn({
                "patches": canonicalized_patch,
                "device": device,
            }), weight, report)
        elif isinstance(loss_fn, Contrastive):
            losses[n] = (loss_fn({
                "features": features,
                "indices": label,
                "pred": A,
                "target": affine_shape
            }), weight, report)
        else:
            losses[n] = (loss_fn({
                "pred": A,
                "target": affine_shape
            }), weight, report)
    return losses
