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


def compute_scale(H: torch.Tensor, size: tuple) -> torch.Tensor:
    N = H.shape[0]
    device = H.device

    # 1. Create coordinate grid
    x = torch.arange(size[0], dtype=torch.float32, device=device)
    y = torch.arange(size[1], dtype=torch.float32, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')

    # Flatten and stack to (3, num_pixels)
    coords = torch.stack([
        x_grid.reshape(-1),
        y_grid.reshape(-1),
        torch.ones_like(x_grid).reshape(-1)
    ], dim=0)

    # 2. Transform coordinates: H @ coords
    # Expand coords to match batch size: (N, 3, num_pixels)
    coords_batched = coords.unsqueeze(0).expand(N, -1, -1)
    transformed = torch.bmm(H, coords_batched)

    Nx = transformed[:, 0, :]  # x'
    Ny = transformed[:, 1, :]  # y'
    D = transformed[:, 2, :]  # w'

    D_sq = D * D

    # 3. Extract matrix elements for the Jacobian
    # Reshaped to (N, 1) for easy broadcasting against (N, num_pixels)
    H00 = H[:, 0:1, 0:1].view(N, 1)
    H01 = H[:, 0:1, 1:2].view(N, 1)
    H10 = H[:, 1:2, 0:1].view(N, 1)
    H11 = H[:, 1:2, 1:2].view(N, 1)
    H20 = H[:, 2:3, 0:1].view(N, 1)
    H21 = H[:, 2:3, 1:2].view(N, 1)

    # 4. Compute Jacobian elements (Quotient Rule)
    du_dx = (H00 * D - H20 * Nx) / D_sq
    du_dy = (H01 * D - H21 * Nx) / D_sq
    dv_dx = (H10 * D - H20 * Ny) / D_sq
    dv_dy = (H11 * D - H21 * Ny) / D_sq

    # 5. Determinant and Scale
    det_J = (du_dx * dv_dy) - (du_dy * dv_dx)
    scale = torch.sqrt(torch.abs(det_J))

    # Reshape back to spatial dimensions
    return scale.view(N, 1, size[0], size[1])


def process_batch_scale_homographic(model, data, criterion, augmentation, device, cfg):
    b, b_t, H, H_inv = map(lambda x: x.to(device), data)
    b = augmentation(b)
    b_t = augmentation(b_t)
    gt = compute_scale(H, b.shape[2:])

    o = model(b)
    o_t = model(b_t)
    o_t = kornia.geometry.transform.warp_perspective(o_t, H_inv, b.shape[2:])

    return criterion(o_t / o, gt)


def process_batch_scale_colmap(model, data, criterion, augmentation, device, cfg):
    scales = data["scales"].to(device)
    keypoints = data["keypoints"].to(device)
    coords = data["keypoint_coords"].to(device)

    img_ids = keypoints[:, 0].long().unique()

    pred_list = []
    gt_list = []
    for i in range(0, len(img_ids), cfg.training.image_batch_size):
        img_batch_ids = img_ids[i:min(i+cfg.training.image_batch_size, len(img_ids))]
        img_batch = augmentation(torch.stack([data["images"][img_id.item()] for img_id in img_batch_ids]).to(device))
        out = model(img_batch)
        for j in range(img_batch.size(0)):
            mask = keypoints[:, 0] == img_batch_ids[j]
            xy = coords[mask].round().long()
            pred_list.append(out[j, 0, xy[:, 1], xy[:, 0]])
            gt_list.append(scales[mask])

    return {
        n: (criterion({"pred_scales": torch.cat(pred_list), "scales": torch.cat(gt_list)}), w, r)
        for n, (criterion, w, r) in criterion.items()
    }
