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


def process_batch_blobs(model, data, criterion, augmentation, device, cfg, **_):
    keypoints = data["keypoints"].to(device)
    coords = data["keypoint_coords"].to(device)

    patches = data["patches"].to(device)

    features = model(patches)
    features = features.view(features.size(0), -1)

    # Drop keypoints that fell outside the image frame after warping — their
    # patches are meaningless white fill. Bound by the actual frame size (the
    # collate attaches it as (H, W)); coords are (x, y), so bound by (W, H).
    img_h, img_w = data["image_size"].tolist()
    bound = torch.tensor([img_w, img_h], device=device)
    in_bound_mask = torch.all((coords >= 0) & (coords < bound), dim=-1)
    features = features[in_bound_mask]
    keypoints = keypoints[in_bound_mask]
    return {n: (c({"features": features, "indices": keypoints[..., 1]}), w, r) for n, (c, w, r) in criterion.items()}
