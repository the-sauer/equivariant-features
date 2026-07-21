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


def process_batch_blobs(model, data, criterion, augmentation, device, cfg,
                        validation=False, **_):
    keypoints = data["keypoints"].to(device)
    coords = data["keypoint_coords"].to(device)

    patches = data["patches"].to(device)

    # Optional learned-mask inputs (only present when the dataset has
    # ``precompute_masks=True``). Absent -> the model is called exactly as before,
    # so plain descriptors (HardNet, cartesian) are untouched.
    #
    # During VALIDATION the GT mask is withheld from the network: at test time the
    # mask is unavailable and the model must predict it, so validation must mirror
    # that (feed patches only, no mask / no is_anchor). This also skips the anchor
    # mask-supervision BCE below, which needs the GT mask.
    mask = data["masks"].to(device) if ("masks" in data and not validation) else None
    is_anchor = data["is_anchor"].to(device) if ("is_anchor" in data and not validation) else None

    if mask is not None or is_anchor is not None:
        out = model(patches, mask=mask, is_anchor=is_anchor)
    else:
        out = model(patches)
    # A mask-aware model returns (descriptor, predicted_mask); everything else a tensor.
    features, m_pred = out if isinstance(out, tuple) else (out, None)
    features = features.view(features.size(0), -1)

    # Drop keypoints that fell outside the image frame after warping — their
    # patches are meaningless white fill. Bound by the actual frame size (the
    # collate attaches it as (H, W)); coords are (x, y), so bound by (W, H).
    img_h, img_w = data["image_size"].tolist()
    bound = torch.tensor([img_w, img_h], device=device)
    in_bound_mask = torch.all((coords >= 0) & (coords < bound), dim=-1)
    features = features[in_bound_mask]
    keypoints = keypoints[in_bound_mask]
    losses = {n: (c({"features": features, "indices": keypoints[..., 1]}), w, r)
              for n, (c, w, r) in criterion.items()}

    # Standalone mask loss: supervise the predictor on the TARGET (warped) views against
    # their true board coverage. The anchor's mask is *given* (used directly, not
    # predicted), so it is excluded here; the predictor exists to supply, at test time,
    # the target mask we no longer have. Restricted to in-bound targets — out-of-frame
    # patches are meaningless white fill.
    if m_pred is not None and mask is not None and is_anchor is not None:
        target = (~is_anchor.view(-1).bool()) & in_bound_mask
        if target.any():
            _, _, a_dim, r_dim = m_pred.shape
            gt = torch.nn.functional.adaptive_avg_pool2d(mask, (a_dim, r_dim))
            bce = torch.nn.functional.binary_cross_entropy(
                m_pred[target].clamp(1e-6, 1 - 1e-6), gt[target].clamp(0.0, 1.0)
            )
        else:
            bce = (m_pred * 0.0).sum()
        weight = float(getattr(getattr(cfg, "training", None), "mask_loss_weight", 1.0))
        losses["mask_bce"] = (bce, weight, True)
    return losses
