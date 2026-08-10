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

    # ``is_pdf`` flags the reference patches — the board rendered from its PDF
    # (`BlobTrackData`) resp. the un-warped identity view (`HomographyData`) — as
    # opposed to the image patches warped/tracked out of real or synthesised views.
    # It is a *loss-side* label (see ``ProxyAnchoredSupCon`` / ``ProxyAnchoredFPR95``)
    # as well as a model input, so it is read on both paths, validation included.
    is_pdf = data["is_pdf"].to(device) if "is_pdf" in data else None

    # Optional learned-mask inputs (only present when the dataset has
    # ``precompute_masks=True``). Absent -> the model is called exactly as before,
    # so plain descriptors (HardNet, cartesian) are untouched.
    #
    # During VALIDATION the GT mask is withheld from the NETWORK: at test time the
    # mask is unavailable and the model must predict it, so validation must mirror
    # that (feed patches only, no mask / no is_pdf). The *loss-side* copies of both
    # the mask and ``is_pdf`` are unaffected — withholding them there would silently
    # make the validation loss a different loss from the training one. Since a
    # mask-aware model returns ``m_pred`` regardless of what it was fed, this lets
    # the supervision BCE below be scored on both paths without leaking the GT.
    masks = data["masks"].to(device) if "masks" in data else None
    model_mask = None if validation else masks
    model_is_pdf = None if validation else is_pdf

    # Only a mask-aware model takes the mask kwargs (it advertises itself with
    # ``learned_mask``); every other descriptor has a plain ``forward(patches)`` and
    # would raise on them. ``training.ignore_mask`` force-disables the mask path even
    # for a mask-aware model (ablation), which is also what makes a mask-aware model
    # runnable on a dataset that carries no masks.
    mask_aware = bool(getattr(model, "learned_mask", False))
    use_mask = mask_aware and not getattr(cfg.training, "ignore_mask", False)
    if use_mask:
        out = model(patches, mask=model_mask, is_pdf=model_is_pdf)
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
    # Same filtering for the loss-side flag, so it stays aligned with `features`;
    # the model-side copy above stays unfiltered (it is indexed by `in_bound_mask`
    # together with the mask supervision below).
    loss_is_pdf = is_pdf[in_bound_mask] if is_pdf is not None else None
    losses = {n: (c({"features": features, "indices": keypoints[..., 1],
                     "is_pdf": loss_is_pdf}), w, r)
              for n, (c, w, r) in criterion.items()}

    # Standalone mask loss: supervise the predictor on the TARGET (warped) views against
    # their true board coverage. The PDF patch's mask is *given* (used directly, not
    # predicted), so it is excluded here; the predictor exists to supply, at test time,
    # the target mask we no longer have. Restricted to in-bound targets — out-of-frame
    # patches are meaningless white fill. Gated on `use_mask` as well, so
    # `ignore_mask` switches the whole mask path off — supervision included; a model
    # that never receives the mask must not be scored against it either.
    #
    # Scored on the validation path too (the GT mask reaches the loss but not the
    # model, see above), so the predictor's quality gets its own curve. There it is
    # reported at weight 0: it is a diagnostic, and letting it into the weighted
    # validation total would change what `best.pth` selects for.
    if use_mask and m_pred is not None and masks is not None and is_pdf is not None:
        target = (~is_pdf.view(-1).bool()) & in_bound_mask
        if target.any():
            _, _, a_dim, r_dim = m_pred.shape
            gt = torch.nn.functional.adaptive_avg_pool2d(masks, (a_dim, r_dim))
            bce = torch.nn.functional.binary_cross_entropy(
                m_pred[target].clamp(1e-6, 1 - 1e-6), gt[target].clamp(0.0, 1.0)
            )
        else:
            bce = (m_pred * 0.0).sum()
        weight = 0.0 if validation else float(
            getattr(getattr(cfg, "training", None), "mask_loss_weight", 1.0))
        losses["mask_bce"] = (bce, weight, True)
    return losses
