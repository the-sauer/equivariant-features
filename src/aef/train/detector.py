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

import kornia
import torch
import torchvision
from tqdm import tqdm


def homogenize(A, b=None):
    *B, H, W = A.size()
    h_dim, w_dim = -2, -1
    if b is None:
        b = torch.zeros(1, 1, dtype=A.dtype, device=A.device).expand(*B, H)
    assert b.size() == (*B, H), f"batch dimension of b {b.shape[:-1]} must match batch dimension of A {B}"

    return torch.cat((
        torch.cat((A, torch.zeros(*(1 for _ in B), 1, 1, dtype=A.dtype, device=A.device).expand(*B, 1, W)), dim=h_dim),
        torch.cat((b.unsqueeze(-1), torch.ones(*(1 for _ in B), 1, 1, dtype=A.dtype, device=A.device).expand(*B, 1, 1)), dim=h_dim),
    ), dim=w_dim)


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
    H = H.unsqueeze(1).unsqueeze(2)
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


def process_batch_homographic_detector_for_image_loss(model, data, criterion, augmentation, device, cfg):
    img, img_t, H, H_inv = data
    img = img.to(device)
    img_t = img_t.to(device)
    img_aug, img_t_aug = augmentation((img, img_t))
    H = H.to(device)
    H_inv = H_inv.to(device)

    feature_map = model(img_aug)
    feature_map_t = model(img_t_aug)

    if "stride" in cfg.training.feature_sampling:
        feature_stride = cfg.training.feature_sampling.stride
    elif "num_features" in cfg.training.feature_sampling:
        feature_stride = img.size(2) * img.size(3) // cfg.training.feature_sampling.num_features
    else:
        raise ValueError("No valid feature sampling method")

    H_inv = H_inv.to(device)
    feature_map_t = kornia.geometry.transform.warp_perspective(
        torch.flatten(feature_map_t, start_dim=1, end_dim=2),
        H_inv,
        dsize=feature_map.shape[-2:]
    ).reshape(feature_map.shape)
    # mask = (kornia.geometry.transform.warp_perspective(
    #     torch.ones(1, 1, 1, 1).to(device).expand(feature_map.shape[0], -1, *feature_map.shape[-2:]),
    #     H_inv,
    #     dsize=feature_map.shape[-2:]
    # ) > 0.5).unsqueeze(1).expand(-1, feature_map.shape[1], feature_map.shape[2], -1, -1)
    features = feature_map.permute(0, 3, 4, 1, 2)#.reshape(feature_map.size(0), -1, 2, 2)
    features_t = feature_map_t.permute(0, 3, 4, 1, 2)#.reshape(feature_map_t.size(0), -1, 2, 2)

    b = torch.stack(torch.meshgrid(
        torch.arange(feature_map.size(3), device=device),
        torch.arange(feature_map.size(4), device=device), indexing="ij"
    ), dim=-1).unsqueeze(0).expand(feature_map.size(0), -1, -1, -1)
    b_t = torch.zeros_like(b)
    for i in range(features.size(0)):
        transformed = (H[i].unsqueeze(0) @ torch.cat((b[i], torch.ones(b.size(1), b.size(2), 1, device=device)), dim=-1).unsqueeze(-1)).squeeze(-1)
        b_t[i] = transformed[..., :2] / transformed[..., 2:3]

    return {
        n: (criterion({
            "pred": (homogenize(features, b), img),
            "target": (homogenize(features_t, b_t), img_t),
            "H": H,
            "descriptor_model": model.descriptor_model
        }), weight, report) for n, (criterion, weight, report) in criterion.items()
    }


def process_batch_homographic_detector_for_transform_loss(model, data, criterion, augmentation, device, cfg):
    img, img_t, H, H_inv = data
    img = img.to(device)
    img_t = img_t.to(device)
    img_aug, img_t_aug = augmentation((img, img_t))
    H = H.to(device)
    H_inv = H_inv.to(device)

    feature_map = model(img_aug)
    feature_map_t = model(img_t_aug)

    if "stride" in cfg.training.feature_sampling:
        feature_stride = cfg.training.feature_sampling.stride
    elif "num_features" in cfg.training.feature_sampling:
        feature_stride = img.size(2) * img.size(3) // cfg.training.feature_sampling.num_features
    else:
        raise ValueError("No valid feature sampling method")

    H_inv = H_inv.to(device)
    feature_map_t = kornia.geometry.transform.warp_perspective(
        torch.flatten(feature_map_t, start_dim=1, end_dim=2),
        H_inv,
        dsize=feature_map.shape[-2:]
    ).reshape(feature_map.shape)
    mask = (kornia.geometry.transform.warp_perspective(
        torch.ones(1, 1, 1, 1).to(device).expand(feature_map.shape[0], -1, *feature_map.shape[-2:]),
        H_inv,
        dsize=feature_map.shape[-2:]
    ) > 0.5).unsqueeze(1).expand(-1, feature_map.shape[1], feature_map.shape[2], -1, -1)
    features = torch.where(mask, feature_map, 0).permute(0, 3, 4, 1, 2).reshape(-1, 2, 2)[::feature_stride, :, :]
    features_t = torch.where(mask, feature_map_t, 0).permute(0, 3, 4, 1, 2).reshape(-1, 2, 2)[::feature_stride, :, :]

    non_singular_mask = torch.linalg.det(features) > 1e-6
    rel_t = features_t[non_singular_mask] @ torch.linalg.inv(features[non_singular_mask])   # use linalg.solve

    gt = linearize_homography(H, feature_map.shape[-2:]).reshape(-1, 2, 2)[::feature_stride, :, :][non_singular_mask]

    return criterion(homogenize(rel_t), homogenize(gt))


def process_batch_gt(model, data, criterion, augmentation, device, cfg):
    img, gt = data
    img = img.to(device)
    gt = gt.to(device)

    img_aug = augmentation(img)

    feature_map = model(img_aug)
    print("Mean values per channel:", feature_map.mean(dim=(0, 3, 4)).tolist())

    return criterion(feature_map, gt)


def process_batch_colmap_detector(model, data, criterion, augmentation, device, cfg, max_imgs_per_batch=56):
    patch_size = model.descriptor_model.patch_size
    keypoints = data["keypoints"].to(torch.long).to(device)
    coords = data["keypoint_coords"].to(device)
    sorting = keypoints[:, 0].argsort()
    keypoints = keypoints[sorting]
    coords = coords[sorting]
    img_ids = keypoints[:, 0].unique().tolist()
    gt_scales = data["scales"].to(device)[sorting]
    if len(img_ids) > max_imgs_per_batch:
        # TODO: Sort by number of features descending
        logging.warning(f"Number of unique images in batch {len(img_ids)} exceeds max_imgs_per_batch {max_imgs_per_batch}")   
        img_ids = img_ids[:max_imgs_per_batch]
        img_mask = torch.any(torch.stack([keypoints[:, 0] == img_id for img_id in img_ids], dim=-1), dim=-1)
        keypoints = keypoints[img_mask]
        coords = coords[img_mask]
        gt_scales = gt_scales[img_mask]

    feature_ids = keypoints[:, 1].unique().tolist()

    imgs = torch.stack([data["images"][img_id] for img_id in img_ids])
    image_batch = cfg.training.image_batch_size

    out = []
    features = []
    indices = []
    detection_list = []
    pts = []
    img_id_list = []
    scales = []
    for i, img in enumerate(imgs.split(image_batch)):
        img_aug = augmentation(img.to(device))
        out = model(img_aug)
        for j, img_id in enumerate(img_ids[i * image_batch:(i+1) * image_batch]):
            kp_mask = keypoints[:, 0] == img_id
            xy = coords[kp_mask]
            xy_rounded = xy.round().int()
            detections = out[j, ..., xy_rounded[:, 1], xy_rounded[:, 0]].permute(2, 0, 1).view(-1, 2, 2)
            transforms = homogenize(detections, b=xy)
            if False:   # TODO: Decide based on losses
                patches = kornia.geometry.transform.warp_perspective(
                    torchvision.transforms.functional.gaussian_blur(torchvision.transforms.functional.rgb_to_grayscale(img_aug[j]), kernel_size=19, sigma=3.0).unsqueeze(0).expand(transforms.size(0), -1, -1, -1),
                    (
                        torch.diag(torch.Tensor([patch_size, patch_size, 1]).to(device)).unsqueeze(0)
                        @ homogenize(torch.eye(2).to(device), b=torch.tensor([0.5, 0.5]).to(device)).unsqueeze(0)
                        @ torch.linalg.inv(transforms)
                    ),
                    dsize=(patch_size, patch_size),
                )

            detection_list.append(transforms)
            if False:   # TODO: Decide based on losses
                features.append(model.descriptor_model(patches.to(device)))
            indices.append(keypoints[kp_mask][:, 1])
            pts.append(xy)
            img_id_list.append(torch.tensor([img_id], dtype=torch.long).to(device).expand(xy.size(0)))
            scales.append(gt_scales[kp_mask])

    matches = []
    for feature_id in feature_ids:
        match_indices = (torch.cat(indices, dim=0) == feature_id).nonzero(as_tuple=False).squeeze(1)
        if match_indices.size(0) < 2:
            continue
        x, y = torch.triu_indices(len(match_indices), len(match_indices), offset=1)
        matches.append(torch.stack((match_indices[x], match_indices[y]), dim=1))

    return {
        n: (criterion({
            "features": torch.cat(features, dim=0) if False else None,
            "indices": torch.cat(indices, dim=0),
            "scales": torch.cat(scales, dim=0).to(device),
            "detections": torch.cat(detection_list, dim=0),
            "img_ids": torch.cat(img_id_list, dim=0),
            "matches": torch.cat(matches, dim=0),
            "pts": torch.cat(pts, dim=0),
            "fundamental": data["fundamental"].to(device),
        }), weight, report) for n, (criterion, weight, report) in criterion.items()
    }


def train_absolute(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    (
        model,
        optimizer,
        _scheduler,
        criterion,
        train_loader,
        _validation_loader,
        _augmentation,
        device,
        _checkpoint_dir
    ) = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)

    for epoch in range(cfg.training.num_epochs):
        loop = tqdm(train_loader, leave=True)
        cumulative_loss = 0.0
        model.train()
        for b, gt, num_blobs in loop:
            for opt in optimizer:
                opt.zero_grad()
            b: torch.Tensor = b.to(device)
            gt: torch.Tensor = gt.to(device)
            num_blobs: torch.Tensor = num_blobs.to(device)
            B = b.size(0)

            transform_params = cfg.training.homography_params if "homography_params" in cfg.training else {}
            H = torch.stack(
                [sample_homography(b.shape[2:], **transform_params) for _ in range(b.size(0))],
                dim=0
            ).to(device)

            b_t = kornia.geometry.transform.warp_perspective(b, H, b.shape[2:], padding_mode="zeros")

            out = model(b_t)

            coords = torch.cat([gt[..., :2], torch.ones(1, 1, 1).expand(*gt.shape[:2], 1).to(device)], dim=-1)
            positions = (H.unsqueeze(1) @ coords.unsqueeze(-1)).squeeze(-1)
            positions = positions[..., :2] / positions[..., 2:3]
            affine_gt = linearize_homography(
                H,
                coords=coords.unsqueeze((2))
            ).reshape(gt.shape[0], gt.shape[1], 2, 2)

            y = []
            y_hat = []
            for i in range(B):
                coords_i = positions[i, :num_blobs[i], :2]
                affine_gt_i = affine_gt[i, :num_blobs[i]]

                mask = (
                    (coords_i[:, 0] >= 0)
                    & (coords_i[:, 0] < b.size(2))
                    & (coords_i[:, 1] >= 0)
                    & (coords_i[:, 1] < b.size(3))
                )

                coords_i = coords_i[mask]
                affine_gt_i = affine_gt_i[mask]

                y.append(affine_gt_i)
                y_hat.append(out[i, :, :, coords_i[:, 0].round().int(), coords_i[:, 1].round().int()].permute(2, 0, 1))

            y = torch.stack(y, dim=0)
            y_hat = torch.stack(y_hat, dim=0)

            loss = criterion(homogenize(y_hat.reshape(-1, 2, 2)), homogenize(y.reshape(-1, 2, 2)))
            loss.backward()
            for opt in optimizer:
                opt.step()
            cumulative_loss += loss.item() * b.size(0)
            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            loop.set_postfix(loss=loss.item())
