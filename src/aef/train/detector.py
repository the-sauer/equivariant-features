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


def process_batch_colmap_detector(model, data, criterion, augmentation, device, cfg):
    img_1 = data["image1"].to(device)
    img_2 = data["image2"].to(device)

    img_1 = augmentation(img_1)
    img_2 = augmentation(img_2)

    feature_map_1 = model(img_1)
    feature_map_2 = model(img_2)

    pts1 = data["pts1"].to(device)
    pts2 = data["pts2"].to(device)

    if pts1.dim() == 2:
        pts1 = pts1.unsqueeze(0)
        pts2 = pts2.unsqueeze(0)
    if pts1.dim() != 3:
        raise ValueError("Expected pts1/pts2 with 2 or 3 dimensions")

    feature_h, feature_w = feature_map_1.shape[-2:]
    pts1_px = pts1.clone()
    pts2_px = pts2.clone()
    pts1_px[..., 0] = pts1_px[..., 0] * (feature_w - 1)
    pts1_px[..., 1] = pts1_px[..., 1] * (feature_h - 1)
    pts2_px[..., 0] = pts2_px[..., 0] * (feature_w - 1)
    pts2_px[..., 1] = pts2_px[..., 1] * (feature_h - 1)

    def sample_affines(feature_map: torch.Tensor, pts_px: torch.Tensor) -> torch.Tensor:
        x = (pts_px[..., 0] / (feature_w - 1)) * 2 - 1
        y = (pts_px[..., 1] / (feature_h - 1)) * 2 - 1
        grid = torch.stack((x, y), dim=-1).unsqueeze(2)
        flat_map = feature_map.reshape(feature_map.size(0), 4, feature_h, feature_w)
        sampled = torch.nn.functional.grid_sample(flat_map, grid, mode="bilinear", align_corners=True)
        sampled = sampled.squeeze(-1).transpose(1, 2)
        return (
            homogenize(sampled.reshape(sampled.size(0), sampled.size(1), 2, 2))
            @ homogenize(
                torch.eye(2, device=feature_map.device).unsqueeze(0).expand(sampled.size(0), sampled.size(1), -1, -1),
                b=torch.stack((x, y), dim=-1)
            )
        )

    aff_1 = sample_affines(feature_map_1, pts1_px)
    aff_2 = sample_affines(feature_map_2, pts2_px)

    valid = (
        (pts1_px[..., 0] >= 0) & (pts1_px[..., 0] < feature_w)
        & (pts1_px[..., 1] >= 0) & (pts1_px[..., 1] < feature_h)
        & (pts2_px[..., 0] >= 0) & (pts2_px[..., 0] < feature_w)
        & (pts2_px[..., 1] >= 0) & (pts2_px[..., 1] < feature_h)
    )

    F = data["fundamental_matrix"].to(device)
    if F.dim() == 2:
        F = F.unsqueeze(0)

    rel_list = []
    pt_list = []
    F_list = []
    detections_1_list = []
    detections_2_list = []
    for i in range(pts1_px.size(0)):
        valid_i = valid[i]
        if not torch.any(valid_i):
            continue
        pts1_i = pts1_px[i, valid_i]
        aff_1_i = aff_1[i, valid_i]
        aff_2_i = aff_2[i, valid_i]

        det = torch.linalg.det(aff_1_i)
        non_singular = (det.abs() > 1e-6) & (torch.linalg.det(aff_2_i).abs() > 1e-6)
        if not torch.any(non_singular):
            continue

        pts1_i = pts1_i[non_singular]
        aff_1_i = aff_1_i[non_singular]
        aff_2_i = aff_2_i[non_singular]

        rel_i = aff_2_i @ torch.linalg.inv(aff_1_i)
        rel_list.append(rel_i)
        pt_list.append(pts1_i)
        F_list.append(F[i].expand(rel_i.size(0), -1, -1))

        detections_1_list.append(aff_1_i)
        detections_2_list.append(aff_2_i)

    F = torch.cat(F_list, dim=0)

    return {
        n: (criterion({
            "img_1": img_1,
            "img_2": img_2,
            "F": F,
            "descriptor_model": model.descriptor_model,
            "detections_1": detections_1_list,
            "detections_2": detections_2_list,
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
