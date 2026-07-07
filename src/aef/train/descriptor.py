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

from enum import Enum
import logging
from typing import Iterable

import kornia
import torch

from .detector import homogenize, linearize_homography


class Detector(Enum):
    DoG = 1
    HARRIS = 2


def detect(img: torch.Tensor, detector: Detector = Detector.HARRIS, threshold: float = 1e-4) -> Iterable[torch.Tensor]:
    img = torch.mean(img, dim=1, keepdim=True)
    if detector == Detector.DoG:
        response_map = kornia.feature.dog_response(img)
    elif detector == Detector.HARRIS:
        response_map = kornia.feature.harris_response(img)
    else:
        raise ValueError("Unknown detector")

    b, _, x, y = torch.where(response_map > threshold)
    logging.info("Detected %d features", b.size(0))
    splits = list(map(lambda i: int(torch.sum(b == i).item()), range(img.shape[0])))

    return torch.split(torch.stack((x, y), dim=1).to(img.device), split_size_or_sections=splits, dim=0)


def warp_detections(detections: Iterable[torch.Tensor], H: torch.Tensor) -> Iterable[torch.Tensor]:
    def coordinate_map(e: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        H, c = e
        if len(c) == 0:
            return c
        c_h = torch.cat(
            (c.to(torch.float32), torch.tensor([[1.0]], device=c.device).expand(c.size(0), 1)),
            dim=1
        ).unsqueeze(2)
        c_warped = (H.unsqueeze(0) @ c_h).squeeze(2)
        return torch.round(c_warped[:, :2] / c_warped[:, 2:]).to(torch.int)
    return map(coordinate_map, zip(H, detections))


def process_batch_homographic_descriptor(model, data, criterion, augmentation, device, cfg):
    img, img_t, H, H_inv = data
    img = augmentation(img.to(device))
    img_t = augmentation(img_t.to(device))

    feature_map = model(img)
    feature_map_t = model(img_t)

    if "detector" in cfg.training.feature_sampling:
        # TODO: Fix this
        H = H.to(device)
        detections = detect(img)
        detections_t = warp_detections(detections, H)

        valid_detections = map(
            lambda ds:
                (ds[:, 0] >= 0) & (ds[:, 0] < img.shape[2])
                & (ds[:, 1] >= 0) & (ds[:, 1] < img.shape[3]),
            detections_t
        )

        detections = map(lambda e: e[0][e[1]], zip(detections, valid_detections))
        detections_t = map(lambda e: e[0][e[1]], zip(detections_t, valid_detections))
        features = torch.empty(sum(map(len, detections)), model.feature_size)
        features_t = torch.empty(sum(map(len, detections)), model.feature_size)

        i = 0
        for j, (ds, ds_t) in enumerate(zip(detections, detections_t)):
            for d, d_t in zip(ds, ds_t):
                features[i, :] = feature_map[j, :, *d].squeeze()
                features_t[i, :] = feature_map_t[j, :, *d_t].squeeze()
                i += 1

        y = torch.cat((features, features_t))
        labels = torch.cat((torch.arange(features.size(0)), torch.arange(features_t.size(0)))).to(device)
    else:
        if "stride" in cfg.training.feature_sampling:
            feature_stride = cfg.training.feature_sampling.stride
        elif "num_features" in cfg.training.feature_sampling:
            feature_stride = img.size(2) * img.size(3) // cfg.training.feature_sampling.num_features
        else:
            raise ValueError("No valid feature sampling method")

        H_inv = H_inv.to(device)
        feature_map_t = kornia.geometry.transform.warp_perspective(
            feature_map_t,
            H_inv,
            dsize=feature_map.shape[2:]
        )
        mask = kornia.geometry.transform.warp_perspective(
            torch.ones(1, 1, 1, 1).to(device).expand(feature_map.size()),
            H_inv,
            dsize=feature_map.shape[2:]
        ) > 0.5
        features = torch.where(mask, feature_map, 0).permute(0, 2, 3, 1).flatten(end_dim=-2)[::feature_stride]
        features_t = torch.where(mask, feature_map_t, 0).permute(0, 2, 3, 1).flatten(end_dim=-2)[::feature_stride]
        y = torch.cat((features, features_t))
        assert y.size(0) % 2 == 0
        labels = torch.cat((
            torch.arange(y.size(0) // 2),
            torch.arange(y.size(0) // 2)
        )).to(device)

    return criterion(y, labels)


def process_batch_colmap_descriptor(model, data, criterion, augmentation, device, cfg):
    img_1 = data["image1"].to(device)
    img_2 = data["image2"].to(device)
    
    batch_size = img_1.shape[0]

    # Apply augmentation
    img_1 = augmentation(img_1)
    img_2 = augmentation(img_2)

    # Get feature maps from model
    feature_map_1 = model(img_1)
    feature_map_2 = model(img_2)

    # Get normalized point coordinates [0, 1]
    pts1 = data["pts1"].to(device)
    pts2 = data["pts2"].to(device)

    # Convert normalized coordinates to feature map coordinates
    feature_h, feature_w = feature_map_1.shape[-2:]
    pts1[:, 0] = pts1[:, 0] * (feature_w - 1)
    pts1[:, 1] = pts1[:, 1] * (feature_h - 1)

    pts2[:, 0] = pts2[:, 0] * (feature_w - 1)
    pts2[:, 1] = pts2[:, 1] * (feature_h - 1)

    # Extract features at point locations
    features_1 = []
    features_2 = []
    labels = []
    offset = 0

    def sample_features(feature_map: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        if pts.numel() == 0:
            return torch.empty((0, feature_map.size(1)), device=feature_map.device)
        x = (pts[:, 0] / (feature_w - 1)) * 2 - 1
        y = (pts[:, 1] / (feature_h - 1)) * 2 - 1
        grid = torch.stack((x, y), dim=-1).view(1, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            align_corners=True
        )
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1)

    for i in range(batch_size):
        if pts1.dim() == 3:
            pts1_i = pts1[i]
            pts2_i = pts2[i]
        elif pts1.dim() == 2:
            pts1_i = pts1
            pts2_i = pts2
        else:
            raise ValueError("Expected pts1/pts2 with 2 or 3 dimensions")

        valid = (
            (pts1_i[:, 0] >= 0) & (pts1_i[:, 0] < feature_w)
            & (pts1_i[:, 1] >= 0) & (pts1_i[:, 1] < feature_h)
            & (pts2_i[:, 0] >= 0) & (pts2_i[:, 0] < feature_w)
            & (pts2_i[:, 1] >= 0) & (pts2_i[:, 1] < feature_h)
        )
        pts1_i = pts1_i[valid]
        pts2_i = pts2_i[valid]
        if pts1_i.numel() == 0:
            continue

        f1 = sample_features(feature_map_1[i].unsqueeze(0), pts1_i)
        f2 = sample_features(feature_map_2[i].unsqueeze(0), pts2_i)

        features_1.append(f1)
        features_2.append(f2)
        labels.append(torch.arange(f1.size(0), device=device) + offset)
        offset += f1.size(0)

    if len(features_1) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    features_1 = torch.cat(features_1, dim=0)
    features_2 = torch.cat(features_2, dim=0)
    labels = torch.cat(labels, dim=0)

    y = torch.cat((features_1, features_2), dim=0)
    labels = torch.cat((labels, labels), dim=0)

    return criterion(y, labels)


def extract_patches(imgs, homographies, coords, scales, patch_size=64, scale_factor=64):
    device = imgs.device
    blob_normalizations = torch.linalg.inv(linearize_homography(homographies, coords=torch.cat([coords, torch.ones((1, 1)).expand(coords.size(0), 1).to(imgs.device)], dim=-1)))
    _, S, Vh = torch.linalg.svd(blob_normalizations)
    Σ = torch.zeros((S.size(0), 2, 2), dtype=torch.float32, device=device)
    Σ[:, 0, 0] = S[..., 0]
    Σ[:, 1, 1] = S[..., 1]
    blob_normalizations = Σ @ Vh

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


def process_batch_blobs(model, data, criterion, augmentation, device, cfg, **_):
    keypoints = data["keypoints"].to(device)
    coords = data["keypoint_coords"].to(device)  

    patches = data["patches"].to(device)

    features = model(patches)
    features = features.view(features.size(0), -1)

    in_bound_mask = torch.all((coords >= 0) & (coords < 1024), dim=-1)
    features = features[in_bound_mask]
    keypoints = keypoints[in_bound_mask]
    return {n: (c({"features": features, "indices": keypoints[..., 1]}), w, r) for n, (c, w, r) in criterion.items()}


def sanity_check(imgs, homographies, coords, scales, patch_size=64, scale_factor=64):
    device = imgs.device
    coords = coords[:1]
    scales = scales[:1]
    blob_normalizations = torch.linalg.inv(linearize_homography(homographies, coords=torch.cat([coords, torch.ones((1, 1)).expand(coords.size(0), 1).to(imgs.device)], dim=-1)))
    print(blob_normalizations.size())
    _, S, Vh = torch.linalg.svd(blob_normalizations)
    Σ = torch.zeros((S.size(0), 2, 2), dtype=torch.float32, device=device)
    Σ[:, 0, 0] = S[..., 0]
    Σ[:, 1, 1] = S[..., 1]
    blob_normalizations = Σ @ Vh
    phi = torch.tensor([0, torch.pi / 4, torch.pi / 2])
    rotations = torch.stack([torch.stack([torch.cos(phi), -torch.sin(phi)], dim=1),
                   torch.stack([torch.sin(phi), torch.cos(phi)], dim=1)], dim=1).to(device)
    print(rotations.size(), blob_normalizations.size())
    blob_normalizations = rotations @ blob_normalizations[:1]

    patches = kornia.geometry.transform.warp_perspective(
        imgs[:1].expand(blob_normalizations.size(0), 3, -1, -1),
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
