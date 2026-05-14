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
import os

import kornia
import torch
from tqdm import tqdm

from ..data.blobboards import BlobBoardAbsoluteScaleData

from ..data import HomographyData, sample_homography

from .losses.geodesic_loss import GeodesicLoss
from .losses.reprojection_loss import HomographyReprojectionLoss
from ..train import prepare_training


def homogenize(A, b=None):
    *B, H, W = A.size()
    h_dim, w_dim = -2, -1
    if b is None:
        b = torch.zeros(1, 1, dtype=A.dtype, device=A.device).expand(*B, H)
    assert b.size() == (*B, H)

    return torch.cat((
        torch.cat((A, torch.zeros(*(1 for _ in B), 1, 1, dtype=A.dtype, device=A.device).expand(*B, 1, W)), dim=h_dim),
        torch.cat((b.unsqueeze(-1), torch.ones(*(1 for _ in B), 1, 1, dtype=A.dtype, device=A.device).expand(*B, 1, 1)), dim=h_dim),
    ), dim=w_dim)


def linearize_homography(H, shape=None, coords=None):
    if coords is None:
        if shape is None:
            raise ValueError("Either shape or coords must be provided")
        coords = torch.stack(
            (
                *torch.meshgrid(
                    torch.arange(shape[0], device=H.device),
                    torch.arange(shape[1], device=H.device), indexing="ij"
                ),
                torch.ones((1, 1), device=H.device).expand(shape[0], shape[1],)
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


def process_batch(model, data, criterion, augmentation, device, cfg):
    img, img_t, H, H_inv = data
    img = img.to(device)
    img_t = augmentation(img_t.to(device))
    H = H.to(device)
    H_inv = H_inv.to(device)

    feature_map = model(img)
    feature_map_t = model(img_t)

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
    features = torch.where(mask, feature_map, 0).permute(0, 3, 4, 1, 2).reshape(feature_map.size(0), -1, 2, 2)[:,::feature_stride, :, :]
    features_t = torch.where(mask, feature_map_t, 0).permute(0, 3, 4, 1, 2).reshape(feature_map_t.size(0), -1, 2, 2)[:,::feature_stride, :, :]

    b = torch.stack(torch.meshgrid(
        torch.arange(feature_map.size(3), device=device),
        torch.arange(feature_map.size(4), device=device), indexing="ij"
    ), dim=-1).reshape(-1, 2)[::feature_stride].unsqueeze(0).expand(feature_map.size(0), -1, -1)

    non_singular_mask = torch.linalg.det(features) > 1e-6
    num_features = torch.sum(non_singular_mask.int(), dim=1)
    features_filtered = torch.empty(features.size(0), int(max(num_features)), 2, 2, device=device)
    features_t_filtered = torch.empty(features.size(0), int(max(num_features)), 2, 2, device=device)
    b_filtered = torch.empty(features.size(0), int(max(num_features)), 2, device=device)
    b_t = torch.empty(features.size(0), int(max(num_features)), 2, device=device)
    for i in range(features.size(0)):
        features_filtered[i, :num_features[i]] = features[i, non_singular_mask[i]]
        features_t_filtered[i, :num_features[i]] = features_t[i, non_singular_mask[i]]
        b_filtered[i, :num_features[i]] = b[i, non_singular_mask[i]]
        transformed = (H[i].unsqueeze(0) @ torch.cat((b[i, non_singular_mask[i]], torch.ones(num_features[i], 1, device=device)), dim=-1).unsqueeze(-1)).squeeze(-1)
        b_t[i, :num_features[i]] = transformed[..., :2] / transformed[..., 2:3]
    features = features_filtered
    features_t = features_t_filtered
    b = b_filtered

    return criterion((homogenize(features, b), img), (homogenize(features_t, b_t), img_t), num_features)


def train(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    if isinstance(train_dataset, HomographyData):
        return train_homographic(model, train_dataset, validation_dataset, cfg, experiment_name)
    elif isinstance(train_dataset, BlobBoardAbsoluteScaleData):
        return train_absolute(model, train_dataset, validation_dataset, cfg, experiment_name)


def train_homographic(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    (
        model,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        validation_loader,
        augmentation,
        device,
        checkpoint_dir
    ) = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)

    best_loss = float("inf")
    for epoch in range(cfg.training.num_epochs):
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
        model.train()
        cumulative_loss = 0.0
        for i, data in enumerate(loop):
            for opt in optimizer:
                opt.zero_grad()

            loss = process_batch(model, data, criterion, augmentation, device, cfg)
            cumulative_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            loss.backward()
            for opt in optimizer:
                opt.step()
            if hasattr(cfg, "logging") and hasattr(cfg.logging, "interval") and i % cfg.logging.interval == 0:
                logging.info("epoch [%d/%d] batch [%d/%d] loss: %f", epoch, cfg.training.num_epochs, i, len(train_loader), (cumulative_loss / cfg.logging.interval))

        loop = tqdm(validation_loader, leave=True)
        loop.set_description(f"Validating [{epoch}/{cfg.training.num_epochs}]")
        del data
        with torch.no_grad():
            model.eval()
            cumulative_loss = 0.0
            for data in loop:
                loss = process_batch(model, data, criterion, lambda x: x, device, cfg)

                cumulative_loss += loss.item() * data[0].size(0)
                loop.set_postfix(loss=loss.item())

            logging.info("finished epoch [%d/%d], avg loss: %f", epoch, cfg.training.num_epochs, cumulative_loss / len(validation_dataset))

            if checkpoint_dir is not None:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": [opt.state_dict() for opt in optimizer],
                    "scheduler_state_dict": [sch.state_dict() for sch in scheduler],
                    "loss": cumulative_loss / len(validation_dataset),
                    "best_loss": best_loss
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))
                if cumulative_loss < best_loss:
                    best_loss = cumulative_loss
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f"best.pth"))
                    msg = f"New best model with loss {best_loss:.6f} at epoch {epoch} saved to {os.path.join(checkpoint_dir, f"best.pth")}"
                    print("\033[1m" + msg + "\033[0m")
                    logging.info("\033[1m" + msg + "\033[0m")
            for sch in scheduler:
                sch.step()


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
