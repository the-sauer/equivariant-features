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
import omegaconf
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from ..train import OPTIMIZERS, LOSSES, GeodesicLoss, HomographyReprojectionLoss


def homogenize(A, b=None):
    B, H, W = A.size()
    batch_dim, height_dim, width_dim = 0, 1, 2
    if b is None:
        b = torch.zeros(1, 1, dtype=A.dtype, device=A.device).expand(B, H)
    assert b.size() == (B, H)

    return torch.cat((
        torch.cat((A, torch.zeros(1, 1, 1, dtype=A.dtype, device=A.device).expand(B, 1, W)), dim=height_dim),
        torch.cat((b.unsqueeze(-1), torch.ones(1, 1, 1, dtype=A.dtype, device=A.device).expand(B, 1, 1)), dim=height_dim),
    ), dim=width_dim)


def linearize_homography(H, shape):
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
    coords = coords.unsqueeze(0).unsqueeze(4)
    H = H.unsqueeze(1).unsqueeze(2)
    proj = H @ coords
    x = proj[..., 0, 0]
    y = proj[..., 1, 0]
    w = proj[..., 2, 0]
    return torch.stack((
        torch.stack(((H[..., 0, 0] * w - H[..., 2, 0] * x) / w ** 2, (H[..., 1, 0] * w - H[..., 2, 0] * y) / w ** 2), dim=2),
        torch.stack(((H[..., 0, 1] * w - H[..., 2, 1] * x) / w ** 2, (H[..., 1, 1] * w - H[..., 2, 1] * y) / w ** 2), dim=2)
    ), dim=3)


def train(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    os.makedirs(os.path.join(cfg.logging.dir, experiment_name), exist_ok=True)
    checkpoint_dir = os.path.join(cfg.logging.dir, experiment_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logfile = os.path.join(cfg.logging.dir, experiment_name, "training.log")
    logging.basicConfig(filename=logfile, level=logging.INFO, force=True)

    with open(os.path.join(cfg.logging.dir, experiment_name, "cfg.yaml"), "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = OPTIMIZERS[cfg.training.optimizer.name](model.parameters(), **cfg.training.optimizer.params)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=True
    )

    criterion = LOSSES[cfg.training.loss]()

    augmentation = v2.Compose([
        v2.ColorJitter(**cfg.training.augmentation.color_jitter),
        v2.GaussianBlur(
            cfg.training.augmentation.gaussian_blur.kernel_size,
            sigma=cfg.training.augmentation.gaussian_blur.sigma
        ),
        v2.GaussianNoise(**cfg.training.augmentation.gaussian_noise),
    ])
    for epoch in range(cfg.training.num_epochs):
        batch_counter = 0
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
        cumulative_loss = 0.0
        model.train()
        for (img, img_t, H, H_inv) in loop:
            img = img.to(device)
            img_t = augmentation(img_t.to(device))
            H = H.to(device)
            H_inv = H_inv.to(device)
            optimizer.zero_grad()

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
                torch.flatten(feature_map, start_dim=1, end_dim=2),
                H_inv,
                dsize=feature_map.shape[-2:]
            ).reshape(feature_map.shape)
            mask = (kornia.geometry.transform.warp_perspective(
                torch.ones(1, 1, 1, 1).to(device).expand(feature_map.shape[0], -1, *feature_map.shape[-2:]),
                H_inv,
                dsize=feature_map.shape[-2:]
            ) > 0.5).unsqueeze(1).expand(-1, feature_map.shape[1], feature_map.shape[2], -1, -1)
            features = torch.where(mask, feature_map, 0).permute(0, 3, 4, 1, 2).reshape(-1, 2, 2)[::feature_stride, :, :]
            features_t = torch.where(mask, feature_map_t, 0).permute(0, 3, 4, 1, 2).reshape(-1, 2, 2)[::feature_stride, : ,:]

            non_singular_mask = torch.linalg.det(features) > 1e-6
            rel_t = features_t[non_singular_mask] @ torch.linalg.inv(features[non_singular_mask])   # use linalg.solve

            gt = linearize_homography(H, feature_map.shape[-2:]).reshape(-1, 2, 2)[::feature_stride, :, :][non_singular_mask]

            loss = criterion(homogenize(rel_t), homogenize(gt))

            reprojection_loss = HomographyReprojectionLoss()(homogenize(rel_t), homogenize(gt))
            geodesic_loss = GeodesicLoss()(rel_t, gt)

            loop.set_postfix(reprojection_loss=reprojection_loss.item(), geodesic_loss=geodesic_loss.item())
            cumulative_loss += loss.item() * img.size(0)
            loss.backward()
            optimizer.step()
            batch_counter += 1
            if batch_counter % cfg.logging.interval == 0:
                checkpoint_name = f"epoch_{epoch:03d}_{batch_counter//cfg.logging.interval:06d}.pth"
                logging.info(f"epoch {epoch}, loss: {loss.item()}, saved_model to: {checkpoint_name}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, checkpoint_name))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))

        loop = tqdm(validation_loader, leave=True)
        loop.set_description(f"Validating [{epoch}/{cfg.training.num_epochs}]")
        model.eval()
        for (img, img_t, H, H_inv) in loop:
            img = img.to(device)
            img_t = augmentation(img_t.to(device))
            H = H.to(device)
            H_inv = H_inv.to(device)

            feature_map = model(img)
            feature_map_t = model(img_t)

            if "stride" in cfg.validation.feature_sampling:
                feature_stride = cfg.validation.feature_sampling.stride
            elif "num_features" in cfg.validation.feature_sampling:
                feature_stride = img.size(2) * img.size(3) // cfg.validation.feature_sampling.num_features
            else:
                raise ValueError("No valid feature sampling method")

            H_inv = H_inv.to(device)
            feature_map_t = kornia.geometry.transform.warp_perspective(
                torch.flatten(feature_map, start_dim=1, end_dim=2),
                H_inv,
                dsize=feature_map.shape[-2:]
            ).reshape(feature_map.shape)
            mask = (kornia.geometry.transform.warp_perspective(
                torch.ones(1, 1, 1, 1).to(device).expand(feature_map.shape[0], -1, *feature_map.shape[-2:]),
                H_inv,
                dsize=feature_map.shape[-2:]
            ) > 0.5).unsqueeze(1).expand(-1, feature_map.shape[1], feature_map.shape[2], -1, -1)
            features = torch.where(mask, feature_map, 0).permute(0, 3, 4, 1, 2).reshape(-1, 2, 2)[::feature_stride, :, :]
            features_t = torch.where(mask, feature_map_t, 0).permute(0, 3, 4, 1, 2).reshape(-1, 2, 2)[::feature_stride, : ,:]

            non_singular_mask = torch.linalg.det(features) > 1e-6
            rel_t = features_t[non_singular_mask] @ torch.linalg.inv(features[non_singular_mask])   # use linalg.solve

            gt = linearize_homography(H, feature_map.shape[-2:]).reshape(-1, 2, 2)[::feature_stride, :, :][non_singular_mask]

            loss = criterion(homogenize(rel_t), homogenize(gt))

            reprojection_loss = HomographyReprojectionLoss()(homogenize(rel_t), homogenize(gt))
            geodesic_loss = GeodesicLoss()(rel_t, gt)

            loop.set_postfix(reprojection_loss=reprojection_loss.item(), geodesic_loss=geodesic_loss.item())

        logging.info(f"finished epoch {epoch}, avg loss: {cumulative_loss / len(train_dataset)}")
