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
from tqdm import tqdm

from aef.data import HomographyData
from aef.data.blobboards import BlobBoardHomographyData, BlobBoardAbsoluteScaleData

from .losses import Loss
from ..train import OPTIMIZERS, SCHEDULERS


def compute_scale(H, size):
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


def train(model, train_dataset, *args, **kwargs):
    if isinstance(train_dataset, HomographyData):
        train_scale_homographic(*args, train_dataset=train_dataset, **kwargs)
    elif isinstance(train_dataset, BlobBoardAbsoluteScaleData):
        train_scale_absolute(model, *args, train_dataset=train_dataset, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type for training scale {type(train_dataset)}")


def train_scale_homographic(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    os.makedirs(os.path.join(cfg.logging.dir, experiment_name), exist_ok=True)
    checkpoint_dir = os.path.join(cfg.logging.dir, experiment_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.logging.dir, experiment_name, "cfg.yaml"), "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))
    logfile = os.path.join(cfg.logging.dir, experiment_name, "training.log")
    logging.basicConfig(filename=logfile, level=logging.INFO, force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = OPTIMIZERS[cfg.training.optimizer.name](model.parameters(), **cfg.training.optimizer.params)
    if "scheduler" in cfg.training.optimizer:
        sched_cfg = cfg.training.optimizer.scheduler
        scheduler = SCHEDULERS[sched_cfg.name](optimizer, **sched_cfg.params)
    else:
        scheduler = None

    criterion = Loss(cfg.training.loss)

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    for epoch in range(cfg.training.num_epochs):
        batch_counter = 0
        loop = tqdm(training_loader, leave=True)
        cumulative_loss = 0.0
        for data in loop:
            b, b_t, H, H_inv = map(lambda x: x.to(device), data)
            gt = compute_scale(H, b.shape[2:])
            optimizer.zero_grad()

            o = model(b)
            o_t = model(b_t)
            o_t = kornia.geometry.transform.warp_perspective(o_t, H_inv, b.shape[2:])

            loss = criterion(o_t / o, gt)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item() * b.size(0)

            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            loop.set_postfix(loss=loss.item())

            batch_counter += 1
            if batch_counter % cfg.logging.interval == 0:
                logging.info(f"epoch {epoch}, loss: {loss.item()}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_batch_{batch_counter:06d}.pth"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pth"))
        logging.info(f"finished epoch {epoch}, avg loss: {cumulative_loss / len(train_dataset)}")

        if scheduler is not None:
            scheduler.step()


def train_scale_absolute(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    os.makedirs(os.path.join(cfg.logging.dir, experiment_name), exist_ok=True)
    checkpoint_dir = os.path.join(cfg.logging.dir, experiment_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.logging.dir, experiment_name, "cfg.yaml"), "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))
    logfile = os.path.join(cfg.logging.dir, experiment_name, "training.log")
    logging.basicConfig(filename=logfile, level=logging.INFO, force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = OPTIMIZERS[cfg.training.optimizer.name](model.parameters(), **cfg.training.optimizer.params)
    if "scheduler" in cfg.training.optimizer:
        sched_cfg = cfg.training.optimizer.scheduler
        scheduler = SCHEDULERS[sched_cfg.name](optimizer, **sched_cfg.params)
    else:
        scheduler = None

    criterion = Loss(cfg.training.loss)

    training_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    for epoch in range(cfg.training.num_epochs):
        batch_counter = 0
        loop = tqdm(training_loader, leave=True)
        cumulative_loss = 0.0
        for b, gt in loop:
            b = b.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()
            o = model(b)
            o = o[..., gt[..., 0].round().int(), gt[..., 1].round().int()].squeeze()  # only compute loss for blob positions

            loss = criterion(o, gt[..., 2])
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item() * b.size(0)

            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            loop.set_postfix(loss=loss.item())

            batch_counter += 1
            if batch_counter % cfg.logging.interval == 0:
                logging.info(f"epoch {epoch}, loss: {loss.item()}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_batch_{batch_counter:06d}.pth"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pth"))
        logging.info(f"finished epoch {epoch}, avg loss: {cumulative_loss / len(train_dataset)}")

        if scheduler is not None:
            scheduler.step()
