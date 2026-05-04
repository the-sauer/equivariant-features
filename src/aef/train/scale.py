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

from ..data import HomographyData
from ..data.blobboards import BlobBoardAbsoluteScaleData

from ..train import prepare_training


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


def train(model, train_dataset, *args, **kwargs):
    if isinstance(train_dataset, HomographyData):
        train_scale_homographic(model, train_dataset, *args, **kwargs)
    elif isinstance(train_dataset, BlobBoardAbsoluteScaleData):
        train_scale_absolute(model, *args, train_dataset=train_dataset, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset type for training scale {type(train_dataset)}")


def train_scale_homographic(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
    model, optimizer, scheduler, criterion, training_loader, validation_loader, augmentation, device, checkpoint_dir = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)

    for epoch in range(cfg.training.num_epochs):
        batch_counter = 0
        loop = tqdm(training_loader, leave=True)
        cumulative_loss = 0.0
        for data in loop:
            b, b_t, H, H_inv = map(lambda x: x.to(device), data)
            gt = compute_scale(H, b.shape[2:])
            for opt in optimizer:
                opt.zero_grad()

            o = model(b)
            o_t = model(b_t)
            o_t = kornia.geometry.transform.warp_perspective(o_t, H_inv, b.shape[2:])

            loss = criterion(o_t / o, gt)
            loss.backward()
            for opt in optimizer:
                opt.step()
            cumulative_loss += loss.item() * b.size(0)

            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            loop.set_postfix(loss=loss.item())

            batch_counter += 1
            if batch_counter % cfg.logging.interval == 0:
                logging.info(f"epoch {epoch}, loss: {loss.item()}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_batch_{batch_counter:06d}.pth"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pth"))
        logging.info(f"finished epoch {epoch}, avg loss: {cumulative_loss / len(train_dataset)}")

        for sch in scheduler:
            sch.step()


def train_scale_absolute(model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, validation_dataset: torch.utils.data.Dataset, cfg, experiment_name="default"):
    model, optimizer, scheduler, criterion, train_loader, validation_loader, augmentation, device, checkpoint_dir = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)

    for epoch in range(cfg.training.num_epochs):
        batch_counter = 0
        loop = tqdm(train_loader, leave=True)
        cumulative_loss = 0.0
        for b, gt, num_blobs in loop:
            b: torch.Tensor = b.to(device)
            gt: torch.Tensor = gt.to(device)
            num_blobs: torch.Tensor = num_blobs.to(device)
            B = b.size(0)

            for opt in optimizer:
                opt.zero_grad()

            b = augmentation(b)
            scale_field: torch.Tensor = model(b)
            out = []
            for i in range(b.size(0)):
                out.append(scale_field[i, 0, gt[i, :num_blobs[i], 0].round().int(), gt[i, :num_blobs[i], 1].round().int()])
            out = torch.cat(out, dim=0).flatten()
            gt = torch.cat([gt[i, :num_blobs[i], 2] for i in range(b.size(0))], dim=0).flatten()

            loss = criterion(out, gt)
            loss.backward()
            for opt in optimizer:
                opt.step()
            cumulative_loss += loss.item() * b.size(0)

            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            loop.set_postfix(loss=loss.item())

            batch_counter += 1
            if batch_counter % cfg.logging.interval == 0:
                logging.info(f"epoch {epoch}, loss: {loss.item()}")
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}_batch_{batch_counter:06d}.pth"))
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pth"))

        loop = tqdm(validation_loader, leave=True)
        cumulative_loss = 0.0
        for b, gt, num_blobs in loop:
            b = b.to(device)
            gt = gt.to(device)
            num_blobs = num_blobs.to(device)

            scale_field: torch.Tensor = model(b)
            out = []
            for i in range(b.size(0)):
                out.append(scale_field[i, 0, gt[i, :num_blobs[i], 0].round().int(), gt[i, :num_blobs[i], 1].round().int()])
            out = torch.cat(out, dim=0).flatten()
            gt = torch.cat([gt[i, :num_blobs[i], 2] for i in range(b.size(0))], dim=0).flatten()

            loss = criterion(out, gt)

            cumulative_loss += loss.item() * b.size(0)

            loop.set_description(f"Validation [{epoch}/{cfg.training.num_epochs}]")
            loop.set_postfix(loss=loss.item())

        logging.info(f"finished epoch {epoch}, avg loss: {cumulative_loss / len(validation_dataset)}")

        for sch in scheduler:
            sch.step()
