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


import blobboards
import kornia
import torch

from ..data.blobboards import BlobBoardData
from ..models.scale import NeuralScaleSpace


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


def train_scale(model, dataset, pattern_size=(256, 256), batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    loss_func = torch.nn.MSELoss()

    training_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    for data in training_loader:
        b, b_t, H, H_inv = map(lambda x: x.to(device), data)
        gt = compute_scale(H, pattern_size)
        optimizer.zero_grad()

        o = model(b)
        o_t = model(b_t)
        o_t = kornia.geometry.transform.warp_perspective(o_t, H_inv, pattern_size)

        loss = loss_func(o_t / o, gt)
        loss.backward()
        optimizer.step()

        print(f"loss {loss.item()}", flush=True)


if __name__ == "__main__":
    model = NeuralScaleSpace(1)
    train_scale(model, 100, batch_size=10)
