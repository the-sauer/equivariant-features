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

import os

import pycolmap
import torch
import torchvision

from ..data import find_images


class ColmapDataset(torch.utils.data.Dataset):
    def __init__(self, images, image_size):
        self.images_files = find_images(images)
        resize = torchvision.transforms.Resize(image_size)
        self.img = torch.stack([
            resize(torchvision.io.read_image(file)).to(torch.float32) / 255.0 for file in self.images_files
        ])
        database_path = os.path.join(images, "database.db")
        if not os.path.exists(database_path):
            # Run reconstruction if not already done
            pycolmap.extract_features(database_path, images)
            pycolmap.match_exhaustive(database_path)
            result, *_ = pycolmap.incremental_mapping(database_path, images, os.path.join(images, "sparse"))

            print(f"Reconstruction finished with {result.num_reg_images()} images and {result.num_points3D()} 3D points.")

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx]
