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

import kagglehub
import numpy as np
import pycolmap
import torch
import torchvision


def compute_epipolar_fundamental(pose1, pose2, cam1, cam2):
    # Retrieve Relative Pose
    R1, t1 = pose1.rotation.matrix(), pose1.translation
    R2, t2 = pose2.rotation.matrix(), pose2.translation
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1

    # Essential Matrix
    t_x = np.array([[0, -t_rel[2], t_rel[1]],
                    [t_rel[2], 0, -t_rel[0]],
                    [-t_rel[1], t_rel[0], 0]])
    E = t_x @ R_rel

    # Fundamental Matrix
    K1 = cam1.calibration_matrix()
    K2 = cam2.calibration_matrix()
    F = np.linalg.inv(K2.T) @ E @ np.linalg.inv(K1)

    return E, F


class ColmapData(torch.utils.data.Dataset):
    def __init__(self, images=None, kaggle_dataset=None, image_size=(256, 256), **_):
        if images is None and kaggle_dataset is None:
            raise ValueError("Either 'images' directory or 'kaggle_dataset' name must be provided.")

        if kaggle_dataset is not None:
            images = kagglehub.dataset_download(kaggle_dataset)

        self.images_dir = os.path.join(images, "images")
        self.image_size = image_size
        self.resize_transform = torchvision.transforms.Resize(image_size)

        database_path = os.path.join(images, "database.db")
        sparse_dir = os.path.join(images, "sparse")

        if not os.path.exists(database_path) or not os.path.exists(sparse_dir):
            os.makedirs(sparse_dir, exist_ok=True)
            pycolmap.extract_features(database_path, images)
            pycolmap.match_exhaustive(database_path)
            # Use incremental mapping
            maps = pycolmap.incremental_mapping(database_path, images, sparse_dir)
            self.reconstruction: pycolmap.Reconstruction = maps[0] if isinstance(maps, dict) else next(iter(maps.values() if isinstance(maps, dict) else maps))
        else:
            self.reconstruction = pycolmap.Reconstruction(sparse_dir)

        self.pairs = []
        img_ids = list(self.reconstruction.images.keys())

        # Build pairs with co-visible points
        for i in range(len(img_ids)):
            for j in range(i + 1, len(img_ids)):
                img1_id, img2_id = img_ids[i], img_ids[j]
                img1, img2 = self.reconstruction.images[img1_id], self.reconstruction.images[img2_id]

                # Intersect valid points 3D
                p3d_1 = {p.point3D_id: idx for idx, p in enumerate(img1.points2D) if p.has_point3D()}
                p3d_2 = {p.point3D_id: idx for idx, p in enumerate(img2.points2D) if p.has_point3D()}

                common_3d = set(p3d_1.keys()).intersection(set(p3d_2.keys()))
                if len(common_3d) > 20:
                    self.pairs.append((img1_id, img2_id, list(common_3d)))
        self.c = 3

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_id, img2_id, common_3d = self.pairs[idx]
        img1_obj = self.reconstruction.image(img1_id)
        img2_obj = self.reconstruction.image(img2_id)

        cam1 = self.reconstruction.camera(img1_obj.camera_id)
        cam2 = self.reconstruction.camera(img2_obj.camera_id)

        img1_path = os.path.join(self.images_dir, img1_obj.name)
        img2_path = os.path.join(self.images_dir, img2_obj.name)

        img1_tensor = torchvision.io.read_image(img1_path, torchvision.io.ImageReadMode.RGB).to(torch.float32) / 255.0
        img2_tensor = torchvision.io.read_image(img2_path, torchvision.io.ImageReadMode.RGB).to(torch.float32) / 255.0

        # Get original image sizes
        orig_h1, orig_w1 = img1_tensor.shape[1], img1_tensor.shape[2]
        orig_h2, orig_w2 = img2_tensor.shape[1], img2_tensor.shape[2]

        # Resize images
        img1_tensor = self.resize_transform(img1_tensor)
        img2_tensor = self.resize_transform(img2_tensor)

        # Calculate scaling factors
        new_h, new_w = self.image_size
        scale_w1 = new_w / orig_w1
        scale_h1 = new_h / orig_h1
        scale_w2 = new_w / orig_w2
        scale_h2 = new_h / orig_h2

        pts1 = []
        pts2 = []
        for p3d_id in common_3d:
            idx1 = next(idx for idx, p in enumerate(img1_obj.points2D) if p.point3D_id == p3d_id)
            idx2 = next(idx for idx, p in enumerate(img2_obj.points2D) if p.point3D_id == p3d_id)
            pts1.append(img1_obj.points2D[idx1].xy)
            pts2.append(img2_obj.points2D[idx2].xy)

        pts1 = torch.tensor(pts1, dtype=torch.float32)
        pts2 = torch.tensor(pts2, dtype=torch.float32)

        # Scale and normalize point coordinates
        pts1[:, 0] = (pts1[:, 0] * scale_w1) / new_w
        pts1[:, 1] = (pts1[:, 1] * scale_h1) / new_h
        pts2[:, 0] = (pts2[:, 0] * scale_w2) / new_w
        pts2[:, 1] = (pts2[:, 1] * scale_h2) / new_h

        E, F = compute_epipolar_fundamental(img1_obj.cam_from_world(), img2_obj.cam_from_world(), cam1, cam2)

        return {
            "image1": img1_tensor,
            "image2": img2_tensor,
            "pts1": pts1,
            "pts2": pts2,
            "essential_matrix": torch.tensor(E, dtype=torch.float32),
            "fundamental_matrix": torch.tensor(F, dtype=torch.float32)
        }
