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
    def __init__(self, images=None, kaggle_dataset=None, image_size=(256, 256), suffix="train", **_):
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

        # self.pairs = []
        if suffix == "train":
            img_ids = list(filter(lambda id: id % 10 > 0, self.reconstruction.images.keys()))
        else:
            img_ids = list(filter(lambda id: id % 10 == 0, self.reconstruction.images.keys()))

        # # Build pairs with co-visible points
        # for i in range(len(img_ids)):
        #     for j in range(i + 1, len(img_ids)):
        #         img1_id, img2_id = img_ids[i], img_ids[j]
        #         img1, img2 = self.reconstruction.images[img1_id], self.reconstruction.images[img2_id]

        #         # Intersect valid points 3D
        #         p3d_1 = {p.point3D_id: idx for idx, p in enumerate(img1.points2D) if p.has_point3D()}
        #         p3d_2 = {p.point3D_id: idx for idx, p in enumerate(img2.points2D) if p.has_point3D()}

        #         common_3d = set(p3d_1.keys()).intersection(set(p3d_2.keys()))
        #         if len(common_3d) > 20:
        #             self.pairs.append((img1_id, img2_id, list(common_3d)))
        self.images = {}
        keypoint_list = []
        keypoint_coords = []
        for img_id in img_ids:
            img_temp = torchvision.io.read_image(os.path.join(self.images_dir, self.reconstruction.image(img_id).name))
            self.images[img_id] = self.resize_transform(img_temp).to(torch.float32) / 255.0
            points2D = list(filter(lambda p: p.has_point3D(), self.reconstruction.image(img_id).points2D))
            if len(points2D) < 20:
                continue
            points3D = torch.Tensor(list(map(lambda p: p.point3D_id, points2D))).to(torch.int64)
            coords = torch.Tensor(np.stack(list(map(lambda p: p.xy, points2D)))) / torch.Tensor([img_temp.shape[-1], img_temp.shape[-2]]).to(torch.float32).unsqueeze(0) * torch.Tensor([*self.image_size]).to(torch.float32).unsqueeze(0)
            img_ids_tensor = torch.Tensor([img_id]).to(torch.int64).expand(points3D.size(0))
            keypoint_list.append(torch.stack([img_ids_tensor, points3D], dim=1))
            keypoint_coords.append(coords)
        self.keypoints = torch.cat(keypoint_list, dim=0)
        self.keypoint_coords = torch.cat(keypoint_coords, dim=0)

        self.c = 3

    def __len__(self):
        return self.keypoints.size(0)

    def __getitem__(self, idx):
        return {
            "keypoint":  self.keypoints[idx],
            "keypoint_coords": self.keypoint_coords[idx],
            # "essential_matrix": torch.tensor(E, dtype=torch.float32),
            # "fundamental_matrix": torch.tensor(F, dtype=torch.float32),
            # "img": {
            #     img_id: self.resize_transform(torchvision.io.read_image(os.path.join(self.images_dir, self.reconstruction.image(img_id).name)))
            #     for img_id in self.reconstruction.images.keys()
            # }
        }
    
    def get_collate_func(self):
        def collate_colmap(batch):
            keypoints = torch.stack([item["keypoint"] for item in batch])
            coords = torch.stack([item["keypoint_coords"] for item in batch])
            return {
                "keypoints": keypoints,
                "keypoint_coords": coords,
                "images":{
                    img_id: self.resize_transform(torchvision.io.read_image(os.path.join(self.images_dir, self.reconstruction.image(img_id).name))).to(torch.float32) / 255.0
                    for img_id in keypoints[:, 0].unique().tolist()
                }
            }
        return collate_colmap
