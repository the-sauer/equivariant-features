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

from aef.train.detector import homogenize
import kagglehub
import numpy as np
import pycolmap
import sqlite3
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


def blob_to_array(blob, dtype, shape=(-1,)):
    np_blob = np.frombuffer(blob, dtype=dtype).reshape(shape)
    return torch.Tensor(np_blob.copy())


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

        conn = sqlite3.connect(os.path.join(images, "database.db"))
        cursor = conn.cursor()

        img_ids = list(self.reconstruction.images.keys())
        if suffix == "train":
            img_ids = img_ids[:int(len(img_ids) * 0.9)]
        elif suffix == "test":
            img_ids = img_ids[int(len(img_ids) * 0.9):]
        else:
            img_ids = img_ids

        self.images = {}
        keypoint_list = []
        keypoint_coords = []
        scale_list = []
        img_scaling_factor = {}
        coord_normalizations = {}
        for img_id in img_ids:
            cursor.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (img_id,))
            rows, cols, blob = cursor.fetchone()
            sift_detections = blob_to_array(blob, dtype=np.float32, shape=(rows, cols))

            img_temp = torchvision.io.read_image(os.path.join(self.images_dir, self.reconstruction.image(img_id).name), torchvision.io.ImageReadMode.GRAY)
            orig_size = img_temp.shape[-2], img_temp.shape[-1]
            if orig_size[0] < orig_size[1]:
                offset_x = (orig_size[1] - orig_size[0]) // 2
                offset_y = 0
            else:
                offset_x = 0
                offset_y = (orig_size[0] - orig_size[1]) // 2
            img_temp = torchvision.transforms.functional.crop(img_temp, offset_y, offset_x, min(orig_size), min(orig_size))

            self.images[img_id] = self.resize_transform(img_temp).to(torch.float32) / 255.0
            points2D = list(filter(lambda p: p.has_point3D(), self.reconstruction.image(img_id).points2D))
            if len(points2D) < 20:
                continue
            points3D = torch.Tensor(list(map(lambda p: p.point3D_id, points2D))).to(torch.int64)
            coords_raw = torch.Tensor(np.stack(list(map(lambda p: p.xy, points2D))))
            scales = torch.empty(coords_raw.size(0), dtype=torch.float32)
            for i, xy in enumerate(coords_raw):
                scales[i] = sift_detections[torch.argsort(torch.norm(coords_raw - xy, dim=1))[0], 2]
            scales = scales * min(self.image_size) / min(orig_size)
            img_scaling_factor[img_id] = torch.Tensor([min(orig_size), min(orig_size)]).to(torch.float32) * torch.Tensor([*self.image_size]).to(torch.float32)
            coordinate_normalization = torch.diag(torch.tensor([self.image_size[0] / min(orig_size), self.image_size[1] / min(orig_size), 1], dtype=torch.float32)) @ homogenize(torch.eye(2, dtype=torch.float32).unsqueeze(0), b=torch.Tensor([-offset_x, -offset_y]).unsqueeze(0))
            coords = coordinate_normalization @ torch.stack((coords_raw[..., 0], coords_raw[..., 1], torch.ones_like(coords_raw[:, 0])), dim=-1).unsqueeze(-1)
            coord_normalizations[img_id] = coordinate_normalization
            coords = (coords[:, :2] / coords[:, 2:]).squeeze(-1)
            coord_mask = (coords.round() >= 0).all(dim=1) & (coords.round() < torch.Tensor(self.image_size)).all(dim=1)
            coords = coords[coord_mask]
            scales = scales[coord_mask]
            img_ids_tensor = torch.Tensor([img_id]).to(torch.int64).expand(points3D.size(0))
            keypoint_list.append(torch.stack([img_ids_tensor, points3D], dim=1)[coord_mask])
            keypoint_coords.append(coords)
            scale_list.append(scales)
        self.keypoints = torch.cat(keypoint_list, dim=0)
        self.keypoint_coords = torch.cat(keypoint_coords, dim=0)
        self.scales = torch.cat(scale_list, dim=0)

        self.fundamental = torch.empty((max(img_ids) + 1, max(img_ids) + 1, 3, 3), dtype=torch.float32)
        for img_id_1 in img_ids:
            for img_id_2 in img_ids:
                if img_id_1 < img_id_2:
                    T_1 = torch.linalg.inv(coord_normalizations[img_id_1])
                    T_2 = torch.linalg.inv(coord_normalizations[img_id_2])
                    self.fundamental[img_id_1, img_id_2] = ((T_2).transpose(-2, -1) @
                        torch.Tensor(compute_epipolar_fundamental(
                            self.reconstruction.image(img_id_1).cam_from_world(),
                            self.reconstruction.image(img_id_2).cam_from_world(),
                            self.reconstruction.camera(self.reconstruction.image(img_id_1).camera_id),
                            self.reconstruction.camera(self.reconstruction.image(img_id_2).camera_id),
                        )[1]) @ T_1
                    )
        self.c = 3
        conn.close()

    def __len__(self):
        return self.keypoints.size(0)

    def __getitem__(self, idx):
        return {
            "keypoint":  self.keypoints[idx],
            "keypoint_coords": self.keypoint_coords[idx],
            "scale": self.scales[idx],
        }

    def get_collate_func(self):
        def collate_colmap(batch):
            keypoints = torch.stack([item["keypoint"] for item in batch])
            coords = torch.stack([item["keypoint_coords"] for item in batch])
            scales = torch.stack([item["scale"] for item in batch])
            return {
                "keypoints": keypoints,
                "keypoint_coords": coords,
                "scales": scales,
                "images": self.images,
                "fundamental": self.fundamental
            }
        return collate_colmap

    def img_id_ordering(self, img_ids):
        min_matches = None
        match_mat = torch.zeros(len(img_ids), len(img_ids), dtype=int)
        for i in range(len(img_ids)):
            img_1_features = set(map(lambda p: p.point3D_id, filter(lambda p: p.has_point3D(), self.reconstruction.image(img_ids[i]).points2D)))
            for j in range(i+1, len(img_ids)):
                img_2_features = set(map(lambda p: p.point3D_id, filter(lambda p: p.has_point3D(), self.reconstruction.image(img_ids[j]).points2D)))
                num_matches = len(img_1_features & img_2_features)
                match_mat[i, j] = num_matches
                match_mat[j, i] = num_matches
                if min_matches is None or num_matches < match_mat[*min_matches]:
                    min_matches = i, j
        ordered_imgs = []
        curr = min_matches[0]
        while len(ordered_imgs) < len(img_ids):
            ordered_imgs.append(img_ids[curr])
            match_mat[:, curr] = 0
            curr = torch.argmax(match_mat[curr])
            print(f"next image {img_ids[curr]} with {torch.max(match_mat[curr])} matches")
        return ordered_imgs
