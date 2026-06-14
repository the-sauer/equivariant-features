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

import functools
import os
from typing import Iterable, Union

import kornia
import torch
import torchvision

from ..transforms.homography import sample_homography


def flip(f):
    def flipped_f(y, x):
        return f(x, y)
    return flipped_f


def curry(f):
    def function(x):
        def inner(y):
            return f(x, y)
        return inner
    return function


def uncurry(f):
    def function(x, y):
        return f(y)(x)
    return function


def fchain(*args):
    def function(x):
        for f in reversed(args):
            x = f(x)
        return x
    return function


def find_images(dir, extensions=[".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]) -> Iterable[str]:
    return fchain(
        curry(filter)(lambda file: any(map(lambda e: file.endswith(e), extensions))),
        curry(functools.reduce)(list.__add__),
        curry(map)(lambda x: list(map(lambda f: os.path.join(x[0], f), x[2]))),
    )(os.walk(dir))


def load_images(dir,  size, extensions=[".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]) -> torch.Tensor:
    resize = torchvision.transforms.Resize(size)
    return fchain(
        torch.stack,
        list,
        curry(map)(resize),
        curry(map)(torchvision.io.decode_image),
    )(find_images(dir, extensions))


class HomographyData(torch.utils.data.Dataset):
    images: torch.Tensor | list[str]
    transforms: torch.Tensor
    transforms_inv: torch.Tensor
    size: tuple[int, int]
    c: int

    def __init__(
        self,
        images: Union[str, torch.Tensor],
        image_size: tuple[int, int] = (512, 512),
        in_memory=True,
        transform_params=None,
        transforms_per_image=1,
        sift_batch_size=100,
        sift_min_response_threshold=0.03,
        features_per_image=500,
        **_
    ):
        super().__init__()
        if transform_params is None:
            transform_params = {}
        self.size = image_size
        self.in_memory = in_memory
        if isinstance(images, torch.Tensor):
            self.images = images
            self.c = self.images.size(1)
        else:
            if in_memory:
                self.images = load_images(images, size=image_size).to(torch.float32) / 255
                self.c = self.images.size(1)
            else:
                self.images = list(find_images(images))
                self.resize = torchvision.transforms.Resize(image_size)
                self.c = torchvision.io.decode_image(self.images[0]).size(0)

        self.transforms = torch.stack([torch.stack([torch.Tensor(sample_homography(image_size, **transform_params)) for _ in range(transforms_per_image)]) for _ in range(len(self.images))])
        self.transforms_inv = torch.linalg.inv(self.transforms)
        if in_memory or isinstance(self.images, torch.Tensor):
            raise RuntimeError("not implemented")
            if self.images.size(1) == 1:
                self.images_transformed = kornia.geometry.transform.warp_perspective(
                    self.images.expand(-1, 3, -1, -1),
                    self.transforms,
                    image_size,
                    padding_mode="fill",
                    fill_value=torch.ones((3,))
                )[:, :1, ...]
            elif self.images.size(1) == 3:
                self.images_transformed = kornia.geometry.transform.warp_perspective(
                    self.images,
                    self.transforms,
                    image_size,
                    padding_mode="fill",
                    fill_value=torch.ones((3,))
                )
            else:
                raise ValueError(f"Unsupported number of image channels: c={self.images.size(1)}")
        
        with torch.no_grad():
            detector = kornia.feature.ScaleSpaceDetector(
                num_features=features_per_image,
                # # scale_pyr_module=kornia.geometry.transform.ScalePyramid(n_levels=3, init_sigma=1.6),
                # resp_module=kornia.feature.BlobDoG(),
                minima_are_also_good=True,
                # mr_size=6.0
            )
            keypoints = []
            keypoint_coord_list = []
            keypoint_scale_list = []
            for i in range(0, len(self.images), sift_batch_size):
                if i + sift_batch_size > len(self.images):
                    actual_sift_batch_size = len(self.images) - i
                else:
                    actual_sift_batch_size = sift_batch_size

                if in_memory:
                    img = self.images[i:i+actual_sift_batch_size].cuda()
                else:
                    img = torch.stack([self.resize(torchvision.io.decode_image(p).to(torch.float32) / 255) for p in self.images[i:i+sift_batch_size]], dim=0).cuda()
                
                img = torchvision.transforms.functional.rgb_to_grayscale(img)
                lafs, responses = detector(img)

                keypoint_coords = kornia.feature.get_laf_center(lafs)
                keypoint_scales = kornia.feature.get_laf_scale(lafs).squeeze()
                keypoint_mask = (responses > sift_min_response_threshold)
                img_ids = torch.stack([torch.full((features_per_image,), (i + j) * (transforms_per_image + 1), dtype=torch.int64).cuda() for j in range(actual_sift_batch_size)], dim=0)
                feature_ids = torch.stack([torch.arange(features_per_image, dtype=torch.int64).cuda() for _ in range(actual_sift_batch_size)], dim=0) + (img_ids << 32)

                keypoints.append(torch.stack([img_ids, feature_ids], dim=-1)[keypoint_mask].cpu())
                keypoint_coord_list.append(keypoint_coords[keypoint_mask].cpu())
                keypoint_scale_list.append(keypoint_scales[keypoint_mask].cpu())

            self.keypoints = torch.cat(keypoints)
            self.keypoint_coords = torch.cat(keypoint_coord_list)
            self.keypoint_scales = torch.cat(keypoint_scale_list)
        avg_keypoints_per_image = self.keypoints.size(0) / len(self.images)
        print(f"{avg_keypoints_per_image=}")

    def __getitem__(self, index):
        keypoint_i = index // (self.transforms.size(1) + 1)
        homography_j = index % (self.transforms.size(1) + 1)
        homography_i = self.keypoints[keypoint_i, 0] // (self.transforms.size(1) + 1)
        if homography_j == self.transforms.size(1):
            keypoint_coords = self.keypoint_coords[keypoint_i]
        else:
            keypoint_coords = (self.transforms[homography_i, homography_j] @ torch.cat([self.keypoint_coords[keypoint_i], torch.ones((1,))], dim=-1).unsqueeze(-1))
            keypoint_coords = (keypoint_coords[:2] / keypoint_coords[2:]).squeeze(-1)
        return {
            "keypoint": torch.stack([self.keypoints[keypoint_i, 0] + homography_j, self.keypoints[keypoint_i, 1]]),
            "keypoint_coords": keypoint_coords,
            "scales": self.keypoint_scales[keypoint_i],   # TODO: Adjust to scale after transformation
            "homographies": self.transforms[homography_i, homography_j] if homography_j < self.transforms.size(1) else None,
        }

    def __len__(self):
        return len(self.keypoints) * (self.transforms.size(1) + 1)

    def get_collate_func(self):
        def collate_homography(batch):
            img_ids = {item["keypoint"][0].item() for item in batch}
            imgs = {}
            for img_id in img_ids:
                if img_id % (self.transforms.size(1) + 1) == 0:
                    imgs[img_id] = self.resize(torchvision.io.decode_image(self.images[img_id // (self.transforms.size(1) + 1)]).to(torch.float32) / 255) if not self.in_memory else self.images[img_id // (self.transforms.size(1) + 1)]
                else:
                    if self.in_memory:
                        img_pretransformed = self.images[img_id // (self.transforms.size(1) + 1)]
                    else:
                        img_pretransformed = self.resize(torchvision.io.decode_image(self.images[img_id // (self.transforms.size(1) + 1)]).to(torch.float32) / 255)
                    imgs[img_id] = kornia.geometry.transform.warp_perspective(
                        img_pretransformed.unsqueeze(0),
                        torch.linalg.inv(self.transforms[img_id // (self.transforms.size(1) + 1), img_id % (self.transforms.size(1) + 1) - 1]).unsqueeze(0),
                        self.size,
                        padding_mode="fill",
                        fill_value=torch.ones(3,)
                    ).squeeze(0)

            return {
                "keypoints": torch.stack([item["keypoint"] for item in batch]),
                "keypoint_coords": torch.stack([item["keypoint_coords"] for item in batch]),
                # "scales": torch.stack([item["scales"] for item in batch]),
                "images": imgs,
            }

        return collate_homography
