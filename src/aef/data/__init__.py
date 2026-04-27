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

from ..transforms import random_affine


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
    def __init__(self, images: Union[str, torch.Tensor], image_size: tuple[int, int] = (128, 128), in_memory=True, transform_params=None):
        super().__init__()
        if transform_params is None:
            transform_params = {}
        self.size = image_size
        if type(images) is torch.Tensor:
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

        self.transforms = random_affine(len(self.images), image_size=image_size, **transform_params)
        self.transforms_inv = torch.linalg.inv(self.transforms)
        if in_memory:
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

    def __getitem__(self, index):
        if type(self.images[index]) is str:
            img = self.resize(torchvision.io.decode_image(self.images[index]).unsqueeze(0).to(torch.float32) / 255)
            if img.size(1) == 1:
                transformed = kornia.geometry.transform.warp_perspective(
                    img.expand(-1, 3, -1, -1),
                    self.transforms,
                    self.size,
                    padding_mode="fill",
                    fill_value=torch.ones((3,))
                )[:, :1, ...]
            elif img.size(1) == 3:
                transformed = kornia.geometry.transform.warp_perspective(
                    img,
                    self.transforms[index].unsqueeze(0),
                    self.size,
                    padding_mode="fill",
                    fill_value=torch.ones((3,))
                )
            else:
                raise ValueError(f"Unsupported number of image channels: c={self.images.size(1)}")
            return (img.squeeze(0), transformed.squeeze(0), self.transforms[index], self.transforms_inv[index])
        else:
            return (self.images[index], self.images_transformed[index], self.transforms[index], self.transforms_inv[index])

    def __len__(self):
        return len(self.images)
