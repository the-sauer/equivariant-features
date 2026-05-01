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

import random

import blobboards
import torch

from ..data import HomographyData


def get_seeds(num_seeds, seed_range=(0, 10000), split="train"):
    seeds = []
    while len(seeds) < num_seeds:
        seed = random.randint(*seed_range)
        if seed not in seeds and ((split == "validation" and seed % 10 == 0) or (split != "test" and seed % 10 == 1) or (split == "train" and seed % 10 > 1)):
            seeds.append(seed)
    return seeds


class BlobBoardHomographyData(HomographyData):
    def __init__(self, num_boards, blobboard_params=None, image_size=(128, 128), split="train", **kwargs):
        if blobboard_params is None:
            blobboard_params = {}
        kwargs["in_memory"] = True
        super().__init__(
            torch.stack([torch.Tensor(blobboards.blob_pattern(*image_size, seed=s, **blobboard_params).pattern).unsqueeze(0) for s in get_seeds(num_boards, split=split)]),
            image_size=image_size,
            **kwargs
        )


class BlobBoardAbsoluteScaleData(torch.utils.data.Dataset):
    def __init__(self, num_boards, blobboard_params=None, image_size=(128, 128), split="train"):
        super().__init__()
        if blobboard_params is None:
            blobboard_params = {}
        boards = [blobboards.blob_pattern(*image_size, seed=s, **blobboard_params) for s in get_seeds(num_boards, split=split)]
        self.data = torch.stack([torch.Tensor(board.pattern).unsqueeze(0) for board in boards])
        self.blobs = [torch.Tensor(board.blobs) for board in boards]
        self.c = self.data.size(1)

    def __getitem__(self, index):
        return self.data[index], self.blobs[index]

    def __len__(self):
        return self.data.shape[0]
