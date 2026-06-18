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

from .homography import HomographyData


def get_seeds(num_seeds, seed_range=(0, 10000), split="train"):
    seeds = []
    while len(seeds) < num_seeds:
        seed = random.randint(*seed_range)
        if (
            seed not in seeds
            and ((split == "validation" and seed % 10 == 0)
                 or (split == "test" and seed % 10 == 1)
                 or (split == "train" and seed % 10 > 1))
        ):
            seeds.append(seed)
    return seeds


class BlobBoardHomographyData(HomographyData):
    def __init__(
        self,
        num_boards,
        blobboard_params=None,
        image_size=(128, 128),
        suffix="train",
        polarity="dark",
        **kwargs
    ):
        if blobboard_params is None:
            blobboard_params = {}
        if blobboard_params is None:
            blobboard_params = {}
        if polarity == "dark":
            polarity = ["dark"] * num_boards
        elif polarity == "light":
            polarity = ["light"] * num_boards
        else:
            assert polarity == "random", "Polarity must be 'dark', 'light', or 'random'."
            polarity = [random.choice(["dark", "light"]) for _ in range(num_boards)]
        kwargs["in_memory"] = True
        boards = [blobboards.blob_pattern(*image_size, seed=s, polarity=p, **blobboard_params) for s, p in zip(get_seeds(num_boards, split=suffix), polarity)]
        blobs = [board.blobs for board in boards]
        max_blobs = max(map(len, blobs))
        blob_mask = torch.stack([
            torch.cat([
                torch.tensor([True], dtype=torch.bool).expand(len(board.blobs)),
                torch.tensor([False], dtype=torch.bool).expand(max_blobs-len(board.blobs))
            ]) for i, board in enumerate(boards)
        ])
        blob_tensor = torch.stack([
            torch.cat([
                torch.tensor(board.blobs, dtype=torch.float32) - 1,     # Julia convention to 0-based indexing
                torch.zeros((max_blobs-len(board.blobs), 3), dtype=torch.float32)
            ]) for i, board in enumerate(boards)
        ])
        super().__init__(
            torch.stack([torch.Tensor(board.pattern).unsqueeze(0) for board in boards]),
            image_size=image_size,
            gt_keypoint_coords=blob_tensor[..., :2],
            gt_keypoint_scales=blob_tensor[..., 2],
            gt_keypoint_mask=blob_mask,
            **kwargs
        )


class BlobBoardAbsoluteScaleData(torch.utils.data.Dataset):
    def __init__(self, num_boards, blobboard_params=None, image_size=(128, 128), split="train", polarity="dark"):
        super().__init__()
        if blobboard_params is None:
            blobboard_params = {}
        if polarity == "dark":
            polarities = ["dark"] * num_boards
        elif polarity == "light":
            polarities = ["light"] * num_boards
        else:
            assert polarity == "random", "Polarity must be 'dark', 'light', or 'random'."
            polarities = [random.choice(["dark", "light"]) for _ in range(num_boards)]
        boards = [
            blobboards.blob_pattern(*image_size, seed=s, polarity=p, **blobboard_params)
            for s, p in zip(get_seeds(num_boards, split=split), polarities)
        ]
        self.data = torch.stack([torch.Tensor(board.pattern).unsqueeze(0) for board in boards])
        self.num_blobs = torch.Tensor([len(board.blobs) for board in boards]).to(torch.int32)
        self.blobs = torch.empty(len(boards), torch.max(self.num_blobs).item(), 3, dtype=torch.float32)
        for i, board in enumerate(boards):
            self.blobs[i, :len(board.blobs)] = torch.Tensor(board.blobs)
        self.c = self.data.size(1)

    def __getitem__(self, index):
        return self.data[index], self.blobs[index], self.num_blobs[index]

    def __len__(self):
        return self.data.shape[0]
