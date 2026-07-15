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

import glob
import os
import random

import blobboards
import omegaconf
import pint
import torch
import torchvision

from .homography import HomographyData, dataset_cache_key


_ureg = pint.UnitRegistry()
_dpi = 1 / _ureg.inch


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
        image_size=(150, 150),
        blobboard_params=None,
        suffix="train",
        polarity="dark",
        resolution=300,
        seeds=None,  # explicit board seeds; pins the board(s) so e.g. scale-split validation sets share one board
        cache_dir=None,  # if set: reuse a prepared dataset from here, else build it and save it there
        **kwargs
    ):
        # Check the cache *before* rendering any boards — board generation (Julia) is
        # the slowest step, so the key must be derived from the params alone. Note the
        # cache freezes this dataset's random draws (homographies, and unpinned board
        # seeds / "random" polarity): a hit reuses the exact same data.
        cache_path = None
        if cache_dir is not None:
            key = dataset_cache_key(dict(
                cls=type(self).__name__, num_boards=num_boards, image_size=image_size,
                blobboard_params=blobboard_params, suffix=suffix, polarity=polarity,
                resolution=resolution, seeds=seeds, **kwargs,
            ))
            cache_path = os.path.join(cache_dir, f"blobboard_{key}.pt")
            if os.path.exists(cache_path):
                super().__init__(None, cache_path=cache_path, **kwargs)
                return

        if blobboard_params is None:
            blobboard_params = {}
        if polarity == "dark":
            polarity = ["dark"] * num_boards
        elif polarity == "light":
            polarity = ["light"] * num_boards
        else:
            assert polarity == "random", "Polarity must be 'dark', 'light', or 'random'."
            polarity = [random.choice(["dark", "light"]) for _ in range(num_boards)]
        kwargs.setdefault("in_memory", False)
        if not isinstance(blobboard_params, dict):
            blobboard_params = omegaconf.OmegaConf.to_container(blobboard_params, resolve=True)
        if "board_size" in blobboard_params:
            blobboard_params["board_size"] = (
                blobboard_params["board_size"][0] * _ureg.mm,
                blobboard_params["board_size"][1] * _ureg.mm,
            )
        if "frame_width" in blobboard_params:
            blobboard_params["frame_width"] = blobboard_params["frame_width"] * _ureg.mm
        blobboard_params["dir"] = os.path.join(kwargs.get("data_dir", "."), "blobboards")
        blobboard_params["format"] = "png"
        out_dir = blobboard_params["dir"]
        os.makedirs(out_dir, exist_ok=True)
        if seeds is not None:
            board_seeds = list(seeds)
            assert num_boards <= len(board_seeds), (
                f"Expected at least {num_boards} seeds, got {len(board_seeds)}"
            )
            # Allow a seed pool larger than num_boards: take the first num_boards
            # so a single shared seed list can back configs with fewer boards.
            board_seeds = board_seeds[:num_boards]
        else:
            board_seeds = get_seeds(num_boards, split=suffix)
        boards = []
        for s, p in zip(board_seeds, polarity):
            board = blobboards.blob_board(
                (image_size[0] * _ureg.mm, image_size[1] * _ureg.mm),
                resolution, seed=s, polarity=p, image_origin="opencv",
                **blobboard_params,
            )
            # Current BlobBoards bindings render the board raster to disk rather
            # than returning it, so board.image is None. Load the just-written
            # PNG (the newest matching file, since boards are created serially)
            # and attach it. Its canvas frame matches the pattern->canvas offset
            # applied to the blob coordinates below.
            if board.image is None:
                pngs = glob.glob(os.path.join(out_dir, "blob_board_*.png"))
                if not pngs:
                    raise RuntimeError(
                        f"blob_board wrote no PNG to {out_dir!r}; cannot recover board image"
                    )
                png = max(pngs, key=os.path.getmtime)
                raster = torchvision.io.decode_image(
                    png, torchvision.io.ImageReadMode.GRAY
                )
                board.image = (raster.squeeze(0).to(torch.float32) / 255.0).numpy()
            boards.append(board)
        print(str(blobboard_params))
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
                torch.tensor(board.blobs, dtype=torch.float32) + torch.tensor([
                    (board.image.shape[1] - board.preamble.pattern_config.width) / 2.0,
                    (board.image.shape[0] - board.preamble.pattern_config.height) / 2.0,
                    0.0,
                ], dtype=torch.float32),
                torch.zeros((max_blobs-len(board.blobs), 3), dtype=torch.float32)
            ]) for i, board in enumerate(boards)
        ])
        super().__init__(
            torch.stack([
                torch.tensor(board.image, dtype=torch.float32).unsqueeze(0) for board in boards
            ]),
            image_size=boards[0].image.shape,
            gt_keypoint_coords=blob_tensor[..., :2],
            gt_keypoint_scales=blob_tensor[..., 2],
            gt_keypoint_mask=blob_mask,
            cache_path=cache_path,
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
