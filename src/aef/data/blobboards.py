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
import torch

from ..data import HomographyData


class BlobBoardHomographyData(HomographyData):
    def __init__(self, num_boards, blobboard_params=None, image_size=(128, 128), **kwargs):
        if blobboard_params is None:
            blobboard_params = {}
        kwargs["in_memory"] = True
        super().__init__(
            torch.stack([torch.Tensor(blobboards.blob_pattern(*image_size, **blobboard_params).pattern).unsqueeze(0) for _ in range(num_boards)]),
            image_size=image_size,
            **kwargs
        )


class BlobBoardAbsoluteScaleData(torch.utils.data.Dataset):
    def __init__(self, num_boards, blobboard_params=None, image_size=(128, 128)):
        super().__init__()
        if blobboard_params is None:
            blobboard_params = {}
        boards = [blobboards.blob_pattern(*image_size, **blobboard_params) for _ in range(num_boards)]
        self.data = torch.stack([torch.Tensor(board.pattern).unsqueeze(0) for board in boards])
        self.blobs = [torch.Tensor(board.blobs) for board in boards]
        self.c = self.data.size(1)

    def __getitem__(self, index):
        return self.data[index], self.blobs[index]

    def __len__(self):
        return self.data.shape[0]
