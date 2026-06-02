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

import torch


class ConstantData(torch.utils.data.Dataset):
    def __init__(self, value, image_size: tuple[int, int], n: int, c: int, **_):
        if value == "identity":
            self.value = torch.eye(2).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(n, -1, -1, *image_size)
        else:
            raise ValueError(f"Unsupported value: {value}")
        self.image_size = image_size
        self.c = c

    def __len__(self):
        return self.value.shape[0]

    def __getitem__(self, idx):
        return torch.rand(self.c, *self.image_size), self.value[idx]
