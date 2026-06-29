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

from .asel.affine import EquivarLayer_affine_resnet32
from .hardnet import HardNet


class BlobCanonicalization:
    def __init__(self):
        self.canonicalizer = EquivarLayer_affine_resnet32((1, 64, 64))
        self.descriptor = HardNet(patch_size=64)

    def to(self, device):
        self.canonicalizer = self.canonicalizer.to(device)
        self.descriptor = self.descriptor.to(device)
        return self
