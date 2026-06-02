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

from .homography import HomographyData


class KaggleHomographyData(HomographyData):
    def __init__(self, name: str, data_dir=None, suffix="", **kwargs):
        if data_dir is None:
            data_dir = "./data/datasets"
        data_dir = os.path.join(data_dir, name)
        if not os.path.exists(data_dir) or os.path.exists(os.path.join(data_dir, f"{name}.archive")):
            data_dir = kagglehub.competition_download(name, output_dir=data_dir, force_download=True)
        data_dir = os.path.join(data_dir, suffix)
        super().__init__(data_dir, **kwargs)
