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

import omegaconf

try:
    from .blobboards import *
except ImportError as e:
    print("Could not import blobboards dataset. Make sure to install the BlobBoards.jl python bindings and to have the Julia package installed. Error was:", e)
from .colmap import *
from .constant import *
from .kaggle import *


def get_dataset(dataset_cfg):
    if isinstance(dataset_cfg, omegaconf.ListConfig):
        return [get_dataset(c) for c in dataset_cfg]
    elif "foreach" in dataset_cfg:
        return [globals[dataset_cfg.name](**dataset_cfg.params, path=os.path.join(dataset_cfg.foreach, d)) for d in os.listdir(dataset_cfg.foreach)]
    else:
        return [eval(f"{dataset_cfg.name}(**dataset_cfg.params)")]
