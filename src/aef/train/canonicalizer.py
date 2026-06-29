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

import kornia

from .detector import homogenize
from .losses.contrastive import Contrastive


def process_batch_canocalize(model, data, criterion, augmentation, device, cfg):
    assert "affine_shape" in data, "canonicalization requires ground truth shape to be available"

    patch = data["patch"]
    label = data["label"]
    affine_shape = data["affine_shape"]

    A = model.canonicalization(patch)
    if any(isinstance(c, Contrastive) for c, _, _ in criterion):
        canonicalized_patch = kornia.transforms.geometry.warp_perspective(
            patch,
            homogenize(A),
            (cfg.training.patch_size, cfg.training.patch_size)
        )
        features = model.descriptor(canonicalized_patch)
        return {
            n: (criterion({
                "features": features,
                "indices": label,
                "detections": A,
                "affine_shape": affine_shape
            }), weight, report) for n, (criterion, weight, report) in criterion.items()
        }
    else:
        return {
            n: (criterion({
                "detections": A,
                "affine_shape": affine_shape
            }), weight, report) for n, (criterion, weight, report) in criterion.items()
        }
