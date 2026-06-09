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


class SiftScaleLoss(torch.nn.Module):
    def forward(self, x, **_):
        scales = x["scales"]
        assert torch.all(scales > 0), "Scales must be positive"
        if "detections" in x:
            detections = x["detections"]
            detected_scales = torch.linalg.det(detections[:, :2, :2])
            detected_scales_mask = torch.isnan(detected_scales)
            scales = scales * 2
        elif "pred_scales" in x:
            detected_scales = x["pred_scales"]
            detected_scales_mask = torch.isnan(detected_scales)
        else:
            raise ValueError("Input must contain either 'detections' or 'pred_scales'")

        return torch.mean((detected_scales[~detected_scales_mask] - scales[~detected_scales_mask]) ** 2)
