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


def process_batch_blobs_graph(model, data, criterion, augmentation, device, cfg, validation=False, **_):
    data["ids0"] = data["ids0"].to(device)
    data["keypoints0"] = data["keypoints0"].to(device)
    data["ids1"] = data["ids1"].to(device)
    data["keypoints1"] = data["keypoints1"].to(device)

    eq = data["ids0"][:, :, None] == data["ids1"][:, None, :]
    data["gt_assignment"] = eq
    data["gt_matches0"] = torch.where(eq.any(2), eq.int().argmax(2), -1)
    data["gt_matches1"] = torch.where(eq.any(1), eq.int().argmax(1), -1)

    def describe(patches, chunk=512):
        B, N = patches.shape[:2]
        flat = patches.flatten(end_dim=1).unsqueeze(1)
        out = torch.cat([model.descriptor(flat[i:i + chunk].to(device)) for i in range(0, flat.shape[0], chunk)])
        return out.reshape(B, N, -1)

    with torch.no_grad():
        desc0 = describe(data["patches0"])
        desc1 = describe(data["patches1"])

    pred = model.graph({
        "descriptors0": desc0,
        "keypoints0": data["keypoints0"],
        "descriptors1": desc1,
        "keypoints1": data["keypoints1"],
        "view0": {
            "image_size": (1, 1)
        },
        "view1": {
            "image_size": (1, 1)
        },
    })

    losses, metrics = model.graph.loss(pred, data)

    if metrics == {}:
        return {"loss": (losses["total"].mean(), 1.0, True)}
    else:
        return {
            k: (v, 1.0, True) for (k, v) in metrics.items()
        }
