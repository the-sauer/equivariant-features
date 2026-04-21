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

from enum import Enum
import logging
from typing import Iterable

import kornia
from pytorch_metric_learning import losses
import torch
import torchvision
from torchvision.transforms import v2


class Detector(Enum):
    DoG = 1
    Harris = 2


def detect(img: torch.Tensor, detector: Detector = Detector.Harris, threshold: float = 0.0001) -> Iterable[torch.Tensor]:
    img = torch.mean(img, dim=1, keepdim=True)
    if detector == Detector.DoG:
        response_map = kornia.feature.dog_response(img)
    elif detector == Detector.Harris:
        response_map = kornia.feature.harris_response(img)
    else:
        raise ValueError("Unknown detector")

    b, _, x, y = torch.where(response_map > threshold)
    logging.info(f"Detected {b.size(0)} features")
    splits = list(map(lambda i: int(torch.sum(b == i).item()), range(img.shape[0])))

    return torch.split(torch.stack((x, y), dim=1).to(img.device), split_size_or_sections=splits, dim=0)


def warp_detections(detections: Iterable[torch.Tensor], H: torch.Tensor) -> Iterable[torch.Tensor]:
    def coordinate_map(e: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        H, c = e
        if len(c) == 0:
            return c
        c_h = torch.cat((c.to(torch.float32), torch.tensor([[1.0]], device=c.device).expand(c.size(0), 1)), dim=1).unsqueeze(2)
        c_warped = (H.unsqueeze(0) @ c_h).squeeze(2)
        return torch.round(c_warped[:, :2] / c_warped[:, 2:]).to(torch.int)
    return map(coordinate_map, zip(H, detections))


def train(model, dataset, batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    criterion = losses.TripletMarginLoss()

    augmentation = v2.Compose([
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        v2.GaussianBlur(kernel_size=3),
        v2.GaussianNoise(),
    ])

    for (img, img_t, H, _) in train_loader:
        img = img.to(device)
        img_t = augmentation(img_t.to(device))
        H = H.to(device)
        optimizer.zero_grad()

        detections = detect(img)
        detections_t = warp_detections(detections, H)
        # TODO: Filter out of bounds detections
        feature_map = model(img)
        feature_map_t = model(img_t)

        features = torch.empty(sum(map(len, detections)), model.feature_size)
        features_t = torch.empty(sum(map(len, detections)), model.feature_size)

        i = 0
        for j, (ds, ds_t) in enumerate(zip(detections, detections_t)):
            for d, d_t in zip(ds, ds_t):
                features[i, :] = feature_map[j, :, *d].squeeze()
                features_t[i, :] = feature_map_t[j, :, *d_t].squeeze()
                i += 1
        y = torch.cat((features, features_t))
        labels = torch.cat((torch.arange(features.size(0)), torch.arange(features_t.size(0)))).to(device)
        loss = criterion(y, labels)

        logging.info(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
