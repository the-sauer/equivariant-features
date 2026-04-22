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
import os
from typing import Iterable

import kornia
import omegaconf
import torch
from torchvision.transforms import v2

from ..train import OPTIMIZERS, LOSSES


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


def train(model, dataset, cfg, experiment_name="default"):
    os.makedirs(os.path.join(cfg.logging.dir, experiment_name), exist_ok=True)
    checkpoint_dir = os.path.join(cfg.logging.dir, experiment_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(cfg.logging.dir, experiment_name, "cfg.yaml"), "w") as f:
        f.write(omegaconf.OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = OPTIMIZERS[cfg.training.optimizer.name](model.parameters(), **cfg.training.optimizer.params)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )

    criterion = LOSSES[cfg.training.loss]()

    augmentation = v2.Compose([
        v2.ColorJitter(**cfg.training.augmentation.color_jitter),
        v2.GaussianBlur(
            cfg.training.augmentation.gaussian_blur.kernel_size,
            sigma=cfg.training.augmentation.gaussian_blur.sigma
        ),
        v2.GaussianNoise(**cfg.training.augmentation.gaussian_noise),
    ])
    for epoch in range(cfg.training.num_epochs):
        batch_counter = 0
        for (img, img_t, H, H_inv) in train_loader:
            img = img.to(device)
            img_t = augmentation(img_t.to(device))
            optimizer.zero_grad()

            feature_map = model(img)
            feature_map_t = model(img_t)

            if "detector" in cfg.training.feature_sampling:
                # TODO: Fix this
                H = H.to(device)
                detections = detect(img)
                detections_t = warp_detections(detections, H)

                valid_detections = map(
                    lambda ds:
                        (ds[:, 0] >= 0) & (ds[:, 0] < img.shape[2])
                        & (ds[:, 1] >= 0) & (ds[:, 1] < img.shape[3]),
                    detections_t
                )

                detections = map(lambda e: e[0][e[1]], zip(detections, valid_detections))
                detections_t = map(lambda e: e[0][e[1]], zip(detections_t, valid_detections))
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
            else:
                if "stride" in cfg.training.feature_sampling:
                    feature_stride = cfg.training.feature_sampling.stride
                elif "num_features" in cfg.training.feature_sampling:
                    feature_stride = img.size(2) * img.size(3) // cfg.training.feature_sampling.num_features
                else:
                    raise ValueError("No valid feature sampling method")

                H_inv = H_inv.to(device)
                feature_map_t = kornia.geometry.transform.warp_perspective(feature_map, H_inv, dsize=feature_map.shape[2:])

                y = torch.cat((
                    feature_map.permute(0, 2, 3, 1).reshape(-1, feature_map.size(1))[::feature_stride, :],
                    feature_map_t.permute(0, 2, 3, 1).reshape(-1, feature_map_t.size(1))[::feature_stride, :]
                ))
                assert y.size(0) % 2 == 0
                labels = torch.cat((
                    torch.arange(y.size(0) // 2),
                    torch.arange(y.size(0) // 2)
                )).to(device)

            loss = criterion(y, labels)

            logging.info(f"Loss: {loss.item()}")
            loss.backward()
            optimizer.step()
            batch_counter += 1
            if batch_counter % cfg.logging.interval == 0:
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:03d}_{batch_counter//cfg.logging.interval:06d}.pth"))  
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))
