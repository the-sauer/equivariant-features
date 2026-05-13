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

import logging
import os

import omegaconf
import torch
from torchvision.transforms import v2

from .losses import Loss


OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD
}


SCHEDULERS = {
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR
}


def prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name):
    """
    Prepares the training by setting up logging, device, model, optimizer, scheduler, data loaders, criterion and
    augmentation.
    """
    if hasattr(cfg, "logging") and cfg.logging is not None:
        os.makedirs(os.path.join(cfg.logging.dir, experiment_name), exist_ok=True)
        checkpoint_dir = os.path.join(cfg.logging.dir, experiment_name, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logfile = os.path.join(cfg.logging.dir, experiment_name, "training.log")
        logging.basicConfig(filename=logfile, level=logging.INFO, force=True)

        with open(os.path.join(cfg.logging.dir, experiment_name, "cfg.yaml"), "w", encoding="utf-8") as f:
            f.write(omegaconf.OmegaConf.to_yaml(cfg))
    else:
        checkpoint_dir = None

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

    model = model.to(device)

    opt_cfg = cfg.training.optimizer
    if hasattr(opt_cfg, "name"):
        optimizer = [OPTIMIZERS[opt_cfg.name](model.parameters(), **opt_cfg.get("params", {}))]
        if hasattr(opt_cfg, "scheduler"):
            sched_cfg = opt_cfg.scheduler
            scheduler = [SCHEDULERS[sched_cfg.name](optimizer[0], **sched_cfg.get("params", {}))]
        else:
            scheduler = []
    else:
        # TODO: Refactor optimizer initialization to be more flexible and less hardcoded.
        optimizer = [
            OPTIMIZERS[opt_cfg.scale_space.name](model.scale_space.parameters(), **opt_cfg.scale_space.get("params", {})),
            OPTIMIZERS[opt_cfg.feature_net.name](model.feature_net.parameters(), **opt_cfg.feature_net.get("params", {}))
        ]
        scheduler = []
        if hasattr(opt_cfg.scale_space, "scheduler"):
            scheduler_kwargs = opt_cfg.scale_space.scheduler.get("params", {})
            scheduler.append(
                SCHEDULERS[opt_cfg.scale_space.scheduler.name](optimizer[0], **scheduler_kwargs)
            )
        if hasattr(opt_cfg.feature_net, "scheduler"):
            scheduler_kwargs = opt_cfg.feature_net.scheduler.get("params", {})
            scheduler.append(
                SCHEDULERS[opt_cfg.feature_net.scheduler.name](optimizer[1], **scheduler_kwargs)
            )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=True
    )

    criterion = Loss(cfg.training.loss)

    if hasattr(cfg.training, "augmentation") and cfg.training.augmentation is not None:
        # TODO: Check for all innner augmentations
        augmentation = v2.Compose([
            v2.ColorJitter(**cfg.training.augmentation.color_jitter),
            v2.GaussianBlur(
                cfg.training.augmentation.gaussian_blur.kernel_size,
                sigma=cfg.training.augmentation.gaussian_blur.sigma
            ),
            v2.GaussianNoise(**cfg.training.augmentation.gaussian_noise),
        ])
    else:
        augmentation = lambda x: x

    return model, optimizer, scheduler, criterion, train_loader, validation_loader, augmentation, device, checkpoint_dir
