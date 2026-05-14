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

import itertools
import logging
import os
from typing import Callable

import omegaconf
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from .losses import Loss


ProcessBatchType = Callable[
    [
        torch.nn.Module,
        tuple[torch.Tensor, ...],
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
        torch.device,
        omegaconf.DictConfig
    ],
    torch.Tensor
]


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
            OPTIMIZERS[opt_cfg.feature_net.name](itertools.chain(model.feature_net.parameters(), model.scale_gain.parameters()), **opt_cfg.feature_net.get("params", {}))
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
    validation_criterion = Loss(cfg.validation.loss)

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
        augmentation = lambda x: x  # noqa: E731

    return model, optimizer, scheduler, criterion, validation_criterion, train_loader, validation_loader, augmentation, device, checkpoint_dir


def train_func(process_batch: ProcessBatchType):
    def train(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
        (
            model,
            optimizer,
            scheduler,
            criterion,
            validation_criterion,
            train_loader,
            validation_loader,
            augmentation,
            device,
            checkpoint_dir
        ) = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)

        best_loss = float("inf")
        for epoch in range(cfg.training.num_epochs):
            loop = tqdm(train_loader, leave=True)
            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            model.train()
            cumulative_loss = 0.0
            for i, data in enumerate(loop):
                for opt in optimizer:
                    opt.zero_grad()

                loss = process_batch(model, data, criterion, augmentation, device, cfg)
                cumulative_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                loss.backward()
                for opt in optimizer:
                    opt.step()
                if hasattr(cfg, "logging") and hasattr(cfg.logging, "interval") and i % cfg.logging.interval == 0:
                    average_loss = cumulative_loss / (i + 1)
                    logging.info("epoch [%d/%d] batch [%d/%d] loss: %f", epoch, cfg.training.num_epochs, i, len(train_loader), average_loss)

            loop = tqdm(validation_loader, leave=True)
            loop.set_description(f"Validating [{epoch}/{cfg.training.num_epochs}]")

            with torch.no_grad():
                model.eval()
                cumulative_loss = 0.0
                for data in loop:
                    loss = process_batch(model, data, validation_criterion, lambda x: x, device, cfg)

                    cumulative_loss += loss.item() * data[0].size(0)
                    loop.set_postfix(loss=loss.item())

                logging.info("finished epoch [%d/%d], avg loss: %f", epoch, cfg.training.num_epochs, cumulative_loss / len(validation_dataset))

                if checkpoint_dir is not None:
                    average_loss = cumulative_loss / len(validation_dataset)
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": [opt.state_dict() for opt in optimizer],
                        "scheduler_state_dict": [sch.state_dict() for sch in scheduler],
                        "loss": average_loss,
                        "best_loss": best_loss
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))
                    if average_loss < best_loss:
                        best_loss = average_loss
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"best.pth"))
                        msg = f"New best model with loss {best_loss:.6f} at epoch {epoch} saved to {os.path.join(checkpoint_dir, f"best.pth")}"
                        print("\033[1m" + msg + "\033[0m")
                        logging.info("\033[1m" + msg + "\033[0m")
                for sch in scheduler:
                    sch.step()

    return train
