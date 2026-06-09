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

from matplotlib import pyplot as plt
import omegaconf
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from . import losses

from .descriptor import *
from .detector import *
from .scale import *


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
        logging.basicConfig(filename=logfile, level=logging.DEBUG, force=True)

        with open(os.path.join(cfg.logging.dir, experiment_name, "cfg.yaml"), "w", encoding="utf-8") as f:
            f.write(omegaconf.OmegaConf.to_yaml(cfg))
    else:
        checkpoint_dir = None

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))

    model = model.to(device)

    opt_cfg = cfg.training.optimizer
    if hasattr(opt_cfg, "name"):
        optimizer = {"main": OPTIMIZERS[opt_cfg.name](model.parameters(), **opt_cfg.get("params", {}))}
        if hasattr(opt_cfg, "scheduler"):
            sched_cfg = opt_cfg.scheduler
            scheduler = {"main": SCHEDULERS[sched_cfg.name](optimizer["main"], **sched_cfg.get("params", {}))}
        else:
            scheduler = {}
    else:
        optimizer = {
            n: OPTIMIZERS[o.name](
                (getattr(model, o.model_params).parameters()
                    if isinstance(o.model_params, str)
                    else itertools.chain(*(getattr(model, mp).parameters() for mp in o.model_params)))
                if hasattr(o, "model_params") else model.parameters(),
                **o.get("params", {})
            )
            for n, o in opt_cfg.items()
        }
        scheduler = {}
        for n, o in opt_cfg.items():
            if hasattr(o, "scheduler"):
                sched_cfg = o.scheduler
                scheduler[n] = SCHEDULERS[sched_cfg.name](optimizer[n], **sched_cfg.get("params", {}))

    if hasattr(cfg.training, "continue_from_checkpoint"):
        checkpoint = torch.load(cfg.training.continue_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        for n, o in optimizer.items():
            if n in checkpoint["optimizer_state_dict"]:
                try:
                    o.load_state_dict(checkpoint["optimizer_state_dict"][n])
                except ValueError as e:
                    logging.error(f"Error loading optimizer state dict for {n}: {e}")
        for n, s in scheduler.items():
            if n in checkpoint["scheduler_state_dict"]:
                try:
                    s.load_state_dict(checkpoint["scheduler_state_dict"][n])
                except ValueError as e:
                    logging.error(f"Error loading scheduler state dict for {n}: {e}")

        start_epoch = checkpoint["epoch"] + 1
        best_loss = float("inf") #checkpoint["best_loss"]
    else:
        start_epoch = 0
        best_loss = float("inf")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=train_dataset.get_collate_func()
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=cfg.validation.batch_size,
        shuffle=True,
        collate_fn=validation_dataset.get_collate_func()
    )

    criterion = {
        loss_cfg.name: (getattr(losses, loss_cfg.name)(**loss_cfg.get("params", {})), getattr(loss_cfg, "weight", 1.0), getattr(loss_cfg, "report", True)) for loss_cfg in cfg.training.loss
    }

    validation_criterion = {
        loss_cfg.name: (getattr(losses, loss_cfg.name)(**loss_cfg.get("params", {})), getattr(loss_cfg, "weight", 1.0), getattr(loss_cfg, "report", True)) for loss_cfg in cfg.validation.loss
    }

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

    return model, optimizer, scheduler, criterion, validation_criterion, train_loader, validation_loader, augmentation, device, checkpoint_dir, start_epoch, best_loss


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
            checkpoint_dir,
            start_epoch,
            best_loss
        ) = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)

        x = []
        y_train = {n: [] for n in criterion.keys() if criterion[n][2]}
        y_val = {n: [] for n in validation_criterion.keys() if validation_criterion[n][2]}

        for epoch in range(start_epoch, cfg.training.num_epochs):
            loop = tqdm(train_loader, leave=True)
            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            model.train()
            cumulative_loss = 0.0
            cumulative_losses = {n: 0.0 for n in criterion.keys() if criterion[n][2]}
            for i, data in enumerate(loop):
                for opt in optimizer.values():
                    opt.zero_grad()
                # try:
                losses = process_batch(model, data, criterion, augmentation, device, cfg)
                # except Exception as e:
                #     logging.error(f"Error processing batch {i} in epoch {epoch}: {e}")
                #     continue
                loop.set_postfix(**{n: l.item() for n, (l, _, r) in losses.items() if r})
                loss = torch.sum(torch.stack([l.view(1) * w for (l, w, _) in losses.values()]))
                cumulative_losses = {n: cumulative_losses[n] + l.item() * data["keypoints"].size(0) for n, (l, _, r) in losses.items() if r}
                cumulative_loss += loss.item()
                loss.backward()
                for opt in optimizer.values():
                    opt.step()
                if hasattr(cfg, "logging") and hasattr(cfg.logging, "interval") and i % cfg.logging.interval == 0 and i > 0:
                    logging.info("epoch [%d/%d] batch [%d/%d] losses: %s", epoch, cfg.training.num_epochs, i, len(train_loader), ", ".join(f"{n}: {v.item():.6f}" for n, (v, _, _) in losses.items()))
            avg_losses = {n: v / len(train_dataset) for n, v in cumulative_losses.items()}
            logging.info("finished epoch [%d/%d], avg losses: %s", epoch, cfg.training.num_epochs, ", ".join(f"{n}: {v:.6f}" for n, v in avg_losses.items()))
            x += [epoch]
            for n, v in y_train.items():
                v += [avg_losses[n]]
            _, ax = plt.subplots()
            for n, v in y_train.items():
                ax.plot(x, v, label=n)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Training Loss")
            ax.set_xlim(0, cfg.training.num_epochs)
            ax.set_ylim(bottom=0)
            ax.legend()
            plt.savefig(os.path.join(checkpoint_dir, "..", "train_losses.svg"))
            plt.close()

            loop = tqdm(validation_loader, leave=True)
            loop.set_description(f"Validating [{epoch}/{cfg.training.num_epochs}]")

            with torch.no_grad():
                model.eval()
                cumulative_loss = 0.0
                cumulative_losses = {n: 0.0 for n in validation_criterion.keys() if validation_criterion[n][2]}
                for data in loop:
                    # try:
                    losses = process_batch(model, data, validation_criterion, lambda x: x, device, cfg)
                    # except Exception as e:
                    #     logging.error(f"Error processing validation batch in epoch {epoch}: {e}")
                    #     continue
                    loss = torch.sum(torch.stack([l.view(1) * w for (l, w, _) in losses.values()]))

                    cumulative_losses = {n: cumulative_losses[n] + l.item() * data["keypoints"].size(0) for n, (l, _, r) in losses.items() if r}
                    cumulative_loss += loss.item() * data["keypoints"].size(0)
                    loop.set_postfix(**{n: l.item() for n, (l, _, r) in losses.items() if r})
                y_val = {n: y_val[n] + [v / len(validation_dataset)] for n, v in cumulative_losses.items()}
                logging.info("finished epoch [%d/%d], avg losses: %s", epoch, cfg.training.num_epochs, ", ".join(f"{n}: {v / len(validation_dataset)}" for n, v in cumulative_losses.items()))

                _, ax = plt.subplots()
                for n, v in y_val.items():
                    ax.plot(x, v, label=n)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Average Validation Loss")
                ax.set_xlim(0, cfg.training.num_epochs)
                ax.set_ylim(bottom=0)
                ax.legend()
                plt.savefig(os.path.join(checkpoint_dir, "..", "validation_losses.svg"))
                plt.close()

                if checkpoint_dir is not None:
                    average_loss = cumulative_loss / len(validation_dataset)
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": {n: o.state_dict() for n, o in optimizer.items()},
                        "scheduler_state_dict": {n: s.state_dict() for n, s in scheduler.items()},
                        "loss": average_loss,
                        "best_loss": best_loss,
                        "plots": {
                            "y_train": y_train,
                            "y_val": y_val,
                            "x": x
                        }
                    }
                    torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))
                    if average_loss < best_loss:
                        best_loss = average_loss
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"best.pth"))
                        msg = f"New best model with loss {best_loss:.6f} at epoch {epoch} saved to {os.path.join(checkpoint_dir, f"best.pth")}"
                        print("\033[1m" + msg + "\033[0m")
                        logging.info("\033[1m" + msg + "\033[0m")
                for sch in scheduler.values():
                    sch.step()

    return train
