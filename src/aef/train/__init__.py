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
from .canonicalizer import *
from .descriptor import *
from .detector import *
from .scale import *


class chain:
    def __init__(self, *args):
        self.iterators = args

    def __iter__(self):
        for it in self.iterators:
            for b in it:
                yield b

    def __len__(self):
        return sum(map(len, self.iterators))


def prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name):
    """
    Prepares the training by setting up logging, device, model, optimizer, scheduler, data loaders, criterion and
    augmentation.
    """
    print(f"Preparing experiment \033[1m{experiment_name}\033[0m")
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
        optimizer = {"main": getattr(torch.optim, opt_cfg.name)(model.parameters(), **opt_cfg.get("params", {}))}
        if hasattr(opt_cfg, "scheduler"):
            if isinstance(opt_cfg.scheduler, omegaconf.ListConfig):
                scheduler = { "main":
                    torch.optim.lr_scheduler.ChainedScheduler([
                        (
                            getattr(torch.optim.lr_scheduler, sched_cfg.name)(optimizer["main"], T_max=100, **sched_cfg.get("params", {}))
                            if sched_cfg.name == "CosineAnnealingLR"
                            else getattr(torch.optim.lr_scheduler, sched_cfg.name)(optimizer["main"], **sched_cfg.get("params", {}))
                        )
                        for sched_cfg in opt_cfg.scheduler
                    ])
                }
            else:
                sched_cfg = opt_cfg.scheduler
                if sched_cfg.name == "CosineAnnealingLR":
                    scheduler = {"main": getattr(torch.optim.lr_scheduler, sched_cfg.name)(optimizer["main"], 
                                                                                    T_max=100,
                                                                                    **sched_cfg.get("params", {}))}
                else:
                    scheduler = {"main": getattr(torch.optim.lr_scheduler, sched_cfg.name)(optimizer["main"], 
                                                                                   **sched_cfg.get("params", {}))}
        else:
            scheduler = {}
    else:
        optimizer = {
            o_name: getattr(torch.optim, o_cfg.name)(
                (getattr(model, o_cfg.model_params).parameters()
                    if isinstance(o_cfg.model_params, str)
                    else itertools.chain(*(getattr(model, mp).parameters() for mp in o_cfg.model_params)))
                if hasattr(o_cfg, "model_params") else model.parameters(),
                **o_cfg.get("params", {})
            )
            for o_name, o_cfg in opt_cfg.items()
        }
        scheduler = {}
        for n, o in opt_cfg.items():
            if hasattr(o, "scheduler"):
                sched_cfg = o.scheduler
                scheduler[n] = getattr(torch.optim.lr_scheduler, sched_cfg.name)(optimizer[n], **sched_cfg.get("params", {}))

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

    # Optional class-balanced sampling (m views per keypoint per batch) for
    # contrastive training; enabled by setting `training.m_per_class` in the
    # config. Falls back to plain shuffling for datasets/tasks that don't use it.
    m_per_class = getattr(cfg.training, "m_per_class", None)
    train_loader = []
    for d in train_dataset:
        sampler = (
            d.get_sampler(cfg.training.batch_size, m=int(m_per_class))
            if m_per_class and hasattr(d, "get_sampler")
            else None
        )
        train_loader.append(torch.utils.data.DataLoader(
            d,
            batch_size=cfg.training.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=d.get_collate_func()
        ))
    validation_loader = [torch.utils.data.DataLoader(
        d,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        collate_fn=d.get_collate_func()
    ) for d in validation_dataset]

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


def train_func(process_batch):
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
        print(f"Running experiment \033[1m{experiment_name}\033[0m")

        print("Training datasets", *(f"{d.__class__.__name__}: {len(d)}"for d in train_dataset))
        print("Validation datasets", *(f"{d.__class__.__name__}: {len(d)}"for d in validation_dataset))
        x = []
        y_train = {n: [] for n in criterion.keys() if criterion[n][2]}
        y_val = {n: [] for n in validation_criterion.keys() if validation_criterion[n][2]}

        lr = {sch.__class__.__name__: [] for sch in scheduler.values()}

        for epoch in range(start_epoch, cfg.training.num_epochs):
            loop = tqdm(chain(*train_loader), leave=True)
            loop.set_description(f"Training [{epoch}/{cfg.training.num_epochs}]")
            model.train()
            cumulative_loss = 0.0
            cumulative_losses = {n: 0.0 for n in criterion.keys() if criterion[n][2]}
            n_items = 0
            for i, data in enumerate(loop):
                for opt in optimizer.values():
                    opt.zero_grad()
                # try:
                losses = process_batch(model, data, criterion, augmentation, device, cfg, optimizer=optimizer)
                # except Exception as e:
                #     logging.error(f"Error processing batch {i} in epoch {epoch}: {e}")
                #     continue
                loop.set_postfix(**{n: l.item() for n, (l, _, r) in losses.items() if r})
                loss = torch.sum(torch.stack([l.view(1) * w for (l, w, _) in losses.values()]))
                
                # Check for NaN/Inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning("Loss is NaN or Inf at batch %d, skipping this batch", i)
                    continue
                    
                cumulative_losses = {n: cumulative_losses[n] + l.item() * data["keypoints"].size(0) for n, (l, _, r) in losses.items() if r}
                cumulative_loss += loss.item() * data["keypoints"].size(0)
                n_items += data["keypoints"].size(0)
                if loss.requires_grad:
                    try:
                        loss.backward()
                        # Check for NaN in gradients after backward
                        has_nan_grad = False
                        for param in model.parameters():
                            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                                has_nan_grad = True
                                break
                        if has_nan_grad:
                            logging.warning("NaN detected in gradients at batch %d, skipping update", i)
                            model.zero_grad()
                            continue
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        for opt in optimizer.values():
                            opt.step()
                    except RuntimeError as e:
                        logging.warning("Error during backward at batch %d: %s", i, str(e))
                        model.zero_grad()
                        continue
                if hasattr(cfg, "logging") and hasattr(cfg.logging, "interval") and i % cfg.logging.interval == 0 and i > 0:
                    logging.info("epoch [%d/%d] batch [%d/%d] losses: %s", epoch, cfg.training.num_epochs, i, len(train_loader), ", ".join(f"{n}: {v.item():.6f}" for n, (v, _, _) in losses.items()))
            avg_losses = {n: v / n_items for n, v in cumulative_losses.items()}
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

            loop = tqdm(chain(*validation_loader), leave=True)
            loop.set_description(f"Validating [{epoch}/{cfg.training.num_epochs}]")

            with torch.no_grad():
                model.eval()
                cumulative_loss = 0.0
                cumulative_losses = {n: 0.0 for n in validation_criterion.keys() if validation_criterion[n][2]}
                n_items = 0
                for data in loop:
                    # try:
                    losses = process_batch(model, data, validation_criterion, lambda x: x, device, cfg)
                    # except Exception as e:
                    #     logging.error(f"Error processing validation batch in epoch {epoch}: {e}")
                    #     continue
                    loss = torch.sum(torch.stack([l.view(1) * w for (l, w, _) in losses.values()]))

                    cumulative_losses = {n: cumulative_losses[n] + l.item() * data["keypoints"].size(0) for n, (l, _, r) in losses.items() if r}
                    cumulative_loss += loss.item() * data["keypoints"].size(0)
                    n_items += data["keypoints"].size(0)
                    loop.set_postfix(**{n: l.item() for n, (l, _, r) in losses.items() if r})
                y_val = {n: y_val[n] + [v / n_items] for n, v in cumulative_losses.items()}
                logging.info("finished epoch [%d/%d], avg losses: %s", epoch, cfg.training.num_epochs, ", ".join(f"{n}: {v / n_items}" for n, v in cumulative_losses.items()))

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
                    average_loss = cumulative_loss / n_items
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
                    # get_last_lr() works for composite schedulers (e.g.
                    # ChainedScheduler), whose get_lr() raises NotImplementedError.
                    lr[sch.__class__.__name__] += sch.get_last_lr()
                    sch.step()

        _, ax = plt.subplots()
        for n, v in lr.items():
            ax.plot(x, v, label=n)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.legend()
        plt.savefig(os.path.join(checkpoint_dir, "..", "learning_rate.svg"))
        plt.close()

    return train
