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
import math
import os
from typing import Callable

from matplotlib import pyplot as plt
import omegaconf
import torch
from tqdm import tqdm

from . import losses
from .canonicalizer import *
from .descriptor import *


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
        # Defaults to <log_dir>/<experiment_name>/checkpoints; override with
        # `logging.checkpoint_dir`.
        checkpoint_dir = getattr(cfg.logging, "checkpoint_dir", None) or os.path.join(
            cfg.logging.dir, experiment_name, "checkpoints"
        )
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
                                                                                    T_max=50,
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
    elif getattr(cfg.training, "init_from_checkpoint", None):
        # Warm-start: load ONLY the model weights and start a FRESH run (epoch 0, new
        # optimizer/scheduler). Distinct from `continue_from_checkpoint`, which resumes a
        # run in place. Used to seed track training from a synthetic-descriptor checkpoint
        # (see launch_track_matrix.sh). strict=False tolerates head/shape drift between
        # the source model and this run's model, reporting what was skipped.
        checkpoint = torch.load(cfg.training.init_from_checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=False)
        msg = (f"Warm-started model from {cfg.training.init_from_checkpoint} "
               f"(missing={len(missing)}, unexpected={len(unexpected)} keys)")
        print("\033[1m" + msg + "\033[0m")
        logging.info(msg)
        start_epoch = 0
        best_loss = float("inf")
    else:
        start_epoch = 0
        best_loss = float("inf")

    # Optional class-balanced sampling (m views per keypoint per batch) for
    # contrastive training; enabled by setting `training.m_per_class` in the
    # config. Falls back to plain shuffling for datasets/tasks that don't use it.
    m_per_class = getattr(cfg.training, "m_per_class", None)

    # DataLoader worker config. With num_workers>0 the data pipeline (index the
    # precomputed patches, collate, H2D copy) overlaps with the GPU forward/backward
    # instead of running synchronously in the main process — the main lever when the
    # GPU is starved by single-process loading. persistent_workers keeps the train
    # loader's workers alive across epochs to avoid re-forking the large in-memory
    # dataset each epoch; validation uses fewer, non-persistent workers since its
    # several small loaders are iterated sequentially. Sizes come from
    # `training.num_workers` / `validation.num_workers` (match Slurm `-c` to them).
    def _loader_kwargs(num_workers, persistent):
        num_workers = int(num_workers)
        kw = {"num_workers": num_workers, "pin_memory": True}
        if num_workers > 0:
            kw["persistent_workers"] = persistent
            kw["prefetch_factor"] = 2
        return kw

    train_workers = int(getattr(cfg.training, "num_workers", 8))
    val_workers = int(getattr(cfg.validation, "num_workers", 4))

    train_loader = []
    for d in train_dataset:
        sampler = (
            d.get_sampler(cfg.training.batch_size, m=int(m_per_class))
            if m_per_class and hasattr(d, "get_sampler")
            else None
        )
        # A dataset may return a *batch* sampler (yields index lists) rather than a
        # per-sample sampler — BlobTrackData does, to guarantee positive pairs and
        # avoid the singleton-confuser duplication MPerClassSampler would introduce.
        # Those are mutually exclusive with batch_size/shuffle/sampler on DataLoader.
        if getattr(sampler, "is_batch_sampler", False):
            train_loader.append(torch.utils.data.DataLoader(
                d,
                batch_sampler=sampler,
                collate_fn=d.get_collate_func(),
                **_loader_kwargs(train_workers, persistent=True),
            ))
        else:
            train_loader.append(torch.utils.data.DataLoader(
                d,
                batch_size=cfg.training.batch_size,
                shuffle=sampler is None,
                sampler=sampler,
                collate_fn=d.get_collate_func(),
                **_loader_kwargs(train_workers, persistent=True),
            ))
    def build_criterion(loss_cfgs):
        return {
            loss_cfg.name: (
                getattr(losses, loss_cfg.name)(**loss_cfg.get("params", {})),
                getattr(loss_cfg, "weight", 1.0),
                getattr(loss_cfg, "report", True),
            )
            for loss_cfg in loss_cfgs
        }

    criterion = build_criterion(cfg.training.loss)

    # ``validation_dataset`` is a list of (label, [dataset], [loss_cfg]) specs.
    # Pair each dataset with its own criterion + loader(s) so their metrics are
    # accumulated and reported separately (keyed ``<loss>@<label>``) rather than
    # pooled into a single number across all validation datasets.
    #
    # A dataset that exposes ``get_eval_batch_sampler`` (BlobTrackData) drives a
    # class-balanced batch_sampler so every validation batch contains positive
    # pairs — a plain contiguous ``shuffle=False`` slice of a track file has its
    # matching observations scattered across sequences, so most batches would have
    # no positive and FPR95 would be NaN. ``validation.confuser_fraction`` (default
    # 0.5) sets how much of each batch is hard-negative confusers.
    val_conf_frac = float(getattr(cfg.validation, "confuser_fraction", 0.5))

    def _val_loader(d):
        if hasattr(d, "get_eval_batch_sampler"):
            batch_sampler = d.get_eval_batch_sampler(
                cfg.validation.batch_size, confuser_fraction=val_conf_frac
            )
            return torch.utils.data.DataLoader(
                d, batch_sampler=batch_sampler, collate_fn=d.get_collate_func(),
                **_loader_kwargs(val_workers, persistent=False))
        return torch.utils.data.DataLoader(
            d, batch_size=cfg.validation.batch_size, shuffle=False,
            collate_fn=d.get_collate_func(),
            **_loader_kwargs(val_workers, persistent=False))

    validation_entries = [
        (label, [_val_loader(d) for d in datasets], build_criterion(loss_cfgs))
        for label, datasets, loss_cfgs in validation_dataset
    ]

    # Augmentation is applied by the dataset (`training.dataset.params.augmentation`,
    # see HomographyData.compute_patches), which is also the only place a config sets
    # it — the process_batch functions ignore this argument.
    augmentation = lambda x: x  # noqa: E731

    return model, optimizer, scheduler, criterion, validation_entries, train_loader, augmentation, device, checkpoint_dir, start_epoch, best_loss


def train_func(process_batch):
    def train(model, train_dataset, validation_dataset, cfg, experiment_name="default"):
        (
            model,
            optimizer,
            scheduler,
            criterion,
            validation_entries,
            train_loader,
            augmentation,
            device,
            checkpoint_dir,
            start_epoch,
            best_loss
        ) = prepare_training(model, train_dataset, validation_dataset, cfg, experiment_name)
        print(f"Running experiment \033[1m{experiment_name}\033[0m")

        def report_key(label, name):
            return f"{name}@{label}" if label else name

        print("Training datasets", *(f"{d.__class__.__name__}: {len(d)}"for d in train_dataset))
        print("Validation datasets", *(
            f"{label or 'val'}[{d.__class__.__name__}]: {len(d)}"
            for label, loaders, _ in validation_entries for d in (ld.dataset for ld in loaders)))
        x = []
        y_train = {n: [] for n in criterion.keys() if criterion[n][2]}
        # One curve per (validation dataset, reported metric), keyed <metric>@<label>.
        val_keys = [report_key(label, n)
                    for label, _, crit in validation_entries
                    for n, (_, _, r) in crit.items() if r]
        y_val = {k: [] for k in val_keys}

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
                    
                cumulative_losses = {n: (cumulative_losses[n] if n in cumulative_losses else 0) + l.item() * data["keypoints"].size(0) for n, (l, _, r) in losses.items() if r}
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
            # Figure heading: model type + the key scale hyperparameter (the single
            # ``scale`` config key when present, else the patch/log-polar scale param).
            model_name = getattr(getattr(cfg, "model", None), "name", type(model).__name__)
            scale_desc = getattr(cfg, "scale", None)
            if scale_desc is None:
                _p = cfg.training.dataset.params
                scale_desc = _p.get("patch_scale_factors", _p.get("logpolar_outer_factor", "n/a"))
            plot_title = f"{model_name} {"FFT" if model.head_type == "fft" else "MaxPool"} {"& Mask " if model.learned_mask else ""} (scale={scale_desc})"
            _, ax = plt.subplots()
            for n, v in y_train.items():
                ax.plot(x, v, label=n)
            ax.set_title(plot_title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Training Loss")
            ax.set_xlim(0, cfg.training.num_epochs)
            ax.set_ylim(bottom=0)
            ax.legend()
            plt.savefig(os.path.join(checkpoint_dir, "..", "train_losses.svg"))
            plt.close()

            with torch.no_grad():
                model.eval()
                cumulative_loss = 0.0
                n_items = 0
                # Per-metric-key sums and item counts: each validation dataset
                # contributes only to its own keys, so an FPR reported for the
                # small-blob set is divided by the small-blob item count, not the
                # pooled total across all validation datasets.
                cumulative_losses = {k: 0.0 for k in val_keys}
                cumulative_items = {k: 0 for k in val_keys}
                for label, loaders, val_crit in validation_entries:
                    loop = tqdm(chain(*loaders), leave=True)
                    loop.set_description(f"Validating {label or ''} [{epoch}/{cfg.training.num_epochs}]")
                    for data in loop:
                        losses = process_batch(model, data, val_crit, lambda x: x, device, cfg, validation=True)
                        batch_items = data["keypoints"].size(0)
                        loss = torch.sum(torch.stack([l.view(1) * w for (l, w, _) in losses.values()]))
                        # Weighted total drives best.pth selection; with the
                        # small/large FPRs at weight 0 this tracks the overall FPR.
                        # Non-finite batches (e.g. a batch with no positive pair, where
                        # FPR@recall is undefined) are skipped rather than averaged in.
                        if torch.isfinite(loss).all():
                            cumulative_loss += loss.item() * batch_items
                            n_items += batch_items
                        for n, (l, _, r) in losses.items():
                            if r and torch.isfinite(l).all():
                                key = report_key(label, n)
                                if key not in cumulative_losses:
                                    cumulative_losses[key] = 0.0
                                    cumulative_items[key] = 0
                                cumulative_losses[key] += l.item() * batch_items
                                cumulative_items[key] += batch_items
                        loop.set_postfix(**{report_key(label, n): l.item() for n, (l, _, r) in losses.items() if r})
                for k in val_keys:
                    y_val[k].append(cumulative_losses[k] / cumulative_items[k] if cumulative_items[k] > 0 else float("nan"))
                logging.info("finished epoch [%d/%d], avg val losses: %s", epoch, cfg.training.num_epochs, ", ".join(f"{k}: {y_val[k][-1]}" for k in val_keys))

                _, ax = plt.subplots()
                # Mark each series' best (lowest) value with a dashed rule in its
                # own colour, plus a small label of that value pinned to the right
                # edge. Labels are collected first, then nudged apart in log space
                # so that near-equal bests don't overprint each other.
                best_marks = []
                for n, v in y_val.items():
                    (line,) = ax.semilogy(x, v, label=n)
                    finite = [val for val in v if math.isfinite(val)]
                    if finite:
                        best = min(finite)
                        ax.axhline(best, color=line.get_color(), linestyle="--", linewidth=0.8, alpha=0.7)
                        best_marks.append((best, line.get_color()))
                # Minimum vertical gap between labels, in decades (log10 units);
                # walking bottom-to-top, push each label up if it crowds the one
                # below it.
                min_gap = 0.16
                prev = None
                for best, color in sorted(best_marks, key=lambda m: m[0]):
                    log_best = math.log10(best) if best > 0 else math.log10(1e-5)
                    if prev is not None and log_best - prev < min_gap:
                        log_best = prev + min_gap
                    prev = log_best
                    ax.annotate(f"{best:.4g}", xy=(1, 10 ** log_best), xycoords=("axes fraction", "data"),
                                xytext=(2, 0), textcoords="offset points", va="center", ha="left",
                                fontsize=7, color=color, clip_on=False)
                ax.set_title(plot_title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Average Validation Loss")
                ax.set_xlim(0, cfg.training.num_epochs)
                # Lower bound defaults to 1e-3, but expands downward (with a little
                # headroom) whenever a series dips below it so the curve stays visible.
                data_min = min((b for b, _ in best_marks), default=1e-3)
                lower = 1e-3 if data_min >= 1e-3 else data_min * 0.8
                ax.set_ylim(lower, 1)
                ax.legend()
                plt.savefig(os.path.join(checkpoint_dir, "..", "validation_losses.svg"))
                plt.close()

                if cfg.logging.model_checkpoints and checkpoint_dir is not None:
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
                    # `latest.pth` is always refreshed (resume point); per-epoch
                    # snapshots are opt-in via `logging.checkpoint_every_epoch`.
                    if getattr(cfg.logging, "checkpoint_every_epoch", False):
                        torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pth"))
                    torch.save(checkpoint, os.path.join(checkpoint_dir, "latest.pth"))
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
