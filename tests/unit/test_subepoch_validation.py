"""Sub-epoch validation: `validation.validate_every_n_batches`.

The training loop normally produces one validation point per epoch. With the option
set it runs the whole validation suite every N training batches as well, and those
extra points are plotted at the *fractional* epoch they were measured at — the two
things pinned here:

* how many validation runs happen (the mid-epoch ones plus the end-of-epoch one, with
  no duplicate run when N divides the epoch's batch count exactly);
* where they sit on the x axis: ``epoch + batches_done / batches_per_epoch``, strictly
  increasing, and exactly as many positions as every curve has points — a mismatch
  there is what makes matplotlib raise at plotting time.

With the option off nothing moves: one point per epoch at integer x, as before.

The curves are read back out of ``latest.pth``, which is where ``train_func`` parks
them (``checkpoint["plots"]``).
"""

import matplotlib
import pytest
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")

from aef.train import train_func  # noqa: E402  (import after choosing the Agg backend)


class _Dataset(torch.utils.data.Dataset):
    """Minimal dataset whose batches carry the one key the loop itself reads."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.tensor([float(i)])

    def get_collate_func(self):
        def collate(items):
            return {"keypoints": torch.stack(items)}
        return collate


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.lin(x)


def _cfg(tmp_path, every_n_batches, num_epochs, batch_size):
    return OmegaConf.create({
        "logging": {"dir": str(tmp_path), "model_checkpoints": True},
        "training": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "num_workers": 0,
            "optimizer": {"name": "SGD", "params": {"lr": 0.01}},
            "loss": [],
            "dataset": {"params": {"patch_scale_factors": [1.0]}},
        },
        "validation": {
            "batch_size": batch_size,
            "num_workers": 0,
            "validate_every_n_batches": every_n_batches,
        },
    })


def _run(tmp_path, every_n_batches, n_train=20, batch_size=4, num_epochs=2):
    """Train on a toy problem; return (x_val, y_val, number of validation batches)."""
    val_batches = []

    def process_batch(model, data, criterion, augmentation, device, cfg, **kwargs):
        del criterion, augmentation, cfg
        if kwargs.get("validation"):
            val_batches.append(1)
        loss = model(data["keypoints"].to(device)).pow(2).mean()
        return {"toy": (loss, 1.0, True)}

    train_func(process_batch)(
        model=_Model(),
        train_dataset=[_Dataset(n_train)],
        validation_dataset=[("val", [_Dataset(batch_size)], [])],
        cfg=_cfg(tmp_path, every_n_batches, num_epochs, batch_size),
        experiment_name="test",
    )
    plots = torch.load(tmp_path / "test" / "checkpoints" / "latest.pth",
                       map_location="cpu", weights_only=False)["plots"]
    return plots["x_val"], plots["y_val"], len(val_batches)


def test_epoch_resolution_by_default(tmp_path):
    """Option off (null) => the historical one point per epoch at integer x."""
    x_val, y_val, n_val_batches = _run(tmp_path, None)
    assert x_val == [0.0, 1.0]
    assert n_val_batches == 2                      # one validation batch per epoch
    assert all(len(v) == len(x_val) for v in y_val.values())


def test_subepoch_points_land_on_fractional_epochs(tmp_path):
    """20 items / batch 4 = 5 batches per epoch; validating every 2 gives 0.4, 0.8, 1.0."""
    x_val, y_val, n_val_batches = _run(tmp_path, 2)
    assert x_val == pytest.approx([0.4, 0.8, 1.0, 1.4, 1.8, 2.0])
    assert n_val_batches == 6
    # Every curve stays as long as the axis — including `toy@val`, which the config
    # never declared and which is only discovered once a batch has run.
    assert set(y_val) == {"toy@val"}
    assert all(len(v) == len(x_val) for v in y_val.values())


def test_no_duplicate_run_at_the_epoch_boundary(tmp_path):
    """An interval that divides the epoch exactly must not validate twice in a row."""
    x_val, _, n_val_batches = _run(tmp_path, 5)
    assert x_val == pytest.approx([1.0, 2.0])
    assert n_val_batches == 2
