"""Automatic ONNX export at the end of training: `logging.export_onnx`.

`aef.export.export_after_training` runs once the last epoch's plots are written and
turns each configured checkpoint into `<stem>.onnx` beside its `<stem>.pth`. Pinned
here:

* the gate — off by default, and a no-op (with a warning, not a crash) when no
  checkpoints are being written at all, since there would be nothing to load;
* what it exports — the model rebuilt from `cfg.model` and the weights read from
  `<checkpoint_dir>/<stem>.pth`, *not* the last-epoch module still in memory, because
  `best.pth` is generally a different epoch;
* that a failing export never propagates: a finished run must not be lost to it;
* that `train_func` actually reaches the hook, with the real checkpoint directory.

The export itself is stubbed out (it needs `onnx`, which the unit env does not have —
see `test_onnx_friendly_ops.py` for the op-level checks that do run here); what these
tests cover is the plumbing and the gating around it.
"""

import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use("Agg")

from aef import export  # noqa: E402  (import after choosing the Agg backend)
from aef.train import train_func  # noqa: E402


class _Dataset(torch.utils.data.Dataset):
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


def _cfg(tmp_path, **logging_overrides):
    logging_cfg = {
        "dir": str(tmp_path),
        "model_checkpoints": True,
        "export_onnx": True,
        "export_onnx_checkpoints": ["best"],
    }
    logging_cfg.update(logging_overrides)
    return OmegaConf.create({
        "logging": logging_cfg,
        "model": {"name": "HardNetLogPolar", "params": {"in_channels": 1, "patch_size": 64}},
        "training": {
            "num_epochs": 1,
            "batch_size": 4,
            "num_workers": 0,
            "optimizer": {"name": "SGD", "params": {"lr": 0.01}},
            "loss": [],
            "dataset": {"params": {"patch_scale_factors": [1.0]}},
        },
        "validation": {"batch_size": 4, "num_workers": 0},
    })


def _record(monkeypatch, fail=False):
    """Replace the real export with a recorder; returns the list of calls made."""
    calls = []

    def fake_export_checkpoint(model_name, params, weights, path, **kwargs):
        calls.append({"model": model_name, "params": params, "weights": weights,
                      "path": path, **kwargs})
        if fail:
            raise RuntimeError("unsupported op")
        # `export_after_training` only reads `.model_proto` off the first return value.
        proto = type("Proto", (), {"graph": type("Graph", (), {"node": []})()})()
        return type("Program", (), {"model_proto": proto})(), None, None

    monkeypatch.setattr(export, "export_checkpoint", fake_export_checkpoint)
    return calls


def test_off_by_default(tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    cfg = _cfg(tmp_path)
    del cfg.logging.export_onnx
    assert export.export_after_training(cfg, str(tmp_path)) == []
    assert calls == []


def test_exports_the_configured_checkpoints(tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    for stem in ("best", "latest"):
        (tmp_path / f"{stem}.pth").write_bytes(b"")
    cfg = _cfg(tmp_path, export_onnx_checkpoints=["best", "latest"])

    written = export.export_after_training(cfg, str(tmp_path))

    assert written == [str(tmp_path / "best.onnx"), str(tmp_path / "latest.onnx")]
    assert [c["weights"] for c in calls] == [str(tmp_path / "best.pth"),
                                            str(tmp_path / "latest.pth")]
    # Architecture comes from the config, so the export matches what was trained.
    assert calls[0]["model"] == "HardNetLogPolar"
    assert calls[0]["params"] == {"in_channels": 1, "patch_size": 64}


def test_missing_checkpoint_is_skipped_not_fatal(tmp_path, monkeypatch):
    """`best.pth` is absent when every epoch's validation was non-finite."""
    calls = _record(monkeypatch)
    assert export.export_after_training(_cfg(tmp_path), str(tmp_path)) == []
    assert calls == []


def test_without_checkpoints_there_is_nothing_to_export(tmp_path, monkeypatch):
    calls = _record(monkeypatch)
    cfg = _cfg(tmp_path, model_checkpoints=False)
    assert export.export_after_training(cfg, str(tmp_path)) == []
    assert calls == []


def test_export_failure_does_not_propagate(tmp_path, monkeypatch):
    """A model that will not export must not take the finished run down with it."""
    calls = _record(monkeypatch, fail=True)
    (tmp_path / "best.pth").write_bytes(b"")
    assert export.export_after_training(_cfg(tmp_path), str(tmp_path)) == []
    assert len(calls) == 1


class _NamedArg(torch.nn.Module):
    """The shape-sensitive tail of every descriptor, under a chosen argument name.

    `.view(B, -1)` + a per-row L2 norm is exactly what a frozen batch axis breaks: with
    B specialized to 1 the whole batch collapses into one globally normalized vector.
    """

    def __init__(self, arg_name):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 4, 3, padding=1)
        # `torch.export` reads the real signature, so the name has to be genuine.
        src = (f"def forward(self, {arg_name}):\n"
               f"    y = self.conv({arg_name})\n"
               f"    y = y.view({arg_name}.size(0), -1)\n"
               f"    return y / y.norm(dim=1, keepdim=True)\n")
        scope = {}
        exec(src, {"torch": torch}, scope)  # noqa: S102  # pylint: disable=exec-used
        self.forward = scope["forward"].__get__(self)


def test_batch_axis_stays_dynamic_whatever_the_arg_is_called():
    """`dynamic_shapes` keyed by name only fits models whose input is called `patches`.

    `HardNetLogPolar.forward(self, patches)` is the odd one out — the steerable and
    efficient blob descriptors take `x`, and a name-keyed dict rejects them outright
    (`UserError: its top-level keys must be the arg names`). This is the export step that
    runs unattended after every training run now, so it has to hold for all of them.
    """
    for arg_name in ("patches", "x"):
        model = _NamedArg(arg_name).eval()
        exported = torch.export.export(
            model, (torch.randn(2, 1, 8, 8),), dynamic_shapes=export.batch_dynamic(),
        )
        out_meta = list(exported.graph.nodes)[-1].args[0]
        shape = (out_meta[0] if isinstance(out_meta, (tuple, list)) else out_meta).meta["val"].shape
        # Symbolic, not the literal 2 the sample happened to use.
        assert not isinstance(shape[0], int), f"batch axis specialized for arg {arg_name!r}"
        assert int(shape[1]) == 4 * 8 * 8


def test_train_func_reaches_the_hook(tmp_path, monkeypatch):
    """End-to-end: the loop calls the export with its own checkpoint directory."""
    calls = _record(monkeypatch)

    def process_batch(model, data, criterion, augmentation, device, cfg, **kwargs):
        del criterion, augmentation, cfg, kwargs
        return {"toy": (model(data["keypoints"].to(device)).pow(2).mean(), 1.0, True)}

    train_func(process_batch)(
        model=_Model(),
        train_dataset=[_Dataset(8)],
        validation_dataset=[("val", [_Dataset(4)], [])],
        cfg=_cfg(tmp_path),
        experiment_name="test",
    )

    checkpoints = tmp_path / "test" / "checkpoints"
    assert [c["weights"] for c in calls] == [str(checkpoints / "best.pth")]
    assert calls[0]["path"] == str(checkpoints / "best.onnx")
