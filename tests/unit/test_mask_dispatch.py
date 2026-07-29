"""How `process_batch_blobs` routes the board-validity mask to the model.

Two failure modes this pins, both silent-ish and both expensive:

* a descriptor that is NOT mask-aware has a plain ``forward(patches)`` and raises on
  ``mask=``/``is_anchor=`` kwargs — so the mask must only be handed to models that
  advertise ``learned_mask`` (HardNetLogPolar, BlobDescriptorEfficient,
  BlobDescriptorNoStride);
* the mask predictor is supervised on the TARGET views only. The anchor's mask is
  *given* to the model, so including anchors in the BCE would train the predictor on
  the one case where it is never used — and validation withholds the GT mask entirely
  (test time has none), so no mask loss may appear there.
"""

import pytest
import torch
from omegaconf import OmegaConf

from aef.train.descriptor import process_batch_blobs


class _Plain(torch.nn.Module):
    """A descriptor with the ordinary signature — kwargs would be a TypeError."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 4)
        self.calls = []

    def forward(self, patches):
        self.calls.append("plain")
        return self.lin(patches.reshape(patches.size(0), -1))


class _MaskAware(torch.nn.Module):
    """Mask-aware descriptor: advertises ``learned_mask`` and returns (d, m_pred)."""

    learned_mask = True

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(16, 4)
        self.seen = []

    def forward(self, patches, mask=None, is_anchor=None):
        self.seen.append((mask is not None, is_anchor is not None))
        d = self.lin(patches.reshape(patches.size(0), -1))
        m_pred = torch.full((patches.size(0), 1, 2, 2), 0.5, requires_grad=True)
        return d, m_pred


def _batch(n=4, with_mask=True):
    data = {
        "keypoints": torch.stack([torch.zeros(n, dtype=torch.long),
                                  torch.arange(n) // 2], dim=1),
        "keypoint_coords": torch.zeros(n, 2),
        "image_size": torch.tensor([1.0, 1.0]),
        "patches": torch.rand(n, 1, 4, 4),
    }
    if with_mask:
        data["masks"] = torch.rand(n, 1, 4, 4)
        # Alternating anchor / target, so both branches of the routing are exercised.
        data["is_anchor"] = torch.tensor([1, 0] * (n // 2))
    return data


def _cfg(**training):
    return OmegaConf.create({"training": training})


def _criterion():
    return {"supcon": (lambda out: out["features"].square().mean(), 1.0, True)}


def _run(model, data, cfg, validation=False):
    return process_batch_blobs(model, data, _criterion(), None, "cpu", cfg,
                               validation=validation)


def test_plain_model_never_receives_mask_kwargs():
    model = _Plain()
    losses = _run(model, _batch(), _cfg(ignore_mask=False))
    assert model.calls == ["plain"]        # would have raised TypeError otherwise
    assert "mask_bce" not in losses


def test_plain_model_without_ignore_mask_key():
    # Configs that predate the flag must keep working (getattr default).
    model = _Plain()
    _run(model, _batch(with_mask=False), _cfg())
    assert model.calls == ["plain"]


def test_mask_aware_model_gets_mask_and_is_supervised():
    model = _MaskAware()
    losses = _run(model, _batch(), _cfg(ignore_mask=False, mask_loss_weight=0.25))
    assert model.seen == [(True, True)]
    assert "mask_bce" in losses
    bce, weight, report = losses["mask_bce"]
    assert weight == 0.25 and report
    assert torch.isfinite(bce)


def test_ignore_mask_forces_the_plain_call():
    model = _MaskAware()
    losses = _run(model, _batch(), _cfg(ignore_mask=True))
    assert model.seen == [(False, False)]   # called as model(patches)
    assert "mask_bce" not in losses


def test_validation_withholds_the_gt_mask():
    # At test time there is no mask, so validation must not feed one (nor score the
    # predictor against it) — otherwise the reported metric is not reproducible.
    model = _MaskAware()
    losses = _run(model, _batch(), _cfg(ignore_mask=False), validation=True)
    assert model.seen == [(False, False)]
    assert "mask_bce" not in losses


def test_mask_bce_ignores_anchors():
    # All-anchor batch: nothing to supervise, and the loss must stay finite (and
    # gradient-connected) rather than reduce over an empty selection -> NaN.
    data = _batch()
    data["is_anchor"] = torch.ones(4, dtype=torch.long)
    losses = _run(_MaskAware(), data, _cfg(ignore_mask=False))
    bce = losses["mask_bce"][0]
    assert bce == 0.0 and torch.isfinite(bce)


@pytest.mark.parametrize("weight", [0.0, 2.0])
def test_mask_loss_weight_is_read_from_the_config(weight):
    losses = _run(_MaskAware(), _batch(), _cfg(ignore_mask=False, mask_loss_weight=weight))
    assert losses["mask_bce"][1] == weight
