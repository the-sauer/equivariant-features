"""Loading a checkpoint into an escnn model for export: `aef.export.load_weights`.

`escnn.nn.R2Conv` stores basis coefficients, not a filter. It materializes the expanded
`<layer>.filter` / `<layer>.expanded_bias` on the train->eval transition and deletes them
again on the way back, so whether a checkpoint contains them is decided by the mode the
model was in when `torch.save` ran — training, for every checkpoint this repo writes.

`build_model` used to `eval()` *before* loading, which gave the module those buffers and
made a perfectly good checkpoint fail with `Missing key(s) ... net.1.filter`; that is what
made the whole steerable family un-exportable via `to_onnx.py`. Pinned here: the order
(weights first, `eval()` after, so the expansion is derived from the loaded coefficients),
the tolerance for those derived buffers in both directions, and that a *real* architecture
mismatch still raises rather than exporting a half-loaded model.
"""

import re

import pytest
import torch

from aef import export


class _Expanding(torch.nn.Module):
    """Stand-in for `escnn.nn.R2Conv`: `filter` exists in eval mode only.

    Mirrors escnn's `_RdConv.train()` — expand on the way into eval, drop on the way out —
    so the buffer is a function of `weights` at the moment the mode flips.
    """

    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(4))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def train(self, mode=True):
        if mode:
            for name in ("filter", "expanded_bias"):
                if name in self._buffers:
                    del self._buffers[name]
        elif self.training:
            self.register_buffer("filter", self.weights.detach() * 2)
            self.register_buffer("expanded_bias", self.bias.detach() * 2)
        return super().train(mode)

    def forward(self, x):
        return x


def _trained_checkpoint():
    """A checkpoint as training writes one: model in train mode, so no expansion buffers."""
    model = _Expanding()
    with torch.no_grad():
        model.weights.copy_(torch.arange(4, dtype=torch.float32))
        model.bias.fill_(3.0)
    assert "filter" not in model.state_dict()
    return model.state_dict()


def test_load_weights_accepts_a_checkpoint_without_the_expansion_buffers(tmp_path):
    path = tmp_path / "best.pth"
    torch.save({"model_state_dict": _trained_checkpoint()}, path)

    model = _Expanding()
    export.load_weights(model, str(path))

    assert torch.equal(model.weights.detach(), torch.arange(4, dtype=torch.float32))


def test_load_weights_accepts_a_checkpoint_that_has_them(tmp_path):
    """The mirror image: saved from an eval-mode model, loaded into a train-mode one."""
    model = _Expanding()
    model.load_state_dict(_trained_checkpoint())
    model.eval()
    state = model.state_dict()
    assert "filter" in state
    path = tmp_path / "eval.pth"
    torch.save(state, path)

    export.load_weights(_Expanding(), str(path))


def test_load_weights_still_rejects_a_real_mismatch(tmp_path):
    state = _trained_checkpoint()
    state["stray"] = torch.zeros(1)
    del state["bias"]
    path = tmp_path / "wrong.pth"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match=re.compile(r"does not match.*stray", re.S)):
        export.load_weights(_Expanding(), str(path))


def test_build_model_expands_from_the_loaded_weights(tmp_path, monkeypatch):
    """The whole point of the ordering: `filter` must follow the checkpoint, not the init."""
    path = tmp_path / "best.pth"
    torch.save({"model_state_dict": _trained_checkpoint()}, path)
    monkeypatch.setattr(export.models, "_Expanding", _Expanding, raising=False)

    model = export.build_model("_Expanding", {}, str(path))

    assert not model.training
    assert torch.equal(model.filter, torch.arange(4, dtype=torch.float32) * 2)
    assert torch.equal(model.expanded_bias, torch.tensor([6.0]))
