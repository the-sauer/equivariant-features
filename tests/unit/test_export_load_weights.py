"""Loading a checkpoint into a model for export: `aef.export.load_weights`.

A checkpoint stores weights only, so the constructor params handed in alongside it are
the sole description of the architecture. A partial load would therefore export a
half-initialized graph that still *looks* like a successful export, which is why any
missing or unexpected key raises here instead of being tolerated.

Also pinned: `build_model` loads the weights *before* `eval()`, and `load_state_dict`
unwraps both checkpoint flavours (a training checkpoint dict, or a bare state_dict).
"""

import re

import pytest
import torch

from aef import export


class _Tiny(torch.nn.Module):
    def __init__(self, **_):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(4))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x


def _trained_state():
    model = _Tiny()
    with torch.no_grad():
        model.weights.copy_(torch.arange(4, dtype=torch.float32))
        model.bias.fill_(3.0)
    return model.state_dict()


def test_load_state_dict_unwraps_a_training_checkpoint(tmp_path):
    path = tmp_path / "best.pth"
    torch.save({"model_state_dict": _trained_state(), "epoch": 7}, path)

    state = export.load_state_dict(str(path))

    assert set(state) == {"weights", "bias"}


def test_load_state_dict_accepts_a_bare_state_dict(tmp_path):
    path = tmp_path / "bare.pth"
    torch.save(_trained_state(), path)

    assert set(export.load_state_dict(str(path))) == {"weights", "bias"}


def test_load_weights_loads_the_checkpoint(tmp_path):
    path = tmp_path / "best.pth"
    torch.save({"model_state_dict": _trained_state()}, path)

    model = _Tiny()
    export.load_weights(model, str(path))

    assert torch.equal(model.weights.detach(), torch.arange(4, dtype=torch.float32))


def test_load_weights_rejects_a_real_mismatch(tmp_path):
    state = _trained_state()
    state["stray"] = torch.zeros(1)
    del state["bias"]
    path = tmp_path / "wrong.pth"
    torch.save(state, path)

    with pytest.raises(RuntimeError, match=re.compile(r"does not match.*stray", re.S)):
        export.load_weights(_Tiny(), str(path))


def test_build_model_loads_the_weights_and_leaves_the_model_in_eval(tmp_path, monkeypatch):
    path = tmp_path / "best.pth"
    torch.save({"model_state_dict": _trained_state()}, path)
    monkeypatch.setattr(export.models, "_Tiny", _Tiny, raising=False)

    model = export.build_model("_Tiny", {}, str(path))

    assert not model.training
    assert torch.equal(model.weights.detach(), torch.arange(4, dtype=torch.float32))
