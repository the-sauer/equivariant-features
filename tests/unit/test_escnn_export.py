"""Making the steerable (escnn) descriptors ONNX-exportable.

`escnn.nn.R2Conv` expands its filter from basis coefficients on every forward and reads
the basis' raw storage while doing it, so `torch.export` — which traces with FakeTensors —
dies with `Cannot access data pointer`. `aef.models.escnn_export` bakes the expanded
filters into plain `Conv2d`s once training is over, which is what makes the export work.

The unit env mocks escnn away (see `tests/conftest.py`), so what is pinned here is
everything that does *not* need the real library:

* the shims for the three layers escnn itself refuses to export — they are the parts
  hand-written from the upstream `forward`, so they are the parts that can be wrong;
* the safety net in `aef.export.deploy_for_export`, which re-runs the converted model and
  refuses to write a graph that no longer matches the trained one.

The end-to-end conversion (bit-exact for `BlobDescriptorEfficient` in all four
head/learned_mask combinations, and for `BlobDescriptorNoStride` with and without the
mask head) needs real escnn and therefore the pixi env, which has no pytest — the
`available()` guard below skips those there rather than pretending to check them.
"""

import pytest
import torch
import torch.nn.functional as F

from aef import export
from aef.models import escnn_export


needs_escnn = pytest.mark.skipif(
    not escnn_export.available(),
    reason="needs the real escnn (the unit env mocks it); run in the pixi env",
)


def test_available_is_false_under_the_mock():
    """The guard the rest of the module leans on to stay importable in this env."""
    assert escnn_export.available() is False
    # ...and the entry point stays callable rather than blowing up on the MagicMock.
    assert escnn_export.is_escnn(torch.nn.Linear(1, 1)) is False


def test_fixed_mask_multiplies():
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 1:3, 1:3] = 1.0
    module = escnn_export.FixedMask(mask)
    x = torch.randn(2, 3, 4, 4)
    assert torch.equal(module(x), x * mask)
    # A buffer, not a parameter: it travels with the module but is not trainable.
    assert "mask" in dict(module.named_buffers())
    assert not list(module.parameters())


def test_dropout_shim_is_the_identity():
    """`FieldDropout.forward` returns its input untouched when not training."""
    module = escnn_export._export_dropout(object())        # pylint: disable=protected-access
    x = torch.randn(2, 3, 4, 4)
    assert torch.equal(module(x), x)


class _FakeBlurPool:
    """Duck type of `escnn.nn.PointwiseAvgPoolAntialiased2D`'s exported attributes.

    Upstream does `F.conv2d(x, self.filter, stride, padding, groups=channels)` with a
    fixed `filter` buffer; only those four attributes matter to the shim.
    """

    def __init__(self, channels, k=5, stride=2, padding=2):
        torch.manual_seed(0)
        self.filter = torch.randn(channels, 1, k, k)
        self.kernel_size = (k, k)
        self.stride = (stride, stride)
        self.padding = (padding, padding)


def test_blurpool_shim_matches_the_grouped_conv_it_replaces():
    """The antialiased pool is a fixed depthwise conv — same numbers, no escnn."""
    fake = _FakeBlurPool(channels=6)
    conv = escnn_export._export_blurpool(fake)             # pylint: disable=protected-access
    x = torch.randn(2, 6, 16, 16)
    want = F.conv2d(x, fake.filter, stride=fake.stride, padding=fake.padding, groups=6)
    with torch.no_grad():
        got = conv(x)
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-6)
    # Stride 2 with a 16 px input: the pool is what takes 64 -> 32 -> 16 -> 8.
    assert got.shape[-1] == 8


class _Toy(torch.nn.Module):
    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def forward(self, x):
        return x.mean(dim=(1, 2, 3)) + self.bias


def _patch_deploy(monkeypatch, replacement):
    monkeypatch.setattr(escnn_export, "is_escnn", lambda model: True)
    monkeypatch.setattr(escnn_export, "deploy", lambda model: replacement)


def test_deploy_is_skipped_for_plain_models(monkeypatch):
    """A HardNet has no equivariant layer, so nothing should be touched."""
    monkeypatch.setattr(escnn_export, "deploy",
                        lambda model: pytest.fail("deploy() called on a plain model"))
    model = _Toy()
    example = (torch.randn(2, 1, 4, 4),)
    reference = (model(*example),)
    got_model, got_ref = export.deploy_for_export(model, example, reference)
    assert got_model is model and got_ref is reference


def test_deploy_accepts_an_equivalent_conversion(monkeypatch):
    _patch_deploy(monkeypatch, _Toy())
    model = _Toy()
    example = (torch.randn(2, 1, 4, 4),)
    reference = (model(*example),)
    _, deployed = export.deploy_for_export(model, example, reference)
    assert torch.equal(deployed[0], reference[0])


def test_deploy_refuses_a_conversion_that_changes_the_descriptor(monkeypatch):
    """The whole point of the check: a wrong graph is worse than no graph."""
    _patch_deploy(monkeypatch, _Toy(bias=1e-2))
    model = _Toy()
    example = (torch.randn(2, 1, 4, 4),)
    reference = (model(*example),)
    with pytest.raises(RuntimeError, match=r"max \|plain - escnn\|"):
        export.deploy_for_export(model, example, reference)


def test_deploy_refuses_a_conversion_that_changes_the_output_count(monkeypatch):
    class _TwoOut(torch.nn.Module):
        def forward(self, x):
            return x.mean(dim=(1, 2, 3)), x.sum(dim=(1, 2, 3))

    _patch_deploy(monkeypatch, _TwoOut())
    model = _Toy()
    example = (torch.randn(2, 1, 4, 4),)
    with pytest.raises(RuntimeError, match="number of outputs"):
        export.deploy_for_export(model, example, (model(*example),))


@needs_escnn
@pytest.mark.parametrize("head", ["attention", "dense"])
@pytest.mark.parametrize("learned_mask", [False, True])
def test_efficient_descriptor_survives_deployment(head, learned_mask):
    """Real-escnn check: converting must not move the descriptor at all."""
    from aef.models import BlobDescriptorEfficient      # pylint: disable=import-outside-toplevel

    torch.manual_seed(0)
    model = BlobDescriptorEfficient(
        n_rotations=4, head=head, scale_factors=[16.0], in_channels=1,
        learned_mask=learned_mask,
    ).eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        reference = model(x)
    reference = reference if isinstance(reference, tuple) else (reference,)

    escnn_export.deploy(model)
    assert not escnn_export.is_escnn(model)
    with torch.no_grad():
        got = model(x)
    got = got if isinstance(got, tuple) else (got,)
    for a, b in zip(got, reference):
        assert torch.equal(a, b)
