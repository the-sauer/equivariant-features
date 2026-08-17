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

"""Make the escnn-based descriptors ONNX-exportable.

``escnn.nn.R2Conv`` does not hold a weight tensor: it holds basis coefficients and
expands them into a filter on every forward pass, reaching into the basis' raw storage as
it goes. Under ``torch.export`` the inputs are FakeTensors, so that expansion dies with
``RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor)`` — which is why
the whole steerable family used to be un-exportable. At eval time, though, the expanded
filter is *constant*, so the fix is to expand it once and keep the result: that is what
escnn's own ``EquivariantModule.export()`` does, turning an ``R2Conv`` into a plain
``torch.nn.Conv2d``.

escnn only implements ``export()`` for some of its modules; of the ones this repo uses,
``MaskModule``, ``FieldDropout`` and ``PointwiseAvgPoolAntialiased2D`` raise
``NotImplementedError``. All three are trivial once training is over, and :data:`SHIMS`
supplies them (a fixed multiply, an identity, and a fixed depthwise blur conv).

The converted layers keep taking and returning :class:`~escnn.nn.GeometricTensor`, via
:class:`GeometricWrapper`. That is deliberate: the descriptors' ``forward`` methods wrap,
unwrap and re-wrap geometric tensors throughout (``self.att(x).tensor``,
``GeometricTensor(x.tensor * w, x.type)``), so preserving the interface means their
forward code — the part that has to stay numerically identical — is not touched at all.
Only the leaves change, and a ``GeometricTensor`` is a plain Python wrapper that
``torch.export`` traces straight through.

Entry point: :func:`deploy`, which every escnn descriptor exposes as ``.deploy()``.
:func:`aef.export.export_model` calls it and verifies the descriptors still match.
"""

import escnn
import torch

# Everything escnn is reached as an attribute of `escnn.nn`, never by importing its
# submodules: the unit-test env replaces escnn with a MagicMock (see tests/conftest.py),
# and a deep `from escnn.nn.modules... import` would break collection there. `available()`
# tells the two apart.


def available():
    """Whether the real escnn is importable, as opposed to the unit tests' MagicMock."""
    return isinstance(getattr(escnn.nn, "EquivariantModule", None), type)


class GeometricWrapper(torch.nn.Module):
    """Runs a plain-torch module on a ``GeometricTensor``'s payload.

    Keeps `in_type`/`out_type` so the surrounding escnn containers still type-check, and
    re-wraps the result so the caller cannot tell the layer was replaced.
    """

    def __init__(self, plain, in_type, out_type):
        super().__init__()
        self.plain = plain
        self.in_type = in_type
        self.out_type = out_type

    def forward(self, x):
        return escnn.nn.GeometricTensor(self.plain(x.tensor), self.out_type)


class FixedMask(torch.nn.Module):
    """`escnn.nn.MaskModule` with its mask frozen — an element-wise multiply.

    The mask is a `requires_grad=False` Parameter upstream; a buffer here, so it travels
    with the module but is not mistaken for something trainable.
    """

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        return x * self.mask


def _export_mask(module):
    return FixedMask(module.mask.detach().clone())


def _export_dropout(module):
    """`FieldDropout` returns its input untouched in eval mode (see its `forward`)."""
    del module
    return torch.nn.Identity()


def _export_blurpool(module):
    """`PointwiseAvgPoolAntialiased2D` is a fixed depthwise Gaussian conv + stride.

    Upstream it is an `F.conv2d(x, self.filter, stride, padding, groups=channels)` with
    `filter` a non-learned buffer, which is exactly a grouped `Conv2d` with frozen
    weights.
    """
    channels = module.filter.shape[0]
    conv = torch.nn.Conv2d(
        channels, channels, kernel_size=module.kernel_size, stride=module.stride,
        padding=module.padding, groups=channels, bias=False,
    )
    with torch.no_grad():
        conv.weight.copy_(module.filter)
    return conv.eval()


# escnn modules whose `export()` raises NotImplementedError but which are constant (or
# absent) at eval time. Keyed by escnn class *name*, resolved against `escnn.nn` only when
# a conversion actually runs, so importing this module under the mocked escnn is safe.
# Matched on exact type — a subclass with different semantics should fail loudly rather
# than silently reuse one of these.
SHIMS = {
    "MaskModule": _export_mask,
    "FieldDropout": _export_dropout,
    "PointwiseAvgPoolAntialiased2D": _export_blurpool,
}


def _export_leaf(module):
    """One equivariant layer -> a plain-torch module over raw tensors."""
    shim = SHIMS.get(type(module).__name__)
    if shim is not None and type(module) is getattr(escnn.nn, type(module).__name__, None):
        return shim(module)
    try:
        return module.export()
    except NotImplementedError as exc:
        raise NotImplementedError(
            f"{type(module).__name__} has no escnn export() and no shim in "
            f"aef.models.escnn_export.SHIMS — add one to make this model exportable"
        ) from exc


def convert(module):
    """Recursively replace every equivariant layer in `module` with a plain-torch one.

    Containers are rebuilt child-by-child, one output child per input child, so child
    order and count are preserved — `BlobDescriptorNoStride.forward` slices
    `list(self.net.children())` by index, and that indexing has to keep meaning the same
    layers.
    """
    if isinstance(module, escnn.nn.SequentialModule):
        return torch.nn.Sequential(*(convert(child) for child in module.children()))
    if isinstance(module, escnn.nn.EquivariantModule):
        return GeometricWrapper(_export_leaf(module), module.in_type, module.out_type)
    return module


def deploy(model):
    """Convert an escnn model to plain torch **in place**, and return it.

    Every direct child that is an equivariant module is converted, which covers how these
    descriptors are built (`net` plus a handful of heads: `readout`, `att`, `group_pool`,
    `mask_head`).

    In place, and one-way — so call it on a model you own. It cannot copy first: after a
    forward pass an escnn module holds its expanded filters as cached *non-leaf* tensors,
    and `copy.deepcopy` refuses those ("Only Tensors created explicitly by the user (graph
    leaves) support the deepcopy protocol"). The export path builds its own model from the
    config and the checkpoint, so it owns one; never hand it a model that is still
    training.

    Requires eval mode: `InnerBatchNorm.export()` folds the *running* statistics into the
    plain `BatchNorm2d`, and `FieldDropout` is only an identity when not training.
    """
    model.eval()
    for name, child in list(model.named_children()):
        if isinstance(child, (escnn.nn.EquivariantModule, escnn.nn.SequentialModule)):
            setattr(model, name, convert(child))
    return model


def is_escnn(model):
    """Whether `model` has any equivariant layer left in it (i.e. needs `deploy`)."""
    if not available():
        return False
    return any(isinstance(m, escnn.nn.EquivariantModule) for m in model.modules())
