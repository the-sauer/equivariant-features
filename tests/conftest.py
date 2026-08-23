import os
import sys
import types
from importlib.machinery import ModuleSpec

import torch
from torch import nn


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


class _IdentityTransform:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, value):
        return value


class _Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


class _FakeTqdm:
    def __init__(self, iterable, *args, **kwargs):
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def set_description(self, *args, **kwargs):
        return None

    def set_postfix(self, *args, **kwargs):
        return None


class _FakeLoss(nn.Module):
    def forward(self, embeddings, labels):
        del labels
        return embeddings.float().pow(2).mean()


def _install_fake_modules():
    if "pytorch_metric_learning" not in sys.modules:
        pml = types.ModuleType("pytorch_metric_learning")
        pml_losses = types.ModuleType("pytorch_metric_learning.losses")
        pml_losses.TripletMarginLoss = _FakeLoss
        pml_losses.NPairsLoss = _FakeLoss
        pml_losses.SupConLoss = _FakeLoss
        pml.losses = pml_losses
        sys.modules["pytorch_metric_learning"] = pml
        sys.modules["pytorch_metric_learning.losses"] = pml_losses

    if "omegaconf" not in sys.modules:
        omegaconf = types.ModuleType("omegaconf")

        class _OmegaConf:
            @staticmethod
            def to_yaml(cfg):
                return "stub: true\n"

        omegaconf.OmegaConf = _OmegaConf
        sys.modules["omegaconf"] = omegaconf

    if "tqdm" not in sys.modules:
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.tqdm = _FakeTqdm
        tqdm_module.__spec__ = ModuleSpec("tqdm", loader=None)
        sys.modules["tqdm"] = tqdm_module

    if "h5py" not in sys.modules:
        # aef.data.track imports h5py at module level, but the pure-logic tests drive
        # `_load_all_sequences` with an in-memory stand-in for an HDF5 file (dicts of
        # numpy arrays, which already support the `[:]` / fancy-index reads it does).
        h5py_module = types.ModuleType("h5py")

        class _File:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("h5py is stubbed out in the unit tests")

        h5py_module.File = _File
        sys.modules["h5py"] = h5py_module

    if "torchvision" not in sys.modules:
        torchvision = types.ModuleType("torchvision")
        transforms = types.ModuleType("torchvision.transforms")
        v2 = types.ModuleType("torchvision.transforms.v2")
        v2.Compose = _Compose
        v2.ColorJitter = _IdentityTransform
        v2.GaussianBlur = _IdentityTransform

        class _GaussianNoise(_IdentityTransform):
            pass

        v2.GaussianNoise = _GaussianNoise
        transforms.v2 = v2
        torchvision.transforms = transforms
        sys.modules["torchvision"] = torchvision
        sys.modules["torchvision.transforms"] = transforms
        sys.modules["torchvision.transforms.v2"] = v2

    if "kornia" not in sys.modules:
        kornia = types.ModuleType("kornia")
        feature = types.ModuleType("kornia.feature")
        geometry = types.ModuleType("kornia.geometry")
        transform = types.ModuleType("kornia.geometry.transform")

        def _response(image):
            return torch.zeros_like(image)

        def _warp_perspective(tensor, matrix, dsize):
            del matrix, dsize
            return tensor

        feature.dog_response = _response
        feature.harris_response = _response
        transform.warp_perspective = _warp_perspective
        geometry.transform = transform
        kornia.feature = feature
        kornia.geometry = geometry
        sys.modules["kornia"] = kornia
        sys.modules["kornia.feature"] = feature
        sys.modules["kornia.geometry"] = geometry
        sys.modules["kornia.geometry.transform"] = transform


_install_fake_modules()
