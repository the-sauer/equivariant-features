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

    if "asel" not in sys.modules:
        asel = types.ModuleType("asel")
        affine = types.ModuleType("asel.affine")

        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                del kwargs
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, value):
                return self.conv(value)

        class EquivarLayer(nn.Module):
            def __init__(self, in_channels, out_channels, **kwargs):
                super().__init__()
                del kwargs
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, value):
                return self.conv(value)

        affine.BasicBlock = BasicBlock
        affine.EquivarLayer = EquivarLayer
        asel.affine = affine
        sys.modules["asel"] = asel
        sys.modules["asel.affine"] = affine

    if "escnn" not in sys.modules:
        # blob_descriptor.py only touches escnn at import time via a class annotation
        # (`field_type: escnn.nn.FieldType`); the real layers are built at runtime. A
        # MagicMock surface is enough to import the package for the pure-torch models.
        from unittest.mock import MagicMock

        escnn = types.ModuleType("escnn")
        escnn.nn = MagicMock()
        escnn.gspaces = MagicMock()
        escnn.group = MagicMock()
        sys.modules["escnn"] = escnn
        sys.modules["escnn.nn"] = escnn.nn
        sys.modules["escnn.gspaces"] = escnn.gspaces
        sys.modules["escnn.group"] = escnn.group

    if "sesn" not in sys.modules:
        sesn = types.ModuleType("sesn")

        class _SESConv(nn.Module):
            def __init__(self, in_channels, out_channels, *args, **kwargs):
                super().__init__()
                del args, kwargs
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, value):
                return self.conv(value)

        class SESArgMaxProjection(nn.Module):
            def __init__(self, scales):
                super().__init__()
                self.scales = scales

            def forward(self, value):
                return value

        sesn.SESConv_Z2_H = _SESConv
        sesn.SESConv_H_H = _SESConv
        sesn.SESConv_H_H_1x1 = _SESConv
        sesn.SESArgMaxProjection = SESArgMaxProjection
        sys.modules["sesn"] = sesn


_install_fake_modules()