"""The log-polar trunk/head is written against ONNX-exportable primitives on purpose.

Three ops that read naturally in torch export to something onnxruntime only runs on the
CPU execution provider, which in a GPU session means a device round trip per node:

    F.pad(mode="circular")   -> Pad(mode="wrap")   no CUDA kernel
    F.pad(mode="replicate")  -> Pad(mode="edge")   ditto in older ORT builds
    torch.fft.rfft           -> DFT               CPU EP only, and ONNX has no complex dtype

So ``LogPolarPad``/``LogPolarBlurPool`` wrap via slice+concat and the angular head runs
the DFT as a matmul. These tests pin the rewrites to the reference torch ops they
replaced — the whole point is that the substitution is invisible, so nothing else in the
suite would catch a drift.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from aef.models.hardnet import (AngularRFFTMag, LogPolarBlurPool, LogPolarPad,
                                angular_rdft)


@pytest.mark.parametrize("pad", [0, 1, 2, 4])
def test_logpolar_pad_matches_f_pad(pad):
    """Wrap on the angular axis, zeros on the radial one — as F.pad would have done."""
    x = torch.randn(2, 3, 16, 10)
    expected = F.pad(x, (0, 0, pad, pad), mode="circular") if pad else x
    expected = F.pad(expected, (pad, pad, 0, 0), mode="constant", value=0)
    assert torch.equal(LogPolarPad(pad)(x), expected)


def test_blur_pool_matches_f_pad_version():
    x = torch.randn(2, 5, 16, 12)
    pool = LogPolarBlurPool(5)
    padded = F.pad(x, (0, 0, 1, 1), mode="circular")
    padded = F.pad(padded, (1, 1, 0, 0), mode="replicate")
    expected = F.conv2d(padded, pool.kernel, groups=5)[..., ::2, ::2]
    assert torch.equal(pool(x), expected)


@pytest.mark.parametrize("angular", [8, 16, 32])
@pytest.mark.parametrize("n_harmonics", [None, 1, 5, 99])
def test_angular_rdft_matches_torch_fft(angular, n_harmonics):
    """The matmul DFT must agree with rfft, including the harmonic truncation.

    ``n_harmonics=99`` covers asking for more bins than exist: rfft-then-slice silently
    clamps to ``A // 2 + 1``, and so must the matmul.
    """
    x = torch.randn(2, 3, angular, 7)
    re, im = angular_rdft(x, n_harmonics)
    keep = angular // 2 + 1 if n_harmonics is None else min(n_harmonics, angular // 2 + 1)
    spec = torch.fft.rfft(x, dim=-2)[:, :, :keep, :]
    assert re.shape == spec.shape
    assert torch.allclose(re, spec.real, atol=1e-4)
    assert torch.allclose(im, spec.imag, atol=1e-4)


def test_angular_rdft_uses_no_complex_dtype():
    """ONNX has no complex dtype; a complex intermediate would not survive the export."""
    re, im = angular_rdft(torch.randn(1, 2, 16, 4), 5)
    assert not re.is_complex() and not im.is_complex()


def test_fft_head_matches_the_rfft_reference():
    x = torch.randn(2, 3, 16, 7)
    expected = torch.fft.rfft(x, dim=-2).abs()[:, :, :5, :]
    assert torch.allclose(AngularRFFTMag(n_harmonics=5)(x), expected, atol=1e-4)


def test_magnitude_gradient_survives_a_vanishing_bin():
    """|X_k| is a sqrt; the eps is what keeps its gradient finite where the bin is 0.

    A constant angular profile zeroes every k >= 1 exactly — the case a real board patch
    with a flat ring hits — and a NaN there would poison the whole batch.
    """
    x = torch.ones(1, 1, 16, 4, requires_grad=True)
    AngularRFFTMag(n_harmonics=5)(x).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_dft_basis_precision_is_not_the_bottleneck():
    """The basis is built in float64 and rounded once; a float32 basis is ~6x worse.

    Cheap at this size, and it pins the choice rather than leaving it to look like a
    stray cast.
    """
    x = torch.randn(2, 4, 16, 8)
    re, im = angular_rdft(x, 5)
    ref = torch.fft.rfft(x.double(), dim=-2)[:, :, :5, :]
    error = max((re.double() - ref.real).abs().max(), (im.double() - ref.imag).abs().max())

    a = 16
    freq = torch.arange(5, dtype=torch.float32).view(1, 5)
    step = torch.arange(a, dtype=torch.float32).view(a, 1)
    ang = (-2.0 * math.pi / a) * (step * freq)
    xt = x.transpose(-1, -2)
    naive_re = torch.matmul(xt, torch.cos(ang)).transpose(-1, -2)
    naive = (naive_re.double() - ref.real).abs().max()
    assert error < naive
