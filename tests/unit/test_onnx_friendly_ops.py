"""The log-polar trunk/head is written against ONNX-exportable primitives on purpose.

Three ops that read naturally in torch export to something onnxruntime only runs on the
CPU execution provider, which in a GPU session means a device round trip per node:

    F.pad(mode="circular")   -> Pad(mode="wrap")   no CUDA kernel
    F.pad(mode="replicate")  -> Pad(mode="edge")   ditto in older ORT builds
    torch.fft.rfft           -> DFT               CPU EP only, and ONNX has no complex dtype

So ``LogPolarPad``/``LogPolarBlurPool`` wrap via slice+concat and the angular heads run
the DFT as a matmul. These tests pin the rewrites to the reference torch ops they
replaced — the whole point is that the substitution is invisible, so nothing else in the
suite would catch a drift.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from aef.models.hardnet import (AngularBispectrum, AngularRelPhase, AngularRFFTMag,
                                LogPolarBlurPool, LogPolarPad, angular_rdft)


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


def _relphase_reference(x, n_harmonics, eps=1e-6):
    """The head as it reads with complex tensors — the version being replaced."""
    spec = torch.fft.rfft(x, dim=-2)[:, :, :n_harmonics, :]
    mag = spec.abs()
    if spec.shape[-2] < 3:
        return mag
    ref = spec[:, :, 1:2, :]
    ref = ref.conj() / (ref.abs() + eps)
    k = torch.arange(2, spec.shape[-2]).view(1, 1, -1, 1)
    c = spec[:, :, 2:, :] * ref.pow(k)
    return torch.cat([mag, c.real, c.imag], dim=-2)


def _bispectrum_reference(x, n_harmonics, normalize=True, eps=1e-6):
    spec = torch.fft.rfft(x, dim=-2)[:, :, :n_harmonics, :]
    mag = spec.abs()
    pairs = AngularBispectrum.pairs(spec.shape[-2])
    if not pairs:
        return mag
    k1 = torch.tensor([p[0] for p in pairs])
    k2 = torch.tensor([p[1] for p in pairs])
    x1, x2 = spec.index_select(-2, k1), spec.index_select(-2, k2)
    x12 = spec.index_select(-2, k1 + k2)
    bisp = x1 * x2 * x12.conj()
    if normalize:
        bisp = bisp / (x1.abs() * x2.abs() * x12.abs() + eps)
    return torch.cat([mag, bisp.real, bisp.imag], dim=-2)


@pytest.mark.parametrize("n_harmonics", [2, 5, 9])
def test_relphase_matches_the_complex_reference(n_harmonics):
    """Row order included: the final conv is sized for [mag ; Re(c_k) ; Im(c_k)]."""
    x = torch.randn(2, 3, 16, 7)
    head = AngularRelPhase(n_harmonics=n_harmonics)
    assert torch.allclose(head(x), _relphase_reference(x, n_harmonics), atol=1e-4)


@pytest.mark.parametrize("normalize", [True, False])
def test_bispectrum_matches_the_complex_reference(normalize):
    x = torch.randn(2, 3, 16, 7)
    head = AngularBispectrum(n_harmonics=5, normalize=normalize)
    expected = _bispectrum_reference(x, 5, normalize=normalize)
    assert torch.allclose(head(x), expected, atol=1e-3 if normalize else 1e-2)


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

    The phase heads multiply three coefficients together, so basis error compounds — this
    pins the choice rather than leaving it to look like a stray cast.
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
