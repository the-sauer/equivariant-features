import torch

from aef.models.hardnet import AngularRFFTMag, HardNetLogPolar, input_norm


def test_angular_rfft_mag_is_cyclic_shift_invariant():
    """|rfft| along the angular axis (dim -2) is exactly invariant to integer rolls."""
    x = torch.randn(2, 4, 16, 16)
    head = AngularRFFTMag()
    base = head(x)
    for k in (1, 5, 8, 15):
        rolled = head(torch.roll(x, shifts=k, dims=-2))
        assert torch.allclose(base, rolled, atol=1e-5), f"not invariant to roll {k}"


def test_angular_rfft_mag_truncates_harmonics():
    x = torch.randn(1, 3, 16, 8)
    assert AngularRFFTMag()(x).shape == (1, 3, 9, 8)          # A // 2 + 1
    assert AngularRFFTMag(n_harmonics=5)(x).shape == (1, 3, 5, 8)


def test_input_norm_mask_none_matches_original():
    """mask=None must reproduce the original all-pixel normalization bit-for-bit."""
    x = torch.randn(3, 1, 32, 32)
    flat = x.view(x.size(0), -1)
    mp = torch.mean(flat, dim=1)
    sp = torch.std(flat, dim=1) + 1e-7
    expected = (x - mp.view(-1, 1, 1, 1)) / sp.view(-1, 1, 1, 1)
    assert torch.allclose(input_norm(x), expected, atol=1e-6)


def test_input_norm_ignores_masked_pixels():
    """Corrupting the invalid region must not change the normalized valid pixels."""
    x = torch.randn(2, 1, 16, 16)
    mask = torch.ones_like(x)
    mask[..., 8:, :] = 0.0
    a = input_norm(x, mask=mask)
    x2 = x.clone()
    x2[..., 8:, :] = 999.0                                     # trash the invalid half
    b = input_norm(x2, mask=mask)
    assert torch.allclose(a * mask, b * mask, atol=1e-5)
    assert torch.allclose((a * (1 - mask)).abs().sum(), torch.tensor(0.0))  # invalid -> 0


def test_default_path_returns_tensor_and_keeps_features_keys():
    """head='maxpool', learned_mask=False: unchanged structure and return type."""
    m = HardNetLogPolar(patch_size=64).eval()
    assert hasattr(m, "features") and not hasattr(m, "trunk")
    out = m(torch.randn(2, 1, 64, 64))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 128)


def test_fft_head_only_returns_tensor():
    m = HardNetLogPolar(patch_size=64, head="fft", n_harmonics=5).eval()
    assert hasattr(m, "trunk") and hasattr(m, "head")
    out = m(torch.randn(2, 1, 64, 64))
    assert isinstance(out, torch.Tensor) and out.shape == (2, 128)


def test_learned_mask_returns_descriptor_and_mask():
    m = HardNetLogPolar(patch_size=64, head="fft", learned_mask=True).eval()
    patches = torch.rand(4, 1, 64, 64)
    mask = torch.ones(4, 1, 64, 64)
    is_pdf = torch.tensor([True, False, True, False])
    d, m_pred = m(patches, mask=mask, is_pdf=is_pdf)
    assert d.shape == (4, 128)
    assert m_pred.shape[0] == 4 and m_pred.shape[1] == 1
    assert (m_pred >= 0).all() and (m_pred <= 1).all()


def test_learned_mask_forward_without_mask_still_runs():
    """A mask-aware model must still forward when the dataset supplies no mask."""
    m = HardNetLogPolar(patch_size=64, head="fft", learned_mask=True).eval()
    d, m_pred = m(torch.rand(2, 1, 64, 64))
    assert d.shape == (2, 128) and m_pred.shape[0] == 2
