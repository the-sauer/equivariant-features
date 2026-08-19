"""`mask_resolution`: tap the mask head at a finer trunk stage than the output grid.

The properties that matter downstream: the default is untouched, the tap really produces
a finer m_pred, the late weighting still consumes it at field resolution, the cascade
gate uses it without interpolation when it already matches the patch, and the trunk
itself is unchanged (same parameters, same descriptor when the mask is neutral).
"""
import pytest
import torch

from aef.models.hardnet import HardNetLogPolar


def _net(**kw):
    return HardNetLogPolar(patch_size=64, head="fft", n_harmonics=4,
                           learned_mask=True, **kw).eval()


def test_default_is_the_trunk_grid():
    m = _net()
    assert m.mask_tap is None
    _, m_pred = m(torch.randn(2, 1, 64, 64))
    assert m_pred.shape == (2, 1, 16, 16)


@pytest.mark.parametrize("res,ch", [(64, 32), (32, 64), (16, 128)])
def test_tap_resolution_and_channels(res, ch):
    m = _net(mask_resolution=res)
    assert m.mask_head[0].in_channels == ch
    d, m_pred = m(torch.randn(2, 1, 64, 64))
    assert m_pred.shape == (2, 1, res, res)
    assert d.shape == (2, 128)


def test_rejects_a_grid_the_trunk_never_has():
    with pytest.raises(ValueError, match="mask_resolution"):
        HardNetLogPolar(patch_size=64, head="fft", learned_mask=True, mask_resolution=8)


def test_tapping_does_not_change_the_trunk():
    """Same trunk parameters, and with the mask forced neutral the descriptor is the
    same as the untapped model's — the tap only adds a read-out point."""
    plain, tapped = _net(), _net(mask_resolution=64)
    # Same keys, same shapes everywhere but the predictor's input channels — that is what
    # lets a warm start transfer everything except `mask_head`.
    assert set(plain.state_dict()) == set(tapped.state_dict())
    differing = {k for k, v in plain.state_dict().items()
                 if v.shape != tapped.state_dict()[k].shape}
    assert differing == {"mask_head.0.weight"}, differing
    tapped.load_state_dict({k: v for k, v in plain.state_dict().items()
                            if not k.startswith("mask_head")}, strict=False)
    x = torch.randn(2, 1, 64, 64)
    ones = torch.ones(2, 1, 64, 64)
    pdf = torch.tensor([1, 1])           # is_pdf -> weight is the (all-valid) GT mask
    assert torch.allclose(plain(x, mask=ones, is_pdf=pdf)[0],
                          tapped(x, mask=ones, is_pdf=pdf)[0], atol=1e-5)


def test_late_weight_consumes_a_finer_mask():
    """A 64x64 m_pred must still weight a 16x16 field (averaged down, not broadcast)."""
    m = _net(mask_resolution=64)
    x = torch.randn(2, 1, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    d, m_pred = m(x, mask=mask, is_pdf=torch.tensor([0, 0]))
    assert m_pred.shape[-2:] == (64, 64) and d.shape == (2, 128)


def test_cascade_gate_needs_no_upsampling_at_full_resolution():
    m = _net(mask_resolution=64, cascade=True)
    x = torch.randn(2, 1, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    mask[..., 40:, :] = 0.0
    d, m_pred = m(x, mask=mask, is_pdf=torch.tensor([1, 1]))
    assert m_pred.shape == (2, 1, 64, 64)
    x2 = x.clone()
    x2[..., 40:, :] = 50.0                      # GT-gated pdf view ignores the junk
    assert torch.allclose(d, m(x2, mask=mask, is_pdf=torch.tensor([1, 1]))[0], atol=1e-4)
