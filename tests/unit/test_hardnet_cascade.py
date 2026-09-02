"""`cascade=True`: the mask gates the INPUT (two trunk passes) instead of the pooling.

The properties worth pinning are the ones a warm start and the training loop rely on:
same parameters as the non-cascade model, the same `(descriptor, m_pred)` contract, the
gate actually reaching the descriptor, and gradients flowing into `mask_head` through the
gate (that is the whole point of training with the gate in the loop).
"""
import pytest
import torch

from aef.models.hardnet import HardNetLogPolar


def _net(**kw):
    return HardNetLogPolar(patch_size=64, head="fft", n_harmonics=4,
                           learned_mask=True, **kw).eval()


def _sharpen(m, seed=0):
    """Give `mask_head` a spatially varying output.

    A freshly initialised predictor emits an almost constant m_pred (~0.5), and a
    *constant* weight is invisible downstream: every conv block ends in a
    `BatchNorm2d(affine=False)` and the descriptor is L2-normalised, so a global scale
    cancels exactly. Only spatial variation can change the descriptor, so any test of
    what the mask does has to make the predictor non-degenerate first.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = m.mask_head[0].weight
        w.copy_(torch.randn(w.shape, generator=g) * 5.0)
    return m


def test_cascade_requires_learned_mask():
    with pytest.raises(ValueError, match="learned_mask"):
        HardNetLogPolar(patch_size=64, head="fft", cascade=True)


def test_cascade_adds_no_parameters_so_a_warm_start_transfers():
    """No new module — trunk and mask_head serve both passes."""
    plain, casc = _net(), _net(cascade=True)
    assert set(plain.state_dict()) == set(casc.state_dict())
    casc.load_state_dict(plain.state_dict())          # exact, not strict=False


def test_cascade_returns_descriptor_and_mask():
    m = _net(cascade=True)
    d, m_pred = m(torch.randn(2, 1, 64, 64))
    assert d.shape == (2, 128)
    assert m_pred.shape == (2, 1, 16, 16)
    assert (m_pred >= 0).all() and (m_pred <= 1).all()
    assert torch.allclose(d.norm(dim=1), torch.ones(2), atol=1e-5)


def test_cascade_gate_changes_the_descriptor():
    """A cascade descriptor must differ from the late-weighted one — else the gate is
    not reaching the trunk and the run would silently measure the old model."""
    x = torch.randn(2, 1, 64, 64)
    plain = _sharpen(_net())
    casc = _net(cascade=True)
    casc.load_state_dict(plain.state_dict())
    d_late, _ = plain(x)
    d_casc, _ = casc(x)
    assert not torch.allclose(d_late, d_casc, atol=1e-4)


def test_pdf_patches_are_gated_by_their_ground_truth_mask():
    """is_pdf=1 -> the GT mask gates the input, so trashing the invalid region must not
    move the descriptor; is_pdf=0 -> the prediction gates it and the junk gets through."""
    torch.manual_seed(0)
    m = _net(cascade=True)
    x = torch.randn(2, 1, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    mask[..., 40:, :] = 0.0
    x2 = x.clone()
    x2[..., 40:, :] = 50.0                                  # trash the off-board region

    pdf = torch.tensor([1, 1])
    d_a, _ = m(x, mask=mask, is_pdf=pdf)
    d_b, _ = m(x2, mask=mask, is_pdf=pdf)
    assert torch.allclose(d_a, d_b, atol=1e-4)

    tgt = torch.tensor([0, 0])
    d_c, _ = m(x, mask=mask, is_pdf=tgt)
    d_d, _ = m(x2, mask=mask, is_pdf=tgt)
    assert not torch.allclose(d_c, d_d, atol=1e-4)


def test_gradients_reach_the_mask_head_through_the_gate():
    """Without a mask loss at all, the descriptor alone must still train the predictor —
    that is what `cascade` changes about the mask path."""
    m = _net(cascade=True).train()
    d, _ = m(torch.randn(4, 1, 64, 64))
    d.sum().backward()
    g = m.mask_head[0].weight.grad
    assert g is not None and g.abs().sum() > 0


def test_late_weight_is_opt_in():
    x = torch.randn(2, 1, 64, 64)
    off = _sharpen(_net(cascade=True))
    on = _net(cascade=True, cascade_late_weight=True)
    on.load_state_dict(off.state_dict())
    assert not torch.allclose(off(x)[0], on(x)[0], atol=1e-4)
