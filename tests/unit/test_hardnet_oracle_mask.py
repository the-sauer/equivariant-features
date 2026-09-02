"""`oracle_mask=True`: the GT mask reaches every view, not only the PDF one.

The point of the flag is to measure the ceiling — what a *perfect* predictor would be
worth — so the properties that matter are that the prediction is genuinely bypassed on
the targets, that `m_pred` is still emitted (it is still supervised, as a diagnostic),
and that nothing about the parameters changes, so an oracle run and a normal one are the
same network trained under different information.
"""
import pytest
import torch

from aef.models.hardnet import HardNetLogPolar


def _scramble(net, seed=0):
    """Give `mask_head` a different, SPATIALLY VARYING output.

    Not a bias shift: that scales m_pred almost uniformly, and a uniform scale of the
    feature field cancels in the descriptor's L2 normalization — the perturbation would
    be invisible whether the oracle bypassed the predictor or not.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        w = net.mask_head[0].weight
        w.copy_(torch.randn(w.shape, generator=g) * 5.0)


def _net(oracle, **kw):
    return HardNetLogPolar(patch_size=64, head="fft", n_harmonics=4,
                           learned_mask=True, oracle_mask=oracle, **kw).eval()


def test_needs_learned_mask():
    with pytest.raises(ValueError, match="oracle_mask"):
        HardNetLogPolar(patch_size=64, head="fft", oracle_mask=True)


def test_same_parameters_as_the_normal_model():
    plain, oracle = _net(False), _net(True)
    assert set(plain.state_dict()) == set(oracle.state_dict())
    oracle.load_state_dict(plain.state_dict())
    x = torch.randn(2, 1, 64, 64)
    mask = torch.ones(2, 1, 64, 64)
    pdf = torch.tensor([1, 1])
    # On an all-PDF batch with an all-valid mask the two paths coincide.
    assert torch.allclose(plain(x, mask=mask, is_pdf=pdf)[0],
                          oracle(x, mask=mask, is_pdf=pdf)[0], atol=1e-6)


@pytest.mark.parametrize("cascade", [False, True])
def test_target_uses_the_gt_not_the_prediction(cascade):
    """Perturbing `mask_head` must not move the descriptor of a TARGET view."""
    net = _net(True, cascade=cascade)
    x = torch.randn(4, 1, 64, 64)
    mask = (torch.rand(4, 1, 64, 64) > 0.3).float()
    target = torch.zeros(4, dtype=torch.long)          # is_pdf = 0 everywhere
    before, _ = net(x, mask=mask, is_pdf=target)
    _scramble(net)
    after, m_pred = net(x, mask=mask, is_pdf=target)
    assert torch.allclose(before, after, atol=1e-6)
    assert m_pred is not None and m_pred.shape == (4, 1, 16, 16)


def test_without_oracle_the_prediction_does_move_the_target():
    """The complementary check — otherwise the test above would pass vacuously."""
    net = _net(False)
    x = torch.randn(4, 1, 64, 64)
    mask = (torch.rand(4, 1, 64, 64) > 0.3).float()
    target = torch.zeros(4, dtype=torch.long)
    before, _ = net(x, mask=mask, is_pdf=target)
    _scramble(net)
    after, _ = net(x, mask=mask, is_pdf=target)
    assert not torch.allclose(before, after, atol=1e-4)
