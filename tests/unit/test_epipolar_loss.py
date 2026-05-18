import pytest
import torch

from aef.train.losses.epipolar import EpipolarLoss


def test_epipolar_loss_zero_matrix_returns_zero():
    loss = EpipolarLoss(n_samples=8, reduction="mean")
    A = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    E = torch.zeros_like(A)
    pt = torch.tensor([[0.25, -0.5]], dtype=torch.float32)

    value = loss(A, E, pt)

    assert value.item() == pytest.approx(0.0, abs=1e-7)


def test_epipolar_loss_matches_scalar_with_deterministic_sampling():
    loss = EpipolarLoss(n_samples=4, reduction="mean")
    loss.sampling_distribution.sample = lambda shape: torch.zeros(shape)

    A = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    E = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    pt = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    value = loss(A, E, pt)

    assert value.item() == pytest.approx(6.0, abs=1e-7)


@pytest.mark.parametrize("reduction", ["median", "min"])
def test_epipolar_loss_rejects_unknown_reduction(reduction):
    with pytest.raises(ValueError, match="Unsupported reduction"):
        EpipolarLoss(n_samples=1, reduction=reduction)


def test_epipolar_loss_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="Unsupported sampling distribution"):
        EpipolarLoss(n_samples=1, sampling_distribution="uniform")


def test_epipolar_loss_requires_expected_shapes():
    loss = EpipolarLoss(n_samples=1)
    A = torch.eye(3, dtype=torch.float32)
    E = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    pt = torch.zeros((1, 2), dtype=torch.float32)

    with pytest.raises(AssertionError):
        loss(A, E, pt)
