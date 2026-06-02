import pytest
import torch

from aef.train.losses.epipolar import EpipolarLoss


def _build_epipolar_input(A_1, A_2, pt_1, pt_2, F):
    detections = torch.stack([A_1, A_2], dim=0)
    pts = torch.stack([pt_1, pt_2], dim=0)
    matches = torch.tensor([[0, 1]], dtype=torch.long)
    img_ids = torch.zeros(2, dtype=torch.long)
    fundamental = F.reshape(1, 1, 3, 3)
    return {
        "matches": matches,
        "detections": detections,
        "img_ids": img_ids,
        "fundamental": fundamental,
        "pts": pts,
    }


def _build_epipolar_input_multi(A_list, pt_list, F, matches):
    detections = torch.stack(list(A_list), dim=0)
    pts = torch.stack(list(pt_list), dim=0)
    img_ids = torch.zeros(len(A_list), dtype=torch.long)
    fundamental = F.reshape(1, 1, 3, 3)
    return {
        "matches": matches,
        "detections": detections,
        "img_ids": img_ids,
        "fundamental": fundamental,
        "pts": pts,
    }


def _mismatch_mask(pt_1, pt_2, F):
    p1 = torch.cat([pt_1, torch.ones(1, dtype=pt_1.dtype)], dim=0).view(3, 1)
    p2 = torch.cat([pt_2, torch.ones(1, dtype=pt_2.dtype)], dim=0).view(1, 3)
    return (p2 @ F @ p1).squeeze() > 2


def _manual_epipolar_scalar(A_1, A_2, pt_1, pt_2, F):
    A_rel = A_2 @ torch.linalg.inv(A_1)
    A_rel_inv = torch.linalg.inv(A_rel)
    p1 = torch.cat([pt_1, torch.ones(1, dtype=pt_1.dtype)], dim=0).view(3, 1)
    p2 = torch.cat([pt_2, torch.ones(1, dtype=pt_2.dtype)], dim=0).view(3, 1)
    first = (A_rel @ p1).transpose(0, 1) @ F @ p1
    second = p2.transpose(0, 1) @ F @ A_rel_inv @ p2
    return (first + second).abs().squeeze()


def test_epipolar_loss_zero_matrix_returns_zero():
    loss = EpipolarLoss(n_samples=8, reduction="mean")
    A = torch.eye(3, dtype=torch.float32)
    F = torch.zeros((3, 3), dtype=torch.float32)
    pt_1 = torch.tensor([0.25, -0.5], dtype=torch.float32)
    pt_2 = torch.tensor([-1.25, 1.5], dtype=torch.float32)

    value = loss(_build_epipolar_input(A, A, pt_1, pt_2, F))

    assert value.item() == pytest.approx(0.0, abs=1e-7)


def test_epipolar_loss_matches_scalar_with_deterministic_sampling():
    loss = EpipolarLoss(n_samples=4, reduction="mean")
    loss.sampling_distribution.sample = lambda shape: torch.zeros(shape)

    A = torch.eye(3, dtype=torch.float32)
    F = torch.eye(3, dtype=torch.float32)
    pt_1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    pt_2 = torch.tensor([0.5, -1.5], dtype=torch.float32)

    value = loss(_build_epipolar_input(A, A, pt_1, pt_2, F))

    expected = (
        pt_1.pow(2).sum().item() + 1.0 +
        pt_2.pow(2).sum().item() + 1.0
    )
    assert value.item() == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("reduction", ["median", "min"])
def test_epipolar_loss_rejects_unknown_reduction(reduction):
    with pytest.raises(ValueError, match="Unsupported reduction"):
        EpipolarLoss(n_samples=1, reduction=reduction)


def test_epipolar_loss_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="Unsupported sampling distribution"):
        EpipolarLoss(n_samples=1, sampling_distribution="uniform")


def test_epipolar_loss_sum_reduction_adds_batch_terms():
    loss = EpipolarLoss(n_samples=2, reduction="sum")
    loss.sampling_distribution.sample = lambda shape: torch.zeros(shape)

    A = torch.eye(3, dtype=torch.float32)
    F = torch.eye(3, dtype=torch.float32)
    pt_1 = torch.tensor([1.0, 2.0], dtype=torch.float32)
    pt_2 = torch.tensor([0.5, -1.5], dtype=torch.float32)
    pt_3 = torch.tensor([-0.25, 0.75], dtype=torch.float32)
    pt_4 = torch.tensor([3.0, -2.0], dtype=torch.float32)

    matches = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    payload = _build_epipolar_input_multi(
        [A, A, A, A],
        [pt_1, pt_2, pt_3, pt_4],
        F,
        matches,
    )

    value = loss(payload)

    expected = (
        pt_1.pow(2).sum().item() + 1.0 + pt_2.pow(2).sum().item() + 1.0 +
        pt_3.pow(2).sum().item() + 1.0 + pt_4.pow(2).sum().item() + 1.0
    )
    assert value.item() == pytest.approx(expected, abs=1e-7)


def test_epipolar_loss_rejects_points_with_wrong_shape():
    loss = EpipolarLoss(n_samples=1)
    A = torch.eye(3, dtype=torch.float32)
    F = torch.eye(3, dtype=torch.float32)
    pt_1 = torch.zeros(3, dtype=torch.float32)
    pt_2 = torch.zeros(3, dtype=torch.float32)

    with pytest.raises(AssertionError):
        loss(_build_epipolar_input(A, A, pt_1, pt_2, F))


def test_epipolar_loss_nontrivial_f_and_relative_affine_matches_manual():
    loss = EpipolarLoss(n_samples=1, reduction="mean")
    loss.sampling_distribution.sample = lambda shape: torch.zeros(shape)

    A_1 = torch.tensor([
        [1.25, 0.1, 0.0],
        [0.2, 0.9, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    A_2 = torch.tensor([
        [0.8, -0.3, 0.0],
        [0.15, 1.1, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    F = torch.tensor([
        [0.0, -1.0, 0.5],
        [1.25, 0.0, -0.25],
        [-0.75, 0.4, 1.0],
    ], dtype=torch.float32)
    pt_1 = torch.tensor([0.6, -1.2], dtype=torch.float32)
    pt_2 = torch.tensor([-0.4, 2.0], dtype=torch.float32)

    assert not _mismatch_mask(pt_1, pt_2, F)

    value = loss(_build_epipolar_input(A_1, A_2, pt_1, pt_2, F))

    expected = _manual_epipolar_scalar(A_1, A_2, pt_1, pt_2, F)
    assert value.item() == pytest.approx(expected.item(), abs=1e-6)


def test_epipolar_loss_nontrivial_relative_affine_sum_reduction():
    loss = EpipolarLoss(n_samples=1, reduction="sum")
    loss.sampling_distribution.sample = lambda shape: torch.zeros(shape)

    F = torch.tensor([
        [0.0, -0.9, 0.3],
        [0.8, 0.0, -0.2],
        [-0.6, 0.1, 1.0],
    ], dtype=torch.float32)

    A_1 = torch.eye(3, dtype=torch.float32)
    A_2 = torch.tensor([
        [1.1, 0.2, 0.0],
        [-0.1, 0.95, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    A_3 = torch.tensor([
        [0.9, -0.15, 0.0],
        [0.25, 1.2, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)
    A_4 = torch.tensor([
        [1.05, 0.05, 0.0],
        [0.0, 0.85, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float32)

    pt_1 = torch.tensor([1.0, -0.5], dtype=torch.float32)
    pt_2 = torch.tensor([-0.25, 0.75], dtype=torch.float32)
    pt_3 = torch.tensor([0.4, 1.1], dtype=torch.float32)
    pt_4 = torch.tensor([-1.2, -0.6], dtype=torch.float32)

    assert not _mismatch_mask(pt_1, pt_2, F)
    assert not _mismatch_mask(pt_3, pt_4, F)

    matches = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    payload = _build_epipolar_input_multi(
        [A_1, A_2, A_3, A_4],
        [pt_1, pt_2, pt_3, pt_4],
        F,
        matches,
    )

    value = loss(payload)

    expected = (
        _manual_epipolar_scalar(A_1, A_2, pt_1, pt_2, F) +
        _manual_epipolar_scalar(A_3, A_4, pt_3, pt_4, F)
    )
    assert value.item() == pytest.approx(expected.item(), abs=1e-6)
