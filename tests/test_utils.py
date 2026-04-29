import math

import pytest
import torch

from aef.evaluate import fpr
from aef.train.losses.geodesic_loss import GeodesicLoss
from aef.train.losses.rel_scale_loss import RELScaleLoss
from aef.train.losses.reprojection_loss import HomographyReprojectionLoss
from aef.train.detector import linearize_homography
from aef.train.scale import compute_scale
from aef.transforms import random_affine


def test_random_affine_without_scaling_or_rotation_is_identity():
    matrix = random_affine(2, scale=False, rotate=False, image_size=(8, 10))

    expected = torch.eye(3, dtype=torch.float32).expand(2, -1, -1)
    torch.testing.assert_close(matrix, expected)


def test_random_affine_scales_around_image_center(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda n, dtype=None: torch.zeros(n, dtype=dtype or torch.float32))

    matrix = random_affine(1, scale=True, min_scale=0.5, max_scale=1.0, rotate=False, image_size=(8, 10))

    expected = torch.tensor(
        [
            [0.5, 0.0, 2.5],
            [0.0, 0.5, 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    torch.testing.assert_close(matrix, expected)


def test_compute_scale_returns_ones_for_identity_homography():
    homography = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    scale = compute_scale(homography, (4, 5))

    torch.testing.assert_close(scale, torch.ones(1, 1, 4, 5))


def test_linearize_homography_identity_is_identity_jacobian():
    homography = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    jacobian = linearize_homography(homography, (3, 4))

    expected = torch.eye(2, dtype=torch.float32).view(1, 1, 2, 2, 1).expand(1, 3, -1, -1, 4)
    torch.testing.assert_close(jacobian, expected)


def test_geodesic_loss_matches_right_angle_rotation():
    identity = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    quarter_turn = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    loss = GeodesicLoss(reduction="mean")(quarter_turn, identity)

    assert loss.item() == pytest.approx(math.pi / 2, rel=1e-5)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_homography_reprojection_loss_identity_is_zero(reduction):
    identity = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)
    loss = HomographyReprojectionLoss(reduction=reduction)(identity, identity)

    if reduction == "none":
        assert loss.shape == (1, 100)
        torch.testing.assert_close(loss, torch.zeros_like(loss))
    else:
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_homography_reprojection_loss_rejects_unknown_metric():
    identity = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3)

    with pytest.raises(ValueError, match="Unsupported distance metric"):
        HomographyReprojectionLoss(distance_metric="cosine")(identity, identity)


def test_relative_scale_loss_matches_mean_absolute_relative_error():
    loss = RELScaleLoss()(torch.tensor([2.0, 4.0]), torch.tensor([1.0, 2.0]))

    assert loss.item() == pytest.approx(1.0)


def test_fpr_is_zero_when_positive_scores_rank_first():
    preds = torch.tensor([0.1, 0.9])
    labels = torch.tensor([1, 0])

    assert fpr(preds, labels, target_recall=0.5) == pytest.approx(0.0)