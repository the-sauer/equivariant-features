import pytest
import torch

from aef.evaluate import fpr, fpr_from_distances
from aef.geometry import linearize_homography


def test_linearize_homography_identity_is_identity_jacobian():
    homography = torch.eye(3, dtype=torch.float32).unsqueeze(0)

    jacobian = linearize_homography(homography, (3, 4))

    expected = torch.eye(2, dtype=torch.float32).view(1, 1, 1, 2, 2).expand(1, 3, 4, -1, -1)
    torch.testing.assert_close(jacobian, expected)


def test_fpr_is_zero_when_positive_scores_rank_first():
    preds = torch.tensor([0.1, 0.9])
    labels = torch.tensor([1, 0])

    assert fpr_from_distances(preds, labels, target_recall=0.5) == pytest.approx(0.0)


def test_fpr_from_distances_is_zero_when_positive_scores_rank_first():
    dists = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])

    assert fpr()(dists, labels, target_recall=0.5) == pytest.approx(0.0)
