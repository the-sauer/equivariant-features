import math

import pytest
import torch

from aef.evaluate import fpr, fpr_from_distances
from aef.geometry import homogenize, linearize_homography
from aef.transforms.affine import random_affine


def test_random_affine_without_scaling_or_rotation_is_identity():
    n = 5
    matrix = torch.stack([
        random_affine(image_size=(8, 10), scaling=False, rotation=False, translation=False) for _ in range(n)
    ])

    expected = torch.eye(3, dtype=torch.float32).expand(n, -1, -1)
    torch.testing.assert_close(matrix, expected)


def test_random_affine_scales_around_image_center(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda n, dtype=None: torch.zeros(n, dtype=dtype or torch.float32))
    n = 5
    matrix = torch.stack([
        random_affine(image_size=(8, 10), scaling=True, scaling_amplitude=1, rotation=False, translation=False)
        for _ in range(n)
    ])
    print(matrix)
    scales = torch.linalg.det(matrix[:, :2, :2]) ** (1/2)
    expected = (torch.tensor([
            [1.0, 0.0, 5.0],
            [0.0, 1.0, 4.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32).unsqueeze(0)
        @ torch.stack([torch.tensor([
            [s, 0.0, 0],
            [0.0, s, 0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32) for s in scales])
        @ torch.tensor([
            [1.0, 0.0, -5.0],
            [0.0, 1.0, -4.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32).unsqueeze(0))
    torch.testing.assert_close(matrix, expected)


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


@pytest.mark.parametrize("A, b, expected", [
    (torch.eye(2), torch.zeros(2), torch.eye(3)),
    (torch.eye(2).reshape(1, 1, 2, 2).expand(4, 5, -1, -1), torch.zeros(1, 1, 2).expand(4, 5, -1), torch.eye(3).reshape(1, 1, 3, 3).expand(4, 5, -1, -1)),
])
def test_homogenize(A, b, expected):
    torch.testing.assert_close(homogenize(A, b), expected)
