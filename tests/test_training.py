import omegaconf
import pytest
import torch

import aef
from aef.train import train_func
from aef.transforms.affine import random_affine
from aef.train.detector import process_batch_homographic_detector_for_transform_loss


DEFAULT_CFG = omegaconf.DictConfig(
    {
        "training": {
            "num_epochs": 1,
            "batch_size": 1,
            "feature_sampling": {"num_features": 1, "stride": None, "detector": None},
            "loss": "assertion_loss",
            "optimizer": {"name": "mock_optimizer"},
            "augmentation": None,
            "dataset": None,
        },
        "validation": {
            "batch_size": 1,
            "feature_sampling": {"num_features": 1},
            "loss": "assertion_loss",
        },
        "model": None,
        "logging": None,
    }
)


class MockModel(torch.nn.Module):
    def __init__(self, gt_transform):
        super().__init__()
        self.gt_transform = gt_transform
        self.params = torch.nn.Parameter(torch.eye(3, 3, dtype=torch.float32))

    def forward(self, x):
        if torch.abs(torch.mean(x - 1)) < 1e-3:
            return torch.eye(2, 2).reshape(1, 2, 2, 1, 1).expand(x.size(0), -1, -1, x.size(-2), x.size(-1)).to(x.device)
        else:
            return ((self.gt_transform.to(x.device) @ self.params)[:2, :2]
                    .reshape(1, 2, 2, 1, 1)
                    .expand(x.size(0), -1, -1, x.size(-2), x.size(-1)))


class MockOptimizer:
    def __init__(self, parameters, **_):
        pass

    def zero_grad(self, set_to_none=False):
        pass

    def step(self):
        pass


def assert_loss(pred, gt):
    assert pred.shape == gt.shape
    torch.testing.assert_close(pred, gt)
    return torch.nn.MSELoss()(pred, gt)


@pytest.mark.parametrize("transform", [
    "identity",
    "rotation",
    "scaling",
    "affine"
])
def test_detector_training_homographic(transform, monkeypatch):
    monkeypatch.setattr(aef.train.losses, "_LOSSES", {"assertion_loss": lambda **_: assert_loss})
    monkeypatch.setattr(aef.train, "OPTIMIZERS", {"mock_optimizer": MockOptimizer})

    torch.manual_seed(1337)

    if transform == "identity":
        gt_transform_inv = gt_transform = torch.eye(3, dtype=torch.float32)
    elif transform == "rotation":
        gt_transform = random_affine(image_size=(8, 10), scaling=False, rotation=True, translation=False)
        gt_transform_inv = torch.linalg.inv(gt_transform)
    elif transform == "scaling":
        gt_transform = random_affine(image_size=(8, 10), scaling=True, rotation=False, translation=False)
        gt_transform_inv = torch.linalg.inv(gt_transform)
    elif transform == "affine":
        gt_transform = random_affine(image_size=(8, 10), scaling=True, rotation=True, translation=False)
        gt_transform_inv = torch.linalg.inv(gt_transform)
    else:
        assert False, f"Unknown transform type: {transform}"

    model = MockModel(gt_transform)

    dataset = [(torch.ones(1, 8, 10), torch.zeros(1, 8, 10), gt_transform, gt_transform_inv)]

    train_func(process_batch_homographic_detector_for_transform_loss)(model, dataset, dataset, DEFAULT_CFG)
