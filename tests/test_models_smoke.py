import pytest
import torch
from torch import nn

from aef.models import MODELS


@pytest.mark.skip(reason="Smoke test are currently not implemented correctly")
@pytest.mark.parametrize(
    ("model_name", "model_kwargs", "input_shape"),
    [
        ("scale_space_sesn", {"in_channels": 1, "factor": 1.25, "num_scales": 3, "min_scale": 0.5}, (4, 1, 12, 12)),
        ("affine_feature_net_one", {"in_channels": 2, "feature_size": 8, "conv_depths": [4, 4]}, (4, 2, 12, 12)),
        ("affine_feature_net_canonical_one", {"in_channels": 1, "conv_depths": [4, 4]}, (4, 1, 12, 12)),
    ],
)
def test_model_factories_run_for_a_few_optimization_steps(model_name, model_kwargs, input_shape):
    model_factory, _ = MODELS[model_name]
    model = model_factory(**model_kwargs)
    model.train()

    inputs = torch.randn(input_shape)

    with torch.no_grad():
        sample_output = model(inputs[:1])

    targets = torch.zeros((input_shape[0], sample_output.shape[1], input_shape[2], input_shape[3]))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    for index in range(0, inputs.size(0), 2):
        batch = inputs[index : index + 2]
        target = targets[index : index + 2]

        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
        assert output.shape == target.shape
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in model.parameters()
        )
