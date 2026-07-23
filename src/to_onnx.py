import argparse
import os

import torch
from torch.export import Dim

from aef import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", metavar="M")
    parser.add_argument("weights", metavar="W")
    parser.add_argument("--resolution", type=int, default=64)

    args = parser.parse_args()

    model_params = {
        "head": "fft",
        "n_harmonics": 5,
        # "learned_mask": True,
    }

    model = getattr(models, args.model)(**model_params)
    model.eval()
    model.load_state_dict(torch.load(args.weights, weights_only=False, map_location=torch.device("cpu"))["model_state_dict"])
    # Batch MUST be > 1 in the example: torch.export applies 0/1 specialization,
    # so a size-1 batch axis is frozen to 1 and cannot be made dynamic — the export
    # then bakes `.view(1, -1)`, and at inference N patches collapse into one
    # flattened, globally-normalized vector (descriptors scaled ~1/sqrt(N) and
    # interleaved). Two rows keep the axis genuinely dynamic.
    example_inputs = (torch.randn((2, 1, args.resolution, args.resolution)),)
    # torch>=2.9 defaults to the dynamo exporter, which ignores `dynamic_axes`
    # (the legacy TorchScript arg) and consumes `dynamic_shapes` instead. A named
    # Dim keeps `patches.size(0)` symbolic, so `.view(B, -1)` + per-row L2Norm stay
    # per-descriptor and the output shape is exported as ['batch', 128].
    onnx_program = torch.onnx.export(
        model,
        args=example_inputs,
        input_names=["patches"],
        output_names=["div_1"],
        dynamic_shapes={"patches": {0: Dim("batch")}},
        dynamo=True,
    )
    onnx_program.save(f"{os.path.splitext(os.path.basename(args.weights))[0]}.onnx")
