# Affine Equivariant Features, the main implementation of my master thesis.
# Copyright (C) 2026 Hendrik Sauer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Export a trained descriptor to ONNX.

The model is built by name from ``aef.models`` and its constructor kwargs come from the
command line — nothing about the architecture is baked in here, because a checkpoint
stores only weights, so the caller has to restate the hyperparameters the run used::

    python src/to_onnx.py HardNetLogPolar runs/xyz/checkpoints/best.pth \\
        --resolution 64 --head fft --n-harmonics 5

Anything without a dedicated flag rides through ``-p/--param KEY=VALUE`` (values are
parsed as Python literals, falling back to a plain string), so a new model kwarg needs
no change here.

Restating them by hand is the error-prone part, though, and a training run already wrote
down exactly what it built — so point at the run directory instead and let it read
``cfg.yaml`` (explicit flags still win over what the config says)::

    python src/to_onnx.py --run runs/xyz

``--summary`` prints the exported op histogram and warns about ops that onnxruntime
only implements on the CPU execution provider — those silently drag a GPU session back
through host memory. Keep that list empty.
"""

import argparse
import ast
import collections
import os

import torch
from torch.export import Dim

from aef import models


# Ops onnxruntime implements on the CPU EP only: hitting one in a GPU session forces a
# device->host->device round trip around it. `DFT` is the one this repo kept walking
# into (`torch.fft.rfft` in the log-polar angular heads, now a matmul instead), and
# `Pad(mode="wrap")` the other (circular padding, now slice+concat).
CPU_ONLY_OPS = {"DFT", "STFT", "MelWeightMatrix", "HannWindow", "HammingWindow", "BlackmanWindow"}
CPU_ONLY_PAD_MODES = {b"wrap", b"reflect"}


def parse_value(text):
    """``"5"`` -> 5, ``"true"`` -> True, ``"fft"`` -> ``"fft"``."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("none", "null"):
            return None
        return text


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", metavar="MODEL", nargs="?",
                        help="class name in `aef.models`, e.g. HardNetLogPolar")
    parser.add_argument("weights", metavar="WEIGHTS", nargs="?",
                        help=".pth checkpoint (or a bare state_dict)")
    parser.add_argument("--run", metavar="DIR",
                        help="training run directory: take MODEL/params from its cfg.yaml and "
                             "WEIGHTS from its checkpoints/. Pass no positionals with this.")
    parser.add_argument("--checkpoint", default="best",
                        help="which checkpoint under --run to export (default: best)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="patch side length of the exported dummy input "
                             "(default: the model's patch_size, else 64)")

    arch = parser.add_argument_group(
        "model params",
        "Constructor kwargs. Unset flags are not passed at all, so the model's own "
        "defaults apply. `--patch-size` defaults to `--resolution`.",
    )
    arch.add_argument("--patch-size", type=int, default=None)
    arch.add_argument("--in-channels", type=int, default=None)
    arch.add_argument("--head", choices=["maxpool", "fft", "relphase", "bispectrum"], default=None)
    arch.add_argument("--n-harmonics", type=int, default=None)
    arch.add_argument("--learned-mask", action="store_true", default=None)
    arch.add_argument("--slim", action="store_true", default=None)
    arch.add_argument("--no-circular-pad", dest="circular_pad", action="store_false", default=None)
    arch.add_argument("--no-antialias", dest="antialias", action="store_false", default=None)
    arch.add_argument("-p", "--param", metavar="KEY=VALUE", action="append", default=[],
                      help="any other constructor kwarg; repeatable")

    out = parser.add_argument_group("output")
    out.add_argument("-o", "--output", default=None,
                     help="output .onnx path (default: next to the weights with --run, "
                          "else <weights basename>.onnx in the cwd)")
    out.add_argument("--output-name", default="descriptors",
                     help="name of the descriptor output tensor (default: descriptors)")
    out.add_argument("--opset", type=int, default=None, help="ONNX opset version")
    out.add_argument("--summary", action="store_true",
                     help="print the op histogram and flag CPU-only onnxruntime ops")
    out.add_argument("--check", action="store_true",
                     help="run the exported graph through onnxruntime and diff against torch")

    args = parser.parse_args()
    if args.run:
        if args.model or args.weights:
            parser.error("--run supplies MODEL and WEIGHTS itself; use --checkpoint to pick "
                         "a different .pth than best")
        args.model, args.weights, args.run_params = from_run(args.run, args.checkpoint)
    else:
        if not args.model or not args.weights:
            parser.error("MODEL and WEIGHTS are required unless --run is given")
        args.run_params = {}
    return args


def from_run(run_dir, checkpoint):
    """Read a training run's own `cfg.yaml` — the record of what it actually built.

    Returns ``(model_name, weights_path, params)``. Only the `model:` block is resolved:
    the rest of the config keeps interpolations into keys that only exist at train time
    (`${track_path}` and friends), which would raise here for no reason.
    """
    from omegaconf import OmegaConf                          # pylint: disable=import-outside-toplevel

    cfg_path = os.path.join(run_dir, "cfg.yaml")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"no cfg.yaml in {run_dir!r} — pass MODEL and WEIGHTS explicitly")
    weights = os.path.join(run_dir, "checkpoints", f"{checkpoint}.pth")
    if not os.path.isfile(weights):
        raise SystemExit(f"no such checkpoint: {weights}")

    cfg = OmegaConf.load(cfg_path)
    params = OmegaConf.to_container(cfg.model.get("params", {}), resolve=True)
    return cfg.model.name, weights, params


def build_params(args):
    """Run config < CLI flags < `-p`, so the more explicit source always wins."""
    named = {
        "patch_size": args.patch_size,
        "in_channels": args.in_channels,
        "head": args.head,
        "n_harmonics": args.n_harmonics,
        "learned_mask": args.learned_mask,
        "slim": args.slim,
        "circular_pad": args.circular_pad,
        "antialias": args.antialias,
    }
    params = dict(args.run_params)
    params.update({k: v for k, v in named.items() if v is not None})
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        params[key.strip()] = parse_value(value)
    params.setdefault("patch_size", args.resolution or 64)
    return params


def load_state_dict(path):
    """Accept a training checkpoint or a bare state_dict, either pickle flavour."""
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:                                        # pylint: disable=broad-except
        # Older checkpoints pickle non-tensor objects (e.g. the resolved config).
        blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def report(model_proto):
    """Op histogram, plus a warning for anything onnxruntime keeps on the CPU EP."""
    counts = collections.Counter(node.op_type for node in model_proto.graph.node)
    print("\nexported ops:")
    for op_type, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:4d}  {op_type}")

    offenders = []
    for node in model_proto.graph.node:
        if node.op_type in CPU_ONLY_OPS:
            offenders.append(f"{node.op_type} ({node.name})")
        elif node.op_type == "Pad":
            mode = next((a.s for a in node.attribute if a.name == "mode"), b"constant")
            if mode in CPU_ONLY_PAD_MODES:
                offenders.append(f"Pad(mode={mode.decode()}) ({node.name})")
    if offenders:
        print("\nWARNING: onnxruntime has no CUDA kernel for these — the graph will fall")
        print("back to the CPU around each one:")
        for entry in offenders:
            print(f"  {entry}")
    else:
        print("\nno CPU-only onnxruntime ops in the graph.")


def check(path, example_inputs, expected):
    """Numeric parity between torch and onnxruntime on the export sample."""
    try:
        import onnxruntime as ort                            # pylint: disable=import-outside-toplevel
    except ImportError:
        print("\n--check skipped: onnxruntime is not installed in this environment.")
        return
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {session.get_inputs()[0].name: example_inputs[0].numpy()}
    outputs = session.run(None, feeds)
    for name, got, want in zip([o.name for o in session.get_outputs()], outputs, expected):
        delta = (torch.from_numpy(got) - want).abs().max()
        print(f"\n{name}: max |onnx - torch| = {delta:.3e}")
    # A second call with a different batch size proves the dynamic axis survived.
    doubled = torch.cat([example_inputs[0], example_inputs[0]], dim=0)
    session.run(None, {session.get_inputs()[0].name: doubled.numpy()})
    print(f"dynamic batch ok: ran with batch {doubled.size(0)} as well as {example_inputs[0].size(0)}")


def main():
    args = parse_args()
    params = build_params(args)
    resolution = args.resolution or params["patch_size"]
    print(f"{args.model}({', '.join(f'{k}={v!r}' for k, v in params.items())})")
    print(f"weights: {args.weights}")

    model = getattr(models, args.model)(**params)
    model.eval()
    model.load_state_dict(load_state_dict(args.weights))

    # Batch MUST be > 1 in the example: torch.export applies 0/1 specialization,
    # so a size-1 batch axis is frozen to 1 and cannot be made dynamic — the export
    # then bakes `.view(1, -1)`, and at inference N patches collapse into one
    # flattened, globally-normalized vector (descriptors scaled ~1/sqrt(N) and
    # interleaved). Two rows keep the axis genuinely dynamic.
    in_channels = params.get("in_channels", 1)
    example_inputs = (torch.randn((2, in_channels, resolution, resolution)),)
    with torch.no_grad():
        reference = model(*example_inputs)
    reference = reference if isinstance(reference, tuple) else (reference,)
    # `learned_mask` models return (descriptor, m_pred); with no mask/is_pdf given the
    # predicted mask is the one that is applied, which is exactly the inference case.
    output_names = [args.output_name] + [f"aux_{i}" for i in range(1, len(reference))]
    if len(reference) > 1:
        output_names[1] = "mask"

    # torch>=2.9 defaults to the dynamo exporter, which ignores `dynamic_axes`
    # (the legacy TorchScript arg) and consumes `dynamic_shapes` instead. A named
    # Dim keeps `patches.size(0)` symbolic, so `.view(B, -1)` + per-row L2Norm stay
    # per-descriptor and the output shape is exported as ['batch', 128].
    export_kwargs = {"opset_version": args.opset} if args.opset is not None else {}
    onnx_program = torch.onnx.export(
        model,
        args=example_inputs,
        input_names=["patches"],
        output_names=output_names,
        dynamic_shapes={"patches": {0: Dim("batch")}},
        dynamo=True,
        **export_kwargs,
    )
    # With --run the export belongs next to the checkpoint it came from: every run has a
    # `best.pth`, so the cwd default would have eight exports fighting over one name.
    default_out = os.path.splitext(args.weights if args.run else os.path.basename(args.weights))[0]
    path = args.output or f"{default_out}.onnx"
    onnx_program.save(path)
    print(f"\nwrote {path}")

    if args.summary:
        report(onnx_program.model_proto)
    if args.check:
        check(path, example_inputs, reference)


if __name__ == "__main__":
    main()
