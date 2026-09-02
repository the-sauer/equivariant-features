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

Training runs configured with ``logging.export_onnx: true`` do this themselves when they
finish (see ``aef.export.export_after_training``); this CLI is for re-exporting at a
different resolution/opset, for runs that predate that flag, and for runs that died
before their final epoch.
"""

import argparse
import ast
import os

from aef import export


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
        try:
            args.model, args.weights, args.run_params = export.from_run(args.run, args.checkpoint)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        if not args.model or not args.weights:
            parser.error("MODEL and WEIGHTS are required unless --run is given")
        args.run_params = {}
    return args


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


def main():
    args = parse_args()
    params = build_params(args)
    resolution = args.resolution or params["patch_size"]
    print(f"{args.model}({', '.join(f'{k}={v!r}' for k, v in params.items())})")
    print(f"weights: {args.weights}")

    # With --run the export belongs next to the checkpoint it came from: every run has a
    # `best.pth`, so the cwd default would have eight exports fighting over one name.
    default_out = os.path.splitext(args.weights if args.run else os.path.basename(args.weights))[0]
    path = args.output or f"{default_out}.onnx"

    onnx_program, example_inputs, reference = export.export_checkpoint(
        args.model, params, args.weights, path,
        resolution=resolution, opset=args.opset, output_name=args.output_name,
    )
    print(f"\nwrote {path}")

    if args.summary:
        export.report(onnx_program.model_proto)
    if args.check:
        export.check(path, example_inputs, reference)


if __name__ == "__main__":
    main()
