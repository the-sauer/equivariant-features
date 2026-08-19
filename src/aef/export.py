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

"""ONNX export of a trained descriptor, shared by the CLI and the training loop.

`src/to_onnx.py` is the interactive front end (restate the architecture, or point at a
run directory); `aef.train` calls :func:`export_checkpoint` itself once training ends so
every run leaves a deployable graph next to its weights. Both go through the same
:func:`export_model`, so an automatic export succeeding means the CLI would too.

A checkpoint stores weights only, so the architecture always has to come from somewhere
else — the caller's flags, or the `cfg.yaml` a run wrote down (:func:`from_run`).
"""

import collections
import logging
import os

import torch
from torch.export import Dim

from . import models
from .models import escnn_export


# Ops onnxruntime implements on the CPU EP only: hitting one in a GPU session forces a
# device->host->device round trip around it. `DFT` is the one this repo kept walking
# into (`torch.fft.rfft` in the log-polar angular heads, now a matmul instead), and
# `Pad(mode="wrap")` the other (circular padding, now slice+concat).
CPU_ONLY_OPS = {"DFT", "STFT", "MelWeightMatrix", "HannWindow", "HammingWindow", "BlackmanWindow"}
CPU_ONLY_PAD_MODES = {b"wrap", b"reflect"}


def batch_dynamic():
    """`dynamic_shapes` marking the batch axis of a single positional input dynamic.

    Positional (a 1-tuple matching ``args``), *not* a dict: a dict has to be keyed by the
    forward parameter's own name, which differs across this repo's descriptors —
    ``patches`` on :class:`HardNetLogPolar`, ``x`` on the steerable and efficient blob
    descriptors — so a dict silently ties the export to one model family and raises
    ``UserError: its top-level keys must be the arg names`` for the rest. The ONNX graph
    input is named by ``input_names`` regardless.

    A fresh ``Dim`` per call, since one export's symbols do not belong to another's.
    """
    return ({0: Dim("batch")},)


def from_run(run_dir, checkpoint="best", checkpoint_dir=None):
    """Read a training run's own `cfg.yaml` — the record of what it actually built.

    Returns ``(model_name, weights_path, params)``. Only the `model:` block is resolved:
    the rest of the config keeps interpolations into keys that only exist at train time
    (`${track_path}` and friends), which would raise here for no reason.

    ``checkpoint_dir`` overrides where the `.pth` is looked up, for runs that moved it off
    the default `<run_dir>/checkpoints` via `logging.checkpoint_dir`.
    """
    from omegaconf import OmegaConf                          # pylint: disable=import-outside-toplevel

    cfg_path = os.path.join(run_dir, "cfg.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"no cfg.yaml in {run_dir!r} — pass MODEL and WEIGHTS explicitly")
    weights = os.path.join(checkpoint_dir or os.path.join(run_dir, "checkpoints"),
                           f"{checkpoint}.pth")
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"no such checkpoint: {weights}")

    cfg = OmegaConf.load(cfg_path)
    params = OmegaConf.to_container(cfg.model.get("params", {}), resolve=True)
    return cfg.model.name, weights, params


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


# Buffers an escnn conv derives from its basis coefficients rather than storing:
# `expand_parameters()` writes them on the train->eval transition and `train()` deletes
# them again, so whether a checkpoint contains them depends only on the mode the model
# happened to be in when it was saved. They are recomputed from the loaded weights, so a
# mismatch on them either way is not a mismatch of the trained model.
EXPANSION_BUFFERS = {"filter", "expanded_bias"}


def _is_expansion_buffer(key):
    """Matched on the leaf name, so a conv at the top level counts too, not just `net.1.*`."""
    return key.rsplit(".", 1)[-1] in EXPANSION_BUFFERS


def load_weights(model, weights):
    """Load `weights` into `model`, tolerating escnn's derived expansion buffers.

    Must be called while `model` is still in **training** mode: `escnn.nn.R2Conv` only
    materializes `<layer>.filter` / `<layer>.expanded_bias` when it switches to eval, and
    a checkpoint written mid-training has no such entry — loading into an already-eval'd
    model then dies with `Missing key(s) ... net.1.filter`. The reverse (a checkpoint
    saved from an eval-mode model) shows up as unexpected keys. Both are ignored, and the
    caller's later `.eval()` regenerates them from the coefficients just loaded.

    Anything else missing or unexpected is a real architecture mismatch and raises.
    """
    state = load_state_dict(weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    stray_missing = [k for k in missing if not _is_expansion_buffer(k)]
    stray_unexpected = [k for k in unexpected if not _is_expansion_buffer(k)]
    if stray_missing or stray_unexpected:
        raise RuntimeError(
            f"{type(model).__name__} does not match {weights}: "
            f"missing {stray_missing}, unexpected {stray_unexpected} — the constructor "
            f"params do not describe the architecture this checkpoint was trained with"
        )
    return model


def build_model(model_name, params, weights=None):
    """Construct `model_name` from `aef.models` with `params`, optionally loading weights.

    Built on the CPU and left in eval mode, which is what the export needs. The weights go
    in *before* `eval()`, so escnn's expanded filters are derived from them; see
    :func:`load_weights`.
    """
    model = getattr(models, model_name)(**params)
    if weights is not None:
        load_weights(model, weights)
    model.eval()
    return model


def deploy_for_export(model, example_inputs, reference, tolerance=1e-5):
    """Replace a steerable model's escnn layers with plain-torch equivalents.

    ``torch.export`` cannot trace ``escnn.nn.R2Conv``, which expands its filter from basis
    coefficients on every forward and touches the basis' raw storage while doing so — with
    FakeTensor inputs that raises ``Cannot access data pointer``. At eval time the expanded
    filter is constant, so :func:`aef.models.escnn_export.deploy` bakes it into ordinary
    ``Conv2d`` weights; see that module for what each layer becomes.

    Converting is only safe if it changes nothing, so the converted model is re-run on the
    same input and compared against ``reference`` — a mismatch raises rather than writing a
    silently wrong graph. Returns the new ``(model, reference)``; the model is converted in
    place, hence the caller must own it.
    """
    if not escnn_export.is_escnn(model):
        return model, reference
    model = escnn_export.deploy(model)
    with torch.no_grad():
        deployed = model(*example_inputs)
    deployed = deployed if isinstance(deployed, tuple) else (deployed,)
    if len(deployed) != len(reference):
        raise RuntimeError(f"deploy() changed the number of outputs: "
                           f"{len(reference)} -> {len(deployed)}")
    for i, (got, want) in enumerate(zip(deployed, reference)):
        delta = (got - want).abs().max().item()
        if not delta <= tolerance:
            raise RuntimeError(
                f"deploy() changed output {i}: max |plain - escnn| = {delta:.3e} > "
                f"{tolerance:.1e}. Refusing to export a graph that does not match the "
                f"trained model."
            )
    return model, deployed


def export_model(model, path, resolution, in_channels=1, opset=None,
                 output_name="descriptors"):
    """Export an already-built model to `path`.

    Returns ``(onnx_program, example_inputs, reference)``; ``reference`` is the torch
    output on the dummy input, for :func:`check`.

    A steerable (escnn) model is converted to plain torch first — see
    :func:`deploy_for_export`. That conversion is in place, so `model` must be one the
    caller owns; :func:`export_checkpoint` builds a fresh one for exactly this reason.
    """
    model.eval()
    # Batch MUST be > 1 in the example: torch.export applies 0/1 specialization,
    # so a size-1 batch axis is frozen to 1 and cannot be made dynamic — the export
    # then bakes `.view(1, -1)`, and at inference N patches collapse into one
    # flattened, globally-normalized vector (descriptors scaled ~1/sqrt(N) and
    # interleaved). Two rows keep the axis genuinely dynamic.
    example_inputs = (torch.randn((2, in_channels, resolution, resolution)),)
    with torch.no_grad():
        reference = model(*example_inputs)
    reference = reference if isinstance(reference, tuple) else (reference,)
    model, reference = deploy_for_export(model, example_inputs, reference)
    # `learned_mask` models return (descriptor, m_pred); with no mask/is_pdf given the
    # predicted mask is the one that is applied, which is exactly the inference case.
    output_names = [output_name] + [f"aux_{i}" for i in range(1, len(reference))]
    if len(reference) > 1:
        output_names[1] = "mask"

    # torch>=2.9 defaults to the dynamo exporter, which ignores `dynamic_axes`
    # (the legacy TorchScript arg) and consumes `dynamic_shapes` instead. The named
    # Dim from `batch_dynamic()` keeps the input's size(0) symbolic, so `.view(B, -1)`
    # + per-row L2Norm stay per-descriptor and the output is exported as ['batch', 128].
    export_kwargs = {"opset_version": opset} if opset is not None else {}
    onnx_program = torch.onnx.export(
        model,
        args=example_inputs,
        input_names=["patches"],
        output_names=output_names,
        dynamic_shapes=batch_dynamic(),
        dynamo=True,
        **export_kwargs,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    onnx_program.save(path)
    return onnx_program, example_inputs, reference


def export_checkpoint(model_name, params, weights, path, resolution=None, opset=None,
                      output_name="descriptors"):
    """Build `model_name(**params)`, load `weights` into it and export to `path`.

    ``resolution`` defaults to the model's own `patch_size` (else 64), matching how the
    dataset fed it during training.
    """
    resolution = resolution or params.get("patch_size") or 64
    model = build_model(model_name, params, weights)
    return export_model(model, path, resolution,
                        in_channels=params.get("in_channels", 1),
                        opset=opset, output_name=output_name)


def cpu_only_ops(model_proto):
    """Names of graph nodes onnxruntime can only run on the CPU execution provider."""
    offenders = []
    for node in model_proto.graph.node:
        if node.op_type in CPU_ONLY_OPS:
            offenders.append(f"{node.op_type} ({node.name})")
        elif node.op_type == "Pad":
            mode = next((a.s for a in node.attribute if a.name == "mode"), b"constant")
            if mode in CPU_ONLY_PAD_MODES:
                offenders.append(f"Pad(mode={mode.decode()}) ({node.name})")
    return offenders


def report(model_proto):
    """Op histogram, plus a warning for anything onnxruntime keeps on the CPU EP."""
    counts = collections.Counter(node.op_type for node in model_proto.graph.node)
    print("\nexported ops:")
    for op_type, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:4d}  {op_type}")

    offenders = cpu_only_ops(model_proto)
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


def export_after_training(cfg, checkpoint_dir, model_name=None, params=None):
    """Export a finished run's checkpoints to ONNX, driven by the `logging:` config.

    Called at the end of `aef.train.train_func`'s loop. Reads
    ``logging.export_onnx`` (bool, default false) and ``logging.export_onnx_checkpoints``
    (list of checkpoint stems, default ``["best"]``) and writes ``<stem>.onnx`` next to
    each ``<stem>.pth``. ``logging.export_onnx_resolution`` / ``_opset`` override the
    dummy-input side length (default: the model's `patch_size`) and the opset.

    The model is rebuilt from the config and reloaded from the checkpoint rather than
    reusing the in-memory (last-epoch, on-GPU) module — `best.pth` is a different set of
    weights, and this is the same path `to_onnx.py --run` takes.

    Never raises: an export failure at the very end must not lose a finished run. Every
    outcome goes to the log and to stdout.
    """
    log_cfg = getattr(cfg, "logging", None)
    if log_cfg is None or not getattr(log_cfg, "export_onnx", False):
        return []
    if checkpoint_dir is None or not getattr(log_cfg, "model_checkpoints", False):
        msg = ("logging.export_onnx is set but no checkpoints are being written "
               "(logging.model_checkpoints is false) — nothing to export")
        print(f"\033[1m{msg}\033[0m")
        logging.warning(msg)
        return []

    stems = getattr(log_cfg, "export_onnx_checkpoints", None) or ["best"]
    if isinstance(stems, str):
        stems = [stems]

    if model_name is None:
        model_name = cfg.model.name
    if params is None:
        import omegaconf                                     # pylint: disable=import-outside-toplevel
        params = (omegaconf.OmegaConf.to_container(cfg.model.params, resolve=True)
                  if "params" in cfg.model else {})

    written = []
    for stem in stems:
        weights = os.path.join(checkpoint_dir, f"{stem}.pth")
        path = os.path.join(checkpoint_dir, f"{stem}.onnx")
        if not os.path.isfile(weights):
            msg = f"skipping ONNX export of {stem!r}: {weights} does not exist"
            print(msg)
            logging.warning(msg)
            continue
        try:
            onnx_program, _, _ = export_checkpoint(
                model_name, params, weights, path,
                resolution=getattr(log_cfg, "export_onnx_resolution", None),
                opset=getattr(log_cfg, "export_onnx_opset", None),
            )
        except Exception as exc:                             # pylint: disable=broad-except
            # Includes the model simply not being exportable (unsupported op, a
            # non-tensor forward signature, missing `onnx`): report and move on.
            msg = f"ONNX export of {weights} failed: {type(exc).__name__}: {exc}"
            print(f"\033[1m{msg}\033[0m")
            logging.exception(msg)
            continue
        offenders = cpu_only_ops(onnx_program.model_proto)
        msg = f"exported {weights} -> {path}"
        # Past ~2 GB the weights do not fit protobuf and torch parks them in a sidecar.
        # The .onnx is then useless on its own, so say so — copying just the graph file
        # to a deployment target is the obvious mistake.
        sidecar = f"{path}.data"
        if os.path.isfile(sidecar):
            msg += (f" + {os.path.basename(sidecar)} "
                    f"({os.path.getsize(sidecar) / 1e9:.1f} GB of external weights — "
                    f"the .onnx needs it alongside)")
        if offenders:
            msg += f" (WARNING: {len(offenders)} CPU-only onnxruntime op(s): {', '.join(offenders)})"
        print(f"\033[1m{msg}\033[0m")
        logging.info(msg)
        written.append(path)
    return written
