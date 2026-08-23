# Configs, training & bootstrapping

## The four entry points

Everything the repo trains is submitted by one of four Slurm scripts, each naming a
Hydra config in `src/conf/`:

| script | config | data |
| --- | --- | --- |
| `bootstrap_synthetic_network.sh` | `bootstrap_synthetic.yaml` | synthetic blob boards, `ProxyAnchoredSupCon` + `fft` head |
| `bootstrap_synthetic_vanilla.sh` | `bootstrap_synthetic_vanilla.yaml` | same boards, plain `SupCon` + `maxpool` head |
| `bootstrap_real.sh` | `bootstrap_real.yaml` | real `.tracks` footage, `ProxyAnchoredSupCon` + `fft` head |
| `bootstrap_real_vanilla.sh` | `bootstrap_real_vanilla.yaml` | same footage, plain `SupCon` + `maxpool` head |

The two `_vanilla` variants are the baseline the proxy-anchored/FFT pair is measured
against; they differ from their counterparts only in the loss and the angular head.

The real ones take the bootstrap round from the environment:

```sh
BB_ITERATION=4 sbatch bootstrap_real.sh
```

`iteration` is interpolated into `track_path`, so round *n* trains on the `.tracks`
file the previous round's detector produced.

Both real configs compose `track_descriptor_base.yaml`, which holds everything
`BlobTrackData` needs (the train/val board split, the view-angle balancing, the
optimizer, the logging block); the leaves add only `model`, `scale` and the loss. The
two synthetic configs are self-contained.

## The single `scale` hyperparameter

Each config defines one top-level `scale` key threaded everywhere via `${scale}`. For
these log-polar configs `scale` is `logpolar_outer_factor` — the outer radius of the
sampled annulus in units of the blob's own scale — and, on the real path, part of the
`.tracks` filename, so the model can never drift from the patches it was trained on.

Override it in one place:

```sh
python src/run_training.py --config-name bootstrap_synthetic scale=128
```

## `shared_params` validation & Hydra struct mode

`validation.shared_params` is a single `{name, params}` template; each
`validation.datasets` entry supplies only its overrides. `get_validation_specs`
(`src/aef/data/__init__.py`) deep-merges them.

Hydra composes configs in **struct mode**, which forbids adding keys not in the
template. The merge therefore first resolves the template to a plain container
(`OmegaConf.to_container(shared, resolve=True)` — this both bakes in `${scale}` against
the real root **and** drops struct mode) and only then merges the split's extra keys.
When testing config-merge code, compose via `hydra.compose` (struct mode), **not**
`OmegaConf.load` (non-struct) — the latter hides this class of bug.

## Sub-epoch validation

`validation.validate_every_n_batches` (declared as `null`) runs the **whole** validation
suite every N training batches, on top of the end-of-epoch run. Use it when one point
per epoch is too coarse to see what a run is doing — a long epoch over the track data,
or a warm-started run that moves within the first few hundred batches.

The extra points are plotted at the fractional epoch they were measured at, so the
validation figure's x axis becomes "fraction of training consumed": the run after
batch `k` of epoch `e` sits at `e + k/batches_per_epoch`, and the end-of-epoch run at
`e + 1`. With the option off the axis is unchanged — one point per epoch at integer
`e`, matching the training curve. That axis is stored alongside the curves as
`checkpoint["plots"]["x_val"]` (the training curve keeps its own `x`).

What it does *not* change: `best.pth` selection, which still compares whole epochs — a
mid-epoch run only records metrics, it never writes a checkpoint. Nor is it free; each
run is a full pass over every validation split, so N should be a decent fraction of an
epoch (tens to hundreds of batches), not single digits.

## DataLoader workers

Both loaders honour `training.num_workers` / `validation.num_workers` (see
`prepare_training`). With `num_workers > 0` the data pipeline (index cached patches,
collate, H2D copy) overlaps with the GPU step instead of running synchronously in the
main process — the main lever when the GPU is starved by single-process loading; extra
`-c` cores do nothing until this is set. The train loader uses `persistent_workers`
(avoids re-forking the large in-memory dataset each epoch); validation uses fewer,
non-persistent workers (its small loaders run sequentially). Keep Slurm `-c` ≥
`num_workers + ~2` (main + Julia board generation).

## Dataset cache

`cache_dir` (set in both synthetic configs for the training dataset and
`validation.shared_params`) makes a run reuse a previously prepared dataset instead of
re-rendering boards / re-extracting patches — cold ~54 s vs warm ~0 s. The key covers
dataset params only, so all models at a given `scale` share one entry. Set it to `null`
to disable. See [data pipeline → dataset cache](data_pipeline.md#dataset-cache-disk)
for the key, concurrency and the "frozen random draw" caveat.

`src/prebuild_datasets.py` builds a config's datasets and exits without training, so a
cold cache can be warmed by one job before several runs start against it (concurrent
cold jobs would each pay the full build — the cache only turns warm after one finishes):

```sh
python src/prebuild_datasets.py --config-name bootstrap_synthetic          # all of them
python src/prebuild_datasets.py --config-name bootstrap_synthetic +prebuild_target=train
python src/prebuild_datasets.py --config-name bootstrap_synthetic +prebuild_target=overall
```

It needs a GPU: SIFT detection and patch extraction run on CUDA.

## Checkpoints

Controlled by the `logging` block:

- `model_checkpoints` — master switch (on in all four configs; the runs are the
  deliverable).
- `checkpoint_dir` — `null` ⇒ `<logging.dir>/<experiment_name>/checkpoints`.
- `checkpoint_every_epoch` — `false` (default) ⇒ only `latest.pth` (refreshed every
  epoch, the resume point) and `best.pth` (lowest weighted validation loss, i.e. the
  `overall` FPR95) are written. `true` ⇒ additionally keep `epoch_<n>.pth` snapshots.

Resume in place with `+training.continue_from_checkpoint=<path>`. To start a fresh run
from another run's weights instead, use `+training.init_from_checkpoint=<path>` — it
loads model weights only and begins at epoch 0.

### Automatic ONNX export

The same block turns the finished run into a deployable graph, so a run's directory
carries its own artefact instead of needing a manual `to_onnx.py` pass afterwards:

- `export_onnx` — on in all four configs. Needs `model_checkpoints: true`; without
  checkpoints there is nothing to load and it warns instead.
- `export_onnx_checkpoints` — stems to export, default `[best]`. Each becomes
  `<stem>.onnx` beside its `<stem>.pth`.
- `export_onnx_resolution` / `export_onnx_opset` — `null` ⇒ the model's own `patch_size`
  and torch's default opset.

It runs once, after the last epoch's plots are written (`aef.export.export_after_training`).
The model is **rebuilt from `cfg.model` and reloaded from the checkpoint** rather than
serialized straight out of memory, since the in-memory module holds the *last* epoch's
weights while `best.pth` is generally an earlier one — the export is therefore identical
to what `python src/to_onnx.py --run <dir>` produces. A failing export is logged and
swallowed: a finished run must not be lost to it. A run killed before its final epoch
(Slurm timeout, preemption) never reaches the hook, so `to_onnx.py --run` stays the way
to export those.

The log-polar trunk and head are written against ONNX primitives onnxruntime accelerates
— slice+concat instead of `Pad(mode="wrap")`, a matmul instead of `torch.fft.rfft` (which
exports to `DFT`, a CPU-execution-provider-only op). `to_onnx.py --summary` prints the op
histogram and flags anything that would still fall back to the CPU; keep that list empty.
`tests/unit/test_onnx_friendly_ops.py` pins the rewrites against the torch ops they
replaced.

## Contrastive losses: `SupCon` vs `ProxyAnchoredSupCon`

`training.loss: [{name: SupCon}]` (the `_vanilla` configs) is `pytorch_metric_learning`'s
SupCon over every patch in the batch. `ProxyAnchoredSupCon` treats the board's own
rendering as the **proxy anchor** of its blob (vendored implementation in
`train/losses/SupConLoss.py`, wrapper in `contrastive.py`):

- the outer sum runs over the `is_pdf` patches only — the rendering (the GT sequence in a
  `.tracks` file, the identity view in `HomographyData`);
- `A(i)` and `P(i)` hold image patches only, so `pdf<->pdf` and `image<->image` terms
  disappear from the numerator **and** from the log-sum-exp denominator.

The motive is that matching happens against the board's rendering, not between two
observations, so the image<->image terms optimise something the descriptor is never
deployed on — and on track data they are also where the label noise sits.

The structure is that of **Proxy-Anchor Loss** (Kim et al., CVPR 2020): proxies as
anchors, each associated with the whole batch. The difference is where the proxy comes
from — Proxy-Anchor learns one free vector per class, here the proxy is the *embedded
rendering*, so it moves with the encoder and exists for a blob the model has never seen.
(Note "anchor" in that name, and in `SupConLoss`'s `anchor_feature` / `anchor_count`,
means the rows of the logit matrix; it is unrelated to the flag, which is why the flag is
`is_pdf` and the loss argument `is_proxy`.)

The flag comes from the batch key `"is_pdf"`, which `process_batch_blobs` filters
alongside the features and passes to every loss; `BlobTrackData` always emits it, and
`HomographyData` emits it whenever patches are precomputed. `ProxyAnchoredSupCon` raises
rather than silently degenerating to plain SupCon if it is missing. The flag is *not*
withheld during validation — otherwise the validation number would be a different loss
from the training one.

A proxy with no in-batch positive is dropped from the mean instead of contributing a
zero, so the value stays comparable across batches of differing yield. Note this narrows
the loss to roughly one row per track group: with `m_per_class: 4` a batch that fed
SupCon ~12 ordered pairs per group now feeds it ~3, so a larger `training.batch_size` (or
a larger `m_per_class`) buys back the gradient signal.

### The metric follows: `ProxyAnchoredFPR95`

`FPR95` ranks *every* patch pair in the batch, so like plain SupCon it is dominated by
image<->image pairs — a number about a question test time never asks. `ProxyAnchoredFPR95`
applies the same restriction to the metric: only pairs with exactly one proxy endpoint
enter the ranking. The pair set is unordered here (the distance matrix is symmetric and
there is no per-row normaliser), so the restriction is a cross-set mask rather than a
rows/columns split; `fpr()` takes it as `is_proxy=`.

Two consequences worth stating in a thesis table: the number is **not comparable** with
`FPR95` (different pair population — a run that switches starts a fresh series), and a
batch with no proxy<->image pair reports NaN and is skipped, exactly as a positive-free
batch already is. Both metrics can be reported side by side: list them as two entries and
give the one that should drive `best.pth` the weight.

## Loss curves

`train_losses.svg` / `validation_losses.svg` (under `<logging.dir>/<experiment_name>/`)
are titled `<model> <head> (scale=…)`. The validation axis defaults to y ∈ [1e-3, 1] —
the FPR metrics live in [0, 1] — and expands only when a series leaves it.

## Registering something new

`run_training.py` resolves everything by **name** against a package namespace, so a new
model / loss / dataset / process-batch function needs no central registry — only an
export from the relevant `__init__.py`:

| config key | resolved against |
| --- | --- |
| `model.name` | `aef.models` |
| `training.process_batch` | `aef.train` |
| `training.dataset.name`, `validation…name` | `aef.data` |
| `training.loss[].name`, `validation.loss[].name` | `aef.train.losses` |
