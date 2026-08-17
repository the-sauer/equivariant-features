# Configs, training & sweeps

## Config hierarchy

The blob-descriptor configs share one base and differ only in the model + patch-type:

```
config.yaml            # global defaults
└─ blob_descriptor_base.yaml     # everything shared: training/val dataset params,
                                 # compositing, augmentation, num_workers, the whole
                                 # `validation.shared_params` + `datasets` structure
   ├─ blob_descriptor_steerable.yaml   # model: BlobDescriptorNoStride  (cartesian)
   ├─ blob_descriptor_efficient.yaml   # model: BlobDescriptorEfficient (cartesian)
   └─ blob_descriptor_logpolar.yaml    # model: HardNet                 (log-polar)
```

Each leaf keeps only what differs: the model block, any train overrides
(`batch_size`, `num_workers`, optimizer), and the **patch-type-specific** fields
(`patch_type` + the scale hyperparameter). Everything else lives in
`blob_descriptor_base.yaml`. (`blob_descriptor_canonicalization.yaml` inherits from
`config`, not this base.)

## The single `scale` hyperparameter

Each leaf defines one top-level `scale` key threaded everywhere via `${scale}`:

- **steerable / efficient** (cartesian): `scale` = the patch scale factor; feeds
  `model.params.scale_factors`, `training…patch_scale_factors`, and every
  validation split's `patch_scale_factors` (so the model can never drift from the
  dataset).
- **logpolar**: `scale` = `logpolar_outer_factor`.

Override it in one place: `python src/run_training.py --config-name … scale=128`.
Note the YAML quoting `["${scale}"]` — an unquoted `${...}` inside a flow sequence is
invalid YAML.

## `shared_params` validation & Hydra struct mode

`validation.shared_params` is a single `{name, params}` template; each
`validation.datasets` entry supplies only its overrides (`scale_quantile_range`, loss
weight). `get_validation_specs` (`src/aef/data/__init__.py`) deep-merges them.

Hydra composes configs in **struct mode**, which forbids adding keys not in the
template. The merge therefore first resolves the template to a plain container
(`OmegaConf.to_container(shared, resolve=True)` — this both bakes in `${scale}` against
the real root **and** drops struct mode) and only then merges the split's extra keys.
When testing config-merge code, compose via `hydra.compose` (struct mode), **not**
`OmegaConf.load` (non-struct) — the latter hides this class of bug.

## Sub-epoch validation

`validation.validate_every_n_batches` (declared as `null` in both bases) runs the
**whole** validation suite every N training batches, on top of the end-of-epoch run.
Use it when one point per epoch is too coarse to see what a run is doing — a long
epoch over the track data, or a warm-started run that moves within the first few
hundred batches.

The extra points are plotted at the fractional epoch they were measured at, so the
validation figure's x axis becomes "fraction of training consumed": the run after
batch `k` of epoch `e` sits at `e + k/batches_per_epoch`, and the end-of-epoch run at
`e + 1`. With the option off the axis is unchanged — one point per epoch at integer
`e`, matching the training curve. That axis is stored alongside the curves as
`checkpoint["plots"]["x_val"]` (the training curve keeps its own `x`);
`plot_supcon_comparison.py` prefers `x_val` and falls back to `x` for older
checkpoints.

What it does *not* change: `best.pth` selection, which still compares whole epochs —
a mid-epoch run only records metrics, it never writes a checkpoint. Nor is it free;
each run is a full pass over every validation split, so N should be a decent fraction
of an epoch (tens to hundreds of batches), not single digits.

```sh
python src/run_training.py --config-name track_descriptor_logpolar \
  validation.validate_every_n_batches=200
```

## DataLoader workers

Both loaders honour `training.num_workers` / `validation.num_workers` (see
`prepare_training`). With `num_workers > 0` the data pipeline (index cached patches,
collate, H2D copy) overlaps with the GPU step instead of running synchronously in the
main process — the main lever when the GPU is starved by single-process loading; extra
`-c` cores do nothing until this is set. The train loader uses `persistent_workers`
(avoids re-forking the large in-memory dataset each epoch); validation uses fewer,
non-persistent workers (its small loaders run sequentially). Keep Slurm `-c` ≥
`num_workers + ~2` (main + Julia board generation). `steerable` is GPU-bound so uses
fewer workers; the lighter nets are more data-bound.

## Dataset cache

`cache_dir` (set in `blob_descriptor_base.yaml` for both the training dataset and
`validation.shared_params`) makes a run reuse a previously prepared dataset instead of
re-rendering boards / re-extracting patches — cold ~54 s vs warm ~0 s. The key covers
dataset params only, so all models at a given `scale` share one entry. Set it to `null`
to disable. See [data pipeline → dataset cache](data_pipeline.md#dataset-cache-disk)
for the key, concurrency and the "frozen random draw" caveat.

## Checkpoints

Controlled by the `logging` block (`config.yaml`):

- `model_checkpoints` — master switch (currently `false`).
- `checkpoint_dir` — `null` ⇒ `<logging.dir>/<experiment_name>/checkpoints`.
- `checkpoint_every_epoch` — `false` (default) ⇒ only `latest.pth` (refreshed every
  epoch, the resume point) and `best.pth` (lowest weighted validation loss, i.e. the
  `overall` FPR95) are written. `true` ⇒ additionally keep `epoch_<n>.pth` snapshots.

Note escnn checkpoints are large (~240 MB — the basis buffers live in the
`state_dict`), so per-epoch snapshots add up fast. Resume with
`+training.continue_from_checkpoint=<path>`.

## Contrastive losses: `SupCon` vs `ProxyAnchoredSupCon`

`training.loss: [{name: SupCon}]` is the default — `pytorch_metric_learning`'s SupCon over
every patch in the batch. `ProxyAnchoredSupCon` treats the board's own rendering as the
**proxy anchor** of its blob (vendored implementation in `train/losses/SupConLoss.py`,
wrapper in `contrastive.py`):

- the outer sum runs over the `is_pdf` patches only — the rendering (the GT sequence in a
  `.tracks` file, the identity view in `HomographyData`);
- `A(i)` and `P(i)` hold image patches only, so `pdf<->pdf` and `image<->image` terms
  disappear from the numerator **and** from the log-sum-exp denominator.

The motive is that matching happens against the board's rendering, not between two
observations, so the image<->image terms optimise something the descriptor is never
deployed on — and on track data they are also where the label noise sits. See
`docs/figures/proxy_anchored_supcon.tex` for the pair matrix.

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
rather than silently degenerating to plain SupCon if it is missing. Unlike the mask, the
flag is *not* withheld during validation — otherwise the validation number would be a
different loss from the training one.

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
are titled `<model> (scale=…)` and fixed to y ∈ [0, 6] (training) and y ∈ [0, 1]
(validation) for comparability across runs.

## Matrix training (`launch_training_matrix.sh`)

Submits one Slurm job per `(network, scale)`. Edit the `CONFIG` / `SCALES` /
`NET_EXTRA` / `NET_MEM` / `NET_CPUS` maps at the top.

```sh
./launch_training_matrix.sh                          # full matrix
./launch_training_matrix.sh steerable efficient8 efficient4   # subset (e.g. model comparison)
./launch_training_matrix.sh -n my_sweep …            # group all runs under
                                                     #   <logdir>/YYYY_MM_DD_my_sweep/
DRY_RUN=1 ./launch_training_matrix.sh …              # print jobs, submit nothing
```

- `-n NAME` prepends a dated subfolder (`date +%Y_%m_%d`) applied to both the Slurm
  output logs and the training `experiment_name` (checkpoints/plots).
- `NET_EXTRA[net]` injects arbitrary hydra overrides per network — e.g.
  `efficient8`/`efficient4` both use `blob_descriptor_efficient` and differ only by
  `model.params.n_rotations=8|4`.
- Each job encodes `<net>_s<scale>` in its job name and `experiment_name`, so
  `squeue`, logs, and output dirs stay distinct.

### Dataset prebuild

The launcher first submits a **prebuild job array** and gives every training job
`--dependency=afterok:<array id>`, so they all start from a warm dataset cache. Without
it, jobs launched together would each cold-build the same boards: the cache only turns
warm *after* a build finishes, so concurrent cold jobs duplicate the whole
render + composite + SIFT + extract cost.

- **One array task per distinct dataset** (`src/prebuild_datasets.py` with
  `+prebuild_target=train|<split>`), so they build in parallel rather than in sequence.
- **Dataset groups** (`NET_DATASET_GROUP` → `GROUP_CONFIG`) collapse networks that share
  a dataset block: the cache key covers dataset params but *not* the model, so
  `steerable`/`efficient8`/`efficient4` are one `cartesian` group and their datasets are
  built once. A model-comparison sweep therefore prebuilds `4 scales × 5 datasets = 20`
  tasks rather than 12 jobs × 5 redundant builds. If a group mapping is wrong the only
  cost is a cache miss (the training job rebuilds it) — never a wrong dataset.
- Size it with `PREBUILD_MEM` / `PREBUILD_CPUS` / `PREBUILD_TIME`; the array is
  unthrottled, so Slurm runs as many tasks as the cluster allows. It needs a GPU
  (SIFT + patch extraction run on CUDA). `NO_PREBUILD=1` skips the whole thing.
- `VAL_SPLITS` in the launcher must match the split names in `validation.datasets`.
