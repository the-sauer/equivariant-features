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
