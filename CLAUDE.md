# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Affine Equivariant Features — the reference implementation of Hendrik Sauer's master thesis on affine/scale-equivariant local feature detection and description. Licensed AGPLv3.

## Environments

Two Python environments coexist; pick by task:

- **Running code / training / notebooks → the pixi env at the repo root** (`pixi.toml` + `pixi.lock`): `pixi run python src/run_training.py ...`. It carries every Python dep this repo imports *and* the Julia bridge — `juliacall` plus the editable `blobboards` package from `deps/BlobBoards.jl/python`, which activates the `BlobBoards.jl` Julia project itself (two directories up from its own `__init__.py`) and instantiates it on first use. `JULIA_PYTHONCALL_EXE` is pinned so PythonCall reuses pixi's interpreter and `JULIA_CONDAPKG_BACKEND=Null` stops CondaPkg spinning up a second one. Anything touching `BlobBoardHomographyData` must run here.
  - **Python is pinned `>=3.13,<3.14` on purpose.** hydra-core 1.3.5 (its latest release) passes a non-`str` object as `help=`, which 3.14's argparse rejects at `add_argument` time (`ValueError: badly formed help string`) — so `@hydra.main` dies before either entrypoint runs. Raise the ceiling once hydra ships a 3.14-compatible release.
  - The submodule has a `pixi.toml` of its own; that one is BlobBoards.jl's env, not this repo's. Don't add this repo's deps to it.
- **Unit tests → `.venv13`** (`.venv13/bin/python -m pytest`). `tests/conftest.py` stubs out the heavy deps (kornia, torchvision, Julia, pytorch-metric-learning), so tests need only torch + pytest (+ hydra for the config-compose tests). The pixi env does **not** have pytest.

`requirements.txt` is the pip-installable dependency list (used by the Slurm container recipe); `pixi.toml` is the authoritative local env. Keep them in step — `pint` reaches the code through `data/blobboards.py` even though the `blobboards` package does not declare it, and `onnx`/`onnxscript` are needed by the automatic export that ends every run.

## Entry points

Everything the repo trains is submitted by one of **four Slurm scripts**, each naming a config in `src/conf/`:

| script | config | data |
| --- | --- | --- |
| `bootstrap_synthetic_network.sh` | `bootstrap_synthetic.yaml` | synthetic boards, `ProxyAnchoredSupCon` + `fft` head |
| `bootstrap_synthetic_vanilla.sh` | `bootstrap_synthetic_vanilla.yaml` | synthetic boards, `SupCon` + `maxpool` head |
| `bootstrap_real.sh` | `bootstrap_real.yaml` | real `.tracks` footage (`BB_ITERATION=<n>`) |
| `bootstrap_real_vanilla.sh` | `bootstrap_real_vanilla.yaml` | same, vanilla baseline |

The `_vanilla` pair is the baseline the proxy-anchored/FFT pair is measured against. The
two real configs compose `track_descriptor_base.yaml`; the two synthetic ones are
self-contained. **These four plus `src/prebuild_datasets.py` and `src/to_onnx.py` define
what the repo keeps** — the model zoo, the ablation configs, the sweep launchers and the
experiment write-ups that used to sit alongside them were deleted deliberately, so
resurrect from git history rather than assuming something is missing.

## Common commands

```sh
# Training (run inside the pixi env). --config-name picks a file from src/conf/
pixi run python src/run_training.py --config-name bootstrap_synthetic
# Hydra lets you override any config key on the CLI, e.g. training.batch_size=512

# Warm the dataset cache without training (one job, so parallel runs don't each cold-build)
pixi run python src/prebuild_datasets.py --config-name bootstrap_synthetic +prebuild_target=train

# Unit tests
.venv13/bin/python -m pytest tests/unit -q
.venv13/bin/python -m pytest tests/unit/test_utils.py::test_name   # single test

# Lint (config in pylintrc, max-line-length=120)
pylint src/aef

# ONNX export (pixi env — .venv13 has no onnx). A checkpoint stores weights only, so
# the architecture kwargs must be restated on the CLI (`-p KEY=VALUE` for anything
# without a dedicated flag). `--summary` flags ops onnxruntime keeps on the CPU EP.
pixi run python src/to_onnx.py HardNetLogPolar path/to/best.pth \
  --resolution 64 --head fft --n-harmonics 5 --summary
# ...or read the architecture out of the run's own cfg.yaml:
pixi run python src/to_onnx.py --run path/to/run_dir --summary --check
```

All four configs set `logging.export_onnx: true`, so a run writes `best.onnx` next to
`best.pth` when its last epoch ends. Both paths go through `aef.export`, so they produce
the same graph; the CLI is for re-exports and for runs killed before their final epoch.
See `docs/configs_and_training.md#automatic-onnx-export`.

First-time setup also requires the Julia submodule and registry (see README.md): `git submodule update --init --recursive`, then the pixi env handles Julia instantiation on first use. The Julia deps come from **CauRegistry**; a stale local mirror shows up as `expected package \`X\` to be registered` on the first `blob_board(...)` call, which is a registry refresh, not a Python-env problem.

## Architecture

### Config-driven, string-dispatched wiring

`src/run_training.py` (Hydra entrypoint, configs in `src/conf/`; there is **no** default config — `--config-name` is required) resolves everything by **name** from YAML:

- `model.name` → `eval()`'d against the `aef.models` namespace, constructed with `model.params`.
- `training.process_batch` → `eval()`'d against `aef.train` (`process_batch_blobs` is the only one).
- `training.dataset.name` / `validation.dataset.name` → `eval()`'d against `aef.data`, constructed with `**dataset.params` (the params dict is splatted straight into the dataset constructor — new dataset kwargs need no dataclass change).
- `training.loss` / `validation.loss` → list of `{name, params, weight, report}`, each `getattr`'d from `aef.train.losses`.

Consequence: **to register a new model / loss / dataset / process-batch function, export it from the relevant package `__init__.py`**; no central registry to edit. `src/aef/configuration.py` holds the dataclass schema but is loosely enforced (it is a type annotation, not a registered Hydra schema) — patch/scale kwargs ride through `params` dicts.

`juliacall` **must be imported before torch** — `run_training.py` imports it first on line 1 for this reason. Preserve that ordering in any new entrypoint that touches BlobBoards.

### Generic training loop

`aef.train.train_func(process_batch)` (in `src/aef/train/__init__.py`) returns the one training loop. It builds optimizer(s)/scheduler(s) (supports multiple named optimizers over distinct `model_params` sub-modules), DataLoaders using each dataset's own `get_collate_func()`, and a weighted multi-loss criterion. The loop has built-in NaN/Inf guarding (skips bad batches, clips grad norm to 5.0) and checkpointing to `logging.dir/<experiment>/checkpoints/` — gated by `logging.model_checkpoints` (**true** in all four configs), writing `latest.pth` + `best.pth`, and `epoch_<n>.pth` only if `logging.checkpoint_every_epoch`. The task-specific logic lives entirely in `process_batch_blobs` (`train/descriptor.py`). Shared geometry helpers live in `src/aef/geometry.py`.

The per-patch flag `is_pdf` (batch key `"is_pdf"`) marks the board's own rendering — the GT sequence of a `.tracks` file, the identity view in `HomographyData` — as opposed to the warped/tracked image patches. It drives the proxy-anchored objectives: `ProxyAnchoredSupCon` (the rendering is the blob's **proxy anchor**: outer sum over PDF patches, `A(i)`/`P(i)` over image patches only, in the shape of Proxy-Anchor Loss but with an embedded sample instead of a learned proxy) and its metric counterpart `ProxyAnchoredFPR95` (FPR95 over pdf<->image pairs only, not comparable with plain `FPR95`); see `docs/configs_and_training.md#contrastive-losses-supcon-vs-proxyanchoredsupcon`. It used to be called `is_anchor`, which collided with SupCon's own term for the rows of its logit matrix; **the on-disk `.tracks` attribute is still `is_anchor`** and is renamed on read in `_load_all_sequences`.

### Data pipeline

`aef.data.homography.HomographyData` is the base dataset: it samples random homographies per image, detects SIFT keypoints (kornia `ScaleSpaceDetector`), and — when `in_memory=True` — pre-extracts and caches per-keypoint patches so training avoids re-warping. Each item yields keypoint coords, scale, homography, and (if precomputed) its patch; `get_collate_func()` either stacks cached patches or falls back to on-the-fly warping.

Patch extraction happens in module-level functions in `homography.py`:
- `blob_normalizations(...)` — SVD-whitens the local homography Jacobian so an elliptical blob maps to an isotropic one.
- `extract_logpolar_patches(...)` — log-polar patches (single channel; angular axis = dim −2, radial axis = dim −1) via `grid_sample`; inner/outer radius = `logpolar_{inner,outer}_factor * scale`. This is the only patch type; the cartesian extractor was removed with the cartesian models.

`BlobBoardHomographyData` (synthetic calibration boards via the `BlobBoards.jl` Julia bridge — needs the pixi env) is the only `HomographyData` subclass; it hands `HomographyData` an in-memory tensor of board rasters. `data/track.py`'s `BlobTrackData` (real-image track patches from a `.tracks` HDF5 file, needs only h5py — no Julia) emits the same batch keys, so it trains through the same loop + `process_batch_blobs` (canonicalized patches ⇒ a trivially in-bounds frame; track id ⇒ contrastive index).

`balance_view_angles` (+ `view_angle_band_edges`/`view_angle_band_target`/`view_angle_band_bias`, on by default in `track_descriptor_base.yaml`): real footage is heavily fronto-parallel-biased, so `get_sampler` switches from packing whole track groups to `_pack_band_balanced`, which gives every populated band the same number of in-band patches in **every batch** (SupCon only ever sees one batch, so per-epoch quotas would not help). Positives become disjoint pairs — an in-band observation plus its GT (PDF) patch, or two in-band observations once that one is spent — never a repeated index, since a duplicate inside a batch is a distance-0 fake positive. A band that runs out of disjoint pairs under-fills rather than duplicating; `_band_capacity` predicts that ceiling and the sampler prints it per band on startup.

`training.init_from_checkpoint` loads model weights only and starts fresh, distinct from `training.continue_from_checkpoint`, which resumes a run in place.

`in_memory` is task-specific, not a perf dial: `true` (all four configs) pre-extracts patches and batches carry `"patches"`; `false` batches carry `"images"` and the collate warps per view. Setting `cache_dir` reuses a fully prepared dataset from disk, keyed by a hash of the *effective* constructor params. **`_CACHE_KEY_REMOVED` in `homography.py` pins params that no longer exist in the code at the values the shipped configs ran with**, so already-prepared datasets stay addressable after the cleanup; dropping an entry (and bumping `CACHE_VERSION`) is how you deliberately invalidate them. See `docs/data_pipeline.md`.

`blob_normalizations` carries **shape only** (`det == 1`); the blob's size is `scales`' job alone, and every Jacobian is evaluated at the *source* point (`linearize_homography` differentiates at a point of its map's domain). Both invariants are pinned by `tests/unit/test_blob_normalizations.py` — patches that violate them still look plausible in isolation but stop matching their own reference patch. Note `resolution` selects *which board you get* (the packer snaps to the pixel grid), not just how finely it is sampled. **Patch measurements taken before those fixes (CACHE_VERSION < 4) are void.**

### Model

`src/aef/models/hardnet.py` holds the single model, `HardNetLogPolar`: a HardNet-style trunk with angular-wrapping padding (`LogPolarPad`) and antialiased downsampling (`LogPolarBlurPool`), plus a choice of angular reduction head — `maxpool` (one peak per channel/radius) or `fft` (`AngularRFFTMag`, the lowest `n_harmonics` DFT magnitudes). `maxpool` keeps the historical flat `features` module; `fft` splits `trunk`/`head`, which is how its checkpoints were written — so the two layouts are load-compatible with the runs that produced them but not with each other. See `docs/logpolar_descriptor.md`.

The trunk and head deliberately avoid ONNX ops onnxruntime only runs on the CPU EP: the angular wrap is slice+concat rather than `F.pad(mode="circular")` (`Pad(mode="wrap")`), and the DFT is a matmul rather than `torch.fft.rfft` (`DFT`). `tests/unit/test_onnx_friendly_ops.py` pins both against the torch ops they replaced; `to_onnx.py --summary` reports any that creep back in.

## Notes

- Some source comments are in German; matching that is not required for new code.
- `.env` at the repo root contains real API tokens (Kaggle, W&B) and is loaded via `dotenv` — do not echo, commit, or leak its contents.
