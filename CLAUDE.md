# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Affine Equivariant Features — the reference implementation of Hendrik Sauer's master thesis on affine/scale-equivariant local feature detection and description. Licensed AGPLv3.

## Environments

Two Python environments coexist; pick by task:

- **Running code / training / notebooks → the pixi env in `deps/BlobBoards.jl`.** It bundles `juliacall` + the `BlobBoards.jl` Julia package + all Python deps and pins `JULIA_PYTHONCALL_EXE` so PythonCall reuses pixi's interpreter. Run either `cd deps/BlobBoards.jl && pixi run python ...` or `pixi run --manifest-path deps/BlobBoards.jl/pixi.toml python ...` from the repo root. Anything that touches `BlobBoardHomographyData` (the Julia bridge) must run here.
- **Unit tests → `.venv13`** (`.venv13/bin/python -m pytest`). `tests/conftest.py` stubs out the heavy deps (kornia, torchvision, Julia, escnn/asel, pytorch-metric-learning), so tests need only torch + pytest. The pixi env does **not** have pytest.

`requirements.txt` is the pip-installable dependency list (used by the Slurm container recipe); pixi's `pixi.toml` is the authoritative local env.

## Common commands

```sh
# Training (run inside the pixi env). --config-name picks a file from src/conf/
pixi run --manifest-path deps/BlobBoards.jl/pixi.toml \
  python src/run_training.py --config-name blob_descriptor_steerable
# Hydra lets you override any config key on the CLI, e.g. training.batch_size=512

# Unit tests
.venv13/bin/python -m pytest tests/unit -q
.venv13/bin/python -m pytest tests/unit/test_utils.py::test_name   # single test

# Lint (config in pylintrc, max-line-length=120)
pylint src/aef
```

`tests/unit/` is the whole suite (the old `tests/integration/` + smoke harnesses drove the since-removed detector/scale tasks and were deleted).

First-time setup also requires the Julia submodule and registry (see README.md): `git submodule update --init --recursive`, then the pixi env handles Julia instantiation on first use.

## Architecture

### Config-driven, string-dispatched wiring

`src/run_training.py` (Hydra entrypoint, configs in `src/conf/`; there is **no** default config — `--config-name` is required) resolves everything by **name** from YAML:

- `model.name` → `eval()`'d against the `aef.models` namespace, constructed with `model.params`.
- `training.process_batch` → `eval()`'d against `aef.train` (a `process_batch_*` function).
- `training.dataset.name` / `validation.dataset.name` → `eval()`'d against `aef.data`, constructed with `**dataset.params` (the params dict is splatted straight into the dataset constructor — new dataset kwargs need no dataclass change).
- `training.loss` / `validation.loss` → list of `{name, params, weight, report}`, each `getattr`'d from `aef.train.losses`.

Consequence: **to register a new model / loss / dataset / process-batch function, export it from the relevant package `__init__.py`** (most re-export via `from .module import *`); no central registry to edit. `src/aef/configuration.py` holds the dataclass schema but is loosely enforced — patch/scale kwargs ride through `params` dicts.

`juliacall` **must be imported before torch** — `run_training.py` imports it first on line 1 for this reason. Preserve that ordering in any new entrypoint that touches BlobBoards.

### Generic training loop

`aef.train.train_func(process_batch)` (in `src/aef/train/__init__.py`) returns the one training loop used by every task. It builds optimizer(s)/scheduler(s) (supports multiple named optimizers over distinct `model_params` sub-modules), DataLoaders using each dataset's own `get_collate_func()`, and a weighted multi-loss criterion. The loop has built-in NaN/Inf guarding (skips bad batches, clips grad norm to 5.0) and checkpointing to `logging.dir/<experiment>/checkpoints/` — gated by `logging.model_checkpoints` (currently **false**), writing `latest.pth` + `best.pth`, and `epoch_<n>.pth` only if `logging.checkpoint_every_epoch`. The task-specific logic lives entirely in the chosen `process_batch_*` function; only two remain: `process_batch_blobs` (`train/descriptor.py`) and `process_batch_canonicalize` (`train/canonicalizer.py`). Shared geometry helpers live in `src/aef/geometry.py`.

### Data pipeline

`aef.data.homography.HomographyData` is the base dataset: it samples random homographies per image, detects SIFT keypoints (kornia `ScaleSpaceDetector`), and — when `in_memory=True` — pre-extracts and caches per-keypoint patches so training avoids re-warping. Each item yields keypoint coords, scale, homography, and (if precomputed) its patch; `get_collate_func()` either stacks cached patches or falls back to on-the-fly warping.

Patch extraction happens in module-level functions in `homography.py`:
- `blob_normalizations(...)` — SVD-whitens the local homography Jacobian so an elliptical blob maps to an isotropic one (shared shape normalization).
- `extract_multiscale_patches(...)` — **cartesian** patches, one channel per entry in `patch_scale_factors`, via `warp_perspective`.
- `extract_logpolar_patches(...)` — **log-polar** patches (single channel; angular axis = dim −2, radial axis = dim −1) via `grid_sample`; inner/outer radius = `logpolar_{inner,outer}_factor * scale`.

`HomographyData` selects between them with `patch_type="cartesian"|"logpolar"`. `BlobBoardHomographyData` (synthetic calibration boards via the `BlobBoards.jl` Julia bridge — needs the pixi env) is the only `HomographyData` subclass; it hands `HomographyData` an in-memory tensor of board rasters. `data/track.py`'s `BlobTrackData` (real-image track patches from a `.tracks` HDF5 file, needs only h5py — no Julia) is wired up too: its `get_collate_func()` emits the same batch keys as `HomographyData`, so it trains through the same loop + `process_batch_blobs` (canonicalized patches ⇒ a trivially in-bounds frame; track id ⇒ contrastive index). `view_angle_range: [lo_deg, hi_deg)` filters observations by the per-frame viewing obliquity the `.tracks` file records (`view_angles`/`homography_frame_ids`, joined via each observation's `frame_id`; anchors have no pose and are kept so bands keep their positives) — `conf/track_angle_validation.yaml` uses it to build one validation set per 10° band, and `track_descriptor_logpolar_angles.yaml` is the composed leaf. Track training uses the `track_descriptor_*` configs; `launch_track_matrix.sh` sweeps `(network, scale)` and **warm-starts** each run from a synthetic-descriptor checkpoint found under its `--from-dir` (via `training.init_from_checkpoint`, which loads model weights only and starts fresh — distinct from `training.continue_from_checkpoint`, which resumes a run in place).

`in_memory` is task-specific, not a perf dial: `true` (blob descriptor) pre-extracts patches and batches carry `"patches"`; `false` (canonicalization) batches carry `"images"` and the collate warps per view. Setting `cache_dir` reuses a fully prepared dataset from disk, keyed by a hash of the *effective* constructor params. See `docs/data_pipeline.md`.

`blob_normalizations` carries **shape only** (`det == 1`); the blob's size is `scales`' job alone, and every Jacobian is evaluated at the *source* point (`linearize_homography` differentiates at a point of its map's domain). Both invariants are pinned by `tests/unit/test_blob_normalizations.py` — patches that violate them still look plausible in isolation but stop matching their own anchor. `docs/scale_budget_and_jitter.md` covers the blob-scale budget, the measured detector error, and the `keypoint_jitter`/`scale_jitter` settings; **patch measurements taken before those fixes (CACHE_VERSION < 4) are void**. Note `resolution` selects *which board you get* (the packer snaps to the pixel grid), not just how finely it is sampled.

### Models

`src/aef/models/` holds the equivariant architectures: `blob_descriptor.py` (`BlobDescriptorNoStride`, `BlobDescriptorEfficient` + older `HardNet`/`Deep`/`Robust`/`Hierarchical` variants), `hardnet.py` (`HardNet` and `HardNetLogPolar` — the log-polar-aware variant), `blob_canon.py`. `asel/` is a **vendored** equivariant-conv library (affine steerable convs), integrated into the tree rather than pip-installed; it is a dependency of `BlobCanonicalization`. See `docs/steerable_descriptors.md` (cartesian/steerable) and `docs/logpolar_descriptor.md` (log-polar; note plain `HardNet` is **not** rotation-invariant on log-polar patches). Learned board-validity masking (`model.params.learned_mask`) is shared by both families — `HardNetLogPolar` and the two steerable descriptors — with the same `(descriptor, m_pred)` contract; `docs/steerable_masking.md` has the derivation and the config/launcher surface. Only a model advertising `learned_mask` is called with the mask kwargs (`process_batch_blobs`); `training.ignore_mask=true` disables the path entirely.

## Notes

- Some source comments are in German; matching that is not required for new code.
- `.env` at the repo root contains real API tokens (Kaggle, W&B) and is loaded via `dotenv` — do not echo, commit, or leak its contents.
