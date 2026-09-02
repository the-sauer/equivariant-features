# affine-equivariant-features

Affine Equivariant Features, the main implementation of my master thesis.

## Getting started

The environment is managed by [pixi](https://pixi.sh); `pixi.toml` at the repository
root is authoritative. Initialise the `BlobBoards.jl` submodule first — the environment
installs its Python bridge from there as an editable package, so `pixi install` fails
without it.

```sh
git submodule update --init --recursive
pixi install
```

The environment resolves to Python 3.12, and cannot go higher: `lie-learn` (a dependency
of `escnn`) publishes wheels only up to `cp312` and has no sdist. `pixi.lock` holds it
there; a from-scratch re-solve that picks 3.13+ will fail on that package.

`requirements.txt` mirrors the same dependency set as a plain pip list and is what the
Slurm container recipe uses; keep the two in sync.

Julia needs no separate setup. The first `import juliacall` downloads a private Julia
into `.pixi/envs/default/julia_env/`, and the first call into `blobboards` registers
`CauRegistry` and instantiates `deps/BlobBoards.jl` on its own — it just needs GitHub
access and pays the download/precompile cost once.

Board *authoring* needs one env more. Since BlobBoards 0.22 the real `blob_board` method
lives in `BlobBoardsAuthoringExt`, which triggers on CairoMakie + Makie; both are
BlobBoards `[weakdeps]`, and Julia will not load a package's own weakdeps from that
package's project env. `deps/julia/Project.toml` is that env — it dev's the checkout and
lists CairoMakie as an ordinary dep. `aef.data.blobboards._ensure_authoring()`
instantiates and loads it before the first board is rendered, so the only visible cost is
a one-off Makie precompile (a few minutes) on the first synthetic-board run.

Optionally, drop a `.env` at the repository root with your Kaggle credentials
(`KAGGLE_USERNAME`, `KAGGLE_KEY`). It is loaded via `dotenv` at startup and is only
needed so the synthetic-board configs can fetch their background images; a
`./backgrounds` directory that already exists is used instead.

## Training

Every run goes through the one Hydra entrypoint. There is **no** default config, so
`--config-name` is required — it names a file in `src/conf/`, without the `.yaml`:

```sh
pixi run python src/run_training.py --config-name blob_descriptor_logpolar
```

Run it from the repository root, and always through `pixi run`: `run_training.py`
imports `juliacall` before torch, and the board generator needs the environment's own
Julia.

There are two families of leaf configs:

- **`blob_descriptor_*`** train on synthetic calibration boards rendered by
  `BlobBoards.jl`. They need no data on disk — boards are generated, backgrounds are
  downloaded — but they do need a GPU, and the first run spends a while rendering boards
  and extracting patches into `./homography_cache`.
- **`track_descriptor_*`** train on real-image patches from a `.tracks` HDF5 file
  prepared beforehand by BlobBoards. The file has no default, so pass it:

  ```sh
  pixi run python src/run_training.py --config-name track_descriptor_logpolar \
      track_path=/path/to/my.tracks
  ```

Hydra lets you override any config key from the command line, including the single
`scale` hyperparameter the descriptor configs are built around:

```sh
pixi run python src/run_training.py --config-name blob_descriptor_efficient \
    scale=64 training.batch_size=512
```

Checkpoints and logs land under `<logging.dir>/<experiment>/`; with
`logging.export_onnx` on, the final epoch also writes `best.onnx` next to `best.pth`.

When several runs share a dataset, build the cache once up front instead of letting each
job cold-build the same boards:

```sh
pixi run python src/prebuild_datasets.py --config-name blob_descriptor_efficient
```

A whole matrix of runs (one Slurm job per network × scale, prebuild job first) is
submitted with

```sh
./launch_training_matrix.sh -n my_sweep efficient8 efficient4
./launch_track_matrix.sh --track /path/to/my.tracks -n track_ft
```

Both take `DRY_RUN=1` to print the jobs instead of submitting them.

## Documentation

- [Blob-board data pipeline](docs/data_pipeline.md) — views/patches, background
  compositing, garbage keypoints, the clean identity view, scale bands and equal-sized
  validation splits.
- [Configs, training & sweeps](docs/configs_and_training.md) — the base/leaf config
  hierarchy, the single `scale` hyperparameter, `shared_params` + Hydra struct mode,
  DataLoader workers, loss curves, and the matrix launcher.
- [Steerable blob descriptors](docs/steerable_descriptors.md) — the `escnn` model
  variants, why `NoStride` is heavy, and `BlobDescriptorEfficient` (design, benchmarks,
  the equivariance trade-off).
- [Steerable board-validity masking](docs/steerable_masking.md) — `learned_mask` on the
  `escnn` descriptors: why weighting by a scalar field stays equivariant, the
  given-on-`is_pdf` / predicted-on-target split, and how the predictor is supervised.
- [Log-polar descriptor](docs/logpolar_descriptor.md) — the log-polar geometry, why
  `HardNet` is not rotation-invariant on it, and `HardNetLogPolar` (circular angular
  padding + antialiased stride, with an ablation).
- [The mask-ceiling experiment](docs/mask_ceiling_experiment.md) — what board-validity
  masking is worth, measured: the four-arm `oracle_mask` sweep, the full ablation matrix,
  and why training on ground-truth masks produces a model that is not a descriptor.

## Gotchas

If `juliacall` chooses an incompatible version for `BlobBoards.jl` (for example 1.11) set
```sh
export PYTHON_JULIACALL_EXE=/path/to/julia-1.12.6+0.x64.linux.gnu/bin/julia
export PYTHON_JULIACALL_PROJECT=/path/to/BlobBoards.jl
```

## License

This project is licensed under [AGPLv3](./LICENSE.md).
