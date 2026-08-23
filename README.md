# affine-equivariant-features

Affine Equivariant Features, the main implementation of my master thesis.

## Getting started

Initialise the `BlobBoards.jl` submodule
```sh
git submodule update --init --recursive
```
then build the environment. [pixi](https://pixi.sh) is the supported path — `pixi.toml`
at the repo root carries every Python dependency *and* the Julia bridge, and `pixi.lock`
pins the exact resolution:
```sh
pixi install
```
The Julia side instantiates itself on the first call through the Python interface. It
pulls its dependencies from **CauRegistry**, so register that once (a stale or missing
mirror surfaces later as `expected package ... to be registered`):
```sh
julia --project=./deps/BlobBoards.jl -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url="git@github.com:prittjam/CauRegistry.jl.git")); Pkg.instantiate()'
```

Without pixi, `requirements.txt` is the equivalent pip list (this is what the Slurm
container recipe uses). Python 3.13 — **not** 3.14, where hydra's `@hydra.main` fails to
build its argument parser:
```sh
pip install -r requirements.txt
```

## Training

Four Slurm scripts cover everything the repo trains — a synthetic pre-training pass and
a real-footage bootstrap round, each with a `_vanilla` baseline:

```sh
sbatch bootstrap_synthetic_network.sh      # synthetic boards, ProxyAnchoredSupCon + fft head
sbatch bootstrap_synthetic_vanilla.sh      # synthetic boards, SupCon + maxpool head
BB_ITERATION=4 sbatch bootstrap_real.sh          # real .tracks footage, round 4
BB_ITERATION=4 sbatch bootstrap_real_vanilla.sh
```

Submit them from the repo root — each is a one-line `pixi run` wrapper around the Hydra
entrypoint, so a run can also be reproduced directly (any config key is overridable on
the CLI):

```sh
pixi run python src/run_training.py --config-name bootstrap_synthetic scale=128
```

Two supporting entrypoints:

```sh
pixi run python src/prebuild_datasets.py --config-name bootstrap_synthetic  # warm the dataset cache, then exit
pixi run python src/to_onnx.py --run path/to/run_dir --summary --check      # (re-)export a checkpoint to ONNX
```

Runs with `logging.export_onnx: true` (all four configs) export `best.onnx` themselves
when the last epoch ends; `to_onnx.py` is for re-exports and for runs killed early.

## Documentation

- [Blob-board data pipeline](docs/data_pipeline.md) — views/patches, shape
  normalization, background compositing, garbage keypoints, the clean identity view,
  track viewing-angle balance, and the dataset cache.
- [Configs, training & bootstrapping](docs/configs_and_training.md) — the four entry
  points, the single `scale` hyperparameter, `shared_params` + Hydra struct mode,
  checkpoints, ONNX export, and the proxy-anchored losses.
- [Log-polar descriptor](docs/logpolar_descriptor.md) — the log-polar geometry, why a
  plain HardNet trunk is not rotation-invariant on it, and the circular-padding /
  antialiasing / FFT-head fixes, with ablations.

## Gotchas

If `juliacall` chooses an incompatible version for `BlobBoards.jl` (for example 1.11) set
```sh
export PYTHON_JULIACALL_EXE=/path/to/julia-1.12.6+0.x64.linux.gnu/bin/julia
export PYTHON_JULIACALL_PROJECT=/path/to/BlobBoards.jl
```

## License

This project is licensed under [AGPLv3](./LICENSE.md).
