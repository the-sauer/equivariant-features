# affine-equivariant-features

Affine Equivariant Features, the main implementation of my master thesis.

## Getting started

Make sure you have Python installed. Supported versions are 3.12 and 3.13.

Initialise the `BlobBoards.jl` submodule
```sh
git submodule update --init --recursive
```
instantiate `BlobBoards.jl` (optional, will be handled with the first call to the `BlobBoards.jl` through the python interface)
```sh
julia --project=./deps/BlobBoards.jl -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url="git@github.com:prittjam/CauRegistry.jl.git")); Pkg.instantiate()'
```
and install the python dependencies.
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

Each is a one-line wrapper around the Hydra entrypoint, so a run can be reproduced
directly (any config key is overridable on the CLI):

```sh
python src/run_training.py --config-name bootstrap_synthetic scale=128
```

Two supporting entrypoints:

```sh
python src/prebuild_datasets.py --config-name bootstrap_synthetic   # warm the dataset cache, then exit
python src/to_onnx.py --run path/to/run_dir --summary --check       # (re-)export a checkpoint to ONNX
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
