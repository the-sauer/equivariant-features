# affine-equivariant-features

Affine Equivariant Features, the main implementation of my master thesis.

## Getting started

Make sure you have Python installed. Supported version are 3.12 and 3.13.

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

You can now run a training with
```sh
python src/run_training.py --config-name (scale|detector|descriptor)
```

For the blob-descriptor configs, one training run is
```sh
python src/run_training.py --config-name blob_descriptor_steerable scale=64
```
and a whole matrix of runs (one Slurm job per network × scale) is submitted with
```sh
./launch_training_matrix.sh -n my_sweep steerable efficient8 efficient4
```

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

## Gotchas

If `juliacall` chooses an incompatible version for `BlobBoards.jl` (for example 1.11) set
```sh
export PYTHON_JULIACALL_EXE=/path/to/julia-1.12.6+0.x64.linux.gnu/bin/julia
export PYTHON_JULIACALL_PROJECT=/path/to/BlobBoards.jl
```

## License

This project is licensed under [AGPLv3](./LICENSE.md).
