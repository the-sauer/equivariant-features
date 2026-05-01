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
```
pip install -r requirements.txt
```

You can now run a training with
```sh
python src/run_training.py --config-name (scale|detector|descriptor)
```

## Gotchas

If `juliacall` chooses an incompatible version for `BlobBoards.jl` (for example 1.11) set
```sh
export PYTHON_JULIACALL_EXE=/path/to/julia-1.12.6+0.x64.linux.gnu/bin/julia
export PYTHON_JULIACALL_PROJECT=/path/to/BlobBoards.jl
```

## License

This project is licensed under [AGPLv3](./LICENSE.md).
