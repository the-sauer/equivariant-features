# AffEquivarFeatures.jl

Affine Equivariant Features, the main implementation of my master thesis.

## Getting started

Initialise the BlobBoards.jl submodule,
```sh
git submodule update --init --recursive

```
instantiate the BlobBoards, and
```sh
julia --project=./deps/BlobBoards.jl -e 'using Pkg; Pkg.Registry.add(RegistrySpec(url="git@github.com:prittjam/CauRegistry.jl.git")); Pkg.instantiate()'
```
install the python dependencies.
```
pip install -r requirements.txt
```

## License

This project is licensed under [AGPLv3](./LICENSE.md).
