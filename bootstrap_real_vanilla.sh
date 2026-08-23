#!/bin/bash
#SBATCH --job-name=bootstrap_real_vanillq
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH -c10
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=24:00:00

cd affine-equivariant-features/deps/BlobBoards.jl
pixi run python ../../src/run_training.py --config-name bootstrap_real_vanilla.yaml iteration=$BB_ITERATION
