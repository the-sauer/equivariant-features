#!/bin/bash
#SBATCH --job-name=bootstrap_real_network
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH -c10
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=24:00:00

cd deps/BlobBoards.jl
pixi run python ../../src/run_training.py --config-name bootstrap_real.yaml iteration=$BB_ITERATION
