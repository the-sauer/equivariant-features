#!/bin/bash
#SBATCH --job-name=blob_descriptor
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -c16
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=24:00:00

cd deps/BlobBoards.jl
pixi run python ../../src/run_training.py --config-name blob_descriptor_steerable
