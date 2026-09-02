#!/bin/bash
#SBATCH --job-name=blobboards-train
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -c10
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=24:00:00

cd deps/BlobBoards.jl
pixi run python ../../src/run_training.py --config-name $AEF_CONFIG_NAME $AEF_OVERRIDES