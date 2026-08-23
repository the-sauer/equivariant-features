#!/bin/bash
#SBATCH --job-name=bootstrap_synthetic_vanilla
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH -c10
#SBATCH --output=logs/%x-%j.out
#SBATCH --time=24:00:00

# Submit from the repo root: the pixi env lives there now (it carries the Julia
# bridge too — `blobboards` activates deps/BlobBoards.jl itself). Slurm starts the
# job in the directory sbatch was invoked from; the fallback is for a direct run.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
pixi run python src/run_training.py --config-name bootstrap_synthetic_vanilla
