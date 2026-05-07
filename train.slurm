#!/bin/bash
#SBATCH --job-name=affine-equivariant-features
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH -c32
#SBATCH --container-image=/home/hsa/aef
#SBATCH --container-mounts=/home/hsa/workspace:/workspace,/raid/data/hsa:/data
#SBATCH --output=logs/%x-%j.out
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --time=12:00:00

source /etc/profile
source /root/.bashrc
source /root/.profile
source /.venv/bin/activate 
/.venv/bin/python /workspace/affine-equivariant-features/src/run_training.py --config-name scale_dgx
