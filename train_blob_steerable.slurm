#!/bin/bash
#SBATCH --job-name=blob_descriptor
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH -c16
#SBATCH --container-image=/home/hsa/aef
#SBATCH --container-mounts=/home/hsa/workspace:/workspace,/raid/data/hsa:/data
#SBATCH --output=logs/%x-%j.out
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --time=24:00:00

source /etc/profile
source /root/.bashrc
source /root/.profile
source /.venv/bin/activate 
cd /workspace/affine-equivariant-features
apt install -y gfortran
pip install -r requirements.txt
/.venv/bin/python /workspace/affine-equivariant-features/src/run_training.py --config-name blob_descriptor_steerable_base
