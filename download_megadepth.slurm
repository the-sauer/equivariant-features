#!/bin/bash
#SBATCH --job-name=affine-equivariant-features
#SBATCH --mem=8GB
#SBATCH -c8
#SBATCH --container-image=/home/hsa/aef
#SBATCH --container-mounts=/raid/data/hsa:/data
#SBATCH --output=logs/%x-%j.out
#SBATCH --container-remap-root
#SBATCH --container-writable
#SBATCH --time=24:00:00

cd /data/datasets
curl -c MegaDepth_SfM_v1.tar.xz -O https://www.cs.cornell.edu/projects/megadepth/dataset/MegaDepth_SfM/MegaDepth_SfM_v1.tar.xz
# mkdir MegaDepth
tar -xf MegaDepth_SfM_v1.tar.xz -C MegaDepth
