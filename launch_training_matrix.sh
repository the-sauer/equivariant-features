#!/bin/bash
# Launch a matrix of blob-descriptor trainings: for every network type, sweep the
# key `scale` hyperparameter and submit one Slurm job per (network, scale) pair.
#
# The `scale` override maps to the network's most important knob:
#   steerable -> patch_scale_factor     logpolar -> logpolar_outer_factor
# (see the leaf configs; a single `scale=<x>` threads it everywhere).
#
# Usage:
#   ./launch_training_matrix.sh                 # submit the full matrix
#   ./launch_training_matrix.sh steerable       # only the given network(s)
#   ./launch_training_matrix.sh -n my_sweep     # group all runs under
#                                               #   <logdir>/YYYY_MM_DD_my_sweep/
#   DRY_RUN=1 ./launch_training_matrix.sh       # print the jobs without submitting
#
# Run it from the repository root.
set -euo pipefail

# ---- Sweep definition (edit these) ------------------------------------------
# network name -> hydra config-name
declare -A CONFIG=(
  [steerable]=blob_descriptor_steerable
  [logpolar]=blob_descriptor_logpolar
  [efficient8]=blob_descriptor_efficient
  [efficient4]=blob_descriptor_efficient
)
# per-network scale values
declare -A SCALES=(
  [steerable]="32 64 96 128"
  [logpolar]="32 64 96 128"
  [efficient8]="32 64 96 128"
  [efficient4]="32 64 96 128"
)
# extra hydra overrides per network (e.g. C8 vs C4 for the efficient descriptor).
# NoStride-vs-Efficient FPR95 comparison: run `steerable efficient8 efficient4`.
declare -A NET_EXTRA=(
  [efficient8]="model.params.n_rotations=8"
  [efficient4]="model.params.n_rotations=4"
)

# ---- Slurm resources --------------------------------------------------------
GRES="${GRES:-gpu:1}"
MEM="${MEM:-64GB}"
CPUS="${CPUS:-10}"
TIME="${TIME:-24:00:00}"
LOGDIR="${LOGDIR:-logs}"

# Per-network memory override (falls back to $MEM). The light nets need less.
declare -A NET_MEM=(
  [logpolar]=32GB
  [efficient8]=32GB
  [efficient4]=32GB
)
# Per-network core count (falls back to $CPUS). Sized to each config's
# `num_workers` + ~2 (main process + Julia board generation): steerable uses 6
# DataLoader workers (GPU-bound); log-polar and the efficient nets are lighter/more
# data-bound and use the base 8.
declare -A NET_CPUS=(
  [steerable]=8
  [logpolar]=10
  [efficient8]=10
  [efficient4]=10
)

DRY_RUN="${DRY_RUN:-0}"                       # DRY_RUN=1 -> print instead of submit
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- CLI parsing ------------------------------------------------------------
# -n/--name NAME : group this whole matrix under a dated subfolder
#                  "<YYYY_MM_DD>_<NAME>" inside the log dir (both the Slurm output
#                  logs and the training experiment_name). Any other args select
#                  which networks to run (default: all defined ones).
RUN_NAME=""
NETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name) RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") [-n NAME] [network ...]"
      echo "  -n, --name NAME   group runs under <YYYY_MM_DD>_NAME inside the log dir"
      echo "  network ...       restrict to given networks (default: all)"
      echo "  DRY_RUN=1 (env)   print jobs instead of submitting"
      exit 0 ;;
    -*) echo "!! unknown option '$1'" >&2; exit 1 ;;
    *)  NETS+=("$1"); shift ;;
  esac
done
[[ ${#NETS[@]} -eq 0 ]] && NETS=("${!CONFIG[@]}")

# Dated subfolder shared by every run in this matrix (empty -> no grouping).
if [[ -n "$RUN_NAME" ]]; then
  RUN_SUBDIR="$(date +%Y_%m_%d)_${RUN_NAME}"
else
  RUN_SUBDIR=""
fi
LOG_OUTDIR="${LOGDIR}${RUN_SUBDIR:+/$RUN_SUBDIR}"
mkdir -p "$REPO_ROOT/$LOG_OUTDIR"

submit() {
  local net="$1" scale="$2"
  local cfg="${CONFIG[$net]:-}"
  if [[ -z "$cfg" ]]; then
    echo "!! unknown network '$net' (known: ${!CONFIG[*]})" >&2
    return 1
  fi
  local name="${net}_s${scale}"
  local mem="${NET_MEM[$net]:-$MEM}"
  local cpus="${NET_CPUS[$net]:-$CPUS}"
  local extra="${NET_EXTRA[$net]:-}"
  # Prepend the dated subfolder so this run's checkpoints/plots land under it too.
  local exp="${RUN_SUBDIR:+$RUN_SUBDIR/}${name}"
  local job
  job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --gres=${GRES}
#SBATCH --mem=${mem}
#SBATCH -c${cpus}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_OUTDIR}/%x-%j.out

cd deps/BlobBoards.jl
pixi run python ../../src/run_training.py \\
  --config-name ${cfg} scale=${scale} +experiment_name=${exp} ${extra}
EOF
)
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== would submit: ${name} ====="
    echo "$job"
    echo
  else
    echo "$job" | sbatch
  fi
}

for net in "${NETS[@]}"; do
  read -ra scales <<< "${SCALES[$net]:-}"
  if [[ ${#scales[@]} -eq 0 ]]; then
    echo "!! no scales defined for network '$net'" >&2
    continue
  fi
  for scale in "${scales[@]}"; do
    submit "$net" "$scale"
  done
done
