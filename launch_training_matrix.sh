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
#   NO_PREBUILD=1 ./launch_training_matrix.sh   # skip the dataset-prebuild job
#
# A prebuild job runs first and builds every dataset the matrix needs (populating the
# dataset cache); the training jobs are submitted with --dependency=afterok on it, so
# they all start from a warm cache instead of cold-building the same boards in
# parallel. See docs/configs_and_training.md.
#
# Run it from the repository root.
set -euo pipefail

# ---- Sweep definition (edit these) ------------------------------------------
# network name -> hydra config-name
declare -A CONFIG=(
  [steerable]=blob_descriptor_steerable
  [logpolar]=blob_descriptor_logpolar
  [logpolar_circ]=blob_descriptor_logpolar
  [efficient8]=blob_descriptor_efficient
  [efficient4]=blob_descriptor_efficient
)
# per-network scale values
declare -A SCALES=(
  [steerable]="32 64 96 128"
  [logpolar]="32 64 96 128"
  [logpolar_circ]="32 64 96 128"
  [efficient8]="32 64 96 128"
  [efficient4]="32 64 96 128"
)
# Patch anti-aliasing: sub-taps per output pixel per axis, area-averaged. This is a
# *dataset* param, so each value is its own dataset (and its own cache entry) — the
# matrix is the full cross product NETS x SCALES x SUPERSAMPLES, so adding values here
# multiplies both the training jobs and the prebuild tasks. Set to a single value to
# switch the sweep off.
SUPERSAMPLES=(1 2 4)
# extra hydra overrides per network (e.g. C8 vs C4 for the efficient descriptor).
# NoStride-vs-Efficient FPR95 comparison: run `steerable efficient8 efficient4`.
declare -A NET_EXTRA=(
  [efficient8]="model.params.n_rotations=8"
  [efficient4]="model.params.n_rotations=4"
  # log-polar-aware HardNet (circular angular padding + antialiased stride) vs the
  # plain one; same config/dataset, so they share the prebuilt datasets.
  [logpolar_circ]="model.name=HardNetLogPolar"
)

# ---- Dataset groups (drive the prebuild) ------------------------------------
# The dataset cache key covers the dataset params but NOT the model, so networks whose
# configs share a dataset block share their datasets. Group them here and name one
# representative config per group, so the prebuild builds each dataset exactly once.
# If a group mapping is wrong the only cost is a cache miss (the training job rebuilds
# it), never a wrong dataset.
declare -A NET_DATASET_GROUP=(
  [steerable]=cartesian
  [efficient8]=cartesian
  [efficient4]=cartesian
  [logpolar]=logpolar
  [logpolar_circ]=logpolar
)
declare -A GROUP_CONFIG=(
  [cartesian]=blob_descriptor_steerable
  [logpolar]=blob_descriptor_logpolar
)
# Validation split names from `validation.datasets` — one prebuild task each.
VAL_SPLITS=(overall small medium large)

# ---- Slurm resources --------------------------------------------------------
GRES="${GRES:-gpu:1}"
MEM="${MEM:-64GB}"
CPUS="${CPUS:-10}"
TIME="${TIME:-24:00:00}"
LOGDIR="${LOGDIR:-logs}"

# Prebuild job: builds every dataset the matrix needs (populating `cache_dir`) so the
# training jobs all start from a warm cache instead of cold-building the same boards
# in parallel. Training jobs are submitted with --dependency=afterok on it.
# NO_PREBUILD=1 skips it. Needs a GPU (SIFT + patch extraction run on CUDA).
NO_PREBUILD="${NO_PREBUILD:-0}"
PREBUILD_MEM="${PREBUILD_MEM:-64GB}"
PREBUILD_CPUS="${PREBUILD_CPUS:-4}"
PREBUILD_TIME="${PREBUILD_TIME:-4:00:00}"

# Per-network memory override (falls back to $MEM). The light nets need less.
declare -A NET_MEM=(
  [logpolar]=32GB
  [logpolar_circ]=32GB
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
  [logpolar_circ]=10
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

# One prebuild task per DISTINCT dataset: for every (dataset group, scale) the matrix
# touches, the training set plus each validation split. Networks in the same group
# collapse to a single set of tasks. Fills PREBUILD_TASKS as "<config> <scale> <target>".
build_prebuild_tasks() {
  local seen=" " grp cfg scale ss s
  PREBUILD_TASKS=()
  for net in "${NETS[@]}"; do
    [[ -z "${CONFIG[$net]:-}" ]] && continue
    grp="${NET_DATASET_GROUP[$net]:-$net}"
    cfg="${GROUP_CONFIG[$grp]:-${CONFIG[$net]}}"
    read -ra scales <<< "${SCALES[$net]:-}"
    for scale in "${scales[@]}"; do
      for ss in "${SUPERSAMPLES[@]}"; do
        case "$seen" in *" ${grp}:${scale}:${ss} "*) continue ;; esac
        seen+="${grp}:${scale}:${ss} "
        PREBUILD_TASKS+=("${cfg} ${scale} ${ss} train")
        for s in "${VAL_SPLITS[@]}"; do
          PREBUILD_TASKS+=("${cfg} ${scale} ${ss} ${s}")
        done
      done
    done
  done
}

# Submits the prebuild as a Slurm job ARRAY (one task per dataset) so they build in
# parallel; a dependency on the array's job id waits for every task. Echoes the id.
submit_prebuild() {
  build_prebuild_tasks
  local n=${#PREBUILD_TASKS[@]}
  if [[ $n -eq 0 ]]; then echo ""; return; fi

  local tasks=""
  for t in "${PREBUILD_TASKS[@]}"; do tasks+="  \"${t}\""$'\n'; done

  local job
  job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=prebuild
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --gres=${GRES}
#SBATCH --mem=${PREBUILD_MEM}
#SBATCH -c${PREBUILD_CPUS}
#SBATCH --time=${PREBUILD_TIME}
#SBATCH --array=0-$((n - 1))
#SBATCH --output=${LOG_OUTDIR}/%x-%A_%a.out
set -euo pipefail

TASKS=(
${tasks})
read -r cfg scale ss target <<< "\${TASKS[\$SLURM_ARRAY_TASK_ID]}"
echo "prebuilding: config=\$cfg scale=\$scale supersample=\$ss target=\$target"

cd deps/BlobBoards.jl
pixi run python ../../src/prebuild_datasets.py \\
  --config-name "\$cfg" scale="\$scale" +prebuild_target="\$target" \\
  training.dataset.params.supersample="\$ss" \\
  validation.shared_params.params.supersample="\$ss"
EOF
)
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== would submit: prebuild array (${n} datasets) =====" >&2
    echo "$job" >&2
    echo >&2
    echo "<prebuild-jobid>"
  else
    echo "$job" | sbatch --parsable
  fi
}

submit() {
  local net="$1" scale="$2" ss="$3"
  local cfg="${CONFIG[$net]:-}"
  if [[ -z "$cfg" ]]; then
    echo "!! unknown network '$net' (known: ${!CONFIG[*]})" >&2
    return 1
  fi
  local name="${net}_s${scale}_ss${ss}"
  local mem="${NET_MEM[$net]:-$MEM}"
  local cpus="${NET_CPUS[$net]:-$CPUS}"
  local extra="${NET_EXTRA[$net]:-}"
  # Wait for the prebuild job so every run starts from a warm dataset cache.
  local dep="${PREBUILD_ID:+#SBATCH --dependency=afterok:$PREBUILD_ID}"
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
${dep}

cd deps/BlobBoards.jl
pixi run python ../../src/run_training.py \\
  --config-name ${cfg} scale=${scale} +experiment_name=${exp} \\
  training.dataset.params.supersample=${ss} \\
  validation.shared_params.params.supersample=${ss} ${extra}
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

PREBUILD_ID=""
if [[ "$NO_PREBUILD" != "1" ]]; then
  PREBUILD_ID="$(submit_prebuild)"
  echo "prebuild job: ${PREBUILD_ID} (training jobs wait on afterok)"
fi

for net in "${NETS[@]}"; do
  read -ra scales <<< "${SCALES[$net]:-}"
  if [[ ${#scales[@]} -eq 0 ]]; then
    echo "!! no scales defined for network '$net'" >&2
    continue
  fi
  for scale in "${scales[@]}"; do
    for ss in "${SUPERSAMPLES[@]}"; do
      submit "$net" "$scale" "$ss"
    done
  done
done
