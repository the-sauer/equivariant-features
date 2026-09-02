#!/bin/bash
# Launch a matrix of blob-descriptor trainings: for every network type, sweep the
# key `scale` hyperparameter and submit one Slurm job per (network, scale) pair.
#
# The `scale` override maps to the network's most important knob:
#   steerable -> patch_scale_factor     logpolar -> logpolar_outer_factor
# (see the leaf configs; a single `scale=<x>` threads it everywhere).
#
# Usage:
#   ./launch_training_matrix.sh                 # submit the DEFAULT_SWEEP (see below)
#   ./launch_training_matrix.sh steerable       # only the given network(s)
#   ./launch_training_matrix.sh "${NET_ORDER[@]}"  # (from a shell) the full matrix
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
  [logpolar_circ_nomask]=blob_descriptor_logpolar
  [logpolar_fft]=blob_descriptor_logpolar
  [logpolar_relphase]=blob_descriptor_logpolar
  [logpolar_bispectrum]=blob_descriptor_logpolar
  [logpolar_fftmask]=blob_descriptor_logpolar_fftmask
  [efficient8]=blob_descriptor_efficient
  [efficient4]=blob_descriptor_efficient
  # Steerable counterparts of `logpolar_fftmask`: learned board-validity masking on the
  # cartesian (escnn) descriptors. Same GT-on-anchor / predicted-on-target contract.
  [efficient8_mask]=blob_descriptor_efficient_mask
  [efficient4_mask]=blob_descriptor_efficient_mask
  [steerable_mask]=blob_descriptor_steerable_mask
)
# Default submission order — SLOWEST LAST, and the only thing that defers the slow nets
# on this cluster. PriorityType is multifactor but every PriorityWeight* is 0 (TRES
# unset), so `site_factor + SUM(weight * factor)` is identical for every job and Slurm
# falls back to its equal-priority tiebreak: job id. Submission order therefore *is* the
# schedule.
#
# `--nice` does NOT work here, so don't reach for it: with a zero baseline the nice'd
# priority is `0 - N`, which Slurm clamps back up (0 is reserved for held jobs). Measured
# — two jobs differing only by `--nice=10000` both came back PENDING at priority 1.
# It becomes a real lever only once some PriorityWeight* is non-zero.
#
# Bash associative arrays are UNORDERED, so the previous `${!CONFIG[@]}` was hash order —
# steerable landed last by accident, and renaming or adding a network reshuffles it
# silently. Must cover every CONFIG key (checked below).
NET_ORDER=(efficient8 efficient4 efficient8_mask efficient4_mask logpolar logpolar_circ logpolar_fft logpolar_relphase logpolar_bispectrum logpolar_fftmask logpolar_circ_nomask steerable steerable_mask)

# What a bare (no-arg) invocation submits. NET_ORDER above stays the full registry (it
# must list every CONFIG key, and fixes the slowest-last schedule); DEFAULT_SWEEP just
# picks which of them run by default. Any of the other nets is still runnable by name.
# Currently: the log-polar angular-head ablation ladder, in order of how much of the
# angular profile survives the head — max-pool (one peak) -> fft (|X_k|, no phase) ->
# relphase / bispectrum (|X_k| + an invariant phase feature) -> fft+mask.
DEFAULT_SWEEP=(logpolar_circ logpolar_fft logpolar_relphase logpolar_bispectrum logpolar_fftmask)

# per-network scale values
declare -A SCALES=(
  [steerable]="8 16 32 64 96 128"
  [logpolar]="8 16 32 64 96 128"
  [logpolar_circ]="96"
  [logpolar_circ_nomask]="96"
  [logpolar_fft]="96"
  [logpolar_relphase]="96"
  [logpolar_bispectrum]="96"
  [logpolar_fftmask]="96"
  [efficient8]="8 16 32 64 96 128"
  [efficient4]="8 16 32 64 96 128"
  [efficient8_mask]="8 16 32 64 96 128"
  [efficient4_mask]="8 16 32 64 96 128"
  [steerable_mask]="8 16 32 64 96 128"
)
# Patch anti-aliasing: sub-taps per output pixel per axis, area-averaged. This is a
# *dataset* param, so each value is its own dataset (and its own cache entry) — the
# matrix is the full cross product NETS x SCALES x SUPERSAMPLES, so adding values here
# multiplies both the training jobs and the prebuild tasks. Set to a single value to
# switch the sweep off.
SUPERSAMPLES=(3)
# extra hydra overrides per network (e.g. C8 vs C4 for the efficient descriptor).
# NoStride-vs-Efficient FPR95 comparison: run `steerable efficient8 efficient4`.
declare -A NET_EXTRA=(
  [efficient8]="model.params.n_rotations=8"
  [efficient4]="model.params.n_rotations=4"
  # Same C8/C4 switch; the mask itself comes from the `*_mask` config (model
  # `learned_mask` + dataset `precompute_masks`), not from an override here.
  [efficient8_mask]="model.params.n_rotations=8"
  [efficient4_mask]="model.params.n_rotations=4"
  # log-polar-aware HardNet (circular angular padding + antialiased stride) vs the
  # plain one. `precompute_masks=true` here is NOT used by the model (learned_mask is
  # off) — it only makes the dataset cache key match `logpolar_fftmask`, so all three
  # log-polar variants share ONE prebuilt (mask-carrying) dataset. HardNetLogPolar
  # accepts and ignores the mask inputs, so no mask loss is added.
  [logpolar_circ]="model.name=HardNetLogPolar ++training.dataset.params.precompute_masks=true ++validation.shared_params.params.precompute_masks=true"
  [logpolar_circ_nomask]="++training.ignore_mask=true ++validation.shared_params.params.ignore_mask=true"
  # DFT-magnitude angular head instead of the max-pool (keeps the full angular
  # spectrum). Shares the same mask dataset (mask ignored, as above).
  [logpolar_fft]="model.name=HardNetLogPolar ++model.params.head=fft ++model.params.n_harmonics=5 ++training.dataset.params.precompute_masks=true ++validation.shared_params.params.precompute_masks=true"
  # The two phase-keeping heads: same rfft, but they append an invariant phase feature
  # to the magnitudes instead of discarding the phase (docs/fft_theory.md). `relphase`
  # references every harmonic to X_1 (cheap, fragile where |X_1| ~ 0); `bispectrum` uses
  # reference-free triple products (complete, noisier — normalized by default). Both
  # widen the final conv's angular input (5 rows -> 11 / 13), so a checkpoint from one
  # head does not transfer that layer to another. Same shared mask dataset as above.
  [logpolar_relphase]="model.name=HardNetLogPolar ++model.params.head=relphase ++model.params.n_harmonics=5 ++training.dataset.params.precompute_masks=true ++validation.shared_params.params.precompute_masks=true"
  [logpolar_bispectrum]="model.name=HardNetLogPolar ++model.params.head=bispectrum ++model.params.n_harmonics=5 ++training.dataset.params.precompute_masks=true ++validation.shared_params.params.precompute_masks=true"
  # `logpolar_fftmask` (FFT head + learned mask) needs no NET_EXTRA — its dedicated
  # config `blob_descriptor_logpolar_fftmask` sets the model + `precompute_masks`.
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
  # The mask nets need the GT validity mask co-extracted, which changes the dataset
  # cache key — hence their own group rather than the plain `cartesian` one. (The
  # log-polar variants instead all share ONE mask-carrying dataset, via `precompute_masks`
  # overrides on the mask-less nets; do the same here if you want a single cartesian
  # build, at the cost of invalidating the existing mask-less cartesian caches.)
  [steerable_mask]=cartesian_mask
  [efficient8_mask]=cartesian_mask
  [efficient4_mask]=cartesian_mask
  [logpolar]=logpolar
  [logpolar_circ_nomask]=logpolar
  # circ/fft/fftmask all share ONE mask-carrying dataset: the mask channel is a superset
  # the mask-less variants simply ignore, so the boards/SIFT/patches are built once. Only
  # the plain `logpolar` variant (model `HardNet`, which can't take the mask kwargs)
  # keeps the mask-less `logpolar` group.
  [logpolar_circ]=logpolar_mask
  [logpolar_fft]=logpolar_mask
  [logpolar_relphase]=logpolar_mask
  [logpolar_bispectrum]=logpolar_mask
  [logpolar_fftmask]=logpolar_mask
)
declare -A GROUP_CONFIG=(
  [cartesian]=blob_descriptor_steerable
  [cartesian_mask]=blob_descriptor_steerable_mask
  [logpolar]=blob_descriptor_logpolar
  [logpolar_mask]=blob_descriptor_logpolar_fftmask
)
# Validation split names from `validation.datasets` — one prebuild task each. MUST match
# `conf/homography_validation.yaml`: prebuild_datasets.py raises on an unknown split, and
# the training jobs hang on `--dependency=afterok`, so a stale name here silently blocks
# the whole matrix. (Was `overall far medium near` — those splits are long gone.)
VAL_SPLITS=(default affine strong extreme)

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

# Per-network memory override (falls back to $MEM = 64GB for every net). The light nets
# used to be pinned to 32GB; that is no longer worth the OOM risk — in-memory patch
# caches (plus the mask channel for the *_mask nets) dominate host RAM, so everything
# gets the same 64GB. Add an entry here only for a net that genuinely needs MORE.
declare -A NET_MEM=(
  [__none__]=""   # placeholder: an empty assoc array + `set -u` is an error on bash < 4.4
)
# Per-network core count (falls back to $CPUS). Sized to each config's
# `num_workers` + ~2 (main process + Julia board generation): steerable uses 6
# DataLoader workers (GPU-bound); log-polar and the efficient nets are lighter/more
# data-bound and use the base 8.
declare -A NET_CPUS=(
  [steerable]=8
  [steerable_mask]=8
  [efficient8_mask]=10
  [efficient4_mask]=10
  [logpolar]=10
  [logpolar_circ]=10
  [logpolar_circ_nomask]=10
  [logpolar_fft]=10
  [logpolar_relphase]=10
  [logpolar_bispectrum]=10
  [logpolar_fftmask]=10
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
# A CONFIG entry missing from NET_ORDER would silently vanish from the default matrix,
# so fail loudly instead of quietly training four networks when you asked for five.
for net in "${!CONFIG[@]}"; do
  case " ${NET_ORDER[*]} " in
    *" ${net} "*) ;;
    *) echo "!! '${net}' is in CONFIG but not NET_ORDER — add it (slowest last)" >&2
       exit 1 ;;
  esac
done
[[ ${#NETS[@]} -eq 0 ]] && NETS=("${DEFAULT_SWEEP[@]}")

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
  validation.shared_params.params.supersample=${ss} ++training.continue= ${extra}
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
