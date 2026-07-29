#!/bin/bash
# Launch a matrix of TRACK-data descriptor trainings, each WARM-STARTED from a synthetic
# blob-descriptor checkpoint. This is the track counterpart of launch_training_matrix.sh:
# for every network type it sweeps the `scale` hyperparameter and submits one Slurm job
# per (network, scale) pair — but instead of building synthetic boards, each job trains
# on a real `.tracks` file and initializes its model from a prior run's checkpoint.
#
# The point of this script: `--from-dir DIR` chooses a DIRECTORY OF TRAINING RUNS and
# continues from the checkpoints in there. For each (network, scale) it locates the
# matching synthetic run under DIR — a `<net>_s<scale>_ss<ss>/checkpoints/<ckpt>.pth`,
# at any depth (so dated `-n` sub-folders are found too) — and passes it as
# `+training.init_from_checkpoint=<ckpt>`, which loads the model weights and starts a
# FRESH run (epoch 0, new optimizer). See prepare_training in src/aef/train/__init__.py.
#
# Usage:
#   ./launch_track_matrix.sh --from-dir /raid/data/hsa/logs/2026_07_20_sweep \
#                            --track /raid/data/hsa/datasets/only_absolute.tracks
#   ./launch_track_matrix.sh --from-dir DIR --track FILE steerable   # only given network(s)
#   ./launch_track_matrix.sh --from-dir DIR --track FILE --ckpt latest
#   ./launch_track_matrix.sh --from-dir DIR --track FILE -n track_ft # group under a dated subfolder
#   DRY_RUN=1 ./launch_track_matrix.sh --from-dir DIR --track FILE   # print jobs without submitting
#
# There is no prebuild job — the `.tracks` file is prepared beforehand (by BlobBoards).
#
# Run it from the repository root.
set -euo pipefail

# ---- Sweep definition (mirrors launch_training_matrix.sh) -------------------
# network name -> hydra config-name (the track_* leaves)
declare -A CONFIG=(
  [steerable]=track_descriptor_steerable
  [logpolar]=track_descriptor_logpolar
  [logpolar_circ]=track_descriptor_logpolar
  [logpolar_fft]=track_descriptor_logpolar
  [logpolar_relphase]=track_descriptor_logpolar
  [logpolar_bispectrum]=track_descriptor_logpolar
  [logpolar_fftmask]=track_descriptor_logpolar_fftmask
  [logpolar_relphase_mask]=track_descriptor_logpolar_fftmask
  [logpolar_bispectrum_mask]=track_descriptor_logpolar_fftmask
  [efficient8]=track_descriptor_efficient
  [efficient4]=track_descriptor_efficient
  # Learned board-validity masking on the steerable (escnn) descriptors — the cartesian
  # counterpart of the logpolar_*_mask entries. The `.tracks` file already carries the
  # per-patch masks, so these only need `with_mask: true` (set by the configs).
  [efficient8_mask]=track_descriptor_efficient_mask
  [efficient4_mask]=track_descriptor_efficient_mask
  [steerable_mask]=track_descriptor_steerable_mask
)
# Submission order (slowest last); must cover every CONFIG key.
NET_ORDER=(efficient8 efficient4 efficient8_mask efficient4_mask logpolar logpolar_circ logpolar_fft logpolar_relphase logpolar_bispectrum logpolar_fftmask logpolar_relphase_mask logpolar_bispectrum_mask steerable steerable_mask)
# What a bare (no network arg) invocation submits: the log-polar angular-head ladder
# (max-pool/circ -> fft -> relphase/bispectrum -> fft+mask), matching
# launch_training_matrix.sh's default.
DEFAULT_SWEEP=(logpolar_circ logpolar_fft logpolar_relphase logpolar_bispectrum logpolar_fftmask)

# per-network scale values (must match the synthetic runs you warm-start from, since the
# model's scale knob has to agree with the checkpoint).
declare -A SCALES=(
  [steerable]="8 16 32 64 96 128"
  [logpolar]="96"
  [logpolar_circ]="96"
  [logpolar_fft]="96"
  [logpolar_relphase]="96"
  [logpolar_bispectrum]="96"
  [logpolar_fftmask]="96"
  [logpolar_relphase_mask]="96"
  [logpolar_bispectrum_mask]="96"
  [efficient8]="8 16 32 64 96 128"
  [efficient4]="8 16 32 64 96 128"
  [efficient8_mask]="8 16 32 64 96 128"
  [efficient4_mask]="8 16 32 64 96 128"
  [steerable_mask]="8 16 32 64 96 128"
)
# Supersample tag used ONLY to locate the source run (its name is <net>_s<scale>_ss<ss>);
# it is not a track-training parameter (track patches are precomputed).
SRC_SS="${SRC_SS:-3}"

# Per-network memory override (falls back to $MEM, now 64GB for every net — the track
# patches are all held in memory, so no net is meaningfully lighter). Add an entry only
# for a net that needs MORE than the default.
declare -A NET_MEM=(
  [__none__]=""   # placeholder: an empty assoc array + `set -u` is an error on bash < 4.4
)

# extra hydra overrides per network — the model switches that distinguish the log-polar
# ablation variants and the C8/C4 efficient nets. Mirrors launch_training_matrix.sh's
# NET_EXTRA, minus the dataset precompute_masks bits (track patches carry their own).
declare -A NET_EXTRA=(
  [efficient8]="model.params.n_rotations=8"
  [efficient4]="model.params.n_rotations=4"
  [logpolar_circ]="model.name=HardNetLogPolar"
  [logpolar_fft]="model.name=HardNetLogPolar ++model.params.head=fft ++model.params.n_harmonics=4"
  # Magnitude + an invariant phase feature (docs/fft_theory.md). Warm-start these from
  # the SAME head's synthetic run: the phase rows widen the final conv, whose weights
  # therefore do not transfer between heads (the load drops that tensor and reports it).
  [logpolar_relphase]="model.name=HardNetLogPolar ++model.params.head=relphase ++model.params.n_harmonics=4"
  [logpolar_bispectrum]="model.name=HardNetLogPolar ++model.params.head=bispectrum ++model.params.n_harmonics=4"
  [logpolar_relphase_mask]="model.name=HardNetLogPolar ++model.params.head=relphase ++model.params.n_harmonics=4"
  [logpolar_bispectrum_mask]="model.name=HardNetLogPolar ++model.params.head=bispectrum ++model.params.n_harmonics=4"
  [logpolar_fftmask]="++model.params.n_harmonics=4"
  # C8/C4 switch; the masking itself is in the *_mask configs (model `learned_mask` +
  # dataset `with_mask`), not overridden here.
  [efficient8_mask]="model.params.n_rotations=8"
  [efficient4_mask]="model.params.n_rotations=4"
)

# ---- Slurm resources --------------------------------------------------------
GRES="${GRES:-gpu:1}"
MEM="${MEM:-64GB}"
CPUS="${CPUS:-10}"
TIME="${TIME:-24:00:00}"
LOGDIR="${LOGDIR:-logs}"

DRY_RUN="${DRY_RUN:-0}"                       # DRY_RUN=1 -> print instead of submit
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- CLI parsing ------------------------------------------------------------
# --from-dir DIR : directory of prior training runs to warm-start from (required)
# --track FILE   : the .tracks HDF5 file to train on (required)
# --ckpt NAME    : checkpoint basename to load (best|latest, default best)
# --scales "A B" : restrict the scale sweep to these values (default: each net's SCALES)
# -n/--name NAME : group this whole matrix under a dated "<YYYY_MM_DD>_<NAME>" subfolder
# other args     : restrict to the given network(s) (default: DEFAULT_SWEEP)
FROM_DIR=""
TRACK_PATH=""
CKPT="best"
SCALE_FILTER=""
RUN_NAME=""
NETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-dir) FROM_DIR="$2"; shift 2 ;;
    --track)    TRACK_PATH="$2"; shift 2 ;;
    --ckpt)     CKPT="$2"; shift 2 ;;
    --scales)   SCALE_FILTER="$2"; shift 2 ;;
    -n|--name)  RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") --from-dir DIR --track FILE [--ckpt best|latest] [--scales \"A B\"] [-n NAME] [network ...]"
      echo "  --from-dir DIR   directory of prior runs to warm-start each track run from (required)"
      echo "  --track FILE     .tracks HDF5 file to train on (required)"
      echo "  --ckpt NAME      checkpoint basename to load: best (default) or latest"
      echo "  --scales \"A B\"   restrict the scale sweep to these values (default: each net's full sweep)"
      echo "  -n, --name NAME  group runs under <YYYY_MM_DD>_NAME inside the log dir"
      echo "  network ...      restrict to given networks (default: ${DEFAULT_SWEEP[*]})"
      echo "  DRY_RUN=1 (env)  print jobs instead of submitting"
      exit 0 ;;
    -*) echo "!! unknown option '$1'" >&2; exit 1 ;;
    *)  NETS+=("$1"); shift ;;
  esac
done

# [[ -z "$FROM_DIR" ]] && { echo "!! --from-dir is required (directory of runs to warm-start from)" >&2; exit 1; }
[[ -z "$TRACK_PATH" ]] && { echo "!! --track is required (path to the .tracks file)" >&2; exit 1; }
# [[ ! -d "$FROM_DIR" ]] && { echo "!! --from-dir '$FROM_DIR' is not a directory" >&2; exit 1; }
[[ ! -f "$TRACK_PATH" && "$DRY_RUN" != "1" ]] && { echo "!! --track '$TRACK_PATH' not found" >&2; exit 1; }
case "$CKPT" in best|latest) ;; *) echo "!! --ckpt must be 'best' or 'latest'" >&2; exit 1 ;; esac

# Every CONFIG entry must appear in NET_ORDER (so nothing silently vanishes).
for net in "${!CONFIG[@]}"; do
  case " ${NET_ORDER[*]} " in
    *" ${net} "*) ;;
    *) echo "!! '${net}' is in CONFIG but not NET_ORDER — add it (slowest last)" >&2; exit 1 ;;
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

# Find the source checkpoint for a (net, scale): the descriptor run is named
# <net>_s<scale>_ss<ss>; match its checkpoints/<ckpt>.pth at any depth under FROM_DIR.
find_checkpoint() {
  local net="$1" scale="$2"
  local src_name="${net}_s${scale}_ss${SRC_SS}"
  find "$FROM_DIR" -type f -path "*/${src_name}/checkpoints/${CKPT}.pth" 2>/dev/null | sort | head -n1
}

submit() {
  local net="$1" scale="$2"
  local cfg="${CONFIG[$net]:-}"
  if [[ -z "$cfg" ]]; then
    echo "!! unknown network '$net' (known: ${!CONFIG[*]})" >&2
    return 1
  fi

  # local ckpt_path
  # ckpt_path="$(find_checkpoint "$net" "$scale")"
  # if [[ -z "$ckpt_path" ]]; then
  #   echo "!! no ${CKPT}.pth for ${net}_s${scale}_ss${SRC_SS} under ${FROM_DIR} — skipping" >&2
  #   return 0
  # fi

  local name="track_${net}_s${scale}"
  local extra="${NET_EXTRA[$net]:-}"
  # Prepend the dated subfolder so this run's checkpoints/plots land under it too.
  local exp="${RUN_SUBDIR:+$RUN_SUBDIR/}${name}"
  local job
  job=$(cat <<EOF
#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --chdir=${REPO_ROOT}
#SBATCH --gres=${GRES}
#SBATCH --mem=${NET_MEM[$net]:-$MEM}
#SBATCH -c${CPUS}
#SBATCH --time=${TIME}
#SBATCH --output=${LOG_OUTDIR}/%x-%j.out

cd deps/BlobBoards.jl
pixi run python ../../src/run_training.py \\
  --config-name ${cfg} scale=${scale} \\
  "track_path=${TRACK_PATH}" \\
  +experiment_name=${exp} ${extra}
EOF
)
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "===== would submit: ${name} (warm-start) ====="
    echo "$job"
    echo
  else
    
    echo "$job" | sbatch
  fi
}

for net in "${NETS[@]}"; do
  # `--scales` overrides the per-net sweep for every selected network.
  read -ra scales <<< "${SCALE_FILTER:-${SCALES[$net]:-}}"
  if [[ ${#scales[@]} -eq 0 ]]; then
    echo "!! no scales defined for network '$net'" >&2
    continue
  fi
  for scale in "${scales[@]}"; do
    submit "$net" "$scale"
  done
done
