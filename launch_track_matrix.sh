#!/bin/bash
# Launch a matrix of TRACK-data descriptor trainings. This is the track counterpart of
# launch_training_matrix.sh: for every network type it sweeps the `scale` hyperparameter
# and submits one Slurm job per (network, scale) pair — but instead of building synthetic
# boards, each job trains on a real `.tracks` file.
#
# Every run trains FROM SCRATCH. The script used to warm-start each job from a matching
# synthetic run's checkpoint, discovered under a `--from-dir DIR`; that machinery is gone.
# `training.init_from_checkpoint` still exists in the training loop — pass it explicitly
# via EXTRA_OVERRIDES (below) when a run really should start from a checkpoint.
#
# Usage:
#   ./launch_track_matrix.sh --track /raid/data/hsa/datasets/only_absolute.tracks
#   ./launch_track_matrix.sh --track FILE steerable   # only given network(s)
#   ./launch_track_matrix.sh --track FILE -n track_ft # group under a dated subfolder
#   DRY_RUN=1 ./launch_track_matrix.sh --track FILE   # print jobs without submitting
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
  # Mask as an INPUT GATE instead of a late weight on the pooling (two trunk passes).
  # `logpolar_cascade` is the A-arm of that experiment, `logpolar_fftmask` its control —
  # submit BOTH, at the same batch size, or the comparison means nothing. The `_late`
  # variant keeps the late weight on top of the gate (see the config's header for why
  # that is expected to lose). Costs ~2x the activation memory, hence the batch-size
  # note in the header.
  [logpolar_cascade]=track_descriptor_logpolar_cascade
  [logpolar_cascade_late]=track_descriptor_logpolar_cascade
  # CEILING arm: the true mask on every view (`oracle_mask`), late weight resp. input
  # gate. Not deployable — it is the "what would a perfect predictor be worth?"
  # measurement that tells a useless mask apart from an unlearnable one. Submit with
  # `logpolar_fftmask` (predicted mask) and `logpolar_fft` (no mask) or it says nothing.
  [logpolar_oracle]=track_descriptor_logpolar_oracle
  [logpolar_oracle_gate]=track_descriptor_logpolar_oracle
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
NET_ORDER=(efficient8 efficient4 efficient8_mask efficient4_mask logpolar logpolar_circ logpolar_fft logpolar_relphase logpolar_bispectrum logpolar_fftmask logpolar_relphase_mask logpolar_bispectrum_mask logpolar_cascade logpolar_cascade_late logpolar_oracle logpolar_oracle_gate steerable steerable_mask)
# What a bare (no network arg) invocation submits: the log-polar angular-head ladder
# (max-pool/circ -> fft -> relphase/bispectrum -> fft+mask), matching
# launch_training_matrix.sh's default.
DEFAULT_SWEEP=(logpolar_circ logpolar_fft logpolar_relphase logpolar_bispectrum logpolar_fftmask)

# per-network scale values (must match the scale the `.tracks` patches were extracted at
# — the model's scale knob and the patches have to agree).
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
  [logpolar_cascade]="96"
  [logpolar_cascade_late]="96"
  [logpolar_oracle]="96"
  [logpolar_oracle_gate]="96"
  [efficient8]="8 16 32 64 96 128"
  [efficient4]="8 16 32 64 96 128"
  [efficient8_mask]="8 16 32 64 96 128"
  [efficient4_mask]="8 16 32 64 96 128"
  [steerable_mask]="8 16 32 64 96 128"
)

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
  # The config already sets cascade=true / cascade_late_weight=false; only the `_late`
  # arm flips the second one.
  [logpolar_cascade]=""
  [logpolar_cascade_late]="++model.params.cascade_late_weight=true"
  # The config already sets oracle_mask=true; the `_gate` arm moves it to the input.
  [logpolar_oracle]=""
  [logpolar_oracle_gate]="++model.params.cascade=true"
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

# Hydra overrides appended to EVERY job in the matrix. Use it for settings that must be
# IDENTICAL across the arms of a comparison — a batch size pinned here applies to all of
# them, where a per-net NET_EXTRA entry would silently give one arm a different one.
#   EXTRA_OVERRIDES="training.batch_size=2048" ./launch_track_matrix.sh ...
# It is also how to warm-start deliberately, now that the script does not:
#   EXTRA_OVERRIDES="+training.init_from_checkpoint=/path/to/best.pth" ./launch_track_matrix.sh ...
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

DRY_RUN="${DRY_RUN:-0}"                       # DRY_RUN=1 -> print instead of submit
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- CLI parsing ------------------------------------------------------------
# --track FILE   : the .tracks HDF5 file to train on (required)
# --scales "A B" : restrict the scale sweep to these values (default: each net's SCALES)
# -n/--name NAME : group this whole matrix under a dated "<YYYY_MM_DD>_<NAME>" subfolder
# other args     : restrict to the given network(s) (default: DEFAULT_SWEEP)
TRACK_PATH=""
SCALE_FILTER=""
RUN_NAME=""
NETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --track)    TRACK_PATH="$2"; shift 2 ;;
    --scales)   SCALE_FILTER="$2"; shift 2 ;;
    -n|--name)  RUN_NAME="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") --track FILE [--scales \"A B\"] [-n NAME] [network ...]"
      echo "  --track FILE     .tracks HDF5 file to train on (required)"
      echo "  --scales \"A B\"   restrict the scale sweep to these values (default: each net's full sweep)"
      echo "  -n, --name NAME  group runs under <YYYY_MM_DD>_NAME inside the log dir"
      echo "  network ...      restrict to given networks (default: ${DEFAULT_SWEEP[*]})"
      echo "  DRY_RUN=1 (env)  print jobs instead of submitting"
      echo "  EXTRA_OVERRIDES= (env) hydra overrides appended to every job (e.g."
      echo "                   training.batch_size=2048) — use for settings that must"
      echo "                   match across the arms of a comparison"
      exit 0 ;;
    -*) echo "!! unknown option '$1'" >&2; exit 1 ;;
    *)  NETS+=("$1"); shift ;;
  esac
done

[[ -z "$TRACK_PATH" ]] && { echo "!! --track is required (path to the .tracks file)" >&2; exit 1; }
[[ ! -f "$TRACK_PATH" && "$DRY_RUN" != "1" ]] && { echo "!! --track '$TRACK_PATH' not found" >&2; exit 1; }

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

submit() {
  local net="$1" scale="$2"
  local cfg="${CONFIG[$net]:-}"
  if [[ -z "$cfg" ]]; then
    echo "!! unknown network '$net' (known: ${!CONFIG[*]})" >&2
    return 1
  fi

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
  +experiment_name=${exp} ${extra} ${EXTRA_OVERRIDES}
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
