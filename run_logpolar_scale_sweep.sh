#!/bin/bash

DEPENDENCY="${DEPENDENCY:+--dependency=afterok:$DEPENDENCY}"

for lambda in  45 64 96 128 181; do
    TRACK_PATH="${TRACK_BASE_PATH}/tracks_logpolar_s${lambda}.tracks"
    export AEF_CONFIG_NAME="logpolar_scale_sweep"
    export AEF_OVERRIDES="scale=${lambda} track_path=${TRACK_PATH}"
    sbatch $DEPENDENCY run_training.sh
done
