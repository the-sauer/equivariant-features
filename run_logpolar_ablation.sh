#!/bin/bash

DEPENDENCY="${DEPENDENCY:+--dependency=afterok:$DEPENDENCY}"

for head in maxpool fft; do
    for circular_pad in "true" "false"; do
        for antialias in "true" "false"; do
            echo "Launching job with head=${head} circ_pad=${circ_pad} antialias=${antialias}"
            export AEF_CONFIG_NAME="logpolar_ablation.yaml"
            export AEF_OVERRIDES="head=${head} circular_pad=${circular_pad} antialias=${antialias} track_path=${AEF_TRACK_PATH} +model.params.mask=true +model.params.cascade=true logging.dir=/raid/data/evaluation/mask_ablation"
            sbatch $DEPENDENCY run_training.sh
        done
    done
done
