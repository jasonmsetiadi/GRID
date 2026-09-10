#!/bin/bash
# Submit full-pipeline jobs for all (dataset, SID method, SID mode) combinations.
# Each job runs the SID and recommendation stages independently per mode.

DATASETS=(beauty sports toys)
METHODS=(rkmeans rvq rqvae)
MODES=(fixed variable)
RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD:-0.05}
MIN_HIERARCHIES=${MIN_HIERARCHIES:-1}

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        for MODE in "${MODES[@]}"; do
            echo "Submitting job: dataset=${DATASET}, method=${METHOD}, mode=${MODE}"
            qsub -N gr_${MODE}_${METHOD}_${DATASET} \
                 -v DATASET=${DATASET},SID_METHOD=${METHOD},SEMANTIC_ID_MODE=${MODE},RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD},MIN_HIERARCHIES=${MIN_HIERARCHIES} \
                 run_pipeline.pbs
        done
    done
done

echo "All pipeline jobs submitted."
