#!/bin/bash
# Submit full-pipeline jobs for all (dataset, SID method) combinations.
# Each job runs Steps 2-4 independently, so embeddings are regenerated per dataset.

DATASETS=(beauty sports toys)
METHODS=(rkmeans rvq rqvae)

for DATASET in "${DATASETS[@]}"; do
    for METHOD in "${METHODS[@]}"; do
        echo "Submitting job: dataset=${DATASET}, method=${METHOD}"
        qsub -N gr_${METHOD}_${DATASET} \
             -v DATASET=${DATASET},SID_METHOD=${METHOD} \
             run_pipeline.pbs
    done
done

echo "All pipeline jobs submitted."
