#!/bin/bash
# Submit Step 2 embedding-generation jobs for all datasets.

DATASETS=(beauty sports toys)

for DATASET in "${DATASETS[@]}"; do
    echo "Submitting embedding job for dataset: ${DATASET}"
    qsub -N gr_emb_${DATASET} -v DATASET=${DATASET} generate_embeddings.pbs
done

echo "All embedding jobs submitted."
