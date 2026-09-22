#!/bin/bash
set -euo pipefail

module load python/3.10.8
cd /srv/scratch/z5611110
source .venv/bin/activate
cd GRID

DATASET=${DATASET:-beauty}
SID_HIERARCHIES=${SID_HIERARCHIES:-10}
MIN_SID_LENGTH=${MIN_SID_LENGTH:-1}
LENGTH_DIRECTION=${LENGTH_DIRECTION:?LENGTH_DIRECTION is required}
COMPARISON_ROOT=${COMPARISON_ROOT:?COMPARISON_ROOT is required}
PER_ITEM_METHODS=${PER_ITEM_METHODS:-"interaction_count cooccurrence ppmi neighborhood_entropy graph_centrality item2vec bpr lightgcn"}

read -r -a SIGNALS <<< "${PER_ITEM_METHODS//|/ }"
LENGTH_DIR=${COMPARISON_ROOT}/lengths
mkdir -p "${LENGTH_DIR}"
ARTIFACT_ARGS=()

for SIGNAL in "${SIGNALS[@]}"; do
    LENGTH_PATH=${LENGTH_DIR}/item_lengths_${SIGNAL}_${LENGTH_DIRECTION}.pt
    python -m src.utils.item_length_generation \
        --data-dir=data/amazon_data/${DATASET} \
        --method=${SIGNAL} \
        --min-length=${MIN_SID_LENGTH} \
        --max-length=${SID_HIERARCHIES} \
        --direction=${LENGTH_DIRECTION} \
        --output-path=${LENGTH_PATH}
    ARTIFACT_ARGS+=(--artifact "${SIGNAL}=${LENGTH_PATH}")
done

python -m src.analyze_item_lengths \
    "${ARTIFACT_ARGS[@]}" \
    --max-length=${SID_HIERARCHIES} \
    --output-dir=${COMPARISON_ROOT}/length_analysis/${LENGTH_DIRECTION}
