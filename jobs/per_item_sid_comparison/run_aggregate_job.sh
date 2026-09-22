#!/bin/bash
set -euo pipefail

module load python/3.10.8
cd /srv/scratch/z5611110
source .venv/bin/activate
cd GRID

DATASET=${DATASET:-beauty}
SID_HIERARCHIES=${SID_HIERARCHIES:-10}
COMPARISON_ROOT=${COMPARISON_ROOT:?COMPARISON_ROOT is required}
PER_ITEM_METHODS=${PER_ITEM_METHODS:-"interaction_count cooccurrence ppmi neighborhood_entropy graph_centrality item2vec bpr lightgcn"}
SID_METHODS=${SID_METHODS:-"rkmeans"}
DIRECTIONS=${DIRECTIONS:-"direct inverse"}

read -r -a SIGNALS <<< "${PER_ITEM_METHODS//|/ }"
read -r -a DIRECTION_LIST <<< "${DIRECTIONS//|/ }"
read -r -a METHOD_LIST <<< "${SID_METHODS//|/ }"

python -m src.aggregate_per_item_signal_metrics \
    --dataset=${DATASET} \
    --run-root=${COMPARISON_ROOT} \
    --methods "${METHOD_LIST[@]}" \
    --signals "${SIGNALS[@]}" \
    --directions "${DIRECTION_LIST[@]}" \
    --max-length=${SID_HIERARCHIES} \
    --output=${COMPARISON_ROOT}/final_signal_metrics.csv
