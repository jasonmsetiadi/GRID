#!/bin/bash
set -euo pipefail

DATASETS=(beauty) # toys games)
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
SID_HIERARCHIES=${SID_HIERARCHIES:-10}
MIN_SID_LENGTH=${MIN_SID_LENGTH:-1}
PER_ITEM_METHODS=(interaction_count cooccurrence ppmi neighborhood_entropy graph_centrality item2vec bpr lightgcn)
DIRECTIONS=(direct inverse)
SID_METHODS=(rkmeans) #rvq rqvae)
COMPARISON_ROOT_BASE=${COMPARISON_ROOT_BASE:-logs}

join_options() {
    local separator='|'
    local first=$1
    shift
    printf '%s' "${first}"
    printf '%s%s' "${separator}" "$@"
}

PER_ITEM_METHODS_VALUE=$(join_options "${PER_ITEM_METHODS[@]}")
DIRECTIONS_VALUE=$(join_options "${DIRECTIONS[@]}")
SID_METHODS_VALUE=$(join_options "${SID_METHODS[@]}")

for DATASET in "${DATASETS[@]}"; do
    COMPARISON_ROOT=${COMPARISON_ROOT_BASE}/${DATASET}/per_item_comparison/${RUN_ID}
    declare -A LENGTH_JOB_IDS=()
    MODEL_JOB_IDS=()

    for DIRECTION in "${DIRECTIONS[@]}"; do
        LENGTH_JOB_IDS["${DIRECTION}"]=$(qsub \
            -N gr_lengths_${DIRECTION}_${DATASET} \
            -v DATASET=${DATASET},SID_HIERARCHIES=${SID_HIERARCHIES},MIN_SID_LENGTH=${MIN_SID_LENGTH},LENGTH_DIRECTION=${DIRECTION},PER_ITEM_METHODS=${PER_ITEM_METHODS_VALUE},COMPARISON_ROOT=${COMPARISON_ROOT} \
            jobs/per_item_sid_comparison/run_length_stage.pbs)
        echo "Submitted length stage ${DATASET}/${DIRECTION}: ${LENGTH_JOB_IDS[${DIRECTION}]}"
    done

    for DIRECTION in "${DIRECTIONS[@]}"; do
        for SID_METHOD in "${SID_METHODS[@]}"; do
            for SIGNAL in "${PER_ITEM_METHODS[@]}"; do
                JOB_ID=$(qsub \
                    -N gr_${SID_METHOD}_${SIGNAL}_${DIRECTION}_${DATASET} \
                    -W depend=afterok:${LENGTH_JOB_IDS[${DIRECTION}]} \
                    -v DATASET=${DATASET},SID_METHOD=${SID_METHOD},SIGNAL=${SIGNAL},LENGTH_DIRECTION=${DIRECTION},SID_HIERARCHIES=${SID_HIERARCHIES},MIN_SID_LENGTH=${MIN_SID_LENGTH},COMPARISON_ROOT=${COMPARISON_ROOT} \
                    jobs/per_item_sid_comparison/run_signal_model_job.pbs)
                MODEL_JOB_IDS+=("${JOB_ID}")
                echo "Submitted model job ${DATASET}/${SID_METHOD}/${SIGNAL}/${DIRECTION}: ${JOB_ID}"
            done
        done
    done

    MODEL_DEPENDENCY=$(printf ":%s" "${MODEL_JOB_IDS[@]}")
    MODEL_DEPENDENCY=${MODEL_DEPENDENCY#:}
    AGGREGATE_JOB=$(qsub \
        -N gr_aggregate_${DATASET} \
        -W depend=afterok:${MODEL_DEPENDENCY} \
        -v DATASET=${DATASET},SID_HIERARCHIES=${SID_HIERARCHIES},SID_METHODS=${SID_METHODS_VALUE},PER_ITEM_METHODS=${PER_ITEM_METHODS_VALUE},DIRECTIONS=${DIRECTIONS_VALUE},COMPARISON_ROOT=${COMPARISON_ROOT} \
        jobs/per_item_sid_comparison/run_aggregate_job.pbs)

    echo "Submitted final aggregation job ${DATASET}: ${AGGREGATE_JOB}"
    echo "Comparison root ${DATASET}: ${COMPARISON_ROOT}"
done
