#!/bin/bash
set -e

# Submit fixed and variable comparison jobs for beauty + RQ-VAE.
# Run this script from the jobs directory.

DATASET=${DATASET:-beauty}
SID_METHOD=${SID_METHOD:-rqvae}
RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD:-0.05}
MIN_HIERARCHIES=${MIN_HIERARCHIES:-1}

echo "Submitting variable-length pipeline: dataset=${DATASET}, method=${SID_METHOD}"
PIPELINE_JOB=$(qsub \
    -N gr_variable_${SID_METHOD}_${DATASET} \
    -v DATASET=${DATASET},SID_METHOD=${SID_METHOD},SEMANTIC_ID_MODE=variable,RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD},MIN_HIERARCHIES=${MIN_HIERARCHIES} \
    run_pipeline.pbs)

echo "Variable-length pipeline submitted: ${PIPELINE_JOB}"

if [[ "${COMPARE_FIXED:-1}" == "1" ]]; then
    echo "Submitting fixed-length baseline: dataset=${DATASET}, method=${SID_METHOD}"
    FIXED_JOB=$(qsub \
        -N gr_fixed_${SID_METHOD}_${DATASET} \
        -v DATASET=${DATASET},SID_METHOD=${SID_METHOD},SEMANTIC_ID_MODE=fixed \
        run_pipeline.pbs)
    echo "Fixed-length baseline submitted: ${FIXED_JOB}"
fi
