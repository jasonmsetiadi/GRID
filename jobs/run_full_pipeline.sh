#!/bin/bash
set -e

# load python and cuda
module load python/3.10.8
module load cuda/12.1.0
module load gcc/12.2.0

# Create python environment
SCRATCH_DIR=/srv/scratch/z5611110/
cd $SCRATCH_DIR
source .venv/bin/activate

cd GRID

# Dataset and SID method are expected to be set by the caller (e.g., PBS job).
# Defaults are provided for local/manual runs.
DATASET=${DATASET:-beauty}
SID_METHOD=${SID_METHOD:-rkmeans}       # options: rkmeans, rvq, rqvae
SEMANTIC_ID_MODE=${SEMANTIC_ID_MODE:-fixed} # options: fixed, variable
RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD:-0.05}
MIN_HIERARCHIES=${MIN_HIERARCHIES:-1}

EMBEDDING_DIM=2048
SID_HIERARCHIES=3
CODEBOOK_WIDTH=256
SEPARATOR_TOKEN=256

if [[ "${SEMANTIC_ID_MODE}" == "variable" ]]; then
    SID_HIERARCHIES=10
fi

TIGER_HIERARCHIES=$((SID_HIERARCHIES + 1))

# Helper to print elapsed time in HH:MM:SS given a start timestamp (seconds since epoch).
log_duration() {
    local start_time=$1
    local label=$2
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%s took %02d:%02d:%02d\n" "${label}" ${hours} ${minutes} ${seconds}
}

PIPELINE_START=$(date +%s)

# Step 2: Generate embeddings
# -----------------------------------------------------------------------------
STEP_START=$(date +%s)
STEP2_DIR=logs/step2_${DATASET}_embeddings
echo "Running Step 2: Embedding generation -> ${STEP2_DIR}"
# python -m src.inference experiment=sem_embeds_inference_flat \
#     data_dir=data/amazon_data/${DATASET} \
#     hydra.run.dir=${STEP2_DIR}

EMBEDDING_PATH=${STEP2_DIR}/pickle/merged_predictions_tensor.pt

echo "Embedding path: ${EMBEDDING_PATH}"
log_duration ${STEP_START} "Step 2"

# Step 3: Train and generate semantic IDs
# -----------------------------------------------------------------------------
STEP_START=$(date +%s)
echo "Running Step 3a: Train semantic IDs using ${SID_METHOD}"
python -m src.train experiment=${SID_METHOD}_train_flat \
    data_dir=data/amazon_data/${DATASET} \
    embedding_path=${EMBEDDING_PATH} \
    embedding_dim=${EMBEDDING_DIM} \
    num_hierarchies=${SID_HIERARCHIES} \
    codebook_width=${CODEBOOK_WIDTH} \
    semantic_id_mode=${SEMANTIC_ID_MODE} \
    residual_threshold=${RESIDUAL_THRESHOLD} \
    min_hierarchies=${MIN_HIERARCHIES} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest checkpoint from Step 3a under this method's log dir.
SID_CKPT=$(find logs/${DATASET}/${SID_METHOD}/train/runs -name "checkpoint_*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Step 3a complete. Latest checkpoint: ${SID_CKPT}"
log_duration ${STEP_START} "Step 3a"

STEP_START=$(date +%s)
echo "Running Step 3b: Generate semantic IDs"
python -m src.inference experiment=${SID_METHOD}_inference_flat \
    data_dir=data/amazon_data/${DATASET} \
    embedding_path=${EMBEDDING_PATH} \
    embedding_dim=${EMBEDDING_DIM} \
    num_hierarchies=${SID_HIERARCHIES} \
    codebook_width=${CODEBOOK_WIDTH} \
    ckpt_path=${SID_CKPT} \
    semantic_id_mode=${SEMANTIC_ID_MODE} \
    residual_threshold=${RESIDUAL_THRESHOLD} \
    min_hierarchies=${MIN_HIERARCHIES} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest generated semantic IDs from Step 3b under this method's log dir.
SEMANTIC_ID_PATH=$(find logs/${DATASET}/${SID_METHOD}/inference/runs -name "merged_predictions_tensor.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Step 3b complete. Semantic ID path: ${SEMANTIC_ID_PATH}"
log_duration ${STEP_START} "Step 3b"

# Step 4: Train generative recommendation model
# -----------------------------------------------------------------------------
STEP_START=$(date +%s)
echo "Running Step 4 (train): Train generative recommendation model"
python -m src.train experiment=tiger_train_flat \
    data_dir=data/amazon_data/${DATASET} \
    semantic_id_path=${SEMANTIC_ID_PATH} \
    num_hierarchies=${TIGER_HIERARCHIES} \
    semantic_id_mode=${SEMANTIC_ID_MODE} \
    separator_token=${SEPARATOR_TOKEN} \
    min_hierarchies=${MIN_HIERARCHIES} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest checkpoint from Step 4 under this method's log dir.
TIGER_CKPT=$(find logs/${DATASET}/${SID_METHOD}/train/runs -name "checkpoint_*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Step 4 (train) complete. Latest checkpoint: ${TIGER_CKPT}"
log_duration ${STEP_START} "Step 4 (train)"

# Step 4: Generate recommendations (timestamped directory)
# -----------------------------------------------------------------------------
STEP_START=$(date +%s)
echo "Running Step 4 (inference): Generate recommendations"
python -m src.inference experiment=tiger_inference_flat \
    data_dir=data/amazon_data/${DATASET} \
    semantic_id_path=${SEMANTIC_ID_PATH} \
    ckpt_path=${TIGER_CKPT} \
    num_hierarchies=${TIGER_HIERARCHIES} \
    semantic_id_mode=${SEMANTIC_ID_MODE} \
    separator_token=${SEPARATOR_TOKEN} \
    min_hierarchies=${MIN_HIERARCHIES} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest recommendation output under this method's log dir.
REC_OUTPUT=$(find logs/${DATASET}/${SID_METHOD}/inference/runs -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Pipeline complete. Recommendation output: ${REC_OUTPUT}"
log_duration ${STEP_START} "Step 4 (inference)"

log_duration ${PIPELINE_START} "Total pipeline"
