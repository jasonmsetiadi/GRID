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
VARIABLE_LENGTH_SID=${VARIABLE_LENGTH_SID:-false}
RESIDUAL_THRESHOLD=${RESIDUAL_THRESHOLD:-0.1}
MIN_SID_LENGTH=${MIN_SID_LENGTH:-1}
LENGTH_SELECTION_MODE=${LENGTH_SELECTION_MODE:-content_based}
LENGTH_SELECTION_METHOD=${LENGTH_SELECTION_METHOD:-residual_threshold}
ITEM_LENGTHS_PATH=${ITEM_LENGTHS_PATH:-}
LENGTH_DIRECTION=${LENGTH_DIRECTION:-direct}

if [[ "${LENGTH_SELECTION_MODE}" == "per_item" ]]; then
    VARIABLE_LENGTH_SID=true
fi

EMBEDDING_DIM=2048
SID_HIERARCHIES=${SID_HIERARCHIES:-3}
TIGER_HIERARCHIES=${TIGER_HIERARCHIES:-4}
CODEBOOK_WIDTH=256

if [[ "${LENGTH_SELECTION_MODE}" == "per_item" ]]; then
    ITEM_LENGTHS_PATH="logs/${DATASET}/${SID_METHOD}/item_lengths_${LENGTH_SELECTION_METHOD}_${LENGTH_DIRECTION}.pt"
    mkdir -p "$(dirname "${ITEM_LENGTHS_PATH}")"
    python -m src.utils.item_length_generation \
        --data-dir=data/amazon_data/${DATASET} \
        --method=${LENGTH_SELECTION_METHOD} \
        --min-length=${MIN_SID_LENGTH} \
        --max-length=${SID_HIERARCHIES} \
        --direction=${LENGTH_DIRECTION} \
        --output-path="${ITEM_LENGTHS_PATH}"
fi

if [[ "${VARIABLE_LENGTH_SID}" == "true" ]]; then
    TIGER_HIERARCHIES=${SID_HIERARCHIES}
fi

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
    model.variable_length=${VARIABLE_LENGTH_SID} \
    model.residual_threshold=${RESIDUAL_THRESHOLD} \
    model.min_sid_length=${MIN_SID_LENGTH} \
    model.max_sid_length=${SID_HIERARCHIES} \
    model.length_selection_mode=${LENGTH_SELECTION_MODE} \
    model.length_selection_method=${LENGTH_SELECTION_METHOD} \
    model.item_lengths_path=${ITEM_LENGTHS_PATH:-null} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest generated semantic IDs from Step 3b under this method's log dir.
SEMANTIC_ID_PATH=$(find logs/${DATASET}/${SID_METHOD}/inference/runs -name "merged_predictions_tensor.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Step 3b complete. Semantic ID path: ${SEMANTIC_ID_PATH}"
log_duration ${STEP_START} "Step 3b"

# Step 4: Train generative recommendation model
# -----------------------------------------------------------------------------
STEP_START=$(date +%s)
echo "Running Step 4 (train): Train generative recommendation model"
TIGER_EXPERIMENT=tiger_train_flat
TIGER_INFERENCE_EXPERIMENT=tiger_inference_flat
if [[ "${VARIABLE_LENGTH_SID}" == "true" ]]; then
    TIGER_EXPERIMENT=tiger_varlen_train_flat
    TIGER_INFERENCE_EXPERIMENT=tiger_varlen_inference_flat
fi

python -m src.train experiment=${TIGER_EXPERIMENT} \
    data_dir=data/amazon_data/${DATASET} \
    semantic_id_path=${SEMANTIC_ID_PATH} \
    num_hierarchies=${TIGER_HIERARCHIES} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest checkpoint from Step 4 under this method's log dir.
TIGER_CKPT=$(find logs/${DATASET}/${SID_METHOD}/train/runs -name "checkpoint_*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Step 4 (train) complete. Latest checkpoint: ${TIGER_CKPT}"
log_duration ${STEP_START} "Step 4 (train)"

# Step 4: Generate recommendations (timestamped directory)
# -----------------------------------------------------------------------------
STEP_START=$(date +%s)
echo "Running Step 4 (inference): Generate recommendations"
python -m src.inference experiment=${TIGER_INFERENCE_EXPERIMENT} \
    data_dir=data/amazon_data/${DATASET} \
    semantic_id_path=${SEMANTIC_ID_PATH} \
    ckpt_path=${TIGER_CKPT} \
    num_hierarchies=${TIGER_HIERARCHIES} \
    paths.log_dir=logs/${DATASET}/${SID_METHOD}

# Locate the latest recommendation output under this method's log dir.
REC_OUTPUT=$(find logs/${DATASET}/${SID_METHOD}/inference/runs -name "*.pt" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

echo "Pipeline complete. Recommendation output: ${REC_OUTPUT}"
log_duration ${STEP_START} "Step 4 (inference)"

log_duration ${PIPELINE_START} "Total pipeline"
