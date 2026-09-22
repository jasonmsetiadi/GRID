#!/bin/bash
set -euo pipefail

module load python/3.10.8
module load cuda/12.1.0
module load gcc/12.2.0

cd /srv/scratch/z5611110
source .venv/bin/activate
cd GRID

DATASET=${DATASET:-beauty}
SID_METHOD=${SID_METHOD:-rkmeans}
SIGNAL=${SIGNAL:?SIGNAL is required}
LENGTH_DIRECTION=${LENGTH_DIRECTION:?LENGTH_DIRECTION is required}
COMPARISON_ROOT=${COMPARISON_ROOT:?COMPARISON_ROOT is required}
EMBEDDING_PATH=${EMBEDDING_PATH:-logs/step2_${DATASET}_embeddings/pickle/merged_predictions_tensor.pt}
SID_HIERARCHIES=${SID_HIERARCHIES:-10}
TIGER_HIERARCHIES=${TIGER_HIERARCHIES:-${SID_HIERARCHIES}}
CODEBOOK_WIDTH=${CODEBOOK_WIDTH:-256}
EMBEDDING_DIM=${EMBEDDING_DIM:-2048}
MIN_SID_LENGTH=${MIN_SID_LENGTH:-1}

TOKENIZER_DIR=${COMPARISON_ROOT}/tokenizer
TOKENIZER_LOCK=${TOKENIZER_DIR}/.training.lock
LENGTH_PATH=${COMPARISON_ROOT}/lengths/item_lengths_${SIGNAL}_${LENGTH_DIRECTION}.pt
SIGNAL_DIR=${COMPARISON_ROOT}/${SID_METHOD}/${SIGNAL}_${LENGTH_DIRECTION}

mkdir -p "${TOKENIZER_DIR}"
latest_checkpoint() {
    find "$1/train/runs" -name "checkpoint_*.ckpt" -type f -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | cut -d' ' -f2-
}

TOKENIZER_CHECKPOINT=$(latest_checkpoint "${TOKENIZER_DIR}" || true)
if [[ -z "${TOKENIZER_CHECKPOINT}" ]]; then
    if mkdir "${TOKENIZER_LOCK}" 2>/dev/null; then
        trap 'rmdir "${TOKENIZER_LOCK}" 2>/dev/null || true' EXIT
        TOKENIZER_CHECKPOINT=$(latest_checkpoint "${TOKENIZER_DIR}" || true)
        if [[ -z "${TOKENIZER_CHECKPOINT}" ]]; then
            python -m src.train experiment=${SID_METHOD}_train_flat \
                data_dir=data/amazon_data/${DATASET} \
                embedding_path=${EMBEDDING_PATH} \
                embedding_dim=${EMBEDDING_DIM} \
                num_hierarchies=${SID_HIERARCHIES} \
                codebook_width=${CODEBOOK_WIDTH} \
                paths.log_dir=${TOKENIZER_DIR}
        fi
    else
        until [[ -n "${TOKENIZER_CHECKPOINT}" ]]; do
            sleep 15
            TOKENIZER_CHECKPOINT=$(latest_checkpoint "${TOKENIZER_DIR}" || true)
        done
    fi
fi
TOKENIZER_CHECKPOINT=${TOKENIZER_CHECKPOINT:-$(latest_checkpoint "${TOKENIZER_DIR}")}
if [[ -z "${TOKENIZER_CHECKPOINT}" ]]; then
    echo "Unable to locate shared RKMeans tokenizer checkpoint" >&2
    exit 1
fi

python -m src.inference experiment=${SID_METHOD}_inference_flat \
    data_dir=data/amazon_data/${DATASET} \
    embedding_path=${EMBEDDING_PATH} \
    embedding_dim=${EMBEDDING_DIM} \
    num_hierarchies=${SID_HIERARCHIES} \
    codebook_width=${CODEBOOK_WIDTH} \
    ckpt_path=${TOKENIZER_CHECKPOINT} \
    model.variable_length=true \
    model.length_selection_mode=per_item \
    model.length_selection_method=${SIGNAL} \
    model.min_sid_length=${MIN_SID_LENGTH} \
    model.max_sid_length=${SID_HIERARCHIES} \
    model.item_lengths_path=${LENGTH_PATH} \
    paths.log_dir=${SIGNAL_DIR}

SID_ARTIFACT=$(find "${SIGNAL_DIR}/inference/runs" -name "merged_predictions_tensor.pt" -type f \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -z "${SID_ARTIFACT}" ]]; then
    echo "Unable to locate SID artifact for ${SIGNAL}/${LENGTH_DIRECTION}" >&2
    exit 1
fi

python -m src.train experiment=tiger_varlen_train_flat \
    data_dir=data/amazon_data/${DATASET} \
    semantic_id_path=${SID_ARTIFACT} \
    num_hierarchies=${TIGER_HIERARCHIES} \
    paths.log_dir=${SIGNAL_DIR}

TIGER_CHECKPOINT=$(latest_checkpoint "${SIGNAL_DIR}" || true)
if [[ -z "${TIGER_CHECKPOINT}" ]]; then
    echo "Unable to locate TIGER checkpoint for ${SIGNAL}/${LENGTH_DIRECTION}" >&2
    exit 1
fi

python -m src.inference experiment=tiger_varlen_inference_flat \
    data_dir=data/amazon_data/${DATASET} \
    semantic_id_path=${SID_ARTIFACT} \
    ckpt_path=${TIGER_CHECKPOINT} \
    num_hierarchies=${TIGER_HIERARCHIES} \
    paths.log_dir=${SIGNAL_DIR}
