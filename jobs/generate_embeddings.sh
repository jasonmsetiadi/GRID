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

# Dataset can be passed via environment variable; defaults to beauty for manual runs.
DATASET=${DATASET:-beauty}

STEP2_DIR=logs/step2_${DATASET}_embeddings
echo "Generating embeddings for dataset: ${DATASET} -> ${STEP2_DIR}"

python -m src.inference experiment=sem_embeds_inference_flat \
    data_dir=data/amazon_data/${DATASET} \
    hydra.run.dir=${STEP2_DIR}

echo "Embeddings saved at: ${STEP2_DIR}/pickle/merged_predictions_tensor.pt"
