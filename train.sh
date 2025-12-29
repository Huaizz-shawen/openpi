#!/bin/bash
set -e

# Navigate to project directory
cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

# Offline flags for compute nodes
export LEROBOT_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

# Dataset path
DATASET_PATH="/inspire/hdd/project/exploration-topic/public/zzhuai/data/libero"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Config name
CONFIG_NAME="pi05_libero"
EXP_NAME="libero_finetune"

# Compute normalization statistics
echo "Computing normalization statistics..."
.venv/bin/python scripts/compute_norm_stats.py --config-name $CONFIG_NAME --data.repo_id $DATASET_PATH

# Run training
echo "Starting training..."
.venv/bin/python scripts/train.py $CONFIG_NAME --exp_name $EXP_NAME --overwrite --num_workers 0 --data.repo_id $DATASET_PATH