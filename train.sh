#!/bin/bash
set -e

# Navigate to project directory
cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

# Offline flags for compute nodes
export LEROBOT_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Check if dataset exists
DATASET_PATH="/inspire/hdd/project/exploration-topic/public/zzhuai/data/1006_200s_v6_fold_the_sleeve"
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

# Compute normalization statistics
echo "Computing normalization statistics..."
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_fold_sleeve

# Run training
echo "Starting training..."
.venv/bin/python scripts/train.py pi05_fold_sleeve --exp_name agilex_fold_sleeve --overwrite --num_workers 0
