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
CONFIG_NAME="pi05_libero_local"
EXP_NAME="libero_finetune"

# Run training



echo "Starting training..."



.venv/bin/python scripts/train.py $CONFIG_NAME --exp_name $EXP_NAME --overwrite --num_workers 64 --fsdp_devices 1