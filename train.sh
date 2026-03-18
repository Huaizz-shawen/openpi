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
CONFIG_NAME="pi05_libero_plus"
EXP_NAME="libero_finetune_plus"

# Compute normalization statistics
# Since we reuse the dataset, we might already have stats. 
# But for a new config name, openpi might look in a different place?
# Actually, LeRobotLiberoDataConfig uses `assets_dir/repo_id`.
# With `pi05_libero_plus`, assets_dir is `assets/pi05_libero_plus`.
# So we need to compute stats for this new config or copy them.
# Recomputing is safer to ensure correct path.
echo "Computing normalization statistics..."
.venv/bin/python scripts/compute_norm_stats.py --config-name $CONFIG_NAME

# Run training

echo "Starting training..."

.venv/bin/python scripts/train.py $CONFIG_NAME --exp_name $EXP_NAME --overwrite --num_workers 64 --fsdp_devices 8