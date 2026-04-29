#!/bin/bash
set -e

cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

export LEROBOT_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline

DATASET_PATH="/inspire/hdd/project/exploration-topic/public/zzhuai/openpi/data/ur5_tabletop_3obj_frontleft_train_video"
if [ ! -d "$DATASET_PATH" ]; then
  echo "Error: Dataset not found at $DATASET_PATH"
  exit 1
fi

CONFIG_NAME="pi05_ur5e_tabletop_3obj_frontleft_local"
EXP_NAME="ur5e_tabletop_frontleft_40k_4gpu"

if [ ! -f "$DATASET_PATH/norm_stats.json" ]; then
  echo "norm_stats.json missing, computing from config $CONFIG_NAME ..."
  .venv/bin/python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

echo "Starting training with $CONFIG_NAME on $DATASET_PATH ..."
.venv/bin/python scripts/train.py "$CONFIG_NAME" \
  --exp_name "$EXP_NAME" \
  --overwrite \
  --num_workers 64 \
  --fsdp_devices 4 \
  --num_train_steps 40000 \
  --lr-schedule.decay-steps 40000 \
  --save_interval 10000 \
  --data.repo-id "$DATASET_PATH"
