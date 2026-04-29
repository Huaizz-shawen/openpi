#!/bin/bash
set -euo pipefail

cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

export LEROBOT_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE="${WANDB_MODE:-offline}"

DATASET_PATH="/inspire/hdd/global_user/gongjingjing-25039/lqyin/widowX_dataset/lerobot_2_1/bag"
CONFIG_NAME="qsl_train_widowx_coffee_bean"
EXP_NAME="${EXP_NAME:-QSL_TRAIN_widowx_coffee_bean_30k}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
NUM_WORKERS="${NUM_WORKERS:-64}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
DECAY_STEPS="${DECAY_STEPS:-1000000}"
NORM_STATS_PATH="/inspire/hdd/project/exploration-topic/public/zzhuai/openpi/assets/${CONFIG_NAME}/widowx_bag/norm_stats.json"

if [ ! -d "$DATASET_PATH" ]; then
  echo "Error: Dataset not found at $DATASET_PATH"
  exit 1
fi

if [ ! -f "$NORM_STATS_PATH" ]; then
  echo "norm_stats.json missing at $NORM_STATS_PATH, computing from config $CONFIG_NAME ..."
  .venv/bin/python scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi

echo "Starting training with $CONFIG_NAME on $DATASET_PATH ..."
echo "Visual input: top camera only (wrist views masked out)"
.venv/bin/python scripts/train.py "$CONFIG_NAME" \
  --exp_name "$EXP_NAME" \
  --overwrite \
  --num_workers "$NUM_WORKERS" \
  --fsdp_devices "$FSDP_DEVICES" \
  --batch_size "$BATCH_SIZE" \
  --num_train_steps "$NUM_TRAIN_STEPS" \
  --lr-schedule.decay-steps "$DECAY_STEPS" \
  --save_interval "$SAVE_INTERVAL" \
  --data.repo-id "$DATASET_PATH"
