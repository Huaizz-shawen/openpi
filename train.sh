#!/bin/bash
set -e

# Navigate to project directory
cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

# Add uv to PATH
export PATH="/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin:$PATH"

# Run training with debug config
# Note: wandb_enabled is already False in the 'debug' config definition
uv run scripts/train.py debug --exp_name debug_final_sh
