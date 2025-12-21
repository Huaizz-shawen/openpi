#!/bin/bash
set -e

# Navigate to project directory
cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

# Add uv to PATH
export PATH="/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin:$PATH"

# Run training with debug config in offline mode
export UV_OFFLINE=1
uv run --offline scripts/train.py debug --exp_name debug_final_sh
