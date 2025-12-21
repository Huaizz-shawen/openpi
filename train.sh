#!/bin/bash
set -e

# Navigate to project directory
cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

# Run training with debug config using the pre-installed venv python
.venv/bin/python scripts/train.py debug --exp_name debug_final_sh --num_workers 0
