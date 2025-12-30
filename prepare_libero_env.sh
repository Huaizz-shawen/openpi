#!/bin/bash
set -e

cd /inspire/hdd/project/exploration-topic/public/zzhuai/openpi

# Create virtual environment with python 3.8
# Using absolute path for uv to be safe, though "uv" should be in PATH on bridge
/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin/uv venv --python 3.8 examples/libero/.venv

# Activate
source examples/libero/.venv/bin/activate

# Install dependencies
GIT_LFS_SKIP_SMUDGE=1 /inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin/uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

# Install headless opencv to avoid libGL issues
/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin/uv pip install opencv-python-headless

# Install local packages
/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin/uv pip install -e packages/openpi-client
/inspire/hdd/project/exploration-topic/public/zzhuai/.local/bin/uv pip install -e third_party/libero
