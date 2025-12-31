#!/bin/bash
set -e

# Project root
PROJECT_ROOT="/inspire/hdd/project/exploration-topic/public/zzhuai/openpi"
cd $PROJECT_ROOT

# Checkpoint path
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints/pi05_libero_local/libero_finetune/29999"

# Policy config
POLICY_CONFIG="pi05_libero_local"

# Server log
SERVER_LOG="policy_server_goal.log"

echo "Starting policy server..."
# Use the main venv for the server (Python 3.11)
.venv/bin/python scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config=$POLICY_CONFIG \
    --policy.dir=$CHECKPOINT_DIR \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Policy server started with PID $SERVER_PID. Logs at $SERVER_LOG"

# Wait for server to be ready
echo "Waiting for server to initialize..."
sleep 60

echo "Starting Libero evaluation (Goal)..."
# Use the Libero venv for the client (Python 3.8)
source examples/libero/.venv/bin/activate

# Reset PYTHONPATH to prevent system packages from interfering
export PYTHONPATH="$PROJECT_ROOT/third_party/libero"

# Clean LD_LIBRARY_PATH to remove system python 3.10 torch libs which cause conflicts
if [ -n "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/usr/local/lib/python3.10/dist-packages/torch/lib:||g')
    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's|/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:||g')
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH%:} 
fi

# Setup Libero config to avoid interactive prompt
export LIBERO_CONFIG_PATH="$HOME/.libero"
mkdir -p "$LIBERO_CONFIG_PATH"
CONFIG_FILE="$LIBERO_CONFIG_PATH/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating default Libero config at $CONFIG_FILE"
    python3 -c "
import os
import yaml

benchmark_root = '$PROJECT_ROOT/third_party/libero/libero/libero'
config_file = '$CONFIG_FILE'

config = {
    'benchmark_root': benchmark_root,
    'bddl_files': os.path.join(benchmark_root, 'bddl_files'),
    'init_states': os.path.join(benchmark_root, 'init_files'),
    'datasets': os.path.join(benchmark_root, '../datasets'),
    'assets': os.path.join(benchmark_root, 'assets'),
}

with open(config_file, 'w') as f:
    yaml.dump(config, f)
"
fi

# Run Libero client
# Running libero_goal task suite
# Use 1 trial per task for debugging
export MUJOCO_GL=egl
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

python examples/libero/main.py --args.task-suite-name libero_goal --args.host 127.0.0.1 --args.num-trials-per-task 1

EXIT_CODE=$?

# Kill server
echo "Stopping policy server..."
kill $SERVER_PID || true

exit $EXIT_CODE
