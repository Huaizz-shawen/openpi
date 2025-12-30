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
SERVER_LOG="policy_server.log"

echo "Starting policy server..."
# Use the main venv for the server (Python 3.11)
.venv/bin/python scripts/serve_policy.py policy:checkpoint \
    --policy.config=$POLICY_CONFIG \
    --policy.dir=$CHECKPOINT_DIR \
    --host 0.0.0.0 --port 8000 > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo "Policy server started with PID $SERVER_PID. Logs at $SERVER_LOG"

# Wait for server to be ready
echo "Waiting for server to initialize..."
sleep 60

echo "Starting Libero evaluation..."
# Use the Libero venv for the client (Python 3.8)
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT/third_party/libero

# Run Libero client
# Running libero_spatial task suite
python examples/libero/main.py --args.task-suite-name libero_spatial

EXIT_CODE=$?

# Kill server
echo "Stopping policy server..."
kill $SERVER_PID || true

exit $EXIT_CODE
