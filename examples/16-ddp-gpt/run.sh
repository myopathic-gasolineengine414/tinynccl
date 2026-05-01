#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ~/tinynccl/.venv/bin/activate
backend="${1:-verbs}"
steps="${2:-3000}"
python3 train.py 0 127.0.0.1 "$backend" "$steps" > /tmp/r0.log 2>&1 &
PID=$!
sleep 1.5
python3 train.py 1 127.0.0.1 "$backend" "$steps"
wait $PID
echo ""
echo "--- rank 0 ---"
cat /tmp/r0.log
