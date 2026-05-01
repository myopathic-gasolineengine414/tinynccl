#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source ~/tinynccl/.venv/bin/activate
backend="${1:-tcp}"
python3 train.py 0 127.0.0.1 "$backend" > /tmp/r0.log 2>&1 &
PID=$!
sleep 0.5
python3 train.py 1 127.0.0.1 "$backend"
wait $PID
echo "--- rank 0 ---"
cat /tmp/r0.log
