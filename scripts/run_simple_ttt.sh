#!/usr/bin/env bash
# Closed-loop TTT on SIMPLE. Placeholder.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CONFIG="${CONFIG:-configs/ablations/ttt_fast_ff_decay.yaml}"
TASK="${TASK:-configs/tasks/simple_pick.yaml}"

echo "[run_simple_ttt] config=${CONFIG} task=${TASK}"
echo "[run_simple_ttt] TODO: wire up src/linexa/cli entrypoint."
