#!/usr/bin/env bash
# Open-loop TTT evaluation. Placeholder.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CONFIG="${CONFIG:-configs/ablations/ttt_fast_ff.yaml}"

echo "[run_open_loop_ttt] config=${CONFIG}"
echo "[run_open_loop_ttt] TODO: wire up src/linexa/cli entrypoint."
