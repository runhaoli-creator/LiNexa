#!/usr/bin/env bash
# Baseline run (no TTT). Placeholder.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

CONFIG="${CONFIG:-configs/ablations/no_ttt.yaml}"

echo "[run_baseline] config=${CONFIG}"
echo "[run_baseline] TODO: wire up src/linexa/cli entrypoint."
