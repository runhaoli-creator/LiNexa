#!/usr/bin/env bash
# Download psi0 SIMPLE checkpoints into $LINEXA_ROOT/checkpoints/psi0/.
# Runs inside psi0:latest (ships `hf`) — no host Python required.
#
# Usage:
#   bash scripts/eval/download_psi0_ckpt.sh                         # full simple-checkpoints/
#   REMOTE_SUBDIR=psi0/simple-checkpoints/<task> bash ...           # narrower
#
# Env:
#   REMOTE_SUBDIR  default: psi0/simple-checkpoints
#   HF_TOKEN       optional. Falls back to docker/.env.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
export LINEXA_ROOT="${ROOT}"

REMOTE_SUBDIR="${REMOTE_SUBDIR:-psi0/simple-checkpoints}"

mkdir -p "${ROOT}/checkpoints/psi0"

echo "[download_psi0_ckpt] ${REMOTE_SUBDIR}/* → checkpoints/${REMOTE_SUBDIR}/"

docker compose -f docker/docker-compose.yml --env-file docker/.env run --rm \
  -u "$(id -u):$(id -g)" \
  hf-cli \
    download USC-PSI-Lab/psi-model \
    --include "${REMOTE_SUBDIR}/*" \
    --local-dir /workspace/checkpoints \
    --repo-type model

echo "[download_psi0_ckpt] done. Running find_run_dir to populate docker/.env..."
bash scripts/eval/find_run_dir.sh
