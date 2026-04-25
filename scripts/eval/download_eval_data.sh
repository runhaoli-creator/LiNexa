#!/usr/bin/env bash
# Download a SIMPLE eval task bundle from HuggingFace into
# $LINEXA_ROOT/data/evals/simple-eval/<task>/<dr>/.
#
# Runs `hf download` inside the psi0:latest container so no host-level
# Python/pip is needed.
#
# Usage:
#   TASK=G1WholebodyXMovePickTeleop-v0 bash scripts/eval/download_eval_data.sh
#
# Env:
#   TASK        required. SIMPLE task id (e.g. G1WholebodyXMovePickTeleop-v0).
#   HF_TOKEN    optional. Falls back to docker/.env.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
export LINEXA_ROOT="${ROOT}"

TASK="${TASK:?Set TASK, e.g. G1WholebodyXMovePickTeleop-v0}"

mkdir -p "${ROOT}/data/evals/simple-eval"

echo "[download_eval_data] task=${TASK} → data/evals/simple-eval/${TASK}/"

# 1. Pull the zip via `hf` inside the psi0:latest container. --local-dir is
#    relative to the compose volume mount (./data → /workspace/data).
docker compose -f docker/docker-compose.yml --env-file docker/.env run --rm \
  -u "$(id -u):$(id -g)" \
  hf-cli \
    download USC-PSI-Lab/psi-data \
    "simple-eval/${TASK}.zip" \
    --local-dir /workspace/data/evals \
    --repo-type dataset

# 2. Extract on the host (unzip is standard; avoids another container round-trip).
if ! command -v unzip >/dev/null 2>&1; then
  echo "[download_eval_data] host is missing 'unzip'; extracting inside Docker instead."
  docker compose -f docker/docker-compose.yml --env-file docker/.env run --rm \
    --entrypoint /bin/bash -u "$(id -u):$(id -g)" \
    hf-cli \
    -c "cd /workspace/data/evals && apt-get >/dev/null 2>&1 || true; \
        python3 -c 'import zipfile, pathlib; \
z=zipfile.ZipFile(\"simple-eval/${TASK}.zip\"); \
z.extractall(\"simple-eval\"); \
print(\"extracted\")'"
else
  unzip -o "data/evals/simple-eval/${TASK}.zip" -d "data/evals/simple-eval"
fi

echo "[download_eval_data] done. Available levels under data/evals/simple-eval/${TASK}/:"
ls "data/evals/simple-eval/${TASK}/" 2>/dev/null | sed 's/^/  /'
