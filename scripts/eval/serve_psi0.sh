#!/usr/bin/env bash
# Start the psi0 inference server via docker compose (detached).
#
# Prereqs:
#   1. `docker/.env` filled in (HF_TOKEN, PSI0_RUN_DIR, PSI0_CKPT_STEP).
#       PSI0_RUN_DIR + step are normally populated by
#       scripts/eval/download_psi0_ckpt.sh (which calls find_run_dir.sh).
#   2. `psi0:latest` image present on the host (already true on this DGX).
#
# The server listens on ${PSI0_SERVER_PORT:-22085}.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
export LINEXA_ROOT="${ROOT}"

if [ ! -f docker/.env ]; then
  echo "[serve_psi0] docker/.env not found — copy docker/.env.sample first." >&2
  exit 1
fi

if ! docker image inspect psi0:latest >/dev/null 2>&1; then
  echo "[serve_psi0] docker image 'psi0:latest' not found on this host." >&2
  echo "[serve_psi0] See README 'Notes on custom Docker work' for how to build it." >&2
  exit 1
fi

set -a; . docker/.env; set +a
: "${PSI0_RUN_DIR:?PSI0_RUN_DIR is not set in docker/.env — run scripts/eval/find_run_dir.sh}"
: "${PSI0_CKPT_STEP:?PSI0_CKPT_STEP is not set in docker/.env — run scripts/eval/find_run_dir.sh}"

docker compose -f docker/docker-compose.yml --env-file docker/.env up -d psi0-server

PORT="${PSI0_SERVER_PORT:-22085}"
echo "[serve_psi0] psi0-server up. port=${PORT}  run=${PSI0_RUN_DIR}  step=${PSI0_CKPT_STEP}"
echo "[serve_psi0] follow startup:  docker compose -f docker/docker-compose.yml logs -f psi0-server"
echo "[serve_psi0] health (after boot):  curl -s http://localhost:${PORT}/health"
