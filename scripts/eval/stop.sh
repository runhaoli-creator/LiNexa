#!/usr/bin/env bash
# Stop the psi0 server (and any other LiNexa eval compose services).
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
export LINEXA_ROOT="${ROOT}"

docker compose -f docker/docker-compose.yml --env-file docker/.env down "$@"
