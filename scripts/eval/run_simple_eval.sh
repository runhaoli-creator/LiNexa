#!/usr/bin/env bash
# Run one psi0 eval task against a running psi0-server.
#
# Auto-dispatch by task suffix:
#   *Teleop-v0 → simple.cli.eval_decoupled_wbc  +  psi0_decoupled_wbc
#   *MP-v0     → simple.cli.eval                +  psi0
#
# Required:
#   TASK         e.g. G1WholebodyXMovePickTeleop-v0
#
# Optional env:
#   DR                 level-0|level-1|level-2          (default level-0)
#   NUM_EPISODES       number of episodes               (default 10)
#   MAX_EPISODE_STEPS  override task default            (unset)
#   PSI0_HOST          server host                      (default 127.0.0.1)
#   PSI0_PORT          server port                      (default from .env or 22085)
#   HEADLESS           1|0                              (default 1)
#   EXTRA_EVAL_ARGS    extra CLI args appended verbatim (unset)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"
export LINEXA_ROOT="${ROOT}"

if [ ! -f docker/.env ]; then
  echo "[run_simple_eval] docker/.env not found — copy docker/.env.sample first." >&2
  exit 1
fi
set -a; . docker/.env; set +a

TASK="${TASK:?Set TASK, e.g. G1WholebodyXMovePickTeleop-v0}"
DR="${DR:-level-0}"
NUM_EPISODES="${NUM_EPISODES:-10}"
PSI0_HOST="${PSI0_HOST:-127.0.0.1}"
PSI0_PORT="${PSI0_PORT:-${PSI0_SERVER_PORT:-22085}}"
HEADLESS="${HEADLESS:-1}"

if [[ "${TASK}" == *Teleop-v0 ]]; then
  MODULE="simple.cli.eval_decoupled_wbc"
  AGENT="psi0_decoupled_wbc"
elif [[ "${TASK}" == *MP-v0 ]]; then
  MODULE="simple.cli.eval"
  AGENT="psi0"
else
  echo "[run_simple_eval] can't infer entrypoint for TASK='${TASK}' (expect *Teleop-v0 or *MP-v0)." >&2
  exit 1
fi

DATA_REL="data/evals/simple-eval/${TASK}/${DR}"
if [ ! -d "${DATA_REL}" ]; then
  echo "[run_simple_eval] ${DATA_REL} missing — run scripts/eval/download_eval_data.sh first." >&2
  exit 1
fi

if ! docker image inspect simple:latest >/dev/null 2>&1; then
  echo "[run_simple_eval] docker image 'simple:latest' not found on this host." >&2
  echo "[run_simple_eval] See README 'Notes on custom Docker work' for how to build it." >&2
  exit 1
fi

mkdir -p logs/eval

headless_flag=()
[ "${HEADLESS}" = "1" ] && headless_flag+=(--headless)

extra=()
[ -n "${MAX_EPISODE_STEPS:-}" ] && extra+=(--max-episode-steps "${MAX_EPISODE_STEPS}")
if [ -n "${EXTRA_EVAL_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  extra+=(${EXTRA_EVAL_ARGS})
fi

echo "[run_simple_eval] task=${TASK} dr=${DR} module=${MODULE} agent=${AGENT}"
echo "[run_simple_eval] psi0 server=${PSI0_HOST}:${PSI0_PORT}  episodes=${NUM_EPISODES}"

docker compose -f docker/docker-compose.yml --env-file docker/.env run --rm \
  simple-eval \
  /workspace/SIMPLE/.venv/bin/python -m "${MODULE}" \
    "simple/${TASK}" "${AGENT}" "${DR}" \
    --host "${PSI0_HOST}" --port "${PSI0_PORT}" \
    --sim-mode mujoco_isaac \
    --data-format lerobot \
    --data-dir "${DATA_REL}" \
    --num-episodes "${NUM_EPISODES}" \
    "${headless_flag[@]}" \
    "${extra[@]}"

echo "[run_simple_eval] outputs under logs/eval/${AGENT}/${TASK}/"
