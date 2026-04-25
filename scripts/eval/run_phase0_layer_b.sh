#!/usr/bin/env bash
# Phase 0 Layer B orchestrator: baseline (LINEXA_TTT_ENABLED=0) -> linexa (=1, ΔW=0).
# Streams per-episode results to W&B via an ephemeral psi0:latest sidecar.
#
# Required env: WANDB_API_KEY
# Optional env: TS, TASK, DR, NUM_EPISODES, WANDB_PROJECT, GIT_COMMIT
set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
export LINEXA_ROOT="$ROOT"

: "${WANDB_API_KEY:?WANDB_API_KEY must be exported}"

TS="${TS:-$(date +%Y%m%d_%H%M)}"
TASK="${TASK:-G1WholebodyXMovePickTeleop-v0}"
DR="${DR:-level-0}"
NUM_EPISODES="${NUM_EPISODES:-20}"
WANDB_PROJECT="${WANDB_PROJECT:-linexa-phase0}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD)}"
LEGS="${LEGS:-baseline,linexa}"

CONSOLE_LOG="logs/eval/phase0_console_${TS}.log"
mkdir -p logs/eval reports

echo "[orch] ===== Phase 0 Layer B start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$CONSOLE_LOG"
echo "[orch] TS=$TS TASK=$TASK DR=$DR NUM_EPISODES=$NUM_EPISODES" | tee -a "$CONSOLE_LOG"
echo "[orch] commit=$GIT_COMMIT project=$WANDB_PROJECT" | tee -a "$CONSOLE_LOG"

set_env_var() {
  local key="$1" val="$2"
  if grep -q "^${key}=" docker/.env; then
    sed -i "s|^${key}=.*|${key}=${val}|" docker/.env
  else
    echo "${key}=${val}" >> docker/.env
  fi
}

run_one_leg() {
  local leg="$1"
  local enabled
  case "$leg" in
    baseline) enabled=0 ;;
    linexa)   enabled=1 ;;
    *) echo "[orch] bad leg: $leg" >&2; return 2 ;;
  esac

  echo "[orch] ===== leg=$leg enabled=$enabled $(date -u +%H:%M:%SZ) =====" | tee -a "$CONSOLE_LOG"

  set_env_var LINEXA_TTT_ENABLED "$enabled"
  set_env_var LINEXA_TTT_WRITE_SCALE "0.0"
  set_env_var LINEXA_TTT_DECAY "0.0"
  set_env_var LINEXA_TTT_CLIP "0.0"
  set_env_var LINEXA_TTT_LAYERS ""
  set_env_var LINEXA_TTT_LOG_STATS "0"

  bash scripts/eval/stop.sh 2>&1 | tee -a "$CONSOLE_LOG" || true
  bash scripts/eval/serve_psi0.sh 2>&1 | tee -a "$CONSOLE_LOG" || true

  local i healthy=0
  for i in $(seq 1 150); do
    if curl -sf http://localhost:22085/health >/dev/null 2>&1; then
      echo "[orch] server healthy after ${i}s" | tee -a "$CONSOLE_LOG"
      healthy=1
      break
    fi
    sleep 2
  done
  if [ "$healthy" != "1" ]; then
    echo "[orch] FATAL server not healthy after 300s" | tee -a "$CONSOLE_LOG"
    docker logs --tail 200 linexa-psi0-server 2>&1 | tee -a "$CONSOLE_LOG"
    return 1
  fi

  if [ "$enabled" = "1" ]; then
    local install_line
    install_line=$(docker logs linexa-psi0-server 2>&1 | grep -E "linexa: installed FastFFWrapper" | tail -1 || true)
    if [ -z "$install_line" ]; then
      echo "[orch] FATAL linexa enabled but no install line in server log" | tee -a "$CONSOLE_LOG"
      docker logs --tail 200 linexa-psi0-server 2>&1 | tee -a "$CONSOLE_LOG"
      return 1
    fi
    echo "[orch] $install_line" | tee -a "$CONSOLE_LOG"
  fi

  # OpenAPI schema sanity: /act must accept a request body (payload as Pydantic body),
  # not a query parameter. If our subclass dropped the annotation, FastAPI treats
  # `payload` as a query param and every /act call returns 422.
  local schema_ok
  schema_ok=$(curl -s http://localhost:22085/openapi.json | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin)
    post=d.get("paths",{}).get("/act",{}).get("post",{})
    body_present=bool(post.get("requestBody"))
    payload_in_query=any(p.get("in")=="query" and p.get("name")=="payload" for p in post.get("parameters",[]))
    print("ok" if body_present and not payload_in_query else "BAD")
except Exception as e:
    print(f"BAD:{e}")
' 2>/dev/null)
  echo "[orch] /act schema check: $schema_ok" | tee -a "$CONSOLE_LOG"
  if [ "$schema_ok" != "ok" ]; then
    echo "[orch] FATAL /act schema is wrong (payload not in body)." | tee -a "$CONSOLE_LOG"
    return 1
  fi

  local server_log="logs/eval/phase0_${leg}_server_${TS}.log"
  docker logs -f --since 1s linexa-psi0-server > "$server_log" 2>&1 &
  local logs_pid=$!

  local leg_start_ts
  leg_start_ts="$(date +%s.%N)"
  local done_flag_host="logs/eval/.${leg}_${TS}.done"
  rm -f "$done_flag_host"
  local wandb_name="phase0-${leg}-${TS}"
  local sidecar_name="linexa-wandb-${leg}-${TS}"

  echo "[orch] launching wandb sidecar=$sidecar_name leg_start_ts=$leg_start_ts" | tee -a "$CONSOLE_LOG"
  docker run -d --rm \
    --name "$sidecar_name" \
    --network host \
    -v "${LINEXA_ROOT}/logs/eval:/logs/eval:rw" \
    -v "${LINEXA_ROOT}/scripts/eval/log_to_wandb.py:/opt/log_to_wandb.py:ro" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e WANDB_DIR=/logs/eval/wandb \
    psi0:latest \
    /opt/venv-psi/bin/python /opt/log_to_wandb.py \
      --project "$WANDB_PROJECT" \
      --name "$wandb_name" \
      --mode "$leg" \
      --task "$TASK" \
      --dr "$DR" \
      --num-episodes "$NUM_EPISODES" \
      --watch-dir "/logs/eval/psi0_decoupled_wbc/${TASK}/${DR}" \
      --eval-stats "/logs/eval/eval_stats.txt" \
      --done-flag "/logs/eval/.${leg}_${TS}.done" \
      --leg-start-ts "$leg_start_ts" \
      --repo-commit "$GIT_COMMIT" \
      --linexa-layers "" \
      >> "$CONSOLE_LOG" 2>&1 || {
        echo "[orch] WARNING: wandb sidecar failed to launch (continuing eval)" | tee -a "$CONSOLE_LOG"
      }

  local eval_log="logs/eval/phase0_${leg}_eval_${TS}.log"
  local eval_rc=0
  TASK="$TASK" DR="$DR" NUM_EPISODES="$NUM_EPISODES" \
    bash scripts/eval/run_simple_eval.sh 2>&1 | tee "$eval_log" >> "$CONSOLE_LOG" || eval_rc=$?
  echo "[orch] leg=$leg eval_rc=$eval_rc $(date -u +%H:%M:%SZ)" | tee -a "$CONSOLE_LOG"

  touch "$done_flag_host"

  for i in $(seq 1 60); do
    if ! docker ps -q --filter "name=$sidecar_name" | grep -q .; then
      echo "[orch] wandb sidecar exited after $((i*2))s" | tee -a "$CONSOLE_LOG"
      break
    fi
    sleep 2
  done
  docker logs "$sidecar_name" 2>&1 | tail -80 | tee -a "$CONSOLE_LOG" || true
  docker rm -f "$sidecar_name" >/dev/null 2>&1 || true

  kill "$logs_pid" 2>/dev/null || true
  wait "$logs_pid" 2>/dev/null || true
  echo "[orch] ===== leg=$leg DONE eval_rc=$eval_rc $(date -u +%H:%M:%SZ) =====" | tee -a "$CONSOLE_LOG"
  return 0
}

IFS=',' read -r -a LEG_ARR <<< "$LEGS"
for leg in "${LEG_ARR[@]}"; do
  run_one_leg "$leg"
done

bash scripts/eval/stop.sh 2>&1 | tee -a "$CONSOLE_LOG" || true

echo "[orch] ===== Phase 0 Layer B complete $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$CONSOLE_LOG"
echo "[orch] console log: $CONSOLE_LOG"
