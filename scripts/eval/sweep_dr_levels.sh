#!/usr/bin/env bash
# Autonomous DR-level sweep for a single SIMPLE task.
#
# Waits for an already-running eval (if any), then runs the remaining levels
# sequentially — each in its own tmux session. Per-level outputs are archived
# so they don't get intermingled.
#
# Usage (detached):
#   nohup bash scripts/eval/sweep_dr_levels.sh > logs/eval/sweep.log 2>&1 &
#
# Env:
#   TASK           default: G1WholebodyXMovePickTeleop-v0
#   NUM_EPISODES   default: 10
#   LEVELS         default: "level-0 level-1 level-2" (space-separated, in order)
#   WAIT_FIRST     "1" to assume LEVELS[0] is already running and just wait for
#                  its container to exit (default: 1). Set to "0" to launch all.
set -uo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "${ROOT}"

TASK="${TASK:-G1WholebodyXMovePickTeleop-v0}"
NUM_EPISODES="${NUM_EPISODES:-10}"
LEVELS="${LEVELS:-level-0 level-1 level-2}"
WAIT_FIRST="${WAIT_FIRST:-1}"

STATS="${ROOT}/logs/eval/eval_stats.txt"
ORCH_LOG="${ROOT}/logs/eval/sweep.log"
mkdir -p "${ROOT}/logs/eval"

log() {
  local msg="[$(date -Is)] $*"
  echo "${msg}"
  echo "${msg}" >> "${ORCH_LOG}"
}

# Block until no `linexa-eval-simple-eval-run-*` container is running.
wait_for_eval_container_gone() {
  local first=1
  while docker ps --filter "name=linexa-eval-simple-eval-run" -q | grep -q .; do
    if [ "${first}" = "1" ]; then
      log "waiting on active eval container(s)..."
      first=0
    fi
    sleep 30
  done
}

# Extract the last "====" block from eval_stats.txt (resets on every `====`).
extract_last_block() {
  awk '
    /^================$/ { block=$0"\n"; next }
    { block = block $0 "\n" }
    END { printf "%s", block }
  ' "${STATS}"
}

# Count distinct `====` blocks currently in eval_stats.txt.
count_blocks() {
  grep -c "^================$" "${STATS}" 2>/dev/null || echo 0
}

# True if the currently-last block contains N lines matching "episode_i:".
last_block_has_all_episodes() {
  local n="$1"
  local got
  got=$(extract_last_block | grep -cE "^episode_[0-9]+: " 2>/dev/null || echo 0)
  [ "${got}" -ge "${n}" ]
}

launch_level_in_tmux() {
  local level="$1"
  local session="linexa-${level}"

  # If session already exists, kill it (prior failed attempt)
  tmux has-session -t "${session}" 2>/dev/null && tmux kill-session -t "${session}"

  log "launching ${level} in tmux session '${session}'"
  tmux new-session -d -s "${session}" -n eval \
    "cd ${ROOT} && TASK=${TASK} DR=${level} NUM_EPISODES=${NUM_EPISODES} bash scripts/eval/run_simple_eval.sh 2>&1 | tee logs/eval/run_${level}.log; echo '[sweep] ${level} exited. Press enter to close.'; read _"
  sleep 15
}

run_and_wait() {
  local level="$1"

  # Remember how many blocks already exist so we can detect the new one.
  local before
  before=$(count_blocks)

  launch_level_in_tmux "${level}"

  # Wait until the eval container appends a fresh `====` block (run started).
  local t=0
  while [ "$(count_blocks)" -le "${before}" ] && [ ${t} -lt 60 ]; do
    sleep 5; t=$((t+5))
  done

  wait_for_eval_container_gone

  # Archive the (new) last block, which belongs to this level.
  extract_last_block > "${ROOT}/logs/eval/stats_${level}.txt"

  local succ fail
  succ=$(grep -c ": True"  "${ROOT}/logs/eval/stats_${level}.txt" 2>/dev/null || true)
  fail=$(grep -c ": False" "${ROOT}/logs/eval/stats_${level}.txt" 2>/dev/null || true)
  local total=$((succ + fail))
  if [ "${total}" -eq 0 ]; then
    log "${level}: [FAIL] 0 episodes recorded — run likely crashed. See logs/eval/run_${level}.log"
    SWEEP_FAILED+=("${level}")
  elif [ "${total}" -lt "${NUM_EPISODES}" ]; then
    log "${level}: [PARTIAL] only ${total}/${NUM_EPISODES} episodes (succ=${succ}, fail=${fail})"
    SWEEP_FAILED+=("${level}")
  else
    local pct=$(awk -v s="${succ}" -v t="${total}" 'BEGIN{printf "%.2f", s/t}')
    log "${level}: [OK] ${succ}/${total} success (rate ${pct})"
  fi
}

SWEEP_FAILED=()
IFS=' ' read -ra LEVEL_ARR <<< "${LEVELS}"

log "starting sweep task=${TASK} episodes=${NUM_EPISODES} levels=(${LEVELS}) wait_first=${WAIT_FIRST}"

first_level="${LEVEL_ARR[0]}"
rest_levels=("${LEVEL_ARR[@]:1}")

if [ "${WAIT_FIRST}" = "1" ]; then
  log "assuming ${first_level} is already running; waiting for its container to exit"
  wait_for_eval_container_gone
  extract_last_block > "${ROOT}/logs/eval/stats_${first_level}.txt"
  succ=$(grep -c ": True"  "${ROOT}/logs/eval/stats_${first_level}.txt" 2>/dev/null || true)
  fail=$(grep -c ": False" "${ROOT}/logs/eval/stats_${first_level}.txt" 2>/dev/null || true)
  total=$((succ + fail))
  if [ "${total}" -eq 0 ]; then
    log "${first_level}: [FAIL] 0 episodes in archive"
    SWEEP_FAILED+=("${first_level}")
  else
    pct=$(awk -v s="${succ}" -v t="${total}" 'BEGIN{printf "%.2f", s/t}')
    log "${first_level}: [OK] ${succ}/${total} success (rate ${pct})"
  fi
else
  run_and_wait "${first_level}"
fi

for level in "${rest_levels[@]}"; do
  run_and_wait "${level}"
done

log "sweep complete"
log "--- summary ---"
for lvl in "${LEVEL_ARR[@]}"; do
  local_sr=$(grep "success rate:" "${ROOT}/logs/eval/stats_${lvl}.txt" 2>/dev/null | tail -1)
  succ=$(grep -c ": True" "${ROOT}/logs/eval/stats_${lvl}.txt" 2>/dev/null)
  fail=$(grep -c ": False" "${ROOT}/logs/eval/stats_${lvl}.txt" 2>/dev/null)
  log "  ${lvl}: ${succ:-?} True, ${fail:-?} False. ${local_sr}"
done

if [ "${#SWEEP_FAILED[@]}" -gt 0 ]; then
  log "levels with problems: ${SWEEP_FAILED[*]}"
  exit 1
fi
log "all levels finished cleanly"
