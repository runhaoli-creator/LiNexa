#!/usr/bin/env bash
# Scan checkpoints/psi0/ for valid psi0 run directories (those containing
# run_config.json, argv.txt, and at least one checkpoints/ckpt_<step>/ dir),
# pick the first match, and write PSI0_RUN_DIR + PSI0_CKPT_STEP into
# docker/.env.
#
# If TASK is set and multiple run_dirs exist, prefer one whose path contains
# the task name.
#
# Usage:
#   bash scripts/eval/find_run_dir.sh                  # auto
#   TASK=G1WholebodyXMovePickTeleop-v0 bash ...        # task-biased
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "${ROOT}"

CKPT_DIR_HOST="${ROOT}/checkpoints/psi0"
CKPT_DIR_IN_CONTAINER="/workspace/psi0-ckpts"

if [ ! -d "${CKPT_DIR_HOST}" ]; then
  echo "[find_run_dir] ${CKPT_DIR_HOST} does not exist. Run scripts/eval/download_psi0_ckpt.sh first." >&2
  exit 1
fi

# Candidate run_dirs: any dir containing both run_config.json and argv.txt.
mapfile -t runs < <(find "${CKPT_DIR_HOST}" -mindepth 1 -maxdepth 6 -type f -name run_config.json \
                      -exec dirname {} \; | sort -u)

if [ "${#runs[@]}" -eq 0 ]; then
  echo "[find_run_dir] no run_config.json under ${CKPT_DIR_HOST}." >&2
  echo "[find_run_dir] run dirs should contain: run_config.json, argv.txt, checkpoints/ckpt_<step>/." >&2
  exit 1
fi

echo "[find_run_dir] candidate run_dirs:"
for r in "${runs[@]}"; do echo "  ${r}"; done

# Prefer a match for $TASK when provided.
picked=""
if [ -n "${TASK:-}" ]; then
  for r in "${runs[@]}"; do
    if [[ "${r}" == *"${TASK}"* ]]; then
      picked="${r}"
      break
    fi
  done
fi
picked="${picked:-${runs[0]}}"

# Pick the largest step number (or first alphabetically) from checkpoints/.
step=""
if [ -d "${picked}/checkpoints" ]; then
  step=$(ls -1 "${picked}/checkpoints" 2>/dev/null \
         | grep -E '^ckpt_[0-9]+$' \
         | sed 's/^ckpt_//' \
         | sort -n | tail -n 1)
fi

if [ -z "${step}" ]; then
  echo "[find_run_dir] ${picked}/checkpoints has no ckpt_<step> dirs." >&2
  ls -la "${picked}/checkpoints" 2>&1 | sed 's/^/  /' >&2 || true
  exit 1
fi

container_path="${picked/${CKPT_DIR_HOST}/${CKPT_DIR_IN_CONTAINER}}"

echo "[find_run_dir] picked: ${picked}"
echo "[find_run_dir] container path: ${container_path}"
echo "[find_run_dir] step: ${step}"

ENV_FILE="${ROOT}/docker/.env"
if [ ! -f "${ENV_FILE}" ]; then
  cp "${ROOT}/docker/.env.sample" "${ENV_FILE}"
fi

# Update (or append) the two keys in docker/.env, leaving everything else alone.
python3 - <<PY
from pathlib import Path
p = Path("${ENV_FILE}")
lines = p.read_text().splitlines()
want = {"PSI0_RUN_DIR": "${container_path}", "PSI0_CKPT_STEP": "${step}"}
seen = set()
out = []
for line in lines:
    m = line.split("=", 1)
    if len(m) == 2 and m[0] in want:
        out.append(f"{m[0]}={want[m[0]]}")
        seen.add(m[0])
    else:
        out.append(line)
for k, v in want.items():
    if k not in seen:
        out.append(f"{k}={v}")
p.write_text("\n".join(out) + "\n")
print("[find_run_dir] updated docker/.env")
PY
