#!/usr/bin/env bash
# psi0 server entrypoint. The psi0:latest image already has a uv-managed venv
# at /opt/venv-psi with serve_psi0 registered; we just need to exec it with
# the right arguments.
set -euo pipefail

: "${PSI0_RUN_DIR:?PSI0_RUN_DIR must point at a run_dir containing run_config.json + argv.txt + checkpoints/ckpt_<step>/}"
: "${PSI0_CKPT_STEP:?PSI0_CKPT_STEP must be set (e.g. 20000 or 'latest')}"

PSI0_SERVER_HOST="${PSI0_SERVER_HOST:-0.0.0.0}"
PSI0_SERVER_PORT="${PSI0_SERVER_PORT:-22085}"
PSI0_ACTION_HORIZON="${PSI0_ACTION_HORIZON:-24}"
PSI0_RTC="${PSI0_RTC:-1}"

# Guard against common mistakes early so the error is obvious in `docker logs`.
if [ ! -f "${PSI0_RUN_DIR}/run_config.json" ]; then
  echo "[psi0-server] FATAL: ${PSI0_RUN_DIR}/run_config.json missing inside container." >&2
  echo "[psi0-server] The host path checkpoints/psi0/ is mounted read-only at /workspace/psi0-ckpts/." >&2
  ls -la "${PSI0_RUN_DIR%/*}" 2>&1 | head -20 >&2 || true
  exit 1
fi

rtc_args=()
case "${PSI0_RTC,,}" in
  1|true|yes) rtc_args+=(--rtc) ;;
esac

cd /workspace

# psi0_serve_simple.main asserts on `load_dotenv()`, which requires a .env
# to exist in CWD. Every variable it cares about is already injected by
# compose, so we just need an existing file.
if [ ! -f /workspace/.env ]; then
  cat > /workspace/.env <<EOF
HF_TOKEN=${HF_TOKEN:-}
PSI_HOME=${PSI_HOME:-/hfm}
DATA_HOME=${DATA_HOME:-/hfm/data}
HF_HOME=${HF_HOME:-/hfm/cache}
TORCH_HOME=${TORCH_HOME:-/hfm/cache}
UV_CACHE_DIR=${UV_CACHE_DIR:-/hfm/cache}
OMP_NUM_THREADS=8
TOKENIZERS_PARALLELISM=false
TF_CPP_MIN_LOG_LEVEL=3
AV_LOG_LEVEL=quiet
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
EOF
fi

echo "[psi0-server] run=${PSI0_RUN_DIR}"
echo "[psi0-server] step=${PSI0_CKPT_STEP} port=${PSI0_SERVER_PORT} horizon=${PSI0_ACTION_HORIZON} rtc=${PSI0_RTC}"

# LiNexa branch: when LINEXA_TTT_ENABLED=1 launch the in-process monkey-patched
# wrapper from src/linexa/eval/serve_psi0_linexa.py. Default (0) execs the
# upstream serve_psi0 console script unchanged.
linexa_enabled="${LINEXA_TTT_ENABLED:-0}"
case "${linexa_enabled,,}" in
  1|true|yes|on)
    if [ -z "${LINEXA_SRC:-}" ] || [ ! -d "${LINEXA_SRC}" ]; then
      echo "[psi0-server] FATAL: LINEXA_TTT_ENABLED=${linexa_enabled} but LINEXA_SRC is not a mounted directory." >&2
      echo "[psi0-server] Expected docker-compose to mount \${LINEXA_ROOT}/src and export LINEXA_SRC." >&2
      exit 1
    fi
    export PYTHONPATH="${LINEXA_SRC}:${PYTHONPATH:-}"
    echo "[psi0-server] linexa: enabled (PYTHONPATH=${PYTHONPATH})"
    exec /opt/venv-psi/bin/python -m linexa.eval.serve_psi0_linexa \
      --host "${PSI0_SERVER_HOST}" \
      --port "${PSI0_SERVER_PORT}" \
      --run-dir "${PSI0_RUN_DIR}" \
      --ckpt-step "${PSI0_CKPT_STEP}" \
      --action-exec-horizon "${PSI0_ACTION_HORIZON}" \
      "${rtc_args[@]}"
    ;;
esac

exec /opt/venv-psi/bin/serve_psi0 \
  --host "${PSI0_SERVER_HOST}" \
  --port "${PSI0_SERVER_PORT}" \
  --run-dir "${PSI0_RUN_DIR}" \
  --ckpt-step "${PSI0_CKPT_STEP}" \
  --action-exec-horizon "${PSI0_ACTION_HORIZON}" \
  "${rtc_args[@]}"
