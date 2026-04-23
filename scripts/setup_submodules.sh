#!/usr/bin/env bash
# Initialize and update submodules under extern/.
# Placeholder: current logic just defers to `git submodule`.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git submodule update --init --recursive

echo "[setup_submodules] done."
