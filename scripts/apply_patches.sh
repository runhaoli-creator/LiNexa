#!/usr/bin/env bash
# Apply patches from patches/<submodule>/ onto extern/<submodule>.
# Placeholder: no patches exist yet; this is a no-op skeleton.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

for sub in psi0 simple; do
  patch_dir="patches/${sub}"
  target_dir="extern/${sub}"
  if [[ ! -d "${target_dir}" ]]; then
    echo "[apply_patches] skip ${sub}: ${target_dir} missing"
    continue
  fi
  shopt -s nullglob
  patches=("${patch_dir}"/*.patch)
  shopt -u nullglob
  if (( ${#patches[@]} == 0 )); then
    echo "[apply_patches] ${sub}: no patches"
    continue
  fi
  echo "[apply_patches] ${sub}: ${#patches[@]} patch(es)"
  for p in "${patches[@]}"; do
    echo "  applying $(basename "${p}")"
    (cd "${target_dir}" && git apply --check "../../${p}" && git apply "../../${p}")
  done
done

echo "[apply_patches] done."
