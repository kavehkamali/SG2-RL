#!/usr/bin/env bash
# Run SG2-RL Isaac scripts using the same Python env as UWLab training on tai.
# Usage: run_on_tai.sh smoke_random_motion.py --headless --steps 32
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
: "${UWLAB:=${HOME}/projects/API/UWLab}"
: "${SG2_RL:=${HOME}/projects/API/SG2-RL}"
# Peg / hole USD: local tree only (tai often has no Hugging Face). Expected layout:
#   $ROOT/Props/Custom/Peg/peg.usd
#   $ROOT/Props/Custom/PegHole/peg_hole.usd
# Set UWLAB_CLOUD_ASSETS_DIR or SG2_CLOUD_ASSETS_DIR to the directory that contains ``Props/``.
if [[ -z "${UWLAB_CLOUD_ASSETS_DIR:-}" ]]; then
  if [[ -n "${SG2_CLOUD_ASSETS_DIR:-}" ]]; then
    UWLAB_CLOUD_ASSETS_DIR="${SG2_CLOUD_ASSETS_DIR}"
  elif [[ -f "${HOME}/uwlab_sync/Props/Custom/Peg/peg.usd" ]]; then
    UWLAB_CLOUD_ASSETS_DIR="${HOME}/uwlab_sync"
  elif [[ -f "${HOME}/uwlab_hf_assets/Props/Custom/Peg/peg.usd" ]]; then
    UWLAB_CLOUD_ASSETS_DIR="${HOME}/uwlab_hf_assets"
  else
    UWLAB_CLOUD_ASSETS_DIR="${HOME}/uwlab_sync"
  fi
fi
export UWLAB_CLOUD_ASSETS_DIR
PY="${UWLAB}/env_uwlab/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 1
fi
script_name="$1"
shift
cd "${UWLAB}"
exec "${PY}" "${SG2_RL}/scripts/${script_name}" "$@"
