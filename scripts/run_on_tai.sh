#!/usr/bin/env bash
# Run SG2-RL Isaac scripts using the same Python env as UWLab training on tai.
# Usage: run_on_tai.sh smoke_random_motion.py --headless --steps 32
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
: "${UWLAB:=${HOME}/projects/API/UWLab}"
: "${SG2_RL:=${HOME}/projects/API/SG2-RL}"
# Peg USD and other props resolve from the HF mirror (same as omnireset_sg2_uwlab train scripts).
: "${UWLAB_CLOUD_ASSETS_DIR:=${HOME}/uwlab_hf_assets}"
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
