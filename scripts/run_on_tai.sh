#!/usr/bin/env bash
# Run SG2-RL Isaac scripts using the repo-local .venv on tai.
# Usage: run_on_tai.sh smoke_random_motion.py --headless --steps 32
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
: "${SG2_RL:=${HOME}/projects/API/SG2-RL}"

PY="${SG2_RL}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY} — run: cd ${SG2_RL} && ~/.local/bin/uv venv .venv --python 3.10 && ~/.local/bin/uv pip install -e '.[dev]'" >&2
  exit 1
fi
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <script.py> [args...]" >&2
  exit 1
fi
script_name="$1"
shift
cd "${SG2_RL}"
exec "${PY}" "${SG2_RL}/scripts/${script_name}" "$@"
