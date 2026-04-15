#!/usr/bin/env bash
# Multi-GPU PPO training for OmniReset-FFWSG2-PegMLPGraspLift-v0.
#
# Launches a tmux session with torchrun DDP across 2 GPUs (default).
# Uses SG2-RL's own .venv and env config — no external repos needed.
#
# Usage:
#   chmod +x scripts/tmux_train_grasp_lift_ddp.sh
#   ./scripts/tmux_train_grasp_lift_ddp.sh
#
# Override env count:
#   NUM_ENV_TOTAL=28672 ./scripts/tmux_train_grasp_lift_ddp.sh
set -euo pipefail

: "${SG2_RL:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${NPROC:=2}"
: "${NUM_ENV_TOTAL:=30000}"
: "${SESSION:=sg2rl-grasp-ppo-ddp}"

PY="${SG2_RL}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}" >&2
  exit 1
fi

TASK="OmniReset-FFWSG2-PegMLPGraspLift-v0"
SKRL_YAML="${SG2_RL}/configs/skrl_ppo_mlp_grasp_lift_96k.yaml"
LOG="/tmp/sg2rl_grasp_lift_ppo_ddp.log"

NUM_PER_PROC=$(( NUM_ENV_TOTAL / NPROC ))
echo "[sg2_rl] task=${TASK} envs=${NUM_ENV_TOTAL} nproc=${NPROC} per_proc=${NUM_PER_PROC}"
echo "[sg2_rl] log: ${LOG}"

export OMNI_KIT_ACCEPT_EULA=YES

CMD="cd '${SG2_RL}' && \
  ${SG2_RL}/.venv/bin/torchrun --nproc_per_node=${NPROC} \
  ${SG2_RL}/.venv/bin/python -m isaaclab.train \
    --task ${TASK} \
    --num_envs ${NUM_PER_PROC} \
    --headless \
    --distributed \
    --ml_framework skrl \
    --skrl_cfg ${SKRL_YAML} \
  2>&1 | tee ${LOG}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[sg2_rl] tmux session '${SESSION}' already exists — attach with: tmux attach -t ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
echo "[sg2_rl] Started tmux session '${SESSION}' — attach with: tmux attach -t ${SESSION}"
