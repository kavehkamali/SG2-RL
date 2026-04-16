#!/usr/bin/env bash
# 2-GPU torchrun PPO (SKRL) — 32,768 total parallel envs (16,384 per GPU), 96,000 trainer steps,
# checkpoints every 3,000 steps (see configs/skrl_ppo_mlp_stage{1,2}_*.yaml).
#
# Usage:
#   chmod +x scripts/tmux_train_ppo_32768_ddp.sh
#   NUM_ENV_TOTAL=32768 NPROC=2 STAGE=stage1 ./scripts/tmux_train_ppo_32768_ddp.sh
#   NUM_ENV_TOTAL=32768 NPROC=2 STAGE=stage2 ./scripts/tmux_train_ppo_32768_ddp.sh
#
# Notes:
# - Training entry: scripts/train_skrl_ppo.py (torchrun, one Kit process per GPU).
# - Stage2 warm-start: point SKRL to an existing experiment dir or resume from checkpoint.
set -euo pipefail

: "${SG2_RL:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${NPROC:=2}"
: "${NUM_ENV_TOTAL:=32768}"
: "${SESSION:=sg2rl-ppo-32768-ddp}"
: "${STAGE:=stage1}"   # stage1 | stage2

PY="${SG2_RL}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}" >&2
  exit 1
fi

if [[ "${STAGE}" == "stage1" ]]; then
  TASK="FFWSG2-PegGraspLift-v0"
  SKRL_YAML="${SG2_RL}/configs/skrl_ppo_mlp_stage1_grasp_lift.yaml"
  SESSION="${SESSION}-stage1"
  LOG="/tmp/sg2rl_stage1_ppo_32768_ddp.log"
elif [[ "${STAGE}" == "stage2" ]]; then
  TASK="FFWSG2-PegInsert-v0"
  SKRL_YAML="${SG2_RL}/configs/skrl_ppo_mlp_stage2_insert.yaml"
  SESSION="${SESSION}-stage2"
  LOG="/tmp/sg2rl_stage2_ppo_32768_ddp.log"
else
  echo "[error] Unknown STAGE='${STAGE}' (expected stage1|stage2)" >&2
  exit 2
fi

NUM_PER_PROC=$(( NUM_ENV_TOTAL / NPROC ))
if [[ $(( NUM_PER_PROC * NPROC )) -ne "${NUM_ENV_TOTAL}" ]]; then
  echo "[error] NUM_ENV_TOTAL (${NUM_ENV_TOTAL}) must be divisible by NPROC (${NPROC})" >&2
  exit 3
fi

echo "[sg2_rl] stage=${STAGE} task=${TASK} envs_total=${NUM_ENV_TOTAL} nproc=${NPROC} per_proc=${NUM_PER_PROC}"
echo "[sg2_rl] skrl_cfg=${SKRL_YAML} (trainer timesteps=96000, checkpoint_interval=3000)"
echo "[sg2_rl] log: ${LOG}"

export OMNI_KIT_ACCEPT_EULA=YES
# Multi-GPU stability fallbacks (recommended by Isaac Lab docs for some systems).
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
# On some IOMMU systems, GPU P2P/NVLink probing can hang collectives.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
# Make NCCL fail fast + surface better errors (avoid silent hangs).
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CMD="cd '${SG2_RL}' && \
  ${SG2_RL}/.venv/bin/torchrun --nproc_per_node=${NPROC} \
  ${SG2_RL}/scripts/train_skrl_ppo.py \
    --task ${TASK} \
    --num_envs ${NUM_PER_PROC} \
    --headless \
    --skrl_cfg ${SKRL_YAML} \
  2>&1 | tee ${LOG}; \
  ec=\${PIPESTATUS[0]}; \
  echo \"[sg2_rl] train exited with code=\$ec\"; \
  echo \"[sg2_rl] keeping tmux session alive for inspection\"; \
  sleep infinity"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "[sg2_rl] tmux session '${SESSION}' already exists — attach with: tmux attach -t ${SESSION}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" "${CMD}"
echo "[sg2_rl] Started tmux session '${SESSION}' — attach with: tmux attach -t ${SESSION}"

