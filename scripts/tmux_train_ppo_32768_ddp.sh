#!/usr/bin/env bash
# Multi-GPU torchrun PPO (SKRL) + optional Weights & Biases + GIF eval.
#
# GIF eval runs as a synchronous subprocess every 5 000 trainer steps.
# Training blocks (rank-0 waits, rank-1 stalls at NCCL barrier) so both GPUs
# have no active kernels during eval — the subprocess shares VRAM with the
# paused training process on GPU 0 (no spare GPU needed).
# Override which GPU eval uses: SG2RL_WANDB_EVAL_CUDA=<id>
#
# STAGE values: stage1 | stage2 | omnireset
#
# Usage (2 GPUs, both used for training, eval shares GPU 0):
#   NPROC=2 RESERVE_INFERENCE_GPU=0 NUM_ENV_TOTAL=4096 STAGE=omnireset WANDB=1 \
#     WANDB_PROJECT=sg2-rl ./scripts/tmux_train_ppo_32768_ddp.sh
#
#   # 4 GPUs → train on 3, eval on GPU 3
#   NUM_ENV_TOTAL=32766 STAGE=omnireset WANDB=1 ./scripts/tmux_train_ppo_32768_ddp.sh
#
# Env:
#   RESERVE_INFERENCE_GPU=1  (default)  Set to 0 to use ALL GPUs for training.
#   WANDB=1                  Pass --wandb --wandb_gif to train_skrl_ppo.py
#   WANDB_PROJECT, WANDB_ENTITY
#
set -euo pipefail

: "${SG2_RL:=$(cd "$(dirname "$0")/.." && pwd)}"
: "${NUM_ENV_TOTAL:=32768}"
: "${SESSION:=sg2rl-ppo-ddp}"
: "${STAGE:=stage1}"   # stage1 | stage2
: "${RESERVE_INFERENCE_GPU:=1}"
: "${WANDB:=0}"

PY="${SG2_RL}/.venv/bin/python"
TORCHRUN="${SG2_RL}/.venv/bin/torchrun"
if [[ ! -x "${PY}" ]]; then
  echo "[error] Missing ${PY}" >&2
  exit 1
fi

GPU_COUNT=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
GPU_COUNT=$(echo "${GPU_COUNT}" | tr -d '[:space:]')
GPU_COUNT=$((GPU_COUNT + 0))
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[sg2_rl] warn: could not detect GPUs (nvidia-sim?); defaulting NPROC=1"
  GPU_COUNT=1
fi

if [[ -z "${NPROC:-}" ]]; then
  if [[ "${RESERVE_INFERENCE_GPU}" == "1" && "${GPU_COUNT}" -ge 2 ]]; then
    NPROC=$((GPU_COUNT - 1))
    # Last GPU index (0-based): free for wandb_gif_eval subprocess
    export SG2RL_WANDB_EVAL_CUDA="${SG2RL_WANDB_EVAL_CUDA:-$((GPU_COUNT - 1))}"
  else
    NPROC="${GPU_COUNT}"
    if [[ "${GPU_COUNT}" -lt 2 ]]; then
      echo "[sg2_rl] warn: single GPU — unsetting SG2RL_WANDB_EVAL_CUDA (GIF eval needs a spare GPU or run without --wandb_gif)"
      unset SG2RL_WANDB_EVAL_CUDA || true
    fi
  fi
fi

if [[ "${NPROC}" -lt 1 ]]; then
  echo "[error] NPROC must be >= 1 (got NPROC=${NPROC}, GPU_COUNT=${GPU_COUNT})" >&2
  exit 4
fi

if [[ "${STAGE}" == "stage1" ]]; then
  TASK="FFWSG2-PegGraspLift-v0"
  SKRL_YAML="${SG2_RL}/configs/skrl_ppo_mlp_stage1_grasp_lift.yaml"
  SESSION="${SESSION}-stage1"
  LOG="/tmp/sg2rl_stage1_ppo_ddp.log"
elif [[ "${STAGE}" == "stage2" ]]; then
  TASK="FFWSG2-PegInsert-v0"
  SKRL_YAML="${SG2_RL}/configs/skrl_ppo_mlp_stage2_insert.yaml"
  SESSION="${SESSION}-stage2"
  LOG="/tmp/sg2rl_stage2_ppo_ddp.log"
elif [[ "${STAGE}" == "omnireset" ]]; then
  TASK="FFWSG2-OmniReset-BimanualPegInsert-v0"
  SKRL_YAML="${SG2_RL}/configs/skrl_ppo_omnireset_bimanual_peg.yaml"
  SESSION="${SESSION}-omnireset"
  LOG="/tmp/sg2rl_omnireset_ppo_ddp.log"
else
  echo "[error] Unknown STAGE='${STAGE}' (expected stage1|stage2|omnireset)" >&2
  exit 2
fi

NUM_PER_PROC=$(( NUM_ENV_TOTAL / NPROC ))
if [[ $(( NUM_PER_PROC * NPROC )) -ne "${NUM_ENV_TOTAL}" ]]; then
  echo "[error] NUM_ENV_TOTAL (${NUM_ENV_TOTAL}) must be divisible by NPROC (${NPROC})" >&2
  exit 3
fi

echo "[sg2_rl] GPU_COUNT=${GPU_COUNT} reserve_inference_gpu=${RESERVE_INFERENCE_GPU} SG2RL_WANDB_EVAL_CUDA=${SG2RL_WANDB_EVAL_CUDA:-unset}"
echo "[sg2_rl] stage=${STAGE} task=${TASK} envs_total=${NUM_ENV_TOTAL} nproc=${NPROC} per_proc=${NUM_PER_PROC}"
echo "[sg2_rl] skrl_cfg=${SKRL_YAML} (trainer timesteps from yaml, checkpoint_interval in yaml)"
echo "[sg2_rl] log: ${LOG}"

WANDB_ARGS=()
if [[ "${WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--wandb --wandb_gif --wandb_eval_interval 5000 --wandb_eval_episodes 4 --wandb_eval_steps 800)
  [[ -n "${WANDB_PROJECT:-}" ]] && WANDB_ARGS+=(--wandb_project "${WANDB_PROJECT}")
  [[ -n "${WANDB_ENTITY:-}" ]] && WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Helps some wandb + multiprocess setups
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

CMD="cd '${SG2_RL}' && \
  ${TORCHRUN} --nproc_per_node=${NPROC} \
  ${SG2_RL}/scripts/train_skrl_ppo.py \
    --task ${TASK} \
    --num_envs ${NUM_PER_PROC} \
    --headless \
    --skrl_cfg ${SKRL_YAML} \
    ${WANDB_ARGS[*]} \
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
