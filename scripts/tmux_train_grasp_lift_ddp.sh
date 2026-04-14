#!/usr/bin/env bash
# 2-GPU torchrun PPO on FFWSG2 peg (vector obs / MLP): ~30k global parallel envs, SKRL budget in yaml.
#
# Prerequisites: UWLab checkout with ``uwlab.sh`` + ``scripts/reinforcement_learning/skrl/train.py``,
# this repo for ``register_sg2_tasks.py`` + ``configs/skrl_ppo_mlp_grasp_lift_96k.yaml``.
#
# Session name: sg2rl-grasp-ppo-ddp  (attach: tmux attach -t sg2rl-grasp-ppo-ddp)
set -euo pipefail

# Same local USD root as run_on_tai.sh (no Hugging Face on tai).
if [[ -z "${UWLAB_CLOUD_ASSETS_DIR:-}" ]]; then
  if [[ -n "${SG2_CLOUD_ASSETS_DIR:-}" ]]; then
    export UWLAB_CLOUD_ASSETS_DIR="${SG2_CLOUD_ASSETS_DIR}"
  elif [[ -f "${HOME}/uwlab_sync/Props/Custom/Peg/peg.usd" ]]; then
    export UWLAB_CLOUD_ASSETS_DIR="${HOME}/uwlab_sync"
  elif [[ -f "${HOME}/uwlab_hf_assets/Props/Custom/Peg/peg.usd" ]]; then
    export UWLAB_CLOUD_ASSETS_DIR="${HOME}/uwlab_hf_assets"
  else
    export UWLAB_CLOUD_ASSETS_DIR="${HOME}/uwlab_sync"
  fi
else
  export UWLAB_CLOUD_ASSETS_DIR
fi
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"

SG2_RL="${SG2_RL:-${HOME}/projects/API/SG2-RL}"
UWLAB_PATH="${UWLAB_PATH:-}"
if [[ -z "${UWLAB_PATH}" || ! -f "${UWLAB_PATH}/uwlab.sh" ]]; then
  while IFS= read -r -d '' f; do
    d="$(dirname "$f")"
    if [[ -f "${d}/scripts/reinforcement_learning/skrl/train.py" ]]; then
      UWLAB_PATH="$d"
      break
    fi
  done < <(find "${HOME}" -type f -name uwlab.sh -print0 2>/dev/null)
fi
if [[ -z "${UWLAB_PATH}" || ! -f "${UWLAB_PATH}/uwlab.sh" ]]; then
  echo "[ERROR] Set UWLAB_PATH to UWLab root (contains uwlab.sh and skrl train.py)." >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" || ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
  for _v in "${UWLAB_PATH}/env_uwlab" "${HOME}/projects/API/UWLab/env_uwlab"; do
    if [[ -x "${_v}/bin/python" ]]; then export VIRTUAL_ENV="${_v}"; break; fi
  done
fi
if [[ -z "${VIRTUAL_ENV:-}" || ! -x "${VIRTUAL_ENV}/bin/python" ]]; then
  echo "[ERROR] VIRTUAL_ENV must point to UWLab venv python." >&2
  exit 1
fi

PEG_AGENTS="${UWLAB_PATH}/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset_sg2/config/agents"
mkdir -p "${PEG_AGENTS}"
cp -f "${SG2_RL}/configs/skrl_ppo_mlp_grasp_lift_96k.yaml" "${PEG_AGENTS}/"
echo "[tmux_train_grasp_lift_ddp] Copied SKRL yaml -> ${PEG_AGENTS}/skrl_ppo_mlp_grasp_lift_96k.yaml"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29515}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_ENV_TOTAL="${NUM_ENV_TOTAL:-30000}"
TASK="${TASK:-OmniReset-FFWSG2-PegMLPGraspLift-v0}"
SEED="${SEED:-42}"
LOG_REWARD_EVERY="${LOG_REWARD_EVERY:-12}"
NPROC="${NPROC:-2}"
SESSION="${SESSION:-sg2rl-grasp-ppo-ddp}"
LOG="${LOG:-/tmp/sg2rl_grasp_lift_ppo_ddp.log}"

if (( NUM_ENV_TOTAL % NPROC != 0 )); then
  echo "[ERROR] NUM_ENV_TOTAL (${NUM_ENV_TOTAL}) must be divisible by NPROC (${NPROC})." >&2
  exit 1
fi

TORCHRUN="${VIRTUAL_ENV}/bin/torchrun"
if [[ ! -x "${TORCHRUN}" ]]; then
  echo "[ERROR] Missing ${TORCHRUN}" >&2
  exit 1
fi

tmux kill-session -t "${SESSION}" 2>/dev/null || true
sleep 1

tmux new-session -d -s "${SESSION}" bash -lc \
  "cd '${UWLAB_PATH}' && \
   export PYTHONPATH='${SG2_RL}/src':\"\${PYTHONPATH}\" && \
   '${VIRTUAL_ENV}/bin/python' '${SG2_RL}/scripts/register_sg2_tasks.py' && \
   '${TORCHRUN}' --standalone --nnodes=1 --nproc_per_node=${NPROC} \
     scripts/reinforcement_learning/skrl/train.py \
     --headless --distributed --device cuda \
     --task '${TASK}' \
     --num_envs ${NUM_ENV_TOTAL} \
     --seed ${SEED} \
     --log-reward-terms-every ${LOG_REWARD_EVERY} \
     2>&1 | tee '${LOG}'"

echo "[tmux_train_grasp_lift_ddp] Started tmux session: ${SESSION}"
echo "[tmux_train_grasp_lift_ddp] tail -f ${LOG}"
tmux ls || true
