# RL backends (roadmap)

This repo is intentionally small at first: **one Isaac-based smoke script** plus tests. Full training loops will be added incrementally.

## PPO

- **SKRL**: Matches existing UW Lab Hydra entry points (`skrl_cfg_entry_point` in gym registry).
- **Stable-Baselines3**: Good for rapid baselines; wrap `gym.make` env in SB3’s `VecEnv` if needed.

## SAC

- **Stable-Baselines3** `SAC` is the quickest path for continuous control.
- Ensure observation / action spaces match the SG2 env you register.

## DDP (distributed data parallel)

- Use **`torchrun --nproc_per_node=K`** around your training entry point.
- Prefer a single-process Isaac Lab sim per GPU unless you adopt Isaac’s multi-GPU patterns explicitly.

## Shared notes

- Keep **one** source of truth for the FFWSG2 peg **env cfg** (`uwlab_tasks`); avoid copying `ffw_sg2_peg_partial_smoke_env_cfg.py` into this repo unless you intend to fork it.
- Pin Isaac Sim / Isaac Lab / UW Lab **versions** in this README once you lock a stack.
