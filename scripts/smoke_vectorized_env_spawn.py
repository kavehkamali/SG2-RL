#!/usr/bin/env python3
"""Smoke-test vectorized env spawn + a few sim steps (e.g. 16k envs / GPU).

Requires a working Isaac Sim + Isaac Lab install (same as training).

Example:
  export OMNI_KIT_ACCEPT_EULA=YES
  ./scripts/smoke_vectorized_env_spawn.py --task FFWSG2-PegGraspLift-v0 --num_envs 16384 --steps 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_RANK = os.environ.get("LOCAL_RANK", "0")
_PORTABLE_ROOT = f"/tmp/sg2rl_kit_smoke_rank{_LOCAL_RANK}"

if "--portable" not in sys.argv:
    sys.argv.append("--portable")
if not any(arg == "--portable-root" or arg.startswith("--portable-root") for arg in sys.argv):
    sys.argv.extend(["--portable-root", _PORTABLE_ROOT])

parser = argparse.ArgumentParser(description="Smoke spawn for SG2-RL Isaac envs at scale.")
parser.add_argument("--task", type=str, default="FFWSG2-PegGraspLift-v0")
parser.add_argument("--num_envs", type=int, default=16384)
parser.add_argument("--steps", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--skrl_yaml", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()
args_cli.enable_cameras = False
args_cli.device = f"cuda:{_LOCAL_RANK}"

sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from sg2_rl.config_loader import load_task_cfg  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402


def main() -> None:
    yml = args_cli.skrl_yaml.strip() or str(_REPO_ROOT / "configs/skrl_ppo_mlp_stage1_grasp_lift.yaml")
    ensure_task_registered(args_cli.task, skrl_yaml_override=yml)
    env_cfg, _ = load_task_cfg(args_cli.task, "skrl_cfg_entry_point")
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.sim.device = str(args_cli.device)

    print(f"[smoke] task={args_cli.task} num_envs={env_cfg.scene.num_envs} device={env_cfg.sim.device}", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        env.reset(seed=args_cli.seed)
        unwrapped = env.unwrapped
        device = env_cfg.sim.device
        if not (isinstance(device, str) and device.startswith("cuda")):
            device = "cuda:0"
        act_dim = unwrapped.action_manager.total_action_dim
        n = int(args_cli.num_envs)
        for _ in range(int(args_cli.steps)):
            actions = 0.02 * torch.randn(n, act_dim, device=device)
            env.step(actions)
        print("[smoke] OK: reset + steps completed.", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
