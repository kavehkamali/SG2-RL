#!/usr/bin/env python3
"""Minimal physics smoke: random small actions for N steps (no video)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="SG2 peg scene — random motion smoke test.")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--skrl_yaml", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_REPO_ROOT / "src"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


@hydra_task_compose(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = str(_REPO_ROOT / "artifacts" / "smoke_run")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=args_cli.seed)
    device = env_cfg.sim.device
    if not (isinstance(device, str) and device.startswith("cuda")):
        device = "cuda:0"
    act_dim = env.unwrapped.action_manager.total_action_dim
    n = args_cli.num_envs
    max_abs = 0.0
    for step in range(args_cli.steps):
        actions = 0.02 * torch.randn(n, act_dim, device=device)
        obs, _r, _t, _tr, _i = env.step(actions)
        max_abs = max(max_abs, float(actions.abs().max()))
        if isinstance(obs, dict):
            flat = torch.cat([torch.as_tensor(v).reshape(n, -1) for v in obs.values()], dim=-1)
        else:
            flat = torch.as_tensor(obs).reshape(n, -1)
        if torch.isnan(flat).any():
            raise RuntimeError(f"NaN in observations at step={step}")
    env.close()
    print(f"[sg2_rl] smoke_random_motion ok steps={args_cli.steps} max_abs_action={max_abs:.4f}", flush=True)


if __name__ == "__main__":
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    main()
    simulation_app.close()
