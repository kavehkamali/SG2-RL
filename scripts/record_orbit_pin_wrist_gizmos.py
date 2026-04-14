#!/usr/bin/env python3
"""Orbit camera + RGB axis gizmos at pin and right wrist; record MP4 (FFW SG2 peg scene via UWLab)."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Record orbit video with pin + wrist RGB gizmos.")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=360)
parser.add_argument(
    "--video_folder",
    type=str,
    default="",
    help="Output folder (default: <repo>/artifacts/videos/orbit_<timestamp>)",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--orbit_radius", type=float, default=1.65)
parser.add_argument("--orbit_z_offset", type=float, default=0.48)
parser.add_argument("--axis_len", type=float, default=0.12, help="World-axis gizmo length (m).")
parser.add_argument("--print_every", type=int, default=30, help="Print pin/wrist world xyz every N steps (0=off).")
parser.add_argument("--skrl_yaml", type=str, default="", help="Override absolute path to SKRL yaml for Hydra.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_REPO_ROOT / "src"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.usd_gizmo import ensure_rgb_axes  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


@hydra_task_compose(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    vf = args_cli.video_folder.strip() or str(
        _REPO_ROOT / "artifacts" / "videos" / f"orbit_pin_wrist_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    vpath = Path(vf)
    vpath.mkdir(parents=True, exist_ok=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = str(vpath / "run")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(vpath),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )

    obs, _ = env.reset(seed=args_cli.seed)
    _ = obs

    unwrapped = env.unwrapped
    device = env_cfg.sim.device
    if not (isinstance(device, str) and device.startswith("cuda")):
        device = "cuda:0"

    robot = unwrapped.scene["robot"]
    peg = unwrapped.scene["insertive_object"]
    (wrist_ids, wrist_names) = robot.find_bodies("arm_r_link7")
    if len(wrist_ids) != 1:
        raise RuntimeError(f"Expected one arm_r_link7 body, got {wrist_names}")
    wrist_body_idx = wrist_ids[0]

    look = tuple(float(x) for x in env_cfg.viewer.lookat)
    lx, ly, lz = look
    act_dim = unwrapped.action_manager.total_action_dim
    n = env_cfg.scene.num_envs
    n_steps = max(1, args_cli.video_length - 1)

    import omni.usd  # type: ignore

    for step in range(args_cli.video_length):
        theta = math.pi * (step / n_steps)
        r = float(args_cli.orbit_radius)
        eye = (lx + r * math.cos(theta), ly + r * math.sin(theta), lz + float(args_cli.orbit_z_offset))
        unwrapped.sim.set_camera_view(eye=eye, target=look)

        pin_xyz = peg.data.root_pos_w[0].detach().cpu().tolist()
        wrist_xyz = robot.data.body_link_pos_w[:, wrist_body_idx][0].detach().cpu().tolist()

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            ensure_rgb_axes(stage, "/World/SG2RL/gizmo_pin", pin_xyz, axis_length=float(args_cli.axis_len))
            ensure_rgb_axes(stage, "/World/SG2RL/gizmo_wrist", wrist_xyz, axis_length=float(args_cli.axis_len))

        if args_cli.print_every > 0 and (step % args_cli.print_every == 0 or step == args_cli.video_length - 1):
            print(
                f"[sg2_rl] step={step} pin_world_xyz={pin_xyz} wrist_world_xyz={wrist_xyz}",
                flush=True,
            )

        actions = 0.05 * torch.randn(n, act_dim, device=device)
        obs, _rew, _term, _trunc, _info = env.step(actions)

    env.close()

    src = vpath / "rl-video-step-0.mp4"
    if src.exists():
        dst = vpath / f"orbit_pin_wrist_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        dst.write_bytes(src.read_bytes())
        print(f"[sg2_rl] Video folder: {vpath.resolve()}", flush=True)
        print(f"[sg2_rl] Copied recording to: {dst.resolve()}", flush=True)
    else:
        print(f"[sg2_rl] Video folder: {vpath.resolve()} (expected rl-video-step-0.mp4 missing)", flush=True)


if __name__ == "__main__":
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    main()
    simulation_app.close()
