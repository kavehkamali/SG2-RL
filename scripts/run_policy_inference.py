#!/usr/bin/env python3
"""Run a trained SKRL policy checkpoint and optionally record video.

This script is intended for Isaac machines (Isaac Sim/Lab installed).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Run SG2-RL PPO policy inference/eval and record a video.")
parser.add_argument("--task", type=str, default="FFWSG2-PegGraspLift-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--video_length", type=int, default=1440)
parser.add_argument("--video_folder", type=str, default="")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved SKRL checkpoint file.")
parser.add_argument("--skrl_yaml", type=str, default="", help="Override SKRL yaml path (rare).")
parser.add_argument("--orbit_radius", type=float, default=2.05)
parser.add_argument("--orbit_z_offset", type=float, default=0.52)
parser.add_argument("--orbit_lookat_shift_robot", type=float, default=0.26)
parser.add_argument(
    "--peg_offset_x_from_hole",
    type=float,
    default=None,
    help="Override initial peg X offset from hole (m). Stock is +0.10 (pin away from robot). "
    "Use +0.05 to bring the pin closer by half.",
)
parser.add_argument("--peg_offset_y_from_hole", type=float, default=None)
parser.add_argument("--cluster_shift_x", type=float, default=0.0, help="Shift peg/hole/table cluster in world X (m).")
parser.add_argument("--cluster_shift_y", type=float, default=0.0)
parser.add_argument("--cluster_shift_z", type=float, default=0.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.video)
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_REPO_ROOT / "src"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from sg2_rl.config_loader import load_task_cfg  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.render_quality import enable_high_quality, warm_up_renderer  # noqa: E402

if bool(args_cli.video):
    enable_high_quality(resolution=(1920, 1080))


def main() -> None:
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    env_cfg, agent_cfg = load_task_cfg(args_cli.task, "skrl_cfg_entry_point")

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)

    # Optional: override initial cluster shift and peg placement relative to the hole.
    if (
        float(args_cli.cluster_shift_x) != 0.0
        or float(args_cli.cluster_shift_y) != 0.0
        or float(args_cli.cluster_shift_z) != 0.0
        or args_cli.peg_offset_x_from_hole is not None
        or args_cli.peg_offset_y_from_hole is not None
    ):
        from sg2_rl.scene_layout import apply_peg_hole_workspace_shift  # noqa: E402

        apply_peg_hole_workspace_shift(
            env_cfg,
            float(args_cli.cluster_shift_x),
            float(args_cli.cluster_shift_y),
            float(args_cli.cluster_shift_z),
            peg_offset_x_from_hole=float(
                args_cli.peg_offset_x_from_hole if args_cli.peg_offset_x_from_hole is not None else -0.15
            ),
            peg_offset_y_from_hole=float(args_cli.peg_offset_y_from_hole if args_cli.peg_offset_y_from_hole is not None else 0.0),
            shift_viewer=False,
        )

    if bool(args_cli.video):
        vf = args_cli.video_folder.strip() or str(
            _REPO_ROOT / "artifacts" / "videos" / f"inference_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        vpath = Path(vf)
        vpath.mkdir(parents=True, exist_ok=True)
        env_cfg.log_dir = str(vpath / "run")
    else:
        vpath = None
        env_cfg.log_dir = str(_REPO_ROOT / "artifacts" / "inference_run")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if bool(args_cli.video) else None)
    if bool(args_cli.video):
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(vpath),
            step_trigger=lambda step: step == 0,
            video_length=int(args_cli.video_length),
            disable_logger=True,
        )

    from skrl.envs.wrappers.torch import wrap_env  # noqa: E402
    from skrl.utils.runner.torch import Runner  # noqa: E402

    unwrapped = env.unwrapped
    if bool(args_cli.video):
        warm_up_renderer(unwrapped.sim, num_steps=30)

    device = env_cfg.sim.device
    if not (isinstance(device, str) and device.startswith("cuda")):
        device = "cuda:0"

    # Orbit camera setup (video B style)
    robot = unwrapped.scene["robot"]
    peg = unwrapped.scene["insertive_object"]
    if bool(args_cli.video):
        from sg2_rl.orbit_camera import orbit_lookat_shifted_toward_robot  # noqa: E402

        look = orbit_lookat_shifted_toward_robot(env_cfg, robot, peg, shift_xy_m=float(args_cli.orbit_lookat_shift_robot))
        lx, ly, lz = look
        n_steps = max(1, int(args_cli.video_length) - 1)

    # Wrap env for SKRL and build agent from yaml, then load checkpoint.
    skrl_env = wrap_env(env, wrapper="isaaclab")
    cfg = Runner.load_cfg_from_yaml(str(args_cli.skrl_yaml) if str(args_cli.skrl_yaml).strip() else str(agent_cfg))
    runner = Runner(skrl_env, cfg, verbose=False)
    agent = runner.agent

    ckpt_path = Path(str(args_cli.checkpoint)).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    agent.load(str(ckpt_path))
    agent.enable_training_mode(False)

    obs, _ = skrl_env.reset()

    steps = int(args_cli.steps)
    if bool(args_cli.video):
        steps = max(steps, int(args_cli.video_length))

    for step in range(steps):
        if bool(args_cli.video):
            import math

            theta = math.pi * (step / max(n_steps, 1))
            r = float(args_cli.orbit_radius)
            eye = (lx + r * math.cos(theta), ly + r * math.sin(theta), lz + float(args_cli.orbit_z_offset))
            unwrapped.sim.set_camera_view(eye=eye, target=look)

        with torch.no_grad():
            actions, _outputs = agent.act(obs, None, timestep=step, timesteps=steps)
        obs, _rew, _term, _trunc, _info = skrl_env.step(actions)

    skrl_env.close()

    if vpath is not None:
        src = vpath / "rl-video-step-0.mp4"
        if src.exists():
            dst = vpath / f"inference_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
            dst.write_bytes(src.read_bytes())
            print(f"[sg2_rl] Video folder: {vpath.resolve()}", flush=True)
            print(f"[sg2_rl] Wrote: {dst.resolve()}", flush=True)
        else:
            print(f"[sg2_rl] Video folder: {vpath.resolve()} (missing rl-video-step-0.mp4)", flush=True)

    print(f"[sg2_rl] Done. steps={steps} num_envs={args_cli.num_envs}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()

