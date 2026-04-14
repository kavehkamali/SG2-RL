#!/usr/bin/env python3
"""
Video B: same amber **APF path** polyline + **right gripper follows** the path (differential IK).
No RGB coordinate gizmos.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="APF path + right gripper tracks path (video B).")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--video_length",
    type=int,
    default=720,
    help="Longer default (2× prior 360) so the full gripper motion is visible.",
)
parser.add_argument("--video_folder", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--table_z", type=float, default=0.82)
parser.add_argument("--wrist_clearance_m", type=float, default=0.10)
parser.add_argument("--goal_z_above_pin", type=float, default=0.07)
parser.add_argument("--orbit_radius", type=float, default=2.05)
parser.add_argument("--orbit_z_offset", type=float, default=0.52)
parser.add_argument(
    "--orbit_lookat_shift_robot",
    type=float,
    default=0.26,
    help="Shift orbit look-at toward robot in XY (m); 0 disables.",
)
parser.add_argument("--skrl_yaml", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_REPO_ROOT / "src"))

import gymnasium as gym  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from sg2_rl.apf_path import default_workspace_obstacles, plan_apf_polyline  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.orbit_camera import orbit_lookat_shifted_toward_robot  # noqa: E402
from sg2_rl.right_gripper_ik import actions_for_ee_goal, build_right_gripper_ik  # noqa: E402
from sg2_rl.usd_path_curve import draw_planned_path_polyline  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


def _interp_path(path: list[list[float]], s01: float) -> np.ndarray:
    if len(path) < 2:
        return np.array(path[0], dtype=np.float64)
    s01 = float(np.clip(s01, 0.0, 1.0))
    f = s01 * (len(path) - 1)
    i0 = int(math.floor(f))
    i1 = min(i0 + 1, len(path) - 1)
    a = f - i0
    p0 = np.array(path[i0], dtype=np.float64)
    p1 = np.array(path[i1], dtype=np.float64)
    return (1.0 - a) * p0 + a * p1


@hydra_task_compose(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    vf = args_cli.video_folder.strip() or str(
        _REPO_ROOT / "artifacts" / "videos" / f"apf_path_follow_{time.strftime('%Y%m%d_%H%M%S')}"
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
    device = torch.device(env_cfg.sim.device if str(env_cfg.sim.device).startswith("cuda") else "cuda:0")
    n = args_cli.num_envs

    robot = unwrapped.scene["robot"]
    peg = unwrapped.scene["insertive_object"]

    ctx = build_right_gripper_ik(robot, unwrapped.action_manager, device, n)
    wrist_body_idx = ctx.wrist_body_idx
    wrist0 = robot.data.body_link_pos_w[:, wrist_body_idx][0].detach().cpu().numpy()
    peg0 = peg.data.root_pos_w[0].detach().cpu().numpy()
    goal = peg0.copy()
    goal[2] += float(args_cli.goal_z_above_pin)

    path = plan_apf_polyline(
        wrist0,
        goal,
        table_z=float(args_cli.table_z),
        wrist_clearance_m=float(args_cli.wrist_clearance_m),
        sphere_obstacles=default_workspace_obstacles(peg0),
    )
    print(f"[sg2_rl] APF path vertices={len(path)}; IK tracks polyline (video B)", flush=True)

    min_z = torch.full((n,), float(args_cli.table_z) + float(args_cli.wrist_clearance_m), device=device)

    look = orbit_lookat_shifted_toward_robot(
        env_cfg, robot, peg, shift_xy_m=float(args_cli.orbit_lookat_shift_robot)
    )
    lx, ly, lz = look
    n_steps = max(1, args_cli.video_length - 1)

    import omni.usd  # type: ignore

    for step in range(args_cli.video_length):
        theta = math.pi * (step / n_steps)
        r = float(args_cli.orbit_radius)
        eye = (lx + r * math.cos(theta), ly + r * math.sin(theta), lz + float(args_cli.orbit_z_offset))
        unwrapped.sim.set_camera_view(eye=eye, target=look)

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            draw_planned_path_polyline(stage, "/World/SG2RL/apf_gripper_path", path, width=0.006)

        s01 = step / max(args_cli.video_length - 1, 1)
        tgt = _interp_path(path, s01)
        ee_des = torch.tensor(tgt, device=device, dtype=torch.float32).unsqueeze(0).expand(n, -1)

        actions = actions_for_ee_goal(ctx, robot, ee_des, min_z=min_z)
        env.step(actions)

    env.close()

    src = vpath / "rl-video-step-0.mp4"
    if src.exists():
        dst = vpath / f"apf_path_follow_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        dst.write_bytes(src.read_bytes())
        print(f"[sg2_rl] Video folder: {vpath.resolve()}", flush=True)
        print(f"[sg2_rl] Wrote: {dst.resolve()}", flush=True)
    else:
        print(f"[sg2_rl] Video folder: {vpath.resolve()} (missing rl-video-step-0.mp4)", flush=True)


if __name__ == "__main__":
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    main()
    simulation_app.close()
