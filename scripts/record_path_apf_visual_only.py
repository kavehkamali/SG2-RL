#!/usr/bin/env python3
"""
Video A: orbit camera + **APF-planned right-gripper path** (amber polyline only).
No RGB coordinate gizmos. Robot gets small random actions (path is static world overlay).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="APF path visualization only (video A).")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_folder", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--table_z", type=float, default=0.82)
parser.add_argument(
    "--wrist_clearance_m",
    type=float,
    default=0.18,
    help="APF table Z margin (m): path clamp at table_z + this.",
)
parser.add_argument("--goal_z_above_pin", type=float, default=0.07, help="Pre-grasp goal offset above peg (m).")
parser.add_argument("--orbit_radius", type=float, default=2.05)
parser.add_argument("--orbit_z_offset", type=float, default=0.52)
parser.add_argument(
    "--orbit_lookat_shift_robot",
    type=float,
    default=0.26,
    help="Shift orbit look-at toward robot in XY (m); 0 disables.",
)
parser.add_argument(
    "--scene_shift_x",
    type=float,
    default=0.0,
    help="World delta (m) on peg, hole, and table; default 0 keeps packaged hole position.",
)
parser.add_argument("--scene_shift_y", type=float, default=0.0)
parser.add_argument("--scene_shift_z", type=float, default=0.0)
parser.add_argument(
    "--peg_offset_x_from_hole",
    type=float,
    default=-0.10,
    help="Peg X = hole X + this (m) after cluster; -0.10 mirrors stock +0.10 toward the robot.",
)
parser.add_argument("--peg_offset_y_from_hole", type=float, default=0.0)
parser.add_argument(
    "--shift_viewer_with_scene",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Shift viewer eye/lookat with the cluster delta.",
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
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from sg2_rl.apf_path import default_workspace_obstacles, plan_apf_polyline  # noqa: E402
from sg2_rl.arm_avoidance import pick_right_arm_line_base_xyz  # noqa: E402
from sg2_rl.scene_layout import apply_peg_hole_workspace_shift  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.orbit_camera import orbit_lookat_shifted_toward_robot  # noqa: E402
from sg2_rl.usd_path_curve import draw_planned_path_polyline  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


@hydra_task_compose(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    vf = args_cli.video_folder.strip() or str(
        _REPO_ROOT / "artifacts" / "videos" / f"apf_path_visual_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    vpath = Path(vf)
    vpath.mkdir(parents=True, exist_ok=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.log_dir = str(vpath / "run")

    sdx, sdy, sdz = (
        float(args_cli.scene_shift_x),
        float(args_cli.scene_shift_y),
        float(args_cli.scene_shift_z),
    )
    pox = float(args_cli.peg_offset_x_from_hole)
    poy = float(args_cli.peg_offset_y_from_hole)
    if sdx != 0.0 or sdy != 0.0 or sdz != 0.0 or pox != 0.0 or poy != 0.0:
        apply_peg_hole_workspace_shift(
            env_cfg,
            sdx,
            sdy,
            sdz,
            peg_offset_x_from_hole=pox,
            peg_offset_y_from_hole=poy,
            shift_viewer=bool(args_cli.shift_viewer_with_scene),
        )

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
        raise RuntimeError(wrist_names)
    wrist_body_idx = wrist_ids[0]

    wrist0 = robot.data.body_link_pos_w[:, wrist_body_idx][0].detach().cpu().numpy()
    peg0 = peg.data.root_pos_w[0].detach().cpu().numpy()
    goal = peg0.copy()
    goal[2] += float(args_cli.goal_z_above_pin)

    arm_base = pick_right_arm_line_base_xyz(robot)
    spheres = default_workspace_obstacles(peg0)
    path = plan_apf_polyline(
        wrist0,
        goal,
        table_z=float(args_cli.table_z),
        wrist_clearance_m=float(args_cli.wrist_clearance_m),
        sphere_obstacles=spheres,
        arm_repulse_base_xyz=arm_base,
    )
    print(
        f"[sg2_rl] APF path vertices={len(path)} (Khatib APF + spheres + table; "
        f"arm-aware repulse base={'set' if arm_base is not None else 'off'})",
        flush=True,
    )

    look = orbit_lookat_shifted_toward_robot(
        env_cfg, robot, peg, shift_xy_m=float(args_cli.orbit_lookat_shift_robot)
    )
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

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            draw_planned_path_polyline(stage, "/World/SG2RL/apf_gripper_path", path, width=0.006)

        actions = 0.04 * torch.randn(n, act_dim, device=device)
        env.step(actions)

    env.close()

    src = vpath / "rl-video-step-0.mp4"
    if src.exists():
        dst = vpath / f"apf_path_visual_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        dst.write_bytes(src.read_bytes())
        print(f"[sg2_rl] Video folder: {vpath.resolve()}", flush=True)
        print(f"[sg2_rl] Wrote: {dst.resolve()}", flush=True)
    else:
        print(f"[sg2_rl] Video folder: {vpath.resolve()} (missing rl-video-step-0.mp4)", flush=True)


if __name__ == "__main__":
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    main()
    simulation_app.close()
