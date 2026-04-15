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
    default=1440,
    help="Default 1440 frames (2× prior 720) so the full gripper motion is visible.",
)
parser.add_argument("--video_folder", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--table_z", type=float, default=0.82)
parser.add_argument(
    "--wrist_clearance_m",
    type=float,
    default=0.18,
    help="Min wrist Z = table_z + this (m); larger keeps the gripper off the tabletop in APF + IK.",
)
parser.add_argument("--goal_z_above_pin", type=float, default=0.07)
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
    help="World delta (m) on peg, hole, and table together; default 0 keeps packaged hole position.",
)
parser.add_argument("--scene_shift_y", type=float, default=0.0)
parser.add_argument("--scene_shift_z", type=float, default=0.0)
parser.add_argument(
    "--peg_offset_x_from_hole",
    type=float,
    default=-0.10,
    help="After cluster shift, peg world X = hole X + this (m). Stock layout uses +0.10 (pin away from robot); "
    "-0.10 keeps the same spacing with the pin toward the robot.",
)
parser.add_argument(
    "--peg_offset_y_from_hole",
    type=float,
    default=0.0,
    help="Peg world Y = hole Y + this (m) after cluster shift.",
)
parser.add_argument(
    "--shift_viewer_with_scene",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Shift env viewer eye/lookat by the same cluster delta.",
)
parser.add_argument("--skrl_yaml", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_REPO_ROOT / "src"))

from sg2_rl.render_quality import enable_high_quality, warm_up_renderer  # noqa: E402
enable_high_quality(resolution=(1920, 1080))

import gymnasium as gym  # noqa: E402

# (removed: isaaclab_tasks — tasks registered locally)
# (removed: uwlab_tasks — replaced by sg2_rl.env_cfg)
from sg2_rl.apf_path import default_workspace_obstacles, plan_apf_polyline  # noqa: E402
from sg2_rl.arm_avoidance import (  # noqa: E402
    nudge_ee_des_for_arm_spheres,
    pick_right_arm_line_base_xyz,
    right_arm_link_check_indices,
)
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.orbit_camera import orbit_lookat_shifted_toward_robot  # noqa: E402
from sg2_rl.right_gripper_ik import actions_for_ee_goal, build_right_gripper_ik  # noqa: E402
from sg2_rl.scene_layout import apply_peg_hole_workspace_shift  # noqa: E402
from sg2_rl.usd_path_curve import draw_planned_path_polyline  # noqa: E402
from sg2_rl.config_loader import task_config  # noqa: E402


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


@task_config(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    vf = args_cli.video_folder.strip() or str(
        _REPO_ROOT / "artifacts" / "videos" / f"apf_path_follow_{time.strftime('%Y%m%d_%H%M%S')}"
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
    warm_up_renderer(unwrapped.sim, num_steps=30)
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
    arm_link_ids = right_arm_link_check_indices(robot) or [wrist_body_idx]
    print(
        f"[sg2_rl] APF path vertices={len(path)}; IK tracks polyline; "
        f"arm sphere avoidance links={len(arm_link_ids)} base={'set' if arm_base is not None else 'off'} (video B)",
        flush=True,
    )

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
        link_pos = robot.data.body_link_pos_w[:, arm_link_ids].to(device)
        ee_des = nudge_ee_des_for_arm_spheres(ee_des, link_pos, spheres)

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
