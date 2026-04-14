#!/usr/bin/env python3
"""
Video: lower torso (lift rail), approach pin with right-arm IK, close gripper, lift pin.

Phases are time-fractions of ``--video_length`` so length scales with the recording.
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

parser = argparse.ArgumentParser(description="Grasp peg + lift (right arm IK + lift joint + gripper).")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=1200)
parser.add_argument("--video_folder", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--table_z", type=float, default=0.82)
parser.add_argument(
    "--wrist_clearance_m",
    type=float,
    default=0.18,
    help="Min wrist Z = table_z + this for IK clamp.",
)
parser.add_argument(
    "--lift_low",
    type=float,
    default=0.05,
    help="Torso lift_joint target (rad-ish normalized command space) while reaching — lower = body down.",
)
parser.add_argument(
    "--lift_carry",
    type=float,
    default=0.20,
    help="Torso lift target during lift phase (after grasp).",
)
parser.add_argument(
    "--gripper_close_mag",
    type=float,
    default=0.48,
    help="Added to default ``gripper_r_joint*`` positions when closing (same scale as env).",
)
parser.add_argument("--orbit_radius", type=float, default=2.05)
parser.add_argument("--orbit_z_offset", type=float, default=0.52)
parser.add_argument("--orbit_lookat_shift_robot", type=float, default=0.26)
parser.add_argument(
    "--frac_torso",
    type=float,
    default=0.12,
    help="Fraction of steps: lower torso only (no arm IK).",
)
parser.add_argument(
    "--frac_approach",
    type=float,
    default=0.42,
    help="Fraction after torso: IK approach to pre-grasp above pin.",
)
parser.add_argument(
    "--frac_settle",
    type=float,
    default=0.18,
    help="Fraction: move to grasp height, then close gripper in last third of this segment.",
)
parser.add_argument("--scene_shift_x", type=float, default=0.0)
parser.add_argument("--scene_shift_y", type=float, default=0.0)
parser.add_argument("--scene_shift_z", type=float, default=0.0)
parser.add_argument("--peg_offset_x_from_hole", type=float, default=-0.10)
parser.add_argument("--peg_offset_y_from_hole", type=float, default=0.0)
parser.add_argument("--shift_viewer_with_scene", action=argparse.BooleanOptionalAction, default=True)
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
from sg2_rl.arm_avoidance import nudge_ee_des_for_arm_spheres, right_arm_link_check_indices  # noqa: E402
from sg2_rl.apf_path import default_workspace_obstacles  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.orbit_camera import orbit_lookat_shifted_toward_robot  # noqa: E402
from sg2_rl.right_gripper_ik import (  # noqa: E402
    actions_for_ee_goal,
    actions_lift_only,
    build_right_gripper_ik,
)
from sg2_rl.scene_layout import apply_peg_hole_workspace_shift  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


def _phase_bounds(video_length: int, f_torso: float, f_appr: float, f_settle: float) -> tuple[int, int, int]:
    """Return end indices (exclusive): torso [0,p1), approach [p1,p2), settle [p2,p3), lift [p3,N)."""
    n = max(12, int(video_length))
    s = f_torso + f_appr + f_settle
    if s > 0.95:
        scale = 0.92 / s
        f_torso *= scale
        f_appr *= scale
        f_settle *= scale
    p1 = max(2, int(f_torso * n))
    p2 = max(p1 + 2, int((f_torso + f_appr) * n))
    p3 = max(p2 + 2, int((f_torso + f_appr + f_settle) * n))
    p3 = min(p3, n - 2)
    p2 = min(p2, p3 - 1)
    p1 = min(p1, p2 - 1)
    return p1, p2, p3


@hydra_task_compose(args_cli.task, "skrl_cfg_entry_point", [])
def main(env_cfg, agent_cfg):
    vf = args_cli.video_folder.strip() or str(
        _REPO_ROOT / "artifacts" / "videos" / f"grasp_lift_peg_{time.strftime('%Y%m%d_%H%M%S')}"
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
    pox, poy = float(args_cli.peg_offset_x_from_hole), float(args_cli.peg_offset_y_from_hole)
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
    device = torch.device(env_cfg.sim.device if str(env_cfg.sim.device).startswith("cuda") else "cuda:0")
    n = args_cli.num_envs

    robot = unwrapped.scene["robot"]
    peg = unwrapped.scene["insertive_object"]
    ctx = build_right_gripper_ik(robot, unwrapped.action_manager, device, n)
    arm_link_ids = right_arm_link_check_indices(robot) or [ctx.wrist_body_idx]
    peg0 = peg.data.root_pos_w[0, :3].detach().cpu().numpy()
    spheres = default_workspace_obstacles(peg0)

    lift_low = float(args_cli.lift_low)
    lift_carry = float(args_cli.lift_carry)
    gcm = float(args_cli.gripper_close_mag)
    min_z = torch.full((n,), float(args_cli.table_z) + float(args_cli.wrist_clearance_m), device=device)

    n_total = args_cli.video_length
    p1, p2, p3 = _phase_bounds(
        n_total,
        float(args_cli.frac_torso),
        float(args_cli.frac_approach),
        float(args_cli.frac_settle),
    )
    print(
        f"[sg2_rl] grasp_lift_peg phases steps [0,{p1}) torso, [{p1},{p2}) approach, "
        f"[{p2},{p3}) settle+close, [{p3},{n_total}) lift | lift_low={lift_low} carry={lift_carry}",
        flush=True,
    )

    lift_start = float(robot.data.joint_pos[:, ctx.lift_term_ids].mean().item()) if ctx.lift_dim > 0 else lift_low

    look = orbit_lookat_shifted_toward_robot(
        env_cfg, robot, peg, shift_xy_m=float(args_cli.orbit_lookat_shift_robot)
    )
    lx, ly, lz = look
    n_steps = max(1, n_total - 1)

    for step in range(n_total):
        theta = math.pi * (step / n_steps)
        r = float(args_cli.orbit_radius)
        eye = (lx + r * math.cos(theta), ly + r * math.sin(theta), lz + float(args_cli.orbit_z_offset))
        unwrapped.sim.set_camera_view(eye=eye, target=look)

        peg_xyz = peg.data.root_pos_w[:, :3].to(device)
        px, py, pz = peg_xyz[:, 0:1], peg_xyz[:, 1:2], peg_xyz[:, 2:3]

        if step < p1:
            u = step / max(p1 - 1, 1)
            lift_t = float(np.interp(u, [0.0, 1.0], [lift_start, lift_low]))
            actions = actions_lift_only(ctx, robot, lift_t)
        elif step < p2:
            u = (step - p1) / max(p2 - p1 - 1, 1)
            u = float(np.clip(u, 0.0, 1.0))
            ox = float(np.interp(u, [0.0, 1.0], [0.04, 0.0]))
            oy = float(np.interp(u, [0.0, 1.0], [-0.07, -0.06]))
            oz = float(np.interp(u, [0.0, 1.0], [0.20, 0.14]))
            ee_des = torch.cat([px + ox, py + oy, pz + oz], dim=-1)
            link_pos = robot.data.body_link_pos_w[:, arm_link_ids].to(device)
            ee_des = nudge_ee_des_for_arm_spheres(ee_des, link_pos, spheres)
            actions = actions_for_ee_goal(
                ctx, robot, ee_des, min_z=min_z, lift_target=lift_low, gripper_r_close=False
            )
        elif step < p3:
            u = (step - p2) / max(p3 - p2 - 1, 1)
            u = float(np.clip(u, 0.0, 1.0))
            ox = float(np.interp(u, [0.0, 1.0], [0.0, 0.0]))
            oy = float(np.interp(u, [0.0, 1.0], [-0.06, -0.05]))
            oz = float(np.interp(u, [0.0, 1.0], [0.12, 0.07]))
            ee_des = torch.cat([px + ox, py + oy, pz + oz], dim=-1)
            link_pos = robot.data.body_link_pos_w[:, arm_link_ids].to(device)
            ee_des = nudge_ee_des_for_arm_spheres(ee_des, link_pos, spheres)
            close = step > p2 + int(0.65 * max(p3 - p2 - 1, 1))
            actions = actions_for_ee_goal(
                ctx,
                robot,
                ee_des,
                min_z=min_z,
                lift_target=lift_low,
                gripper_r_close=close,
                gripper_r_close_mag=gcm,
            )
        else:
            u = (step - p3) / max(n_total - p3 - 1, 1)
            u = float(np.clip(u, 0.0, 1.0))
            ox, oy = 0.0, -0.04
            oz = float(np.interp(u, [0.0, 1.0], [0.12, 0.28]))
            lift_t = float(np.interp(u, [0.0, 1.0], [lift_low, lift_carry]))
            ee_des = torch.cat([px + ox, py + oy, pz + oz], dim=-1)
            link_pos = robot.data.body_link_pos_w[:, arm_link_ids].to(device)
            ee_des = nudge_ee_des_for_arm_spheres(ee_des, link_pos, spheres)
            actions = actions_for_ee_goal(
                ctx,
                robot,
                ee_des,
                min_z=min_z,
                lift_target=lift_t,
                gripper_r_close=True,
                gripper_r_close_mag=gcm,
            )

        env.step(actions)

    env.close()

    src = vpath / "rl-video-step-0.mp4"
    if src.exists():
        dst = vpath / f"grasp_lift_peg_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4"
        dst.write_bytes(src.read_bytes())
        print(f"[sg2_rl] Video folder: {vpath.resolve()}", flush=True)
        print(f"[sg2_rl] Wrote: {dst.resolve()}", flush=True)
    else:
        print(f"[sg2_rl] Video folder: {vpath.resolve()} (missing rl-video-step-0.mp4)", flush=True)


if __name__ == "__main__":
    ensure_task_registered(args_cli.task, args_cli.skrl_yaml)
    main()
    simulation_app.close()
