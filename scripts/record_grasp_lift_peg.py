#!/usr/bin/env python3
"""
Video: lower torso, approach pin (monitored wrist–pin distance), grasp, lift when peg rises.

Uses distance streaks to advance phases early; monitors peg height vs reset / table.
Writes ``grasp_monitor.csv`` next to the video and prints a one-line summary.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

PH_TORSO, PH_APPROACH, PH_SETTLE, PH_LIFT = 0, 1, 2, 3

parser = argparse.ArgumentParser(description="Grasp peg + lift with wrist–pin monitoring (right arm IK).")
parser.add_argument("--task", type=str, default="OmniReset-FFWSG2-PegPartialAssemblySmoke-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--video_length", type=int, default=1500)
parser.add_argument("--video_folder", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--table_z", type=float, default=0.82)
parser.add_argument(
    "--wrist_clearance_m",
    type=float,
    default=0.18,
    help="Min wrist Z = table_z + this for IK clamp.",
)
parser.add_argument("--lift_low", type=float, default=0.05)
parser.add_argument("--lift_carry", type=float, default=0.20)
parser.add_argument("--gripper_close_mag", type=float, default=0.48)
parser.add_argument("--orbit_radius", type=float, default=2.05)
parser.add_argument("--orbit_z_offset", type=float, default=0.52)
parser.add_argument("--orbit_lookat_shift_robot", type=float, default=0.26)
parser.add_argument(
    "--frac_torso",
    type=float,
    default=0.10,
    help="Minimum time in torso-only (fraction of video_length) before arm IK.",
)
parser.add_argument(
    "--frac_approach",
    type=float,
    default=0.35,
    help="Nominal approach segment length (fraction); also caps wait before forced advance.",
)
parser.add_argument(
    "--frac_settle",
    type=float,
    default=0.22,
    help="Nominal settle+close segment (fraction).",
)
parser.add_argument(
    "--dist_approach_ok",
    type=float,
    default=0.14,
    help="Wrist–pin distance (m) below which we may leave approach after --dist_hold_frames.",
)
parser.add_argument(
    "--dist_grasp_ok",
    type=float,
    default=0.095,
    help="Wrist–pin distance (m) below which we may leave settle after --dist_hold_frames.",
)
parser.add_argument("--dist_hold_frames", type=int, default=5)
parser.add_argument(
    "--lift_dz_success",
    type=float,
    default=0.036,
    help="Peg Z rise vs reset (m) to count as lifted.",
)
parser.add_argument(
    "--lift_z_above_table",
    type=float,
    default=0.10,
    help="Alternate lift detect: peg Z - table_z >= this (m).",
)
parser.add_argument("--lifted_hold_frames", type=int, default=6)
parser.add_argument("--monitor_every", type=int, default=20)
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
from sg2_rl.peg_grasp_monitor import pin_lifted, streak_update  # noqa: E402
from sg2_rl.right_gripper_ik import (  # noqa: E402
    actions_for_ee_goal,
    actions_lift_only,
    build_right_gripper_ik,
)
from sg2_rl.scene_layout import apply_peg_hole_workspace_shift  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


def _bounds(n: int, f_t: float, f_a: float, f_s: float) -> tuple[int, int, int]:
    n = max(24, int(n))
    s = f_t + f_a + f_s
    if s > 0.94:
        k = 0.90 / s
        f_t, f_a, f_s = f_t * k, f_a * k, f_s * k
    p1 = max(3, int(f_t * n))
    p2 = max(p1 + 4, int((f_t + f_a) * n))
    p3 = max(p2 + 4, int((f_t + f_a + f_s) * n))
    p3 = min(p3, n - 8)
    p2 = min(p2, p3 - 2)
    p1 = min(p1, p2 - 2)
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

    n_total = int(args_cli.video_length)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(vpath),
        step_trigger=lambda step: step == 0,
        video_length=n_total,
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
    peg_z0 = float(peg0[2])
    tbl = float(args_cli.table_z)

    lift_low = float(args_cli.lift_low)
    lift_carry = float(args_cli.lift_carry)
    gcm = float(args_cli.gripper_close_mag)
    min_z = torch.full((n,), tbl + float(args_cli.wrist_clearance_m), device=device)

    p1, p2, p3 = _bounds(n_total, float(args_cli.frac_torso), float(args_cli.frac_approach), float(args_cli.frac_settle))
    d_appr = float(args_cli.dist_approach_ok)
    d_grasp = float(args_cli.dist_grasp_ok)
    d_hold = int(args_cli.dist_hold_frames)
    dz_succ = float(args_cli.lift_dz_success)
    z_tbl = float(args_cli.lift_z_above_table)
    lift_hold = int(args_cli.lifted_hold_frames)
    mon = int(args_cli.monitor_every)

    lift_start = float(robot.data.joint_pos[:, ctx.lift_term_ids].mean().item()) if ctx.lift_dim > 0 else lift_low

    look = orbit_lookat_shifted_toward_robot(
        env_cfg, robot, peg, shift_xy_m=float(args_cli.orbit_lookat_shift_robot)
    )
    lx, ly, lz = look
    n_steps = max(1, n_total - 1)

    phase = PH_TORSO
    phase_enter = 0
    appr_streak = grasp_streak = lifted_streak = 0
    min_dist = 1e9
    max_dz = -1e9
    lift_success_step: int | None = None
    rows: list[list[object]] = []

    nom_ap = max(30, p2 - p1 - 1)
    nom_st = max(25, p3 - p2 - 1)
    nom_lf = max(50, n_total - p3 - 2)

    print(
        f"[sg2_rl] grasp_lift_peg monitor caps p1={p1} p2={p2} p3={p3} n={n_total} | "
        f"dist_appr<{d_appr} hold={d_hold} | dist_grasp<{d_grasp} | lift dz>{dz_succ} or z-table>{z_tbl}",
        flush=True,
    )

    for step in range(n_total):
        theta = math.pi * (step / n_steps)
        r = float(args_cli.orbit_radius)
        eye = (lx + r * math.cos(theta), ly + r * math.sin(theta), lz + float(args_cli.orbit_z_offset))
        unwrapped.sim.set_camera_view(eye=eye, target=look)

        wrist = robot.data.body_link_pos_w[:, ctx.wrist_body_idx, :3].to(device)
        peg_xyz = peg.data.root_pos_w[:, :3].to(device)
        dist = float(torch.norm(wrist[0] - peg_xyz[0]).item())
        peg_z = float(peg_xyz[0, 2].item())
        dz = peg_z - peg_z0
        min_dist = min(min_dist, dist)
        max_dz = max(max_dz, dz)
        lifted = pin_lifted(peg_z, peg_z0, table_z=tbl, dz_min=dz_succ, z_clear_above_table=z_tbl)

        peg_np = peg.data.root_pos_w[0, :3].detach().cpu().numpy()
        spheres = default_workspace_obstacles(peg_np)
        px, py, pz = peg_xyz[:, 0:1], peg_xyz[:, 1:2], peg_xyz[:, 2:3]

        if phase == PH_TORSO:
            appr_streak = grasp_streak = lifted_streak = 0
            if step >= p1:
                phase = PH_APPROACH
                phase_enter = step
        elif phase == PH_APPROACH:
            appr_streak = streak_update(dist < d_appr, appr_streak)
            if appr_streak >= d_hold or step >= p2:
                phase = PH_SETTLE
                phase_enter = step
                appr_streak = 0
        elif phase == PH_SETTLE:
            grasp_streak = streak_update(dist < d_grasp, grasp_streak)
            if grasp_streak >= d_hold or step >= p3:
                phase = PH_LIFT
                phase_enter = step
                grasp_streak = 0
        elif phase == PH_LIFT:
            lifted_streak = streak_update(lifted, lifted_streak)
            if lift_success_step is None and lifted_streak >= lift_hold and lifted:
                lift_success_step = step

        if phase == PH_TORSO:
            u = step / max(p1 - 1, 1)
            lift_t = float(np.interp(u, [0.0, 1.0], [lift_start, lift_low]))
            actions = actions_lift_only(ctx, robot, lift_t)
        elif phase == PH_APPROACH:
            t = step - phase_enter
            u = min(1.0, t / max(nom_ap, 1))
            ox = float(np.interp(u, [0.0, 1.0], [0.04, 0.0]))
            oy = float(np.interp(u, [0.0, 1.0], [-0.07, -0.06]))
            oz = float(np.interp(u, [0.0, 1.0], [0.20, 0.14]))
            ee_des = torch.cat([px + ox, py + oy, pz + oz], dim=-1)
            link_pos = robot.data.body_link_pos_w[:, arm_link_ids].to(device)
            ee_des = nudge_ee_des_for_arm_spheres(ee_des, link_pos, spheres)
            actions = actions_for_ee_goal(
                ctx, robot, ee_des, min_z=min_z, lift_target=lift_low, gripper_r_close=False
            )
        elif phase == PH_SETTLE:
            t = step - phase_enter
            u = min(1.0, t / max(nom_st, 1))
            ox = float(np.interp(u, [0.0, 1.0], [0.0, 0.0]))
            oy = float(np.interp(u, [0.0, 1.0], [-0.06, -0.05]))
            oz = float(np.interp(u, [0.0, 1.0], [0.12, 0.07]))
            ee_des = torch.cat([px + ox, py + oy, pz + oz], dim=-1)
            link_pos = robot.data.body_link_pos_w[:, arm_link_ids].to(device)
            ee_des = nudge_ee_des_for_arm_spheres(ee_des, link_pos, spheres)
            close = t > int(0.55 * nom_st)
            actions = actions_for_ee_goal(
                ctx,
                robot,
                ee_des,
                min_z=min_z,
                lift_target=lift_low,
                gripper_r_close=close,
                gripper_r_close_mag=gcm,
            )
        elif phase == PH_LIFT:
            t = step - phase_enter
            if lift_success_step is not None and step >= lift_success_step:
                u = 1.0
            else:
                u = min(1.0, t / max(nom_lf, 1))
            ox, oy = 0.0, -0.04
            oz = float(np.interp(u, [0.0, 1.0], [0.10, 0.28]))
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

        rows.append(
            [
                step,
                phase,
                f"{dist:.5f}",
                f"{peg_z:.5f}",
                f"{dz:.5f}",
                int(lifted),
                appr_streak if phase <= PH_APPROACH else 0,
                grasp_streak if phase == PH_SETTLE else 0,
                lifted_streak if phase >= PH_LIFT else 0,
            ]
        )

        if mon > 0 and (step % mon == 0 or step == n_total - 1):
            print(
                f"[sg2_rl] step={step} phase={phase} dist={dist:.3f} peg_z={peg_z:.3f} dz={dz:.3f} "
                f"lifted={int(lifted)} min_dist={min_dist:.3f} max_dz={max_dz:.3f}",
                flush=True,
            )

        env.step(actions)

    csv_path = vpath / "grasp_monitor.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "step",
                "phase",
                "dist_wrist_pin_m",
                "peg_z_m",
                "dz_vs_reset_m",
                "lifted_flag",
                "appr_streak",
                "grasp_streak",
                "lifted_streak",
            ]
        )
        w.writerows(rows)

    ok = lift_success_step is not None
    print(
        f"[sg2_rl] monitor summary success={ok} lift_step={lift_success_step} "
        f"min_dist_m={min_dist:.4f} max_dz_m={max_dz:.4f} csv={csv_path.resolve()}",
        flush=True,
    )

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
