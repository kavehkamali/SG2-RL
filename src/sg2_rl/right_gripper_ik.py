"""Right wrist (arm_r_link7) differential IK → JointPositionAction raw vector (FFW SG2 env layout)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


@dataclass
class RightGripperIKContext:
    device: torch.device
    num_envs: int
    act_dim: int
    wrist_body_idx: int
    jacobi_body_idx: int
    jacobi_joint_ids: list[int]
    right_joint_ids: list[int]
    right_arm_indices_in_arms: list[int]
    lift_slice: slice
    arms_slice: slice
    wheels_slice: slice
    lift_dim: int
    arms_dim: int
    lift_term_ids: list[int]
    arms_term_ids: list[int]
    lift_joint_id: int | None
    ik: DifferentialIKController
    lift_scale: float = 0.22
    arms_scale: float = 0.08
    lift_target: float = 0.22
    max_joint_step: float = 0.05
    max_ee_step_m: float = 0.012
    raw_clip: float = 20.0


def build_right_gripper_ik(robot, action_manager, device: torch.device, num_envs: int) -> RightGripperIKContext:
    (wrist_ids, wrist_names) = robot.find_bodies("arm_r_link7")
    if len(wrist_ids) != 1:
        raise RuntimeError(f"Expected one arm_r_link7, got {wrist_names}")
    wrist_body_idx = wrist_ids[0]

    right_joint_ids, right_joint_names = robot.find_joints(["arm_r_joint.*"])
    if not right_joint_ids:
        raise RuntimeError("No arm_r_joint.*")

    if robot.is_fixed_base:
        jacobi_body_idx = wrist_body_idx - 1
        jacobi_joint_ids = right_joint_ids
    else:
        jacobi_body_idx = wrist_body_idx
        jacobi_joint_ids = [i + 6 for i in right_joint_ids]

    ik_cfg = DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls")
    ik_cfg.ik_params = {"lambda_val": 0.05}
    ik = DifferentialIKController(cfg=ik_cfg, num_envs=num_envs, device=str(device))

    act_dim = action_manager.total_action_dim
    lift_term_ids, _ = robot.find_joints(["lift_joint"])
    arms_term_ids, arms_term_names = robot.find_joints(
        ["arm_l_joint.*", "arm_r_joint.*", "gripper_l_joint.*", "gripper_r_joint.*"]
    )
    lift_dim = len(lift_term_ids)
    arms_dim = len(arms_term_ids)
    wheels_dim = act_dim - lift_dim - arms_dim
    if wheels_dim < 0:
        raise RuntimeError(f"Action dim mismatch total={act_dim}")

    lift_slice = slice(0, lift_dim)
    arms_slice = slice(lift_dim, lift_dim + arms_dim)
    wheels_slice = slice(lift_dim + arms_dim, act_dim)

    right_arm_indices_in_arms = [i for i, n in enumerate(arms_term_names) if n.startswith("arm_r_joint")]
    if len(right_arm_indices_in_arms) != len(right_joint_names):
        name_to_i = {n: i for i, n in enumerate(arms_term_names)}
        right_arm_indices_in_arms = [name_to_i[n] for n in right_joint_names if n in name_to_i]

    lift_ids, _ = robot.find_joints(["lift_joint"])
    lift_joint_id = lift_ids[0] if lift_ids else None

    return RightGripperIKContext(
        device=device,
        num_envs=num_envs,
        act_dim=act_dim,
        wrist_body_idx=wrist_body_idx,
        jacobi_body_idx=jacobi_body_idx,
        jacobi_joint_ids=jacobi_joint_ids,
        right_joint_ids=right_joint_ids,
        right_arm_indices_in_arms=right_arm_indices_in_arms,
        lift_slice=lift_slice,
        arms_slice=arms_slice,
        wheels_slice=wheels_slice,
        lift_dim=lift_dim,
        arms_dim=arms_dim,
        lift_term_ids=lift_term_ids,
        arms_term_ids=arms_term_ids,
        lift_joint_id=lift_joint_id,
        ik=ik,
    )


def actions_for_ee_goal(
    ctx: RightGripperIKContext,
    robot,
    ee_pos_des: torch.Tensor,
    *,
    min_z: torch.Tensor,
) -> torch.Tensor:
    """One IK solve + raw action vector (torso lift + right arm; other act terms 0)."""
    device = ctx.device
    n = ctx.num_envs
    wrist_body_idx = ctx.wrist_body_idx

    ee_pos = robot.data.body_link_pos_w[:, wrist_body_idx].to(device)
    ee_quat = robot.data.body_link_quat_w[:, wrist_body_idx].to(device)
    jac = robot.root_physx_view.get_jacobians()[:, ctx.jacobi_body_idx, :, ctx.jacobi_joint_ids].to(device)
    q_cur = robot.data.joint_pos[:, ctx.right_joint_ids].to(device)

    delta = ee_pos_des - ee_pos
    d = torch.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
    scale = torch.clamp(ctx.max_ee_step_m / d, max=1.0)
    ee_cmd = ee_pos + delta * scale
    ee_cmd[:, 2] = torch.maximum(ee_cmd[:, 2], min_z)

    ctx.ik.set_command(ee_cmd, ee_pos=ee_pos, ee_quat=ee_quat)
    q_des = ctx.ik.compute(ee_pos=ee_pos, ee_quat=ee_quat, jacobian=jac, joint_pos=q_cur)
    dq = torch.clamp(q_des - q_cur, -ctx.max_joint_step, ctx.max_joint_step)
    q_des = q_cur + dq

    default_joint_pos = robot.data.default_joint_pos.to(device)
    actions = torch.zeros((n, ctx.act_dim), device=device)

    if ctx.lift_dim > 0 and ctx.lift_joint_id is not None:
        lift_default = default_joint_pos[:, ctx.lift_term_ids]
        lift_target = torch.full_like(lift_default, float(ctx.lift_target))
        actions[:, ctx.lift_slice] = (lift_target - lift_default) / ctx.lift_scale

    arms_default = default_joint_pos[:, ctx.arms_term_ids]
    arms_target = arms_default.clone()
    if len(ctx.right_arm_indices_in_arms) != q_des.shape[1]:
        raise RuntimeError("Right-arm index / q_des mismatch")
    for j, local_i in enumerate(ctx.right_arm_indices_in_arms):
        arms_target[:, local_i] = q_des[:, j]
    actions[:, ctx.arms_slice] = (arms_target - arms_default) / ctx.arms_scale

    return torch.clamp(actions, -ctx.raw_clip, ctx.raw_clip)
