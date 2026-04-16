"""Local MDP functions for the SG2-RL env config.

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Small math helpers (torch only, no extra deps)
# ---------------------------------------------------------------------------

def _quat_to_euler_xyz(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to XYZ euler angles (roll, pitch, yaw).

    Notes:
    - Assumes normalized quaternions. We normalize defensively.
    - Output range: roll,yaw in [-pi, pi], pitch in [-pi/2, pi/2].
    """
    if q.shape[-1] != 4:
        raise ValueError(f"Expected (...,4) quaternion, got {tuple(q.shape)}")
    q = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(dim=-1)

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    pitch = torch.asin(t2.clamp(-1.0, 1.0))

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack((roll, pitch, yaw), dim=-1)


def _relative_pose(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (pos, quat) of target expressed in root frame."""
    target = env.scene[target_asset_cfg.name]
    root = env.scene[root_asset_cfg.name]
    rel_pos, rel_quat = subtract_frame_transforms(
        root.data.root_pos_w, root.data.root_quat_w, target.data.root_pos_w, target.data.root_quat_w
    )
    return rel_pos, rel_quat


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedRLEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg,
    rotation_repr: str = "axis_angle",
) -> torch.Tensor:
    """Pose of target asset root expressed in root asset frame (pos + rotation)."""
    target = env.scene[target_asset_cfg.name]
    root = env.scene[root_asset_cfg.name]

    t_pos = target.data.root_pos_w
    t_quat = target.data.root_quat_w
    r_pos = root.data.root_pos_w
    r_quat = root.data.root_quat_w

    rel_pos, rel_quat = subtract_frame_transforms(r_pos, r_quat, t_pos, t_quat)

    if rotation_repr == "axis_angle":
        # Convert quaternion to axis-angle
        angle = 2.0 * torch.acos(rel_quat[:, 0:1].clamp(-1.0, 1.0))
        axis = rel_quat[:, 1:4]
        norm = torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
        axis = axis / norm
        rot = axis * angle
    else:
        rot = rel_quat

    return torch.cat([rel_pos, rot], dim=-1)


def asset_root_lin_ang_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return (lin_vel_w, ang_vel_w) concatenated for a rigid object or articulation root."""
    asset = env.scene[asset_cfg.name]
    # Both Articulation and RigidObject expose these root velocities in Isaac Lab.
    lin = asset.data.root_lin_vel_w
    ang = asset.data.root_ang_vel_w
    return torch.cat([lin, ang], dim=-1).to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class ProgressContext(ManagerTermBase):
    """Track progress for staged rewards.

    We keep lightweight state to enable “progress since reset” terms.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset_cfg: SceneEntityCfg | None = cfg.params.get("insertive_asset_cfg")
        self.receptive_asset_cfg: SceneEntityCfg | None = cfg.params.get("receptive_asset_cfg")

        # Buffers for staged progress.
        self.prev_wrist_min_dist = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)
        self.prev_peg_xy = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)
        self.prev_peg_z = torch.full((env.num_envs,), -1.0e3, device=env.device, dtype=torch.float32)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif env_ids.dtype != torch.long:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)

        if self.insertive_asset_cfg is None or self.receptive_asset_cfg is None:
            # Nothing to initialize.
            self.prev_wrist_min_dist[env_ids] = 1.0e3
            self.prev_peg_xy[env_ids] = 1.0e3
            self.prev_peg_z[env_ids] = -1.0e3
            return

        with torch.no_grad():
            # Peg pose in hole frame (for insertion progress).
            rel_pos, _rel_q = _relative_pose(self._env, self.insertive_asset_cfg, self.receptive_asset_cfg)
            self.prev_peg_xy[env_ids] = torch.norm(rel_pos[env_ids, 0:2], dim=-1)
            self.prev_peg_z[env_ids] = rel_pos[env_ids, 2]
            # Wrist distance set by other term; initialize to large.
            self.prev_wrist_min_dist[env_ids] = 1.0e3

    def __call__(self, env, insertive_asset_cfg=None, receptive_asset_cfg=None) -> torch.Tensor:
        # Context object itself returns 0; it exists so other terms can share buffers if needed.
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def dense_success_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.22,
    insertive_asset_cfg: SceneEntityCfg | None = None,
    receptive_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Dense insertion shaping based on peg pose in hole frame.

    Reward is near 1 when peg pose approaches identity in the hole frame.
    """
    if insertive_asset_cfg is None or receptive_asset_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    rel_pos, rel_quat = _relative_pose(env, insertive_asset_cfg, receptive_asset_cfg)
    pos_err = torch.norm(rel_pos, dim=-1)

    # Orientation error as absolute Euler angles (xyz). We shape all 3 but success will gate tighter tolerances.
    eul = _quat_to_euler_xyz(rel_quat)
    ang_err = torch.norm(eul, dim=-1)

    s = float(std)
    # Separate scales: use std for position, a conservative angular std tied to std.
    ang_std = max(0.15, 0.8 * s)  # radians
    r_pos = torch.exp(-pos_err / max(s, 1e-6))
    r_ang = torch.exp(-ang_err / ang_std)
    return (0.65 * r_pos + 0.35 * r_ang).to(dtype=torch.float32)


def success_reward(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg | None = None,
    receptive_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Sparse success: peg pose matches hole frame within tight tolerances.

    Default tolerances (requested):
    - XY <= 3mm, Z <= 5mm
    - roll/pitch/yaw <= 5deg
    """
    if insertive_asset_cfg is None or receptive_asset_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    rel_pos, rel_quat = _relative_pose(env, insertive_asset_cfg, receptive_asset_cfg)
    xy_ok = torch.norm(rel_pos[:, 0:2], dim=-1) <= 0.003
    z_ok = torch.abs(rel_pos[:, 2]) <= 0.005

    eul = torch.abs(_quat_to_euler_xyz(rel_quat))
    ang_ok = (eul[:, 0] <= torch.deg2rad(torch.tensor(5.0, device=env.device))) & (
        eul[:, 1] <= torch.deg2rad(torch.tensor(5.0, device=env.device))
    ) & (eul[:, 2] <= torch.deg2rad(torch.tensor(5.0, device=env.device)))

    ok = xy_ok & z_ok & ang_ok
    return ok.to(dtype=torch.float32)


def collision_free(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Collision term placeholder.

    For now we return 0 to keep the task self-contained without Warp/SDF.
    """
    return torch.zeros(env.num_envs, device=env.device)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Negative L2 norm of actions (encourages small actions)."""
    return env.action_manager.action.pow(2).sum(dim=-1).clamp(max=1.0)


def action_delta_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Action smoothness penalty: ||a_t - a_{t-1}||^2 clamped."""
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    return (a - a_prev).pow(2).sum(dim=-1).clamp(max=1.0)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step time penalty (constant 1). Use negative weight in config."""
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode timeout."""
    return env.episode_length_buf >= env.max_episode_length


def insertion_success_done(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg | None = None,
    receptive_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Terminate episode on insertion success (same tolerance as `success_reward`)."""
    return success_reward(env, insertive_asset_cfg=insertive_asset_cfg, receptive_asset_cfg=receptive_asset_cfg) > 0.5


# ---------------------------------------------------------------------------
# Rewards used by the ApproachLift / ApproachOnly env configs
# (used by curriculum configs)
# ---------------------------------------------------------------------------

def wrist_min_distance_to_asset_exp(env, robot_cfg=None, target_asset_cfg=None, sigma=0.32) -> torch.Tensor:
    if robot_cfg is None or target_asset_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    robot: Articulation = env.scene[robot_cfg.name]
    tgt: RigidObject = env.scene[target_asset_cfg.name]

    left_idx = robot.find_bodies("arm_l_link7")[0][0]
    right_idx = robot.find_bodies("arm_r_link7")[0][0]
    left = robot.data.body_link_pos_w[:, left_idx]
    right = robot.data.body_link_pos_w[:, right_idx]
    p = tgt.data.root_pos_w

    d = torch.minimum(torch.norm(left - p, dim=-1), torch.norm(right - p, dim=-1))
    s = float(sigma)
    return torch.exp(-d / max(s, 1e-6)).to(dtype=torch.float32)


class WristToInsertiveApproachProgress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot_cfg: SceneEntityCfg | None = cfg.params.get("robot_cfg")
        self.target_asset_cfg: SceneEntityCfg | None = cfg.params.get("target_asset_cfg")
        self.clip_m = float(cfg.params.get("clip_m", 0.012))

        self.prev = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)
        self.robot: Articulation | None = None
        if self.robot_cfg is not None:
            self.robot = env.scene[self.robot_cfg.name]

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif env_ids.dtype != torch.long:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        with torch.no_grad():
            d = self._dist()
            self.prev[env_ids] = d[env_ids].detach()
    def __call__(self, env, robot_cfg=None, target_asset_cfg=None, clip_m=None) -> torch.Tensor:
        d = self._dist()
        clip = max(self.clip_m if clip_m is None else float(clip_m), 1e-6)
        prog = torch.clamp((self.prev - d) / clip, 0.0, 1.0)
        self.prev = d.detach()
        return prog.to(dtype=torch.float32)

    def _dist(self) -> torch.Tensor:
        if self.robot is None or self.target_asset_cfg is None:
            return torch.full((self._env.num_envs,), 1.0e3, device=self._env.device, dtype=torch.float32)
        tgt: RigidObject = self._env.scene[self.target_asset_cfg.name]
        left_idx = self.robot.find_bodies("arm_l_link7")[0][0]
        right_idx = self.robot.find_bodies("arm_r_link7")[0][0]
        left = self.robot.data.body_link_pos_w[:, left_idx]
        right = self.robot.data.body_link_pos_w[:, right_idx]
        p = tgt.data.root_pos_w
        return torch.minimum(torch.norm(left - p, dim=-1), torch.norm(right - p, dim=-1))


def insertive_xy_near_receptor_tanh(env, insertive_asset_cfg=None, receptive_asset_cfg=None, std=0.16) -> torch.Tensor:
    if insertive_asset_cfg is None or receptive_asset_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    rel_pos, _rel_q = _relative_pose(env, insertive_asset_cfg, receptive_asset_cfg)
    d = torch.norm(rel_pos[:, 0:2], dim=-1)
    s = max(float(std), 1e-6)
    # 1 when close, approaches 0 far away
    return (1.0 - torch.tanh(d / s)).to(dtype=torch.float32)


def insertive_height_above_surface(env, insertive_asset_cfg=None, surface_z=0.82, scale=0.07) -> torch.Tensor:
    if insertive_asset_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    peg: RigidObject = env.scene[insertive_asset_cfg.name]
    z = peg.data.root_pos_w[:, 2]
    dz = torch.clamp(z - float(surface_z), min=0.0)
    sc = max(float(scale), 1e-6)
    return torch.tanh(dz / sc).to(dtype=torch.float32)


def gripper_excitation_near_insertive(env, robot_cfg=None, insertive_asset_cfg=None, gripper_joint_cfg=None, proximity_std=0.30, joint_scale=0.35) -> torch.Tensor:
    if robot_cfg is None or insertive_asset_cfg is None or gripper_joint_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    robot: Articulation = env.scene[robot_cfg.name]
    peg: RigidObject = env.scene[insertive_asset_cfg.name]
    peg_p = peg.data.root_pos_w

    # Proximity: min wrist distance to peg.
    left_idx = robot.find_bodies("arm_l_link7")[0][0]
    right_idx = robot.find_bodies("arm_r_link7")[0][0]
    left = robot.data.body_link_pos_w[:, left_idx]
    right = robot.data.body_link_pos_w[:, right_idx]
    d = torch.minimum(torch.norm(left - peg_p, dim=-1), torch.norm(right - peg_p, dim=-1))
    prox = torch.exp(-d / max(float(proximity_std), 1e-6))

    # Gripper "closure" proxy: magnitude of commanded joint positions for the specified joints.
    # This is sign-agnostic; for most grippers, closed corresponds to larger magnitude than open near 0.
    j_ids, _ = robot.find_joints(gripper_joint_cfg.joint_names)  # type: ignore[arg-type]
    if len(j_ids) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    jp = robot.data.joint_pos[:, j_ids]
    clos = torch.tanh(jp.abs().mean(dim=-1) / max(float(joint_scale), 1e-6))
    return (prox * clos).to(dtype=torch.float32)


def wrists_clearance_above_surface_exp(env, robot_cfg=None, surface_z=0.82, min_clearance_m=0.18, sigma_m=0.048) -> torch.Tensor:
    if robot_cfg is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    robot: Articulation = env.scene[robot_cfg.name]
    left_idx = robot.find_bodies("arm_l_link7")[0][0]
    right_idx = robot.find_bodies("arm_r_link7")[0][0]
    z_left = robot.data.body_link_pos_w[:, left_idx, 2]
    z_right = robot.data.body_link_pos_w[:, right_idx, 2]
    z = torch.minimum(z_left, z_right)
    clearance = z - float(surface_z)
    shortfall = torch.clamp(float(min_clearance_m) - clearance, min=0.0)
    sig = max(float(sigma_m), 1e-6)
    return torch.exp(-shortfall / sig).to(dtype=torch.float32)
