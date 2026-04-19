"""OmniReset-style MDP functions for the FFW-SG2 bimanual peg handover+insertion task.

Ports the essential pieces of UWLab's OmniReset approach
(https://github.com/UW-Lab/UWLab ``uwlab_tasks/manager_based/manipulation/omnireset``)
to the existing SG2 scene. We reuse the SG2 scene (robot + peg + peg-hole + desk)
and peg/hole poses verbatim — only the *task logic* follows OmniReset.

Key OmniReset ideas implemented here:

1. ``ee_asset_distance_tanh`` reward:  ``1 - tanh(|p_ee - p_target| / std)``
2. Pose of asset-B expressed in asset-A frame (end-effector frame) observations.
3. Privileged critic observations: asset velocities, material properties, mass,
   joint friction/armature/stiffness/damping, ``time_left``.
4. ``abnormal_robot_state`` termination/reward on excessive joint velocities.
5. Bimanual reset distribution — a procedural analogue to OmniReset's
   ``MultiResetManager`` with 4 reset kinds.  Each reset of each env picks one
   of the 4 kinds uniformly, implemented directly in sim (no recorded dataset):

     - ``PegOnTableArmsHome``    — peg on table, both arms at home pose.
     - ``PegInLeftGripper``      — peg placed in the left gripper, closed.
     - ``PegHandoverPose``       — both hands near peg at chest height.
     - ``PegInRightOverHole``    — peg in right gripper, positioned above hole.

Mostly-reused helpers (``target_asset_pose_in_root_asset_frame``,
``ProgressContext``, ``dense_success_reward``, ``success_reward``,
``asset_root_lin_ang_vel_w``, ``action_l2_clamped``, ``action_delta_l2_clamped``,
``time_out``, ``insertion_success_done``) live in ``sg2_rl.task_mdp`` and are
imported directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_inv, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


# ---------------------------------------------------------------------------
# Body name constants (FFW-SG2 wrist tips)
# ---------------------------------------------------------------------------

_LEFT_EE_BODY = "arm_l_link7"
_RIGHT_EE_BODY = "arm_r_link7"


def _ee_body_idx(robot: Articulation, wrist: str) -> int:
    name = _LEFT_EE_BODY if str(wrist).lower() == "left" else _RIGHT_EE_BODY
    return int(robot.find_bodies(name)[0][0])


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def ee_pose_in_robot_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    wrist: str = "left",
    rotation_repr: str = "axis_angle",
) -> torch.Tensor:
    """Pose of the end-effector (wrist tip) body expressed in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = _ee_body_idx(robot, wrist)
    ee_pos = robot.data.body_link_pos_w[:, idx]
    ee_quat = robot.data.body_link_quat_w[:, idx]
    r_pos = robot.data.root_pos_w
    r_quat = robot.data.root_quat_w
    rel_pos, rel_quat = subtract_frame_transforms(r_pos, r_quat, ee_pos, ee_quat)
    if rotation_repr == "axis_angle":
        angle = 2.0 * torch.acos(rel_quat[:, 0:1].clamp(-1.0, 1.0))
        axis = rel_quat[:, 1:4]
        axis = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
        rot = axis * angle
    else:
        rot = rel_quat
    return torch.cat([rel_pos, rot], dim=-1).to(dtype=torch.float32)


def asset_pose_in_ee_frame(
    env: "ManagerBasedRLEnv",
    target_asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    wrist: str = "left",
    rotation_repr: str = "axis_angle",
) -> torch.Tensor:
    """Pose of a rigid object expressed in the wrist-tip (end-effector) frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_asset_cfg.name]
    idx = _ee_body_idx(robot, wrist)
    ee_pos = robot.data.body_link_pos_w[:, idx]
    ee_quat = robot.data.body_link_quat_w[:, idx]
    t_pos = target.data.root_pos_w
    t_quat = target.data.root_quat_w
    rel_pos, rel_quat = subtract_frame_transforms(ee_pos, ee_quat, t_pos, t_quat)
    if rotation_repr == "axis_angle":
        angle = 2.0 * torch.acos(rel_quat[:, 0:1].clamp(-1.0, 1.0))
        axis = rel_quat[:, 1:4]
        axis = axis / torch.norm(axis, dim=-1, keepdim=True).clamp_min(1e-8)
        rot = axis * angle
    else:
        rot = rel_quat
    return torch.cat([rel_pos, rot], dim=-1).to(dtype=torch.float32)


def ee_velocity_in_robot_frame(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, wrist: str = "left"
) -> torch.Tensor:
    """Linear + angular velocity of EE body expressed in the robot base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    idx = _ee_body_idx(robot, wrist)
    lin_w = robot.data.body_lin_vel_w[:, idx]
    ang_w = robot.data.body_ang_vel_w[:, idx]
    r_quat_inv = quat_inv(robot.data.root_quat_w)
    lin_b = quat_apply(r_quat_inv, lin_w)
    ang_b = quat_apply(r_quat_inv, ang_w)
    return torch.cat([lin_b, ang_b], dim=-1).to(dtype=torch.float32)


def time_left(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Fraction of episode time remaining, shape (N, 1)."""
    max_steps = float(getattr(env, "max_episode_length", 1))
    buf = env.episode_length_buf.to(dtype=torch.float32)
    left = 1.0 - (buf / max(max_steps, 1.0))
    return left.unsqueeze(-1)


# ---- Privileged (critic-only) observations --------------------------------

def get_material_properties(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return per-env material friction/restitution as a (N, 3) tensor.

    Safe fallback: returns zeros if the asset doesn't expose material buffers.
    """
    asset = env.scene[asset_cfg.name]
    try:
        # Isaac Lab exposes material_properties on the PhysX view for some assets.
        mat = asset.root_physx_view.get_material_properties()
        if mat.dim() == 3:  # (num_envs, num_shapes, 3)
            mat = mat.mean(dim=1)
        return mat.to(device=env.device, dtype=torch.float32)
    except Exception:
        return torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)


def get_mass(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    try:
        m = asset.root_physx_view.get_masses()
        if m.dim() == 2:
            m = m.sum(dim=-1, keepdim=True)  # (N, 1)
        else:
            m = m.view(env.num_envs, -1).sum(dim=-1, keepdim=True)
        return m.to(device=env.device, dtype=torch.float32)
    except Exception:
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)


def _joint_attr(robot: Articulation, attr: str) -> torch.Tensor:
    """Fetch a joint-level parameter buffer of shape (N, J). Zeros fallback."""
    data = getattr(robot.data, attr, None)
    if data is None:
        return None
    return data.to(dtype=torch.float32)


def get_joint_friction(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    v = _joint_attr(robot, "joint_friction_coeff")
    if v is None:
        v = _joint_attr(robot, "joint_friction")
    if v is None:
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)
    return v


def get_joint_armature(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    v = _joint_attr(robot, "joint_armature")
    if v is None:
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)
    return v


def get_joint_stiffness(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    v = _joint_attr(robot, "joint_stiffness")
    if v is None:
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)
    return v


def get_joint_damping(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    v = _joint_attr(robot, "joint_damping")
    if v is None:
        return torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32)
    return v


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

def ee_asset_distance_tanh(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    wrist: str = "left",
    std: float = 0.22,
) -> torch.Tensor:
    """``1 - tanh(|p_ee - p_target| / std)`` — smooth EE-to-asset proximity."""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_asset_cfg.name]
    idx = _ee_body_idx(robot, wrist)
    ee_pos = robot.data.body_link_pos_w[:, idx]
    t_pos = target.data.root_pos_w
    d = torch.norm(ee_pos - t_pos, dim=-1)
    return (1.0 - torch.tanh(d / max(float(std), 1e-6))).to(dtype=torch.float32)


def joint_vel_l2_clamped(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, max_val: float = 1.0
) -> torch.Tensor:
    """L2 of joint velocities on the selected joints, clamped."""
    robot: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids is None or asset_cfg.joint_ids == slice(None):
        qd = robot.data.joint_vel
    else:
        qd = robot.data.joint_vel[:, asset_cfg.joint_ids]
    return qd.pow(2).sum(dim=-1).clamp(max=float(max_val)).to(dtype=torch.float32)


def abnormal_robot_state(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    vel_limit_scale: float = 3.0,
) -> torch.Tensor:
    """Flag envs where any joint velocity exceeds ``vel_limit_scale * joint_vel_limits``.

    Used both as a (negative-weight) reward term and as a termination flag.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    qd = robot.data.joint_vel.abs()
    lim = robot.data.joint_vel_limits
    bad_v = qd > (lim * float(vel_limit_scale))
    nan_p = torch.isnan(robot.data.joint_pos).any(dim=-1)
    nan_v = torch.isnan(qd).any(dim=-1)
    flag = bad_v.any(dim=-1) | nan_p | nan_v
    return flag.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def abnormal_robot_state_done(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, vel_limit_scale: float = 3.0
) -> torch.Tensor:
    return abnormal_robot_state(env, robot_cfg=robot_cfg, vel_limit_scale=vel_limit_scale) > 0.5


# ---------------------------------------------------------------------------
# Bimanual reset distribution (OmniReset MultiResetManager analogue, procedural)
# ---------------------------------------------------------------------------

# Home pose deltas applied on top of default joint positions per reset kind.
# Each is a small dict ``joint_regex -> (low, high)`` position offset.
_HOME_DELTAS = {
    "arm_l_joint.*": 0.0,
    "arm_r_joint.*": 0.0,
    "gripper_l_joint.*": 0.0,
    "gripper_r_joint.*": 0.0,
    "lift_joint": 0.25,
}

# Pre-scripted joint targets for each of the 4 reset kinds. Values are *offsets*
# from the articulation's default joint positions, so they stay in a neutral
# bimanual home frame for any USD variant with the same joint names.
# (left arm roll, lift, elbow, wrist, gripper...)
_RESET_JOINT_TARGETS = {
    # A: arms at default home, lift mid, both grippers open.
    "PegOnTableArmsHome": {
        "lift_joint": 0.22,
        "arm_l_joint1": 0.00, "arm_l_joint2": -0.20, "arm_l_joint3": 0.00,
        "arm_l_joint4": -1.40, "arm_l_joint5": 0.00, "arm_l_joint6": 0.60, "arm_l_joint7": 0.00,
        "arm_r_joint1": 0.00, "arm_r_joint2": 0.20, "arm_r_joint3": 0.00,
        "arm_r_joint4": -1.40, "arm_r_joint5": 0.00, "arm_r_joint6": 0.60, "arm_r_joint7": 0.00,
        "gripper_l_joint1": 0.50, "gripper_r_joint1": 0.50,
    },
    # B: left hand positioned around the peg grasp pose, left gripper closed.
    "PegInLeftGripper": {
        "lift_joint": 0.24,
        "arm_l_joint1": 0.10, "arm_l_joint2": 0.10, "arm_l_joint3": 0.00,
        "arm_l_joint4": -1.10, "arm_l_joint5": 0.00, "arm_l_joint6": 0.95, "arm_l_joint7": 0.00,
        "arm_r_joint1": 0.00, "arm_r_joint2": 0.20, "arm_r_joint3": 0.00,
        "arm_r_joint4": -1.40, "arm_r_joint5": 0.00, "arm_r_joint6": 0.60, "arm_r_joint7": 0.00,
        "gripper_l_joint1": 0.00, "gripper_r_joint1": 0.50,
    },
    # C: both hands near each other at chest height around peg (handover pose).
    "PegHandoverPose": {
        "lift_joint": 0.26,
        "arm_l_joint1": 0.00, "arm_l_joint2": 0.30, "arm_l_joint3": 0.20,
        "arm_l_joint4": -1.00, "arm_l_joint5": 0.00, "arm_l_joint6": 0.85, "arm_l_joint7": 0.00,
        "arm_r_joint1": 0.00, "arm_r_joint2": -0.30, "arm_r_joint3": -0.20,
        "arm_r_joint4": -1.00, "arm_r_joint5": 0.00, "arm_r_joint6": 0.85, "arm_r_joint7": 0.00,
        "gripper_l_joint1": 0.00, "gripper_r_joint1": 0.00,
    },
    # D: right hand over the hole, peg inside right gripper.
    "PegInRightOverHole": {
        "lift_joint": 0.26,
        "arm_l_joint1": 0.00, "arm_l_joint2": -0.20, "arm_l_joint3": 0.00,
        "arm_l_joint4": -1.40, "arm_l_joint5": 0.00, "arm_l_joint6": 0.60, "arm_l_joint7": 0.00,
        "arm_r_joint1": -0.10, "arm_r_joint2": 0.10, "arm_r_joint3": 0.00,
        "arm_r_joint4": -1.10, "arm_r_joint5": 0.00, "arm_r_joint6": 0.95, "arm_r_joint7": 0.00,
        "gripper_l_joint1": 0.50, "gripper_r_joint1": 0.00,
    },
}


class BimanualResetSampler(ManagerTermBase):
    """Procedural OmniReset-style 4-path bimanual reset.

    For each env being reset, samples one of 4 reset kinds uniformly and writes
    the corresponding joint pose + peg/hole world pose in-place. The peg/hole
    *nominal* world positions are taken from their config (SG2 scene constants,
    unchanged), and only per-kind relative offsets are applied.
    """

    # kind index to name
    _KINDS = ("PegOnTableArmsHome", "PegInLeftGripper", "PegHandoverPose", "PegInRightOverHole")

    def __init__(self, cfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.robot_cfg: SceneEntityCfg = cfg.params["robot_cfg"]
        self.insertive_cfg: SceneEntityCfg = cfg.params["insertive_cfg"]
        self.receptive_cfg: SceneEntityCfg = cfg.params["receptive_cfg"]
        self.weights = cfg.params.get("weights", (0.25, 0.25, 0.25, 0.25))
        # Scene XYZ constants (from env_cfg; caller passes as params to avoid coupling)
        self.peg_nominal_xyz: tuple[float, float, float] = tuple(cfg.params["peg_nominal_xyz"])
        self.hole_nominal_xyz: tuple[float, float, float] = tuple(cfg.params["hole_nominal_xyz"])
        self._joint_cache: dict[str, torch.Tensor] = {}

    def _resolve_joint_ids(self, robot: Articulation, pattern: str) -> torch.Tensor:
        if pattern in self._joint_cache:
            return self._joint_cache[pattern]
        ids, _ = robot.find_joints(pattern)
        tids = torch.as_tensor(ids, dtype=torch.long, device=robot.device)
        self._joint_cache[pattern] = tids
        return tids

    def _apply_joint_targets(
        self, robot: Articulation, env_ids: torch.Tensor, targets: dict[str, float]
    ) -> torch.Tensor:
        """Start from default joint pos; overwrite selected joints with targets."""
        default = robot.data.default_joint_pos[env_ids].clone()
        qpos = default.clone()
        for pattern, val in targets.items():
            ids = self._resolve_joint_ids(robot, pattern)
            if ids.numel() == 0:
                continue
            qpos[:, ids] = float(val)
        return qpos

    def _peg_hole_world_pose(
        self, env_ids: torch.Tensor, kind_idx: torch.Tensor, device: torch.device
    ):
        """Return (peg_pos_w, peg_quat, hole_pos_w, hole_quat, peg_linvel, peg_angvel)."""
        n = env_ids.shape[0]
        peg = torch.as_tensor(self.peg_nominal_xyz, device=device, dtype=torch.float32).expand(n, 3).clone()
        hole = torch.as_tensor(self.hole_nominal_xyz, device=device, dtype=torch.float32).expand(n, 3).clone()
        # Per-kind offsets for peg:
        #   A: on table at nominal (no offset)
        #   B: lift peg up ~40cm (left hand grasp zone, ~waist-level)
        #   C: bring peg to chest height, centered
        #   D: right hand above hole, ~15cm above
        z_lift = torch.zeros(n, device=device, dtype=torch.float32)
        x_shift = torch.zeros(n, device=device, dtype=torch.float32)
        y_shift = torch.zeros(n, device=device, dtype=torch.float32)
        # kind==1
        m = kind_idx == 1
        z_lift[m] = 0.35
        x_shift[m] = 0.15
        y_shift[m] = 0.12
        # kind==2
        m = kind_idx == 2
        z_lift[m] = 0.40
        x_shift[m] = 0.20
        y_shift[m] = 0.00
        # kind==3
        m = kind_idx == 3
        z_lift[m] = 0.18
        x_shift[m] = 0.00
        y_shift[m] = -0.02  # slightly over the hole
        peg[:, 0] += x_shift
        peg[:, 1] += y_shift
        peg[:, 2] += z_lift
        q = torch.zeros(n, 4, device=device, dtype=torch.float32)
        q[:, 0] = 1.0
        zeros6 = torch.zeros(n, 6, device=device, dtype=torch.float32)
        return peg, q.clone(), hole, q.clone(), zeros6

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        robot_cfg: SceneEntityCfg,
        insertive_cfg: SceneEntityCfg,
        receptive_cfg: SceneEntityCfg,
        peg_nominal_xyz=None,
        hole_nominal_xyz=None,
        weights=None,
    ):
        if env_ids is None or (hasattr(env_ids, "numel") and env_ids.numel() == 0):
            return
        if env_ids.dtype != torch.long:
            env_ids = env_ids.to(dtype=torch.long)
        device = env.device
        n = int(env_ids.shape[0])

        w = torch.as_tensor(self.weights, device=device, dtype=torch.float32).clamp_min(0.0)
        w = w / w.sum().clamp_min(1e-8)
        kind_idx = torch.multinomial(w, num_samples=n, replacement=True)

        robot: Articulation = env.scene[self.robot_cfg.name]
        peg: RigidObject = env.scene[self.insertive_cfg.name]
        hole: RigidObject = env.scene[self.receptive_cfg.name]

        # Joint targets per env by masking kinds.
        num_j = robot.data.default_joint_pos.shape[-1]
        qpos = robot.data.default_joint_pos[env_ids].clone()
        qvel = torch.zeros((n, num_j), device=device, dtype=torch.float32)
        for ki, kname in enumerate(self._KINDS):
            mask = (kind_idx == ki)
            if not bool(mask.any()):
                continue
            sub_ids = env_ids[mask]
            sub_qpos = self._apply_joint_targets(robot, sub_ids, _RESET_JOINT_TARGETS[kname])
            qpos[mask] = sub_qpos

        robot.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)

        # Peg / hole world pose.
        peg_pos_w, peg_quat, hole_pos_w, hole_quat, zeros6 = self._peg_hole_world_pose(env_ids, kind_idx, device)
        # Apply env origins offset (scene replicates envs along the grid).
        origins = env.scene.env_origins[env_ids]
        peg_world = torch.cat([peg_pos_w + origins, peg_quat], dim=-1)
        hole_world = torch.cat([hole_pos_w + origins, hole_quat], dim=-1)
        peg.write_root_pose_to_sim(peg_world, env_ids=env_ids)
        peg.write_root_velocity_to_sim(zeros6, env_ids=env_ids)
        hole.write_root_pose_to_sim(hole_world, env_ids=env_ids)
        hole.write_root_velocity_to_sim(zeros6, env_ids=env_ids)
