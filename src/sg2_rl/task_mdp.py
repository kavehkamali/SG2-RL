"""Local MDP functions replacing uwlab_tasks.manager_based.manipulation.omnireset.mdp.

Only the subset actually used by the SG2-RL env config is implemented here.
Complex functions (ProgressContext, collision_free, etc.) are replaced with
zero-return stubs since they are either weight-0 or unused by recording scripts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


# ---------------------------------------------------------------------------
# Rewards — stubs for env instantiation (recording scripts ignore rewards)
# ---------------------------------------------------------------------------

class ProgressContext(ManagerTermBase):
    """Stub: returns 0. Original tracks peg-hole alignment via USD metadata."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def reset(self, env_ids=None):
        super().reset(env_ids)

    def __call__(self, env, **kwargs) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)


def dense_success_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.22,
    insertive_asset_cfg: SceneEntityCfg | None = None,
    receptive_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Stub: returns 0 (original measures distance-to-assembled-pose)."""
    return torch.zeros(env.num_envs, device=env.device)


def success_reward(
    env: ManagerBasedRLEnv,
    insertive_asset_cfg: SceneEntityCfg | None = None,
    receptive_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Stub: returns 0."""
    return torch.zeros(env.num_envs, device=env.device)


def collision_free(env: ManagerBasedRLEnv, **kwargs) -> torch.Tensor:
    """Stub: returns 0 (original uses Warp-based SDF)."""
    return torch.zeros(env.num_envs, device=env.device)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Negative L2 norm of actions (encourages small actions)."""
    return env.action_manager.action.pow(2).sum(dim=-1).clamp(max=1.0)


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode timeout."""
    return env.episode_length_buf >= env.max_episode_length


# ---------------------------------------------------------------------------
# Rewards used by the ApproachLift / ApproachOnly env configs
# (stubs — only needed if those configs are instantiated)
# ---------------------------------------------------------------------------

def wrist_min_distance_to_asset_exp(env, **kwargs) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


class WristToInsertiveApproachProgress(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
    def reset(self, env_ids=None):
        super().reset(env_ids)
    def __call__(self, env, **kwargs) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)


def insertive_xy_near_receptor_tanh(env, **kwargs) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def insertive_height_above_surface(env, **kwargs) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def gripper_excitation_near_insertive(env, **kwargs) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def wrists_clearance_above_surface_exp(env, **kwargs) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)
