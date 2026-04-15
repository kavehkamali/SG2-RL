"""Approach rewards anchored to a fixed world-frame point (tabletop top center).

Extracted from uwlab_tasks omnireset_sg2 config.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def wrists_min_distance_to_world_point_exp_sum(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    anchor_xyz: tuple[float, float, float],
    left_body_name: str = "arm_l_link7",
    right_body_name: str = "arm_r_link7",
    sigma: float = 0.32,
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    left_idx = robot.find_bodies(left_body_name)[0][0]
    right_idx = robot.find_bodies(right_body_name)[0][0]
    left_pos = robot.data.body_link_pos_w[:, left_idx]
    right_pos = robot.data.body_link_pos_w[:, right_idx]
    anchor = torch.tensor(anchor_xyz, dtype=torch.float32, device=env.device)
    d_left = torch.norm(left_pos - anchor, dim=-1)
    d_right = torch.norm(right_pos - anchor, dim=-1)
    s = float(sigma)
    return torch.exp(-d_left / s) + torch.exp(-d_right / s)


class WristsToWorldPointApproachProgressSum(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot_cfg: SceneEntityCfg = cfg.params.get("robot_cfg")
        self.left = cfg.params.get("left_body_name", "arm_l_link7")
        self.right = cfg.params.get("right_body_name", "arm_r_link7")
        self.clip_m = float(cfg.params.get("clip_m", 0.012))
        anchor = cfg.params.get("anchor_xyz")
        self.robot: Articulation = env.scene[self.robot_cfg.name]
        self.anchor = torch.tensor(anchor, dtype=torch.float32, device=env.device)
        self.prev_left = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)
        self.prev_right = torch.full((env.num_envs,), 1.0e3, device=env.device, dtype=torch.float32)

    def _distances(self):
        left_idx = self.robot.find_bodies(self.left)[0][0]
        right_idx = self.robot.find_bodies(self.right)[0][0]
        left_pos = self.robot.data.body_link_pos_w[:, left_idx]
        right_pos = self.robot.data.body_link_pos_w[:, right_idx]
        return torch.norm(left_pos - self.anchor, dim=-1), torch.norm(right_pos - self.anchor, dim=-1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif env_ids.dtype != torch.long:
            env_ids = env_ids.to(device=self._env.device, dtype=torch.long)
        with torch.no_grad():
            d_left, d_right = self._distances()
            self.prev_left[env_ids] = d_left[env_ids].detach()
            self.prev_right[env_ids] = d_right[env_ids].detach()

    def __call__(self, env, robot_cfg=None, anchor_xyz=None,
                 left_body_name="arm_l_link7", right_body_name="arm_r_link7", clip_m=0.012):
        d_left, d_right = self._distances()
        clip_m = float(clip_m)
        prog_l = torch.clamp((self.prev_left - d_left) / clip_m, 0.0, 1.0)
        prog_r = torch.clamp((self.prev_right - d_right) / clip_m, 0.0, 1.0)
        self.prev_left = d_left.detach()
        self.prev_right = d_right.detach()
        return 0.5 * (prog_l + prog_r)
