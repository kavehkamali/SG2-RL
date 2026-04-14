"""Full right-arm checks against the same workspace spheres used by APF.

The APF planner only integrates a wrist waypoint. We complement it by:

- **Planning:** multi-point repulsion along a frozen shoulder→wrist line (see ``apf_path``).
- **Playback:** each step, nudge the commanded wrist position if any ``arm_r_link*``
  sample sits inside a repulsor's influence radius.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import torch

from sg2_rl.apf_path import SphereObstacle


def _link_index_from_body_name(name: str) -> int | None:
    m = re.search(r"arm_r_link(\d+)", name)
    return int(m.group(1)) if m else None


def pick_right_arm_line_base_xyz(robot, env_idx: int = 0) -> np.ndarray | None:
    """World position of a mid-arm link to use as the base of the shoulder→wrist segment."""
    for name in ("arm_r_link3", "arm_r_link2", "arm_r_link4", "arm_r_link1"):
        ids, _ = robot.find_bodies(name)
        if len(ids) == 1:
            p = robot.data.body_link_pos_w[env_idx, ids[0]].detach().cpu().numpy()
            return np.asarray(p, dtype=np.float64).reshape(3)
    return None


def right_arm_link_check_indices(robot) -> list[int]:
    """Ordered body indices for ``arm_r_link1`` … ``arm_r_link7`` (whatever exists)."""
    found: dict[int, int] = {}
    for k in range(1, 8):
        ids, names = robot.find_bodies(f"arm_r_link{k}")
        if len(ids) >= 1 and names:
            for bid, nm in zip(ids, names):
                li = _link_index_from_body_name(nm)
                if li == k:
                    found[k] = bid
                    break
    return [found[i] for i in sorted(found)]


def nudge_ee_des_for_arm_spheres(
    ee_des: torch.Tensor,
    link_pos_w: torch.Tensor,
    obstacles: Sequence[SphereObstacle],
    *,
    max_total_shift_m: float = 0.02,
) -> torch.Tensor:
    """Push ``ee_des`` slightly if any arm link is inside a sphere's influence region.

    Uses the same inverse-distance repulsion *direction* as APF (away from sphere center),
    aggregated over links then clamped so motion stays smooth for video.
    """
    if not obstacles or link_pos_w.numel() == 0:
        return ee_des
    n = ee_des.shape[0]
    delta = torch.zeros_like(ee_des)
    for o in obstacles:
        c = torch.as_tensor(o.center, device=ee_des.device, dtype=ee_des.dtype).view(1, 1, 3)
        rho0 = float(o.radius)
        eta = float(o.gain)
        dvec = link_pos_w - c
        d = torch.linalg.norm(dvec, dim=-1).clamp_min(1e-9)
        inside = d < rho0
        if not inside.any():
            continue
        term = (1.0 / d) - (1.0 / rho0)
        mag = eta * term * (1.0 / (d * d))
        f = (mag / d).unsqueeze(-1) * dvec
        f = f * inside.unsqueeze(-1).to(ee_des.dtype)
        delta = delta + f.sum(dim=1)
    norms = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
    scale = torch.clamp(max_total_shift_m / norms.squeeze(-1), max=1.0).unsqueeze(-1)
    return ee_des + delta * scale
