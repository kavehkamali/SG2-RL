"""Artificial Potential Field (APF) path sampling in world frame.

Classic formulation: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators
and Mobile Robots", IJRR 1986 (often cited as 1985/1986).

We use:
  - **Attractive** force toward a single Cartesian goal (linear spring).
  - **Spherical repulsors** (Khatib-style inverse-distance repulsion inside a cutoff).
  - **Table half-space** soft constraint: upward restoring force below a safe Z band.

By default repulsion is evaluated at the **wrist** only. If ``arm_repulse_base_xyz`` is
set, repulsive gradients are summed along a **straight segment** from that base point to
the wrist at several fractions ``t in (0, 1]``, each scaled by ``t`` (chain rule for
``p = base + t * (x - base)``). That approximates clearance for the arm bulk versus the
same spheres; it does **not** model self-collision or mesh CAD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Fractions along base→wrist used for repulsive field sampling (includes wrist at 1.0).
DEFAULT_ARM_REPULSE_T: tuple[float, ...] = (0.12, 0.28, 0.45, 0.62, 0.8, 1.0)

# If the base is almost coincident with the wrist, fall back to wrist-only repulsion.
_ARM_SEGMENT_MIN_M = 0.04


@dataclass
class SphereObstacle:
    center: np.ndarray  # (3,)
    radius: float  # influence distance rho_0
    gain: float = 0.35


def _repulsive_force(x: np.ndarray, obs: SphereObstacle) -> np.ndarray:
    """Repulsive contribution (vector) in world frame."""
    c = obs.center.astype(np.float64)
    d_vec = x - c
    d = float(np.linalg.norm(d_vec))
    rho0 = float(obs.radius)
    if d >= rho0 or d < 1e-9:
        return np.zeros(3, dtype=np.float64)
    # Gradient of 0.5*eta*(1/d - 1/rho0)^2 w.r.t. x pushes *away* from obstacle center.
    eta = float(obs.gain)
    term = (1.0 / d) - (1.0 / rho0)
    mag = eta * term * (1.0 / (d * d))
    return (mag / d) * d_vec


def _chain_repulsive_wrist_gradient(
    x: np.ndarray,
    base: np.ndarray,
    t_values: Sequence[float],
    obs_list: Sequence[SphereObstacle],
) -> np.ndarray:
    """Repulsive gradient w.r.t. wrist ``x`` from spheres sampled along base→``x``."""
    sh = np.asarray(base, dtype=np.float64).reshape(3)
    out = np.zeros(3, dtype=np.float64)
    for t in t_values:
        p = sh + float(t) * (x - sh)
        for o in obs_list:
            out += float(t) * _repulsive_force(p, o)
    return out


def _table_restore(x: np.ndarray, z_floor: float, gain: float = 11.0) -> np.ndarray:
    """Soft upward force if below ``z_floor`` (table + margin)."""
    if x[2] >= z_floor:
        return np.zeros(3, dtype=np.float64)
    return np.array([0.0, 0.0, gain * (z_floor - x[2])], dtype=np.float64)


def plan_apf_polyline(
    start_xyz: Sequence[float],
    goal_xyz: Sequence[float],
    *,
    table_z: float,
    wrist_clearance_m: float = 0.18,
    sphere_obstacles: Sequence[SphereObstacle] | None = None,
    arm_repulse_base_xyz: Sequence[float] | None = None,
    arm_repulse_t: Sequence[float] = DEFAULT_ARM_REPULSE_T,
    k_attract: float = 1.2,
    step_m: float = 0.008,
    max_steps: int = 800,
    goal_tol_m: float = 0.015,
) -> list[list[float]]:
    """Return a dense polyline [N,3] from APF gradient integration."""
    x = np.array(start_xyz, dtype=np.float64).reshape(3)
    g = np.array(goal_xyz, dtype=np.float64).reshape(3)
    z_floor = float(table_z) + float(wrist_clearance_m)
    obs_list = list(sphere_obstacles or ())
    arm_base = None if arm_repulse_base_xyz is None else np.asarray(arm_repulse_base_xyz, dtype=np.float64).reshape(3)
    t_vals = tuple(float(t) for t in arm_repulse_t) if arm_repulse_t else (1.0,)

    pts: list[list[float]] = [x.copy().tolist()]
    for _ in range(max_steps):
        if float(np.linalg.norm(x - g)) <= goal_tol_m:
            break
        f_att = -k_attract * (x - g)
        f_rep = np.zeros(3, dtype=np.float64)
        if arm_base is not None and float(np.linalg.norm(x - arm_base)) > _ARM_SEGMENT_MIN_M:
            f_rep = _chain_repulsive_wrist_gradient(x, arm_base, t_vals, obs_list)
        else:
            for o in obs_list:
                f_rep += _repulsive_force(x, o)
        f_tab = _table_restore(x, z_floor)
        v = f_att + f_rep + f_tab
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            # Local minimum: nudge toward goal + small lateral jitter
            jitter = np.array([0.006 * math.sin(len(pts) * 0.7), 0.006 * math.cos(len(pts) * 0.5), 0.002])
            v = f_att + jitter
            vn = float(np.linalg.norm(v))
        if vn < 1e-12:
            break
        x = x + step_m * (v / vn)
        # Hard clamp below table
        x[2] = max(x[2], z_floor)
        pts.append(x.copy().tolist())
    # Ensure exact goal endpoint for IK targeting
    if float(np.linalg.norm(np.array(pts[-1]) - g)) > 1e-4:
        pts.append(g.tolist())
    return pts


def default_workspace_obstacles(peg_xyz: Sequence[float]) -> list[SphereObstacle]:
    """Virtual spheres near the peg to create a non-trivial detour (no mesh required)."""
    p = np.array(peg_xyz[:3], dtype=np.float64)
    return [
        SphereObstacle(center=p + np.array([0.22, 0.0, 0.06]), radius=0.38, gain=0.45),
        SphereObstacle(center=p + np.array([-0.12, 0.22, 0.10]), radius=0.32, gain=0.40),
        SphereObstacle(center=p + np.array([0.0, -0.18, 0.12]), radius=0.28, gain=0.35),
    ]
