"""Shared orbit-camera helpers for SG2 recording scripts."""

from __future__ import annotations

import math


def orbit_lookat_shifted_toward_robot(
    env_cfg,
    robot,
    peg,
    *,
    shift_xy_m: float = 0.26,
) -> tuple[float, float, float]:
    """Move the orbit **look-at** from ``env_cfg.viewer.lookat`` toward the robot (XY only).

    Separates peg vs manipulator in the frame when the default focal point sits too close
    to the pin relative to the arm.
    """
    lx = float(env_cfg.viewer.lookat[0])
    ly = float(env_cfg.viewer.lookat[1])
    lz = float(env_cfg.viewer.lookat[2])
    if shift_xy_m <= 0.0:
        return (lx, ly, lz)

    rx = float(robot.data.root_pos_w[0, 0].item())
    ry = float(robot.data.root_pos_w[0, 1].item())
    px = float(peg.data.root_pos_w[0, 0].item())
    py = float(peg.data.root_pos_w[0, 1].item())
    dx, dy = rx - px, ry - py
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return (lx, ly, lz)
    dx /= n
    dy /= n
    return (lx + shift_xy_m * dx, ly + shift_xy_m * dy, lz)
