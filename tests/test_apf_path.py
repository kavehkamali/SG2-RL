"""Unit tests for APF helpers (no Isaac / torch required)."""

from __future__ import annotations

import numpy as np

from sg2_rl.apf_path import (
    DEFAULT_ARM_REPULSE_T,
    SphereObstacle,
    _chain_repulsive_wrist_gradient,
    plan_apf_polyline,
)


def test_chain_repulsive_nonzero_when_midsegment_near_sphere():
    base = np.array([0.0, 0.0, 0.0])
    x = np.array([1.0, 0.0, 0.0])
    obs = SphereObstacle(center=np.array([0.5, 0.0, 0.0]), radius=0.4, gain=1.0)
    g = _chain_repulsive_wrist_gradient(x, base, DEFAULT_ARM_REPULSE_T, [obs])
    assert np.linalg.norm(g) > 0.05


def test_plan_apf_with_arm_base_differs_from_wrist_only():
    start = [0.55, -0.05, 1.05]
    goal = [0.58, 0.02, 1.12]
    peg = [0.6, 0.0, 1.0]
    obs = [
        SphereObstacle(center=np.asarray(peg) + np.array([0.22, 0.0, 0.06]), radius=0.38, gain=0.45),
    ]
    p0 = plan_apf_polyline(start, goal, table_z=0.82, sphere_obstacles=obs, max_steps=120)
    arm_base = np.array([0.35, -0.12, 0.95])
    p1 = plan_apf_polyline(
        start,
        goal,
        table_z=0.82,
        sphere_obstacles=obs,
        arm_repulse_base_xyz=arm_base,
        max_steps=120,
    )
    a0 = np.asarray(p0, dtype=np.float64)
    a1 = np.asarray(p1, dtype=np.float64)
    m = min(len(a0), len(a1))
    assert m >= 3
    seg = np.linalg.norm(a0[:m] - a1[:m], axis=1)
    assert float(seg.sum()) > 0.02
