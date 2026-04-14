"""PegMLPGraspLift task entry points at bundled SKRL yaml."""

from __future__ import annotations

import pytest

pytest.importorskip("gymnasium")

from sg2_rl.gym_register import REGISTERED_TASKS
from sg2_rl.paths import configs_dir


def test_peg_mlp_grasp_lift_registry_and_yaml():
    tid = "OmniReset-FFWSG2-PegMLPGraspLift-v0"
    assert tid in REGISTERED_TASKS
    yname = REGISTERED_TASKS[tid][1]
    ypath = configs_dir() / yname
    assert ypath.is_file()
