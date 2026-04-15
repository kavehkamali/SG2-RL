"""FFW SG2 articulation config (self-contained, no external package imports).

Robot USD ships in assets/robots/FFW_SG2.usd. Override with SG2_ROBOT_USD env var.
"""
from __future__ import annotations

import os
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import ArticulationRootPropertiesCfg, RigidBodyPropertiesCfg, UsdFileCfg

_ASSETS_DIR = str(Path(__file__).resolve().parents[2] / "assets")
_ROBOT_USD = os.environ.get("SG2_ROBOT_USD", os.path.join(_ASSETS_DIR, "robots", "FFW_SG2.usd"))


FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=_ROBOT_USD,
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},
            "head_joint1": 0.0,
            "head_joint2": 0.0,
            "lift_joint": 0.0,
        },
    ),
    actuators={
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[1-2]", "arm_r_joint[1-2]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[3-6]", "arm_r_joint[3-6]"],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint7", "arm_r_joint7"],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),
        "gripper_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1", "gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "gripper_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]", "gripper_r_joint[2-4]"],
            effort_limit_sim=20.0,
            stiffness=2.0,
            damping=0.5,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    },
)
