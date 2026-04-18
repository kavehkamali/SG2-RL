"""FFW-SG2 peg-in-hole RL env configs (PPO-ready, state-only).

Scene: FFW-SG2 robot + work surface (kinematic cuboid) + peg + peg-hole + ground + skylight.

This file defines only the PPO training environments (curriculum stage1 + stage2) plus a minimal smoke env.
Older approach-only variants were intentionally removed to keep the training surface area small.
"""
from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from sg2_rl import task_mdp
from sg2_rl.physics_material_bind import bind_sg2rl_prop_shared_physics_materials
from sg2_rl.robot_cfg import FFW_SG2_CFG

# ---------------------------------------------------------------------------
# Asset paths — props shipped in-repo, robot USD external
# ---------------------------------------------------------------------------
_ASSETS_DIR = str(Path(__file__).resolve().parents[2] / "assets")
_PROPS_DIR = os.environ.get("SG2_PROPS_DIR", os.path.join(_ASSETS_DIR, "props"))

# ---------------------------------------------------------------------------
# Scene geometry constants (scene geometry constants)
# ---------------------------------------------------------------------------
_TABLE_THICKNESS_M = 0.05
_TABLE_SURFACE_Z = 0.82
_TABLE_CENTER_Z = _TABLE_SURFACE_Z - _TABLE_THICKNESS_M / 2.0
_ROBOT_HEAD_WIDTH_M = 0.20
_SCENE_FORWARD_OFFSET_X = 0.30 + _ROBOT_HEAD_WIDTH_M
# Default layout overrides:
# - Move the hole (and thus the whole cluster) toward the robot.
# - Place the pin even closer toward the robot relative to the hole.
_HOLE_SHIFT_TOWARD_ROBOT_X = -0.10
_PEG_OFFSET_FROM_HOLE_X = -0.15

_PEG_HOLE_XY = (0.12 + _SCENE_FORWARD_OFFSET_X + _HOLE_SHIFT_TOWARD_ROBOT_X, 0.0)
_TABLETOP_TOP_CENTER_XYZ = (_PEG_HOLE_XY[0] + 0.03, _PEG_HOLE_XY[1], _TABLE_SURFACE_Z)
_PEG_TABLE_OFFSET_X = _PEG_OFFSET_FROM_HOLE_X
_RECEPTIVE_SPAWN_Z = _TABLE_SURFACE_Z + 0.14
_PEG_ON_TABLE_Z = _TABLE_SURFACE_Z + 0.025


# ===================================================================
# Scene
# ===================================================================
@configclass
class FfwSg2PegPartialAssemblySceneCfg(InteractiveSceneCfg):
    replicate_physics: bool = True

    robot = FFW_SG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    work_surface: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkSurface",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.75, _TABLE_THICKNESS_M),
            copy_from_source=False,
            physics_material_path="/World/SG2RL_SharedMaterials/TablePhys",
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.75, dynamic_friction=0.75, restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True,
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0] + 0.03, _PEG_HOLE_XY[1], _TABLE_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_PROPS_DIR}/peg.usd",
            copy_from_source=False, scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
                disable_gravity=False, kinematic_enabled=False,
            ),
            # Mass is defined in peg.usd; avoid modify_mass_properties on instanced props at high num_envs.
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0] + _PEG_TABLE_OFFSET_X, _PEG_HOLE_XY[1], _PEG_ON_TABLE_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_PROPS_DIR}/peg_hole.usd",
            copy_from_source=False, scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0,
                disable_gravity=False, kinematic_enabled=False,
            ),
            # Mass is defined in peg_hole.usd; avoid modify_mass_properties on instanced props at high num_envs.
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _RECEPTIVE_SPAWN_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(
            usd_path=f"{_ASSETS_DIR}/environments/default_environment.usd",
            color=(0.15, 0.15, 0.15),
        ),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=600.0,
            color=(1.0, 1.0, 1.0),
        ),
    )

    key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=1000.0,
            color=(1.0, 0.98, 0.95),
            angle=1.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.866, 0.0, 0.25, -0.433),
        ),
    )


# ===================================================================
# Actions (split into three terms to avoid Hydra key-ordering issues)
# ===================================================================
@configclass
class FfwSg2PegPartialAssemblyActionsCfg:
    torso_lift = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["lift_joint"], scale=0.22, use_default_offset=True,
    )
    arms_grippers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint.*", "arm_r_joint.*", "gripper_l_joint.*", "gripper_r_joint.*"],
        scale=0.08, use_default_offset=True,
    )
    wheels_head_misc = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!lift_joint$)(?!arm_[lr]_joint)(?!gripper_[lr]_joint).+$"],
        scale=0.14, use_default_offset=True,
    )


@configclass
class FfwSg2PegStage1ActionsCfg:
    """Stage1: same action vector layout as :class:`FfwSg2PegPartialAssemblyActionsCfg`, but **right arm + right
    gripper commands are multiplied by zero** (network can still output those dimensions; they have no effect).

    Uses per-regex ``scale`` on :class:`~isaaclab.envs.mdp.actions.JointPositionActionCfg` (see Isaac Lab docs).
    """

    torso_lift = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["lift_joint"], scale=0.22, use_default_offset=True,
    )
    arms_grippers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint.*", "arm_r_joint.*", "gripper_l_joint.*", "gripper_r_joint.*"],
        scale={
            "arm_l_joint.*": 0.08,
            "arm_r_joint.*": 0.0,
            "gripper_l_joint.*": 0.08,
            "gripper_r_joint.*": 0.0,
        },
        use_default_offset=True,
    )
    wheels_head_misc = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!lift_joint$)(?!arm_[lr]_joint)(?!gripper_[lr]_joint).+$"],
        scale=0.14,
        use_default_offset=True,
    )


# ===================================================================
# Observations
# ===================================================================
@configclass
class FfwSg2PegPartialAssemblyObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp.last_action)
        peg_vel_w = ObsTerm(func=task_mdp.asset_root_lin_ang_vel_w, params={"asset_cfg": SceneEntityCfg("insertive_object")})
        hole_vel_w = ObsTerm(func=task_mdp.asset_root_lin_ang_vel_w, params={"asset_cfg": SceneEntityCfg("receptive_object")})
        peg_in_hole_frame = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("insertive_object"),
                    "root_asset_cfg": SceneEntityCfg("receptive_object"), "rotation_repr": "axis_angle"},
        )
        peg_in_robot = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("insertive_object"),
                    "root_asset_cfg": SceneEntityCfg("robot"), "rotation_repr": "axis_angle"},
        )
        hole_in_robot = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("receptive_object"),
                    "root_asset_cfg": SceneEntityCfg("robot"), "rotation_repr": "axis_angle"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ===================================================================
# Rewards
# ===================================================================
@configclass
class FfwSg2PegInsertionRewardsCfg:
    progress_context = RewTerm(func=task_mdp.ProgressContext, weight=0.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.35,
        params={"std": 0.22, "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    collision_free = RewTerm(func=task_mdp.collision_free, weight=0.25, params={})
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)
    action_smoothness = RewTerm(func=task_mdp.action_delta_l2_clamped, weight=-3e-4)
    time_penalty = RewTerm(func=task_mdp.time_penalty, weight=-2e-3)


@configclass
class FfwSg2PegInsertionRewardsApproachLiftCfg:
    progress_context = RewTerm(func=task_mdp.ProgressContext, weight=0.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    # Use left wrist only — ``min(left,right)`` lets the policy score approach by moving the right arm.
    wrist_to_peg = RewTerm(
        func=task_mdp.wrist_min_distance_to_asset_exp,
        weight=0.45,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "sigma": 0.32,
            "wrist": "left",
        },
    )
    wrist_approach_progress = RewTerm(
        func=task_mdp.WristToInsertiveApproachProgress,
        weight=0.55,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "clip_m": 0.012,
            "wrist": "left",
        },
    )
    peg_xy_to_hole = RewTerm(func=task_mdp.insertive_xy_near_receptor_tanh, weight=0.14,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object"), "std": 0.16})
    peg_lift = RewTerm(func=task_mdp.insertive_height_above_surface, weight=0.20,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "surface_z": _TABLE_SURFACE_Z, "scale": 0.07})
    left_wrist_table_clearance = RewTerm(
        func=task_mdp.left_wrist_clearance_above_surface_exp,
        weight=0.12,
        params={"robot_cfg": SceneEntityCfg("robot"), "surface_z": _TABLE_SURFACE_Z, "min_clearance_m": 0.10, "sigma_m": 0.055},
    )
    gripper_near_peg = RewTerm(
        func=task_mdp.gripper_excitation_near_insertive,
        weight=0.18,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "gripper_joint_cfg": SceneEntityCfg("robot", joint_names=["gripper_l_joint1"]),
            "proximity_std": 0.30,
            "joint_scale": 0.35,
            "proximity_wrist": "left",
        },
    )
    # Stage1 objective: lift and hold the peg until the end of the episode (no insertion success).
    hold_reward = RewTerm(
        func=task_mdp.left_hold_reward,
        weight=0.45,
        params={"robot_cfg": SceneEntityCfg("robot"), "insertive_asset_cfg": SceneEntityCfg("insertive_object"), "surface_z": _TABLE_SURFACE_Z, "lift_dz_m": 0.03},
    )
    hold_success_terminal = RewTerm(
        func=task_mdp.left_hold_terminal_reward,
        weight=3.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "insertive_asset_cfg": SceneEntityCfg("insertive_object"), "surface_z": _TABLE_SURFACE_Z, "lift_dz_m": 0.03},
    )
    collision_free = RewTerm(func=task_mdp.collision_free, weight=0.14, params={})
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)
    action_smoothness = RewTerm(func=task_mdp.action_delta_l2_clamped, weight=-3e-4)
    # Right arm is not actuated in Stage1; keep a small residual penalty for simulator coupling / drift.
    right_hand_steady = RewTerm(func=task_mdp.right_hand_steady_penalty, weight=-4e-4, params={"robot_cfg": SceneEntityCfg("robot")})
    left_hand_smooth = RewTerm(func=task_mdp.LeftHandSmoothnessPenalty, weight=-1.2e-3, params={"robot_cfg": SceneEntityCfg("robot")})
    time_penalty = RewTerm(func=task_mdp.time_penalty, weight=-2e-3)


@configclass
class FfwSg2PegInsertionRewardsCurriculumFullCfg:
    """Stage2: keep grasp/lift shaping but emphasize insertion."""

    progress_context = RewTerm(
        func=task_mdp.ProgressContext,
        weight=0.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object")},
    )

    # Grasp/lift shaping (keep agent competent at holding/lifting).
    wrist_to_peg = RewTerm(
        func=task_mdp.wrist_min_distance_to_asset_exp,
        weight=0.22,
        params={"robot_cfg": SceneEntityCfg("robot"), "target_asset_cfg": SceneEntityCfg("insertive_object"), "sigma": 0.28},
    )
    wrist_approach_progress = RewTerm(
        func=task_mdp.WristToInsertiveApproachProgress,
        weight=0.28,
        params={"robot_cfg": SceneEntityCfg("robot"), "target_asset_cfg": SceneEntityCfg("insertive_object"), "clip_m": 0.010},
    )
    peg_lift = RewTerm(
        func=task_mdp.insertive_height_above_surface,
        weight=0.26,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "surface_z": _TABLE_SURFACE_Z, "scale": 0.06},
    )
    gripper_near_peg = RewTerm(
        func=task_mdp.gripper_excitation_near_insertive,
        weight=0.20,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "gripper_joint_cfg": SceneEntityCfg("robot", joint_names=["gripper_l_joint1", "gripper_r_joint1"]),
            "proximity_std": 0.26,
            "joint_scale": 0.30,
        },
    )

    # Insertion shaping (dominant).
    peg_xy_to_hole = RewTerm(
        func=task_mdp.insertive_xy_near_receptor_tanh,
        weight=0.55,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object"), "std": 0.06},
    )
    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward,
        weight=0.75,
        params={"std": 0.06, "insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object")},
    )
    success_reward = RewTerm(
        func=task_mdp.success_reward,
        weight=2.5,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object")},
    )

    collision_free = RewTerm(func=task_mdp.collision_free, weight=0.10, params={})
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)
    action_smoothness = RewTerm(func=task_mdp.action_delta_l2_clamped, weight=-3e-4)
    right_hand_steady = RewTerm(func=task_mdp.right_hand_steady_penalty, weight=-8e-4, params={"robot_cfg": SceneEntityCfg("robot")})
    left_hand_smooth = RewTerm(func=task_mdp.LeftHandSmoothnessPenalty, weight=-6e-4, params={"robot_cfg": SceneEntityCfg("robot")})
    time_penalty = RewTerm(func=task_mdp.time_penalty, weight=-3e-3)


# ===================================================================
# Terminations / Events
# ===================================================================
@configclass
class FfwSg2PegInsertionTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.insertion_success_done,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object")},
    )


@configclass
class FfwSg2PegStage1HoldTerminationsCfg:
    """Stage1 success: lift + hold peg until the end of the episode."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    success = DoneTerm(
        func=task_mdp.left_hold_success_at_timeout,
        params={"robot_cfg": SceneEntityCfg("robot"), "insertive_asset_cfg": SceneEntityCfg("insertive_object"), "surface_z": _TABLE_SURFACE_Z, "lift_dz_m": 0.03},
    )

@configclass
class FfwSg2PegInsertionEventsCfgSmokeArmAndPeg:
    bind_prop_shared_physics_materials = EventTerm(
        func=bind_sg2rl_prop_shared_physics_materials,
        mode="startup",
        params={},
    )
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
                "position_range": (-0.18, 0.18), "velocity_range": (-0.08, 0.08)})
    reset_peg_xyyaw = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.06, 0.06), "y": (-0.06, 0.06), "z": (-0.004, 0.004),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.55, 0.55)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("insertive_object")})

@configclass
class FfwSg2PegInsertionEventsCfgHighArmNoise:
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
                "position_range": (-0.18, 0.18), "velocity_range": (-0.08, 0.08)})

@configclass
class FfwSg2PegInsertionEventsCfgApproachOnlyShared:
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
                "position_range": (-0.15, 0.15), "velocity_range": (-0.07, 0.07)})
    reset_lift_joint = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lift_joint"]),
                "position_range": (0.10, 0.24), "velocity_range": (0.0, 0.0)})
    reset_peg_xyyaw = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.004, 0.004),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.45, 0.45)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                                   "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "asset_cfg": SceneEntityCfg("insertive_object")})


@configclass
class FfwSg2PegInsertionEventsCfgPpoStage1:
    """Stage1: keep peg on table near nominal, randomize arm joints."""

    bind_prop_shared_physics_materials = EventTerm(
        func=bind_sg2rl_prop_shared_physics_materials,
        mode="startup",
        params={},
    )
    reset_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
            "position_range": (-0.20, 0.20),
            "velocity_range": (-0.10, 0.10),
        },
    )
    reset_lift_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["lift_joint"]), "position_range": (0.08, 0.26), "velocity_range": (0.0, 0.0)},
    )
    reset_peg_xyyaw = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.004, 0.004), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.55, 0.55)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("insertive_object"),
        },
    )


@configclass
class FfwSg2PegInsertionEventsCfgPpoStage2(FfwSg2PegInsertionEventsCfgPpoStage1):
    """Stage2: slightly tighter peg reset for insertion curriculum."""

    reset_peg_xyyaw = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.003, 0.003), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-0.35, 0.35)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("insertive_object"),
        },
    )


# ===================================================================
# Top-level env configs
# ===================================================================
@configclass
class FfwSg2PegPartialAssemblySmokeEnvCfg(ManagerBasedRLEnvCfg):
    scene: FfwSg2PegPartialAssemblySceneCfg = FfwSg2PegPartialAssemblySceneCfg(num_envs=1, env_spacing=2.0)
    events: FfwSg2PegInsertionEventsCfgSmokeArmAndPeg = FfwSg2PegInsertionEventsCfgSmokeArmAndPeg()
    terminations: FfwSg2PegInsertionTerminationsCfg = FfwSg2PegInsertionTerminationsCfg()
    observations: FfwSg2PegPartialAssemblyObservationsCfg = FfwSg2PegPartialAssemblyObservationsCfg()
    actions: FfwSg2PegPartialAssemblyActionsCfg = FfwSg2PegPartialAssemblyActionsCfg()
    rewards: FfwSg2PegInsertionRewardsCfg = FfwSg2PegInsertionRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.55, 0.0, 1.10),
        lookat=(_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _TABLE_SURFACE_Z + 0.12),
        origin_type="world", env_index=0, asset_name="receptive_object", resolution=(1920, 1080),
    )

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 6.667
        self.sim.dt = 1 / 120.0
        # PhysX: match Isaac Lab Factory peg-insert-style settings (contact-rich GPU sim).
        # GPU PhysX does not grow all buffers dynamically — undersized gpu_* limits at ~16k envs
        # can manifest as malloc/abort during "Starting the simulation…", not only OOM.
        # See: isaaclab_tasks/direct/factory/factory_env_cfg.py (PhysxCfg block).
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_max_num_partitions = 1
        self.sim.render_interval = self.decimation


@configclass
class FfwSg2PegPartialAssemblySmokeApproachLiftEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    actions: FfwSg2PegStage1ActionsCfg = FfwSg2PegStage1ActionsCfg()
    rewards: FfwSg2PegInsertionRewardsApproachLiftCfg = FfwSg2PegInsertionRewardsApproachLiftCfg()
    events: FfwSg2PegInsertionEventsCfgHighArmNoise = FfwSg2PegInsertionEventsCfgHighArmNoise()


@configclass
class FfwSg2PegPartialAssemblyPpoCurriculumEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    """Stage2 PPO env: full shaped reward (grasp/lift + insertion)."""

    rewards: FfwSg2PegInsertionRewardsCurriculumFullCfg = FfwSg2PegInsertionRewardsCurriculumFullCfg()


@configclass
class FfwSg2PegPpoStage1EnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    """Stage1 PPO env."""

    actions: FfwSg2PegStage1ActionsCfg = FfwSg2PegStage1ActionsCfg()
    terminations: FfwSg2PegStage1HoldTerminationsCfg = FfwSg2PegStage1HoldTerminationsCfg()
    rewards: FfwSg2PegInsertionRewardsApproachLiftCfg = FfwSg2PegInsertionRewardsApproachLiftCfg()
    events: FfwSg2PegInsertionEventsCfgPpoStage1 = FfwSg2PegInsertionEventsCfgPpoStage1()


@configclass
class FfwSg2PegPpoStage2EnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    """Stage2 PPO env."""

    rewards: FfwSg2PegInsertionRewardsCurriculumFullCfg = FfwSg2PegInsertionRewardsCurriculumFullCfg()
    events: FfwSg2PegInsertionEventsCfgPpoStage2 = FfwSg2PegInsertionEventsCfgPpoStage2()


# Note: Approach-only env configs removed (deprecated).
