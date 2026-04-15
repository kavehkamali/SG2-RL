"""FFW-SG2 peg-in-hole env configs — self-contained, no uwlab imports.

Scene: FFW-SG2 robot + work surface (kinematic cuboid) + peg + peg-hole + ground + skylight.
The exact geometry, positions, and physics settings are preserved from the original UWLab config.
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
from sg2_rl.robot_cfg import FFW_SG2_CFG
from sg2_rl.tabletop_rewards import (
    WristsToWorldPointApproachProgressSum,
    wrists_min_distance_to_world_point_exp_sum,
)

# ---------------------------------------------------------------------------
# Asset paths — props shipped in-repo, robot USD external
# ---------------------------------------------------------------------------
_ASSETS_DIR = str(Path(__file__).resolve().parents[2] / "assets")
_PROPS_DIR = os.environ.get("SG2_PROPS_DIR", os.path.join(_ASSETS_DIR, "props"))

# ---------------------------------------------------------------------------
# Scene geometry constants (identical to UWLab config)
# ---------------------------------------------------------------------------
_TABLE_THICKNESS_M = 0.05
_TABLE_SURFACE_Z = 0.82
_TABLE_CENTER_Z = _TABLE_SURFACE_Z - _TABLE_THICKNESS_M / 2.0
_ROBOT_HEAD_WIDTH_M = 0.20
_SCENE_FORWARD_OFFSET_X = 0.30 + _ROBOT_HEAD_WIDTH_M
_PEG_HOLE_XY = (0.12 + _SCENE_FORWARD_OFFSET_X, 0.0)
_TABLETOP_TOP_CENTER_XYZ = (_PEG_HOLE_XY[0] + 0.03, _PEG_HOLE_XY[1], _TABLE_SURFACE_Z)
_PEG_TABLE_OFFSET_X = 0.10
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _RECEPTIVE_SPAWN_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(color=None,
            usd_path=f"{_ASSETS_DIR}/environments/default_environment.usd",
        ),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{_ASSETS_DIR}/textures/kloofendal_43d_clear_puresky_4k.hdr",
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


@configclass
class FfwSg2PegInsertionRewardsApproachLiftCfg:
    progress_context = RewTerm(func=task_mdp.ProgressContext, weight=0.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    wrist_to_peg = RewTerm(func=task_mdp.wrist_min_distance_to_asset_exp, weight=0.45,
        params={"robot_cfg": SceneEntityCfg("robot"), "target_asset_cfg": SceneEntityCfg("insertive_object"), "sigma": 0.32})
    wrist_approach_progress = RewTerm(func=task_mdp.WristToInsertiveApproachProgress, weight=0.55,
        params={"robot_cfg": SceneEntityCfg("robot"), "target_asset_cfg": SceneEntityCfg("insertive_object"), "clip_m": 0.012})
    peg_xy_to_hole = RewTerm(func=task_mdp.insertive_xy_near_receptor_tanh, weight=0.14,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "receptive_asset_cfg": SceneEntityCfg("receptive_object"), "std": 0.16})
    peg_lift = RewTerm(func=task_mdp.insertive_height_above_surface, weight=0.20,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"), "surface_z": _TABLE_SURFACE_Z, "scale": 0.07})
    gripper_near_peg = RewTerm(func=task_mdp.gripper_excitation_near_insertive, weight=0.18,
        params={"robot_cfg": SceneEntityCfg("robot"), "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "gripper_joint_cfg": SceneEntityCfg("robot", joint_names=["gripper_l_joint1", "gripper_r_joint1"]),
                "proximity_std": 0.30, "joint_scale": 0.35})
    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.08,
        params={"std": 2.0, "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    collision_free = RewTerm(func=task_mdp.collision_free, weight=0.14, params={})
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


@configclass
class FfwSg2PegInsertionRewardsApproachOnlyCfgA:
    wrists_clearance_above_surface = RewTerm(func=task_mdp.wrists_clearance_above_surface_exp, weight=0.58,
        params={"robot_cfg": SceneEntityCfg("robot"), "surface_z": _TABLE_SURFACE_Z, "min_clearance_m": 0.18, "sigma_m": 0.048})
    both_wrists_to_pin = RewTerm(func=wrists_min_distance_to_world_point_exp_sum, weight=0.38,
        params={"robot_cfg": SceneEntityCfg("robot"), "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ, "sigma": 0.36})
    both_wrists_approach_progress = RewTerm(func=WristsToWorldPointApproachProgressSum, weight=0.32,
        params={"robot_cfg": SceneEntityCfg("robot"), "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ, "clip_m": 0.018})
    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.28,
        params={"std": 0.22, "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


@configclass
class FfwSg2PegInsertionRewardsApproachOnlyCfgB:
    wrists_clearance_above_surface = RewTerm(func=task_mdp.wrists_clearance_above_surface_exp, weight=0.42,
        params={"robot_cfg": SceneEntityCfg("robot"), "surface_z": _TABLE_SURFACE_Z, "min_clearance_m": 0.18, "sigma_m": 0.042})
    both_wrists_to_pin = RewTerm(func=wrists_min_distance_to_world_point_exp_sum, weight=0.52,
        params={"robot_cfg": SceneEntityCfg("robot"), "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ, "sigma": 0.22})
    both_wrists_approach_progress = RewTerm(func=WristsToWorldPointApproachProgressSum, weight=0.40,
        params={"robot_cfg": SceneEntityCfg("robot"), "anchor_xyz": _TABLETOP_TOP_CENTER_XYZ, "clip_m": 0.011})
    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward, weight=0.28,
        params={"std": 0.22, "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    success_reward = RewTerm(func=task_mdp.success_reward, weight=1.0,
        params={"insertive_asset_cfg": SceneEntityCfg("insertive_object"),
                "receptive_asset_cfg": SceneEntityCfg("receptive_object")})
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)


# ===================================================================
# Terminations / Events
# ===================================================================
@configclass
class FfwSg2PegInsertionTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

@configclass
class FfwSg2PegInsertionEventsCfgSmokeArmAndPeg:
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
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120.0
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 1024 * 1024
        self.sim.render_interval = self.decimation


@configclass
class FfwSg2PegPartialAssemblySmokeApproachLiftEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    rewards: FfwSg2PegInsertionRewardsApproachLiftCfg = FfwSg2PegInsertionRewardsApproachLiftCfg()
    events: FfwSg2PegInsertionEventsCfgHighArmNoise = FfwSg2PegInsertionEventsCfgHighArmNoise()


@configclass
class FfwSg2PegPartialAssemblyApproachOnlyAEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    rewards: FfwSg2PegInsertionRewardsApproachOnlyCfgA = FfwSg2PegInsertionRewardsApproachOnlyCfgA()
    events: FfwSg2PegInsertionEventsCfgApproachOnlyShared = FfwSg2PegInsertionEventsCfgApproachOnlyShared()


@configclass
class FfwSg2PegPartialAssemblyApproachOnlyBEnvCfg(FfwSg2PegPartialAssemblySmokeEnvCfg):
    rewards: FfwSg2PegInsertionRewardsApproachOnlyCfgB = FfwSg2PegInsertionRewardsApproachOnlyCfgB()
    events: FfwSg2PegInsertionEventsCfgApproachOnlyShared = FfwSg2PegInsertionEventsCfgApproachOnlyShared()
