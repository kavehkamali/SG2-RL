"""OmniReset-style bimanual peg handover+insertion env for FFW-SG2.

Scene constants (peg / hole / desk / robot positions) are taken verbatim from
the existing ``sg2_rl.env_cfg`` SG2 scene — only the *RL configuration* (obs,
rewards, events, terminations, action groups) follows UWLab OmniReset.

Task: peg starts either on the desk or in one of the grippers; policy must
reach → pick (left) → handover → place into hole with the right gripper.
The 4 reset kinds expose the agent to all stages of this multi-phase task
without needing an expert dataset.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from sg2_rl import task_mdp
from sg2_rl import omnireset_task_mdp as omdp
from sg2_rl.env_cfg import (
    FfwSg2PegPartialAssemblySceneCfg,
    _PEG_HOLE_XY,
    _PEG_ON_TABLE_Z,
    _PEG_TABLE_OFFSET_X,
    _RECEPTIVE_SPAWN_Z,
    _TABLE_SURFACE_Z,
)
from sg2_rl.physics_material_bind import bind_sg2rl_prop_shared_physics_materials

# Nominal peg/hole spawn positions, pulled verbatim from the SG2 scene.
_PEG_NOMINAL_XYZ = (_PEG_HOLE_XY[0] + _PEG_TABLE_OFFSET_X, _PEG_HOLE_XY[1], _PEG_ON_TABLE_Z)
_HOLE_NOMINAL_XYZ = (_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _RECEPTIVE_SPAWN_Z)


# ---------------------------------------------------------------------------
# Actions — bimanual joint-PD (OmniReset uses joint-PD; OSC kept as future work)
# ---------------------------------------------------------------------------
@configclass
class FfwSg2OmniResetBimanualActionsCfg:
    torso_lift = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["lift_joint"], scale=0.22, use_default_offset=True,
    )
    arms_grippers = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["arm_l_joint.*", "arm_r_joint.*", "gripper_l_joint.*", "gripper_r_joint.*"],
        scale=0.08,
        use_default_offset=True,
    )
    wheels_head_misc = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["^(?!lift_joint$)(?!arm_[lr]_joint)(?!gripper_[lr]_joint).+$"],
        scale=0.14,
        use_default_offset=True,
    )


# ---------------------------------------------------------------------------
# Observations — OmniReset policy/critic split
# ---------------------------------------------------------------------------
@configclass
class FfwSg2OmniResetObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        last_action = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        # Left end-effector pose (primary grasp hand)
        left_ee_pose = ObsTerm(
            func=omdp.ee_pose_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "wrist": "left", "rotation_repr": "axis_angle"},
        )
        right_ee_pose = ObsTerm(
            func=omdp.ee_pose_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "wrist": "right", "rotation_repr": "axis_angle"},
        )
        # Peg pose in left-EE frame (grasp), and receptive pose in right-EE frame (insertion)
        peg_in_left_ee = ObsTerm(
            func=omdp.asset_pose_in_ee_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "wrist": "left",
                "rotation_repr": "axis_angle",
            },
        )
        hole_in_right_ee = ObsTerm(
            func=omdp.asset_pose_in_ee_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "wrist": "right",
                "rotation_repr": "axis_angle",
            },
        )
        # Peg pose in hole frame (task-critical)
        peg_in_hole = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        # Everything the policy sees (un-corrupted) + privileged info.
        last_action = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        left_ee_pose = ObsTerm(
            func=omdp.ee_pose_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "wrist": "left", "rotation_repr": "axis_angle"},
        )
        right_ee_pose = ObsTerm(
            func=omdp.ee_pose_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "wrist": "right", "rotation_repr": "axis_angle"},
        )
        peg_in_left_ee = ObsTerm(
            func=omdp.asset_pose_in_ee_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "wrist": "left",
                "rotation_repr": "axis_angle",
            },
        )
        hole_in_right_ee = ObsTerm(
            func=omdp.asset_pose_in_ee_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "robot_cfg": SceneEntityCfg("robot"),
                "wrist": "right",
                "rotation_repr": "axis_angle",
            },
        )
        peg_in_hole = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )
        # Privileged
        time_left = ObsTerm(func=omdp.time_left)
        left_ee_vel = ObsTerm(
            func=omdp.ee_velocity_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "wrist": "left"},
        )
        right_ee_vel = ObsTerm(
            func=omdp.ee_velocity_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "wrist": "right"},
        )
        peg_vel_w = ObsTerm(
            func=task_mdp.asset_root_lin_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("insertive_object")},
        )
        hole_vel_w = ObsTerm(
            func=task_mdp.asset_root_lin_ang_vel_w,
            params={"asset_cfg": SceneEntityCfg("receptive_object")},
        )
        robot_mass = ObsTerm(func=omdp.get_mass, params={"asset_cfg": SceneEntityCfg("robot")})
        peg_mass = ObsTerm(func=omdp.get_mass, params={"asset_cfg": SceneEntityCfg("insertive_object")})
        hole_mass = ObsTerm(func=omdp.get_mass, params={"asset_cfg": SceneEntityCfg("receptive_object")})
        peg_material = ObsTerm(
            func=omdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("insertive_object")}
        )
        hole_material = ObsTerm(
            func=omdp.get_material_properties, params={"asset_cfg": SceneEntityCfg("receptive_object")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards — OmniReset weights verbatim (plus SG2 reuse)
# ---------------------------------------------------------------------------
@configclass
class FfwSg2OmniResetRewardsCfg:
    # Stateful progress tracker (keeps per-env buffers; returns 0 from __call__).
    progress_context = RewTerm(
        func=task_mdp.ProgressContext,
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )
    # EE → peg proximity (left gripper approach).
    ee_peg_distance = RewTerm(
        func=omdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "wrist": "left",
            "std": 0.5,
        },
    )
    # EE → hole proximity (right gripper approach during/after handover).
    ee_hole_distance = RewTerm(
        func=omdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "target_asset_cfg": SceneEntityCfg("receptive_object"),
            "wrist": "right",
            "std": 0.5,
        },
    )
    # Dense insertion shaping (SG2 reuse).
    dense_success_reward = RewTerm(
        func=task_mdp.dense_success_reward,
        weight=0.1,
        params={
            "std": 0.12,
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )
    # Sparse success.
    success_reward = RewTerm(
        func=task_mdp.success_reward,
        weight=1.0,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )
    # OmniReset action / velocity penalties.
    action_magnitude = RewTerm(func=task_mdp.action_l2_clamped, weight=-1e-4)
    action_rate = RewTerm(func=task_mdp.action_delta_l2_clamped, weight=-1e-3)
    arm_joint_vel = RewTerm(
        func=omdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm_l_joint.*", "arm_r_joint.*"]),
            "max_val": 4.0,
        },
    )
    # Heavy penalty for runaway dynamics (mirrors OmniReset abnormal_robot weight).
    abnormal_robot = RewTerm(
        func=omdp.abnormal_robot_state,
        weight=-100.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "vel_limit_scale": 3.0},
    )


# ---------------------------------------------------------------------------
# Events — bimanual 4-path reset sampler + material/mass randomization
# ---------------------------------------------------------------------------
@configclass
class FfwSg2OmniResetEventsCfg:
    bind_prop_shared_physics_materials = EventTerm(
        func=bind_sg2rl_prop_shared_physics_materials,
        mode="startup",
        params={},
    )
    # Material randomization — OmniReset uses startup-mode ranges.
    randomize_peg_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.55, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    randomize_hole_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.55, 1.0),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
        },
    )
    # Mass randomization.
    randomize_peg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    # OmniReset-style bimanual reset distribution (replaces default uniform resets).
    bimanual_reset = EventTerm(
        func=omdp.BimanualResetSampler,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "insertive_cfg": SceneEntityCfg("insertive_object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "peg_nominal_xyz": _PEG_NOMINAL_XYZ,
            "hole_nominal_xyz": _HOLE_NOMINAL_XYZ,
            "weights": (0.25, 0.25, 0.25, 0.25),
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
@configclass
class FfwSg2OmniResetTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(
        func=omdp.abnormal_robot_state_done,
        params={"robot_cfg": SceneEntityCfg("robot"), "vel_limit_scale": 3.0},
    )
    success = DoneTerm(
        func=task_mdp.insertion_success_done,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "receptive_asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )


# ---------------------------------------------------------------------------
# Top-level env config
# ---------------------------------------------------------------------------
@configclass
class FfwSg2OmniResetBimanualPegInsertEnvCfg(ManagerBasedRLEnvCfg):
    scene: FfwSg2PegPartialAssemblySceneCfg = FfwSg2PegPartialAssemblySceneCfg(num_envs=1, env_spacing=2.0)
    actions: FfwSg2OmniResetBimanualActionsCfg = FfwSg2OmniResetBimanualActionsCfg()
    observations: FfwSg2OmniResetObservationsCfg = FfwSg2OmniResetObservationsCfg()
    rewards: FfwSg2OmniResetRewardsCfg = FfwSg2OmniResetRewardsCfg()
    events: FfwSg2OmniResetEventsCfg = FfwSg2OmniResetEventsCfg()
    terminations: FfwSg2OmniResetTerminationsCfg = FfwSg2OmniResetTerminationsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.55, 0.0, 1.10),
        lookat=(_PEG_HOLE_XY[0], _PEG_HOLE_XY[1], _TABLE_SURFACE_Z + 0.12),
        origin_type="world",
        env_index=0,
        asset_name="receptive_object",
        resolution=(1920, 1080),
    )

    def __post_init__(self):
        # OmniReset uses decimation=12, sim.dt=1/120 → 20 Hz control, 16 s episodes.
        self.decimation = 12
        self.episode_length_s = 16.0
        self.sim.dt = 1 / 120.0
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
