# Agent Guide — SG2-RL

Context for AI agents working on this project.

## What This Project Is

A self-contained RL workspace for training a bimanual humanoid robot (Fourier FFW SG2) to perform peg-in-hole insertion on a tabletop. Built on Isaac Lab / Isaac Sim.

## Architecture

```
src/sg2_rl/
├── env_cfg.py          # THE scene definition (robot + table + peg + hole + physics)
├── robot_cfg.py        # FFW SG2 articulation (joints, actuators, USD path)
├── task_mdp.py         # Observation, reward, termination functions
│                       # NOTE: reward stubs return 0 — implement real ones for training
├── tabletop_rewards.py # Wrist-to-anchor approach rewards (functional)
├── config_loader.py    # Loads env cfg from gym registry — replaces Hydra
├── gym_register.py     # Maps task IDs → env config classes
├── right_gripper_ik.py # Differential IK for right arm (arm_r_link7)
├── apf_path.py         # APF path planner (Khatib 1986)
├── arm_avoidance.py    # Arm-link sphere collision nudging
├── render_quality.py   # Resolution for headless recording
├── scene_layout.py     # Peg/hole/table position shifts
├── peg_grasp_monitor.py# Wrist-pin distance / peg height checks
├── orbit_camera.py     # Orbit camera look-at helpers
├── usd_gizmo.py        # RGB axis debug overlays
├── usd_path_curve.py   # Polyline visualization in USD
└── paths.py            # Repo root / configs dir
```

## Key Design Decisions

- **Self-contained**: All USD assets ship in `assets/`. No external repos or Nucleus server.
- **No Hydra**: `config_loader.py` replaces `hydra_task_compose` with direct class instantiation.
- **Reward stubs**: `task_mdp.py` has stub rewards (return 0) for `ProgressContext`, `dense_success_reward`, `success_reward`, `collision_free`. These let the env instantiate for recording but need real implementations for RL training.
- **Recording scripts own their control loop**: They create the env, then run IK / phase logic directly — they don't use the env's reward or observation pipeline during recording.
- **Headless rendering**: The kit file disables RTX reflections/AO/indirect for clean rasterized output. Don't re-enable them — it causes dotted/noisy frames.

## Robot Joint Layout

| Group | Joints | Count | Action Scale |
|-------|--------|-------|-------------|
| Torso lift | `lift_joint` | 1 | 0.22 |
| Left arm | `arm_l_joint1`–`7` | 7 | 0.08 |
| Right arm | `arm_r_joint1`–`7` | 7 | 0.08 |
| Left gripper | `gripper_l_joint1`–`4` | 4 | 0.08 |
| Right gripper | `gripper_r_joint1`–`4` | 4 | 0.08 |
| Head | `head_joint1`, `head_joint2` | 2 | 0.14 |
| Wheels/misc | everything else | varies | 0.14 |

## Scene Geometry (constants in env_cfg.py)

- Table surface Z: **0.82 m**
- Table size: 1.0 × 0.75 × 0.05 m
- Peg-hole XY: **(0.62, 0.0)** (offset 0.5 m in +X from robot)
- Peg starts: 0.10 m in +X from hole, on the table
- Hole spawns: 0.14 m above table, settles under gravity

## Common Tasks

### Add a new recording script
1. Copy `record_grasp_lift_peg.py` as template
2. Import `from sg2_rl.config_loader import task_config`
3. Use `@task_config(args_cli.task, "skrl_cfg_entry_point", [])` decorator
4. Build IK context with `build_right_gripper_ik()`
5. Control the robot with `actions_for_ee_goal()` or `actions_lift_only()`

### Implement real reward functions
Edit `task_mdp.py` — the stubs are marked with docstrings explaining what the original function computed. Key ones:
- `ProgressContext`: Track peg-hole alignment via assembled offset
- `dense_success_reward`: exp(-distance_to_assembled_pose / std)
- `collision_free`: SDF-based collision penalty (needs Warp)

### Change the scene
Edit `env_cfg.py`. The scene is in `FfwSg2PegPartialAssemblySceneCfg`. Positions use the `_PEG_HOLE_XY`, `_TABLE_SURFACE_Z` constants at the top of the file.

### Run on a new machine
See `docs/DEPLOY.md`. Needs: Python 3.10, uv, NVIDIA GPU with CUDA 12+, isaacsim 4.5, isaaclab 2.0.

## Testing

```bash
pytest -q                          # offline tests (no Isaac needed)
SG2_RL_RUN_ISAAC=1 pytest -q       # full tests including Isaac
```

## Environment

- Python 3.10 (uv-managed .venv)
- Isaac Sim 4.5.0, Isaac Lab 2.0.0
- PyTorch 2.5+ with CUDA 12
- Target hardware: 2× NVIDIA RTX 6000 Ada (tai-32)
