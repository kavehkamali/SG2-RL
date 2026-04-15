# SG2-RL

Self-contained reinforcement learning workspace for the **FFW SG2** peg-in-hole task in Isaac Sim.

A bimanual humanoid robot (Fourier FFW SG2) stands at a table with a peg and a hole. The goal is to learn to grasp the peg and insert it. All assets (robot USD, props, environment) ship in-repo — no external dependencies.

## Quick Start

```bash
# On a GPU machine (e.g. tai-32)
cd ~/projects/API/SG2-RL
~/.local/bin/uv venv .venv --python 3.10
~/.local/bin/uv pip install -e '.[dev]'
~/.local/bin/uv pip install 'isaacsim[all]==4.5.0' --extra-index-url https://pypi.nvidia.com
~/.local/bin/uv pip install 'isaaclab==2.0.0' --extra-index-url https://pypi.nvidia.com

# Smoke test
./scripts/run_on_tai.sh smoke_random_motion.py --headless --steps 32
```

## Layout

| Path | Purpose |
|------|---------|
| `src/sg2_rl/` | Python package — env config, robot config, MDP functions, IK, APF planner, utilities |
| `scripts/` | Runnable entry points (Isaac Lab `AppLauncher`) |
| `configs/` | SKRL agent YAML files |
| `assets/` | Robot USD, peg/hole props, grid environment |
| `tests/` | Offline unit tests (`pytest`) |
| `docs/` | Deployment and RL backend notes |

## Key Modules

| Module | What it does |
|--------|-------------|
| `env_cfg.py` | Scene definition — robot, table, peg, hole, lighting, physics |
| `robot_cfg.py` | FFW SG2 articulation — joints, actuators, USD path |
| `right_gripper_ik.py` | Differential IK for the right wrist (arm_r_link7) |
| `apf_path.py` | Artificial Potential Field path planner with sphere obstacles |
| `arm_avoidance.py` | Arm-link collision checks against workspace spheres |
| `task_mdp.py` | Observation/reward/termination functions for the env |
| `config_loader.py` | Loads env configs from gym registry (replaces Hydra) |
| `render_quality.py` | Resolution settings for headless video recording |
| `scene_layout.py` | Peg/hole/table position adjustments |
| `peg_grasp_monitor.py` | Distance/height checks for grasp-lift recording |
| `gym_register.py` | Registers gym task IDs |

## Recording Videos

### Video B — APF path + gripper follows (IK)

```bash
./scripts/run_on_tai.sh record_path_apf_follow_gripper.py --headless --video_length 2160
```

### Grasp + lift sequence

```bash
./scripts/run_on_tai.sh record_grasp_lift_peg.py --headless
```

### Orbit camera with gizmos

```bash
./scripts/run_on_tai.sh record_orbit_pin_wrist_gizmos.py --headless --video_length 360
```

Videos are saved under `artifacts/videos/` (git-ignored).

## RL Training

PPO with MLP policy, 30k parallel envs across 2 GPUs:

```bash
./scripts/tmux_train_grasp_lift_ddp.sh
# Attach: tmux attach -t sg2rl-grasp-ppo-ddp
# Log:    /tmp/sg2rl_grasp_lift_ppo_ddp.log
```

Config: `configs/skrl_ppo_mlp_grasp_lift_96k.yaml`

## Scene

- **Robot**: FFW SG2 — 7-DOF bimanual arms, 4-finger grippers, lift joint, head joints
- **Table**: 1.0 × 0.75 m kinematic cuboid at Z = 0.82 m
- **Peg**: 20 g dynamic rigid body on the table
- **Hole**: 500 g dynamic rigid body, spawns above table and settles under gravity
- **Lighting**: Dome light + distant key light
- **Ground**: Isaac Sim grid plane with wireframe texture

## Tests

```bash
pytest -q
```

Isaac-dependent tests are skipped unless `SG2_RL_RUN_ISAAC=1`.

## Assets

| File | Size | Description |
|------|------|-------------|
| `assets/robots/FFW_SG2.usd` | 40 MB | Fourier FFW SG2 humanoid |
| `assets/props/peg.usd` | 170 KB | Insertive peg |
| `assets/props/peg_hole.usd` | 213 KB | Receptive hole fixture |
| `assets/environments/default_environment.usd` | 8 KB | Grid ground plane |

## License

BSD-3-Clause
