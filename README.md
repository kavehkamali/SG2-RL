# SG2-RL

Isolated reinforcement-learning workspace for the **FFW SG2** peg-in-hole style setup: **same table, pin, and hole** as the existing UW Lab `OmniReset` FFWSG2 peg scene (environment configs live in `uwlab_tasks`; this repo does not fork USD assets).

## Layout

| Path | Purpose |
|------|---------|
| `src/sg2_rl/` | Importable helpers (USD gizmos, gym registration, paths). |
| `scripts/` | Runnable entry points (Isaac Lab `AppLauncher`). |
| `configs/` | Hydra-friendly YAML stubs (e.g. SKRL agent file for registry). |
| `tests/` | Fast checks and optional Isaac smoke tests. |
| `docs/` | Deploy and RL backend notes. |

## Prerequisites (on the GPU machine, e.g. `10.225.68.32`)

- Isaac Sim + **Isaac Lab** on `PYTHONPATH`
- **UW Lab** `uwlab_tasks` (and dependencies) importable the same way you already train `OmniReset-FFWSG2-*` tasks
- Run with Isaac’s Python: e.g. `…/IsaacLab/_isaac_sim/python.sh scripts/<script>.py --headless …`

## On **tai** (`10.225.68.32`)

Use the same **`env_uwlab`** interpreter as UWLab training (paths match a default clone layout).  
**Props (e.g. `peg.usd`)** load from your HF assets mirror — same as `run_m7_train_peg_rl.sh`:

```bash
export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-$HOME/uwlab_hf_assets}"
export UWLAB="${UWLAB:-$HOME/projects/API/UWLab}"
export SG2_RL="${SG2_RL:-$HOME/projects/API/SG2-RL}"
chmod +x "$SG2_RL/scripts/run_on_tai.sh"
"$SG2_RL/scripts/run_on_tai.sh" smoke_random_motion.py --headless --steps 32
```

## Step 1 — Orbit camera + pin / wrist RGB gizmos + video

```bash
"$SG2_RL/scripts/run_on_tai.sh" record_orbit_pin_wrist_gizmos.py \
  --headless \
  --video_length 360 \
  --print_every 30
```

Default `--video_folder` is `<SG2-RL>/artifacts/videos/orbit_pin_wrist_<timestamp>/` if you omit it.

Stdout includes **pin** and **wrist** world coordinates (env 0). The recording wrapper writes `rl-video-step-0.mp4` under `--video_folder`; the script also copies a timestamped `orbit_pin_wrist_*.mp4` next to it.

## Path planning (classic obstacle avoidance)

**Method:** [**Artificial Potential Field (APF)**](https://en.wikipedia.org/wiki/Artificial_potential_field) — the standard attractive / repulsive field formulation associated with **Oussama Khatib** (*International Journal of Robotics Research*, **1986**; often referenced from earlier ICRA work).  

In this repo the planner operates in **3D workspace** (right wrist / gripper proxy):

- **Attractive:** spring-like pull toward a pre-grasp goal above the peg.
- **Repulsive:** a few **spherical “protective hulls”** (virtual obstacles) near the peg to force a visible detour without mesh CAD.
- **Table constraint:** soft upward restoring force below a safe **Z** band so the polyline stays above the tabletop.

### Video A — path only (no RGB coordinate gizmos)

Static **amber polyline** of the planned path; robot still receives small random joint actions so the scene is alive.

```bash
~/projects/API/SG2-RL/scripts/run_on_tai.sh record_path_apf_visual_only.py --headless --video_length 300
```

### Video B — path + gripper follows

Same polyline plus **differential IK** so the **right gripper (`arm_r_link7`)** tracks the polyline over time.

```bash
~/projects/API/SG2-RL/scripts/run_on_tai.sh record_path_apf_follow_gripper.py --headless --video_length 360
```

## Planned RL backends (not wired yet)

| Algorithm | Suggested stack |
|-----------|-----------------|
| **PPO** | SKRL (already used in UWLab) or Stable-Baselines3 |
| **SAC** | Stable-Baselines3 or SKRL |
| **DDP** | `torchrun` + distributed trainer (e.g. skrl / custom) |

Details: `docs/RL_BACKENDS.md`.

## Tests

```bash
pytest -q
```

Isaac-dependent checks are skipped unless `SG2_RL_RUN_ISAAC=1`.

## License

BSD-3-Clause (match UW Lab / Isaac Lab ecosystem unless you specify otherwise).
