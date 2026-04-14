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

## Replicate everything (checklist)

Do these on a machine that already matches your UWLab + Isaac layout (e.g. **tai**).

1. **Repos & env**  
   - `UWLab` at `~/projects/API/UWLab` (or set `UWLAB` / `UWLAB_PATH`).  
   - `SG2-RL` at `~/projects/API/SG2-RL` (this repo).  
   - Use the same **`env_uwlab`** interpreter as training (`…/UWLab/env_uwlab/bin/python`).

2. **Local USD mirror** (peg / hole props — **no Hugging Face required** on air-gapped hosts)  
   Use a directory whose layout matches UWLab’s cloud bundle, i.e. it contains  
   `Props/Custom/Peg/peg.usd` and `Props/Custom/PegHole/peg_hole.usd` (for example an extracted `uwlab_mirror_*.tar.gz` tree, or `~/uwlab_sync` on **tai** if you keep that mirror there).  
   ```bash
   export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-$HOME/uwlab_sync}"
   # or: export SG2_CLOUD_ASSETS_DIR=/path/to/root  # run_on_tai.sh maps this to UWLAB_CLOUD_ASSETS_DIR
   ```

3. **Smoke** (confirms task + PhysX load)  
   ```bash
   export SG2_RL="${SG2_RL:-$HOME/projects/API/SG2-RL}"
   chmod +x "$SG2_RL/scripts/run_on_tai.sh"
   "$SG2_RL/scripts/run_on_tai.sh" smoke_random_motion.py --headless --steps 32
   ```

4. **Videos (optional order)**  
   - Step 1 orbit + gizmos: `record_orbit_pin_wrist_gizmos.py`  
   - APF path A: `record_path_apf_visual_only.py`  
   - APF path B: `record_path_apf_follow_gripper.py`  
   - Grasp / lift script: `record_grasp_lift_peg.py` (writes `grasp_monitor.csv` next to the MP4)

5. **PPO training (MLP, vector obs, ~30k envs on 2 GPUs)**  
   - Registers gym id **`OmniReset-FFWSG2-PegMLPGraspLift-v0`** (same **`FfwSg2PegPartialAssemblySmokeEnvCfg`** as smoke: joints + peg/hole poses, **no** camera pixels).  
   - Scene already uses the shared tabletop **PhysX** material path from UWLab peg config (`FFWSG2_TablePhys`); no extra SG2-RL USD fork.  
   - From `SG2-RL` (adjust `UWLAB_PATH` if needed):  
     ```bash
     chmod +x scripts/tmux_train_grasp_lift_ddp.sh
     UWLAB_PATH="$HOME/projects/API/UWLab" SG2_RL="$HOME/projects/API/SG2-RL" \
       ./scripts/tmux_train_grasp_lift_ddp.sh
     ```  
   - **tmux session:** `sg2rl-grasp-ppo-ddp` → `tmux attach -t sg2rl-grasp-ppo-ddp`  
   - **Log:** `/tmp/sg2rl_grasp_lift_ppo_ddp.log`  
   - **SKRL budget:** `configs/skrl_ppo_mlp_grasp_lift_96k.yaml` sets `trainer.timesteps: 92160000000` as a **~96k PPO-update-scale** budget for `num_envs=30000` and `rollouts=32` (order-of-magnitude: updates ≈ timesteps / (`num_envs` × `rollouts`)); edit the yaml for shorter smoke runs.  
   - **Override env count:** `NUM_ENV_TOTAL=28672 ./scripts/tmux_train_grasp_lift_ddp.sh` (must be divisible by `NPROC`, default **2**).

6. **Sync from laptop** (if you develop on Mac)  
   - See `docs/DEPLOY.md` (`rsync` / `git push` + `git pull` on tai).

## On **tai** (`10.225.68.32`)

Use the same **`env_uwlab`** interpreter as UWLab training (paths match a default clone layout).  
**Props (e.g. `peg.usd`)** load from a **local** USD root (`Props/Custom/...`). `run_on_tai.sh` defaults to `~/uwlab_sync` when that tree is present, else falls back to `~/uwlab_hf_assets`. Override if your mirror lives elsewhere (e.g. after extracting `~/uwlab_mirror_*.tar.gz`):

```bash
export UWLAB_CLOUD_ASSETS_DIR="${UWLAB_CLOUD_ASSETS_DIR:-$HOME/uwlab_sync}"
export UWLAB="${UWLAB:-$HOME/projects/API/UWLab}"
export SG2_RL="${SG2_RL:-$HOME/projects/API/SG2-RL}"
chmod +x "$SG2_RL/scripts/run_on_tai.sh"
"$SG2_RL/scripts/run_on_tai.sh" smoke_random_motion.py --headless --steps 32
```

## Step 1 — Orbit camera + pin / wrist RGB gizmos + video

Orbit defaults pull the camera back a bit (`--orbit_radius` ≈ 2.05 m) and shift the look-at toward the robot in XY (`--orbit_lookat_shift_robot` ≈ 0.26 m) so the pin and arm are not stacked in frame. Set the shift to `0` to use the env viewer look-at unchanged.

```bash
"$SG2_RL/scripts/run_on_tai.sh" record_orbit_pin_wrist_gizmos.py \
  --headless \
  --video_length 360 \
  --print_every 30
```

Default `--video_folder` is `<SG2-RL>/artifacts/videos/orbit_pin_wrist_<timestamp>/` if you omit it.

Recording scripts default to **no cluster shift** (packaged hole/table pose) and place the pin with **`--peg_offset_x_from_hole=-0.10`**: same **0.10 m** spacing as stock UWLab but on the **robot side** of the hole (stock uses ``+0.10`` in **+X**). Optional **`--scene_shift_*`** still moves peg, hole, and table together. APF / IK use a larger tabletop margin via **`--wrist_clearance_m`** (default **0.18** m). Use `--no-shift-viewer-with-scene` to keep the viewer fixed when using a cluster shift.

Stdout includes **pin** and **wrist** world coordinates (env 0). The recording wrapper writes `rl-video-step-0.mp4` under `--video_folder`; the script also copies a timestamped `orbit_pin_wrist_*.mp4` next to it.

## Path planning (classic obstacle avoidance)

**Method:** [**Artificial Potential Field (APF)**](https://en.wikipedia.org/wiki/Artificial_potential_field) — the standard attractive / repulsive field formulation associated with **Oussama Khatib** (*International Journal of Robotics Research*, **1986**; often referenced from earlier ICRA work).  

In this repo the planner operates in **3D workspace** (right wrist / gripper proxy), with **arm-aware** sphere avoidance layered on:

- **Attractive:** spring-like pull toward a pre-grasp goal above the peg.
- **Repulsive:** a few **spherical “protective hulls”** (virtual obstacles) near the peg to force a visible detour without mesh CAD. Repulsion can be sampled along a **shoulder→wrist line** (frozen mid-arm link at reset) so the path is pushed away when intermediate links would intersect those spheres, not only the wrist.
- **Table constraint:** soft upward restoring force below a safe **Z** band so the polyline stays above the tabletop.

During **video B** playback, every ``arm_r_link*`` body is checked against the same spheres and the commanded wrist target is **nudged** if any link enters an influence radius (still analytic spheres, not PhysX mesh contact).

### Video A — path only (no RGB coordinate gizmos)

Static **amber polyline** of the planned path; robot still receives small random joint actions so the scene is alive.

```bash
~/projects/API/SG2-RL/scripts/run_on_tai.sh record_path_apf_visual_only.py --headless --video_length 300
```

### Video B — path + gripper follows

Same polyline plus **differential IK** so the **right gripper (`arm_r_link7`)** tracks the polyline over time.

```bash
~/projects/API/SG2-RL/scripts/run_on_tai.sh record_path_apf_follow_gripper.py --headless
```

The follow script defaults to **`--video_length 1440`** (twice the prior 720 default) so the full gripper motion fits in one clip; override with `--video_length` if you want a shorter file.

### One command — record A then B again

From the repo root on **tai** (after `UWLab` + local USD mirror are in place):

```bash
chmod +x scripts/run_apf_path_videos.sh scripts/run_on_tai.sh
./scripts/run_apf_path_videos.sh
```

Optional: `VIDEO_LENGTH_A=240 VIDEO_LENGTH_B=720 ./scripts/run_apf_path_videos.sh`. Outputs land under `artifacts/videos/` (timestamped `apf_path_*` folders plus copied `apf_path_*.mp4` files).

### Step — grasp pin + lift (torso down, right gripper)

Open-loop phases: lower **``lift_joint``**, right-wrist IK to the moving peg, close **``gripper_r_joint*``**, then raise wrist + torso. Same scene layout flags as the APF recorders.

```bash
"$SG2_RL/scripts/run_on_tai.sh" record_grasp_lift_peg.py --headless
```

Tune reach with ``--lift_low`` (smaller = body lower on the rail), ``--lift_carry`` after grasp, phase fractions ``--frac-torso`` / ``--frac-approach`` / ``--frac-settle``, and ``--gripper-close-mag`` if the fingers do not visibly close.

The script logs **wrist–pin distance** and peg **height vs reset** each ``--monitor-every`` steps, writes ``grasp_monitor.csv`` in the video folder, and advances phases when distance streaks clear thresholds. After peg **lift** is confirmed (``--lift-dz-success`` or ``--lift-z-above-table`` for ``--lifted-hold-frames``), it holds the max lift pose for the rest of the clip so the MP4 still has a fixed length.

## RL (minimal PPO baseline)

| Item | Detail |
|------|--------|
| **Task** | `OmniReset-FFWSG2-PegMLPGraspLift-v0` (registered from `sg2_rl.gym_register`; same smoke env, dedicated SKRL yaml). |
| **Policy** | MLP on vector observations only (`configs/skrl_ppo_mlp_grasp_lift_96k.yaml`). |
| **Multi-GPU** | `scripts/tmux_train_grasp_lift_ddp.sh` → `torchrun` + UWLab `train.py --distributed`, default **30000** parallel envs across **2** ranks. |

Other backends / notes: `docs/RL_BACKENDS.md`.

## Tests

```bash
pytest -q
```

Isaac-dependent checks are skipped unless `SG2_RL_RUN_ISAAC=1`.

## License

BSD-3-Clause (match UW Lab / Isaac Lab ecosystem unless you specify otherwise).
