#!/usr/bin/env python3
"""Short rollouts with a fixed front-facing camera; write one animated GIF per episode.

Runs as a **separate** Isaac process so training can keep using the GPU(s) it already holds.
Set ``SG2RL_WANDB_EVAL_CUDA`` to a free GPU index when training exhausts all GPUs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(description="Policy rollouts → GIFs → wandb (fixed camera).")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--skrl_cfg", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--steps", type=int, default=400, help="Max steps per episode (policy steps).")
parser.add_argument("--seed_base", type=int, default=0)
parser.add_argument("--output_dir", type=str, required=True, help="Directory where rendered episode GIFs are written.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, str(_REPO_ROOT / "src"))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sg2_rl.config_loader import load_task_cfg  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402
from sg2_rl.render_quality import enable_high_quality, warm_up_renderer  # noqa: E402

enable_high_quality(resolution=(640, 360))


def _gif_to_file(
    frames: list[np.ndarray],
    path: Path,
    *,
    fps: int = 10,
    max_height: int = 360,
    palette_colors: int = 128,
    keep_ratio: float = 0.3,
) -> Path | None:
    """Encode ``frames`` as a GIF, shrinking to stay under W&B's size limit.

    W&B rejects GIFs above ~100 MB as "media not supported". We keep the
    playback fps unchanged and uniformly subsample frames down to
    ``keep_ratio`` of the original count (i.e. length is ``keep_ratio`` of
    the episode), then reduce spatial resolution and palette depth.
    """
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        raise RuntimeError("imageio is required for GIF export (pip install imageio)") from exc
    if not frames:
        return None

    # Keep only the first keep_ratio of frames (the opening of the episode).
    if keep_ratio is not None and 0.0 < keep_ratio < 1.0 and len(frames) > 1:
        keep_n = max(1, int(round(len(frames) * float(keep_ratio))))
        frames = frames[:keep_n]

    try:
        from PIL import Image  # imageio dependency; always available
    except Exception:
        Image = None  # type: ignore

    # Downscale each frame so height <= max_height (preserve aspect) and
    # quantize to a small palette for much smaller GIFs.
    processed: list[np.ndarray] = []
    sample = np.asarray(frames[0])
    h, w = sample.shape[:2]
    need_resize = max_height > 0 and h > max_height and Image is not None
    new_w, new_h = w, h
    if need_resize:
        new_h = int(max_height)
        new_w = max(1, int(round(w * (new_h / float(h)))))

    for f in frames:
        arr = np.asarray(f)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]
        if Image is None:
            processed.append(arr.astype(np.uint8))
            continue
        img = Image.fromarray(arr.astype(np.uint8))
        if need_resize:
            img = img.resize((new_w, new_h), Image.BILINEAR)
        if palette_colors and palette_colors > 0:
            img = img.convert("RGB").quantize(
                colors=int(palette_colors), method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG
            ).convert("RGB")
        processed.append(np.asarray(img))

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(path, processed, fps=int(fps), loop=0)
        return path
    except Exception:
        path.unlink(missing_ok=True)
        return None


def main() -> None:
    ensure_task_registered(args_cli.task, args_cli.skrl_cfg)
    env_cfg, _agent_cfg = load_task_cfg(args_cli.task, "skrl_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.seed = int(args_cli.seed_base)

    vf = Path(str(args_cli.output_dir)).expanduser()
    vf.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(vf / "run")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    from skrl.envs.wrappers.torch import wrap_env  # noqa: E402
    from skrl.utils.runner.torch import Runner  # noqa: E402

    unwrapped = env.unwrapped
    warm_up_renderer(unwrapped.sim, num_steps=12)

    eye = tuple(float(x) for x in env_cfg.viewer.eye)
    look = tuple(float(x) for x in env_cfg.viewer.lookat)
    unwrapped.sim.set_camera_view(eye=eye, target=look)

    skrl_env = wrap_env(env, wrapper="isaaclab")
    cfg = Runner.load_cfg_from_yaml(str(args_cli.skrl_cfg))
    runner = Runner(skrl_env, cfg, verbose=False)
    agent = runner.agent
    ckpt = Path(str(args_cli.checkpoint)).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    agent.load(str(ckpt))
    agent.enable_training_mode(False)

    episodes = max(1, int(args_cli.episodes))
    steps_ep = max(1, int(args_cli.steps))

    for ep in range(episodes):
        # The SKRL IsaacLab wrapper only performs a real reset once unless this flag is restored.
        # Re-enable it here so each requested episode produces an independent rollout/GIF.
        if hasattr(skrl_env, "_reset_once"):
            skrl_env._reset_once = True
        if hasattr(skrl_env, "_seed"):
            skrl_env._seed = int(args_cli.seed_base) + ep
        obs, _ = skrl_env.reset()
        unwrapped.sim.set_camera_view(eye=eye, target=look)
        frames: list[np.ndarray] = []
        for t in range(steps_ep):
            unwrapped.sim.set_camera_view(eye=eye, target=look)
            with torch.no_grad():
                actions, _ = agent.act(obs, None, timestep=t, timesteps=steps_ep)
            obs, _rew, term, trunc, _info = skrl_env.step(actions)
            unwrapped.sim.render()
            try:
                rgb = env.render()
            except Exception:
                rgb = None
            if rgb is not None:
                arr = np.asarray(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                frames.append(arr)
            if bool(torch.any(term)) or bool(torch.any(trunc)):
                break

        gif_path = _gif_to_file(frames, vf / f"episode_{ep:02d}.gif", fps=10)
        if gif_path is not None:
            print(f"[wandb_gif_eval] wrote {gif_path}", flush=True)

    skrl_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
