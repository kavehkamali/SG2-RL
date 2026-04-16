#!/usr/bin/env python3
"""Train PPO with SKRL on Isaac Lab envs (supports torchrun DDP).

This is a repo-local replacement for `python -m isaaclab.train`, since the pip
distribution of Isaac Lab doesn't ship that CLI module.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from types import MethodType
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

_REPO_ROOT = Path(__file__).resolve().parents[1]

#
# Kit/Isaac Sim multi-process hardening:
# When running torchrun (one Kit instance per GPU), make per-rank user config paths to avoid
# kvdb/user.config.json locks and related crashes.
#
_LOCAL_RANK = os.environ.get("LOCAL_RANK", "0")
_PORTABLE_ROOT = f"/tmp/sg2rl_kit_portable_rank{_LOCAL_RANK}"

# Enable portable mode to isolate Kit data/cache/logs between ranks.
if "--portable" not in sys.argv:
    sys.argv.append("--portable")
if not any(arg == "--portable-root" or arg.startswith("--portable-root") for arg in sys.argv):
    sys.argv.extend(["--portable-root", _PORTABLE_ROOT])

parser = argparse.ArgumentParser(description="Train SKRL PPO on SG2-RL tasks (DDP via torchrun).")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, required=True, help="Num envs per process (torchrun rank).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skrl_cfg", type=str, required=True, help="Path to SKRL YAML config.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()

# Training should not enable cameras by default.
args_cli.enable_cameras = False
# Ensure per-rank GPU selection (otherwise all ranks default to cuda:0).
args_cli.device = f"cuda:{_LOCAL_RANK}"

# Make sure repo modules are importable
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Reduce UI spam in torchrun
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

from sg2_rl.config_loader import load_task_cfg  # noqa: E402
from sg2_rl.gym_register import ensure_task_registered  # noqa: E402


_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "white": "\033[37m",
}


def _use_color() -> bool:
    return os.environ.get("SG2RL_COLOR", "1") != "0" and os.environ.get("TERM", "dumb") != "dumb"


def _style(text: str, *styles: str) -> str:
    if not _use_color() or not styles:
        return text
    prefix = "".join(_ANSI[style] for style in styles)
    return f"{prefix}{text}{_ANSI['reset']}"


def _fmt_scalar(value: Any, *, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple)) and value:
        value = float(np.mean(value))
    if hasattr(value, "item"):
        value = value.item()
    if not isinstance(value, (int, float)):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{precision}f}"


def _mean_metric(metrics: dict[str, Any], key: str) -> float | None:
    if key not in metrics:
        return None
    value = metrics[key]
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return float(np.mean(value))
    if isinstance(value, np.generic):
        return float(value.item())
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def _progress_bar(progress: float, width: int = 28) -> str:
    progress = min(max(progress, 0.0), 1.0)
    filled = int(round(progress * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def _find_reward_manager(env: Any) -> Any | None:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reward_manager = getattr(current, "reward_manager", None)
        if reward_manager is not None:
            return reward_manager
        next_current = getattr(current, "unwrapped", None)
        if next_current is current:
            break
        current = next_current
    return None


def _step_reward_terms(env: Any) -> list[tuple[str, float]]:
    reward_manager = _find_reward_manager(env)
    if reward_manager is None:
        return []
    names = list(getattr(reward_manager, "active_terms", getattr(reward_manager, "_term_names", [])))
    step_reward = getattr(reward_manager, "_step_reward", None)
    if step_reward is None or not names:
        return []
    try:
        means = step_reward.detach().mean(dim=0).cpu().tolist()
    except Exception:
        return []
    return [(str(name), float(value)) for name, value in zip(names, means)]


def _install_console_reporter(*, runner: Any, env: Any) -> None:
    trainer = runner.trainer
    agent = runner.agent

    if hasattr(trainer.cfg, "disable_progressbar"):
        trainer.cfg.disable_progressbar = True

    if int(os.environ.get("RANK", "0")) != 0:
        return

    start_time = time.monotonic()
    world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    num_envs = int(getattr(env, "num_envs", 1))
    checkpoint_interval = int(getattr(agent, "checkpoint_interval", 0) or 0)
    original_write_tracking_data = agent.write_tracking_data

    def _console_write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        elapsed = max(time.monotonic() - start_time, 1e-6)
        steps_done = int(timestep)
        total_steps = max(1, int(timesteps))
        progress = steps_done / total_steps
        steps_per_second = steps_done / elapsed
        global_env_steps = steps_done * num_envs * world_size
        env_steps_per_second = global_env_steps / elapsed
        remaining = max(total_steps - steps_done, 0) / max(steps_per_second, 1e-6)
        metrics = dict(self.tracking_data)

        reward_total = _mean_metric(metrics, "Reward / Total reward (mean)")
        reward_total_min = _mean_metric(metrics, "Reward / Total reward (min)")
        reward_total_max = _mean_metric(metrics, "Reward / Total reward (max)")
        reward_instant = _mean_metric(metrics, "Reward / Instantaneous reward (mean)")
        reward_instant_min = _mean_metric(metrics, "Reward / Instantaneous reward (min)")
        reward_instant_max = _mean_metric(metrics, "Reward / Instantaneous reward (max)")
        value_loss = _mean_metric(metrics, "Loss / Value loss")
        learning_rate = _mean_metric(metrics, "Learning / Learning rate")
        stddev = _mean_metric(metrics, "Policy / Standard deviation")
        env_step_ms = _mean_metric(metrics, "Stats / Env stepping time (ms)")
        algo_ms = _mean_metric(metrics, "Stats / Algorithm update time (ms)")
        infer_ms = _mean_metric(metrics, "Stats / Inference time (ms)")
        episode_length = _mean_metric(metrics, "Episode / Total timesteps (mean)")

        reward_terms = [f"{name}={_fmt_scalar(value)}" for name, value in _step_reward_terms(env)]
        episode_terms = []
        for key in sorted(metrics):
            if key.startswith("Episode_Reward/"):
                episode_terms.append(f"{key.split('/', 1)[1]}={_fmt_scalar(_mean_metric(metrics, key))}")

        separator = _style("=" * 118, "dim", "blue")
        checkpoint_note = ""
        if checkpoint_interval > 0 and steps_done > 0 and steps_done % checkpoint_interval == 0:
            checkpoint_note = f"  {_style('[checkpoint]', 'bold', 'magenta')}"

        print(separator, flush=True)
        print(
            f"{_style('SG2-RL PPO Monitor', 'bold', 'cyan')}  "
            f"{_style(_progress_bar(progress), 'cyan')}  "
            f"{_style(f'{progress * 100:6.2f}%', 'bold', 'green')}  "
            f"step {_style(f'{steps_done:,}', 'bold', 'white')}/{_style(f'{total_steps:,}', 'white')}{checkpoint_note}",
            flush=True,
        )
        print(
            f"  speed     it/s={_style(_fmt_scalar(steps_per_second), 'yellow')}  "
            f"env-step/s={_style(_fmt_scalar(env_steps_per_second, precision=1), 'bold', 'yellow')}  "
            f"elapsed={_style(_format_duration(elapsed), 'green')}  "
            f"eta={_style(_format_duration(remaining), 'green')}",
            flush=True,
        )
        print(
            f"  reward    total={_style(_fmt_scalar(reward_total), 'bold', 'green')} "
            f"[{_fmt_scalar(reward_total_min)},{_fmt_scalar(reward_total_max)}]  "
            f"instant={_style(_fmt_scalar(reward_instant), 'bold', 'cyan')} "
            f"[{_fmt_scalar(reward_instant_min)},{_fmt_scalar(reward_instant_max)}]  "
            f"ep_len={_style(_fmt_scalar(episode_length), 'white')}",
            flush=True,
        )
        print(
            f"  trainer   value_loss={_style(_fmt_scalar(value_loss), 'red')}  "
            f"lr={_style(_fmt_scalar(learning_rate, precision=6), 'magenta')}  "
            f"policy_std={_style(_fmt_scalar(stddev), 'blue')}  "
            f"infer_ms={_style(_fmt_scalar(infer_ms), 'white')}  "
            f"env_ms={_style(_fmt_scalar(env_step_ms), 'white')}  "
            f"update_ms={_style(_fmt_scalar(algo_ms), 'white')}",
            flush=True,
        )

        if reward_terms:
            for index, chunk in enumerate(_chunked(reward_terms, 4)):
                label = "  reward terms" if index == 0 else "              "
                print(f"{label}  " + " | ".join(chunk), flush=True)

        if episode_terms:
            for index, chunk in enumerate(_chunked(episode_terms, 4)):
                label = "  episode    " if index == 0 else "              "
                print(f"{label}  " + " | ".join(chunk), flush=True)

        original_write_tracking_data(timestep=timestep, timesteps=timesteps)

    agent.write_tracking_data = MethodType(_console_write_tracking_data, agent)


def main() -> None:
    ensure_task_registered(args_cli.task, skrl_yaml_override=str(args_cli.skrl_cfg))
    env_cfg, _agent_cfg = load_task_cfg(args_cli.task, "skrl_cfg_entry_point")

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    # Ensure each torchrun rank uses a distinct GPU for simulation.
    env_cfg.sim.device = str(args_cli.device)

    env = gym.make(args_cli.task, cfg=env_cfg)

    from skrl.envs.wrappers.torch import wrap_env  # noqa: E402
    from skrl.utils.runner.torch import Runner  # noqa: E402

    env = wrap_env(env, wrapper="isaaclab")

    cfg = Runner.load_cfg_from_yaml(str(args_cli.skrl_cfg))
    runner = Runner(env, cfg, verbose=False)
    _install_console_reporter(runner=runner, env=env)
    runner.trainer.train()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
