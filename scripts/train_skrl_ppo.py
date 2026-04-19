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
parser.add_argument("--wandb", action="store_true", help="Log training scalars to Weights & Biases (rank 0 only).")
parser.add_argument(
    "--wandb_project",
    type=str,
    default=os.environ.get("WANDB_PROJECT", "sg2-rl"),
    help="wandb project (default: WANDB_PROJECT or sg2-rl).",
)
parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", ""), help="wandb entity/team.")
parser.add_argument("--wandb_group", type=str, default=os.environ.get("WANDB_GROUP", ""), help="wandb run group (e.g. experiment family).")
parser.add_argument("--wandb_name", type=str, default="", help="wandb run name (default: auto).")
parser.add_argument(
    "--wandb_eval_interval",
    type=int,
    default=1000,
    help="Trainer steps between periodic GIF eval subprocess runs (0 disables). Uses scripts/wandb_gif_eval.py.",
)
parser.add_argument("--wandb_eval_episodes", type=int, default=4, help="Rollouts per GIF eval (one GIF per episode).")
parser.add_argument(
    "--wandb_eval_steps",
    type=int,
    default=2000,
    help="Max policy steps per GIF rollout episode (NOT wandb scalar log spacing — that is SKRL agent.experiment.write_interval in the yaml).",
)
parser.add_argument(
    "--wandb_gif",
    action="store_true",
    help="If set with --wandb, run GIF eval on --wandb_eval_interval (set SG2RL_WANDB_EVAL_CUDA to a free GPU).",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Path to a checkpoint .pt (e.g. artifacts/wandb_eval_ckpt/milestone_XXX_trainer_YYY.pt) to resume from. Loads networks + optimizer state into the agent before training begins. Trainer step counter still starts at 0.",
)
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
from sg2_rl import wandb_utils  # noqa: E402


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


def _get_optimizer_lr(agent: Any) -> float | None:
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is None or not getattr(optimizer, "param_groups", None):
        return None
    return float(optimizer.param_groups[0]["lr"])


def _set_optimizer_lr(agent: Any, lr: float) -> None:
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _resolve_lr_schedule(cfg: dict[str, Any], *, total_timesteps: int) -> dict[str, float] | None:
    sg2rl_cfg = cfg.get("sg2rl", {}) if isinstance(cfg, dict) else {}
    lr_cfg = sg2rl_cfg.get("lr_schedule", {}) if isinstance(sg2rl_cfg, dict) else {}
    if not isinstance(lr_cfg, dict) or not lr_cfg.get("enabled", False):
        return None
    base_lr = float(lr_cfg.get("base_lr", 0.0) or 0.0)
    decay_start = int(lr_cfg.get("decay_start", 0) or 0)
    slowdown_factor = float(lr_cfg.get("slowdown_factor", 1.0) or 1.0)
    min_scale = float(lr_cfg.get("min_scale", 0.0) or 0.0)
    if base_lr <= 0.0 or total_timesteps <= 0:
        return None
    return {
        "base_lr": base_lr,
        "decay_start": max(0, decay_start),
        "slowdown_factor": max(1.0, slowdown_factor),
        "min_scale": min(max(min_scale, 0.0), 1.0),
        "total_timesteps": int(total_timesteps),
    }


def _scheduled_lr(*, absolute_step: int, schedule_cfg: dict[str, float]) -> float:
    base_lr = float(schedule_cfg["base_lr"])
    decay_start = int(schedule_cfg["decay_start"])
    total_timesteps = max(int(schedule_cfg["total_timesteps"]), 1)
    slowdown_factor = float(schedule_cfg["slowdown_factor"])
    min_scale = float(schedule_cfg["min_scale"])
    if absolute_step <= decay_start or decay_start >= total_timesteps:
        return base_lr
    decay_window = max(total_timesteps - decay_start, 1)
    progress = max(0.0, float(absolute_step - decay_start) / float(decay_window))
    scaled_progress = min(progress / slowdown_factor, 1.0)
    scale = max(min_scale, 1.0 - scaled_progress)
    return base_lr * scale


def _install_lr_controller(*, agent: Any, schedule_cfg: dict[str, float] | None, wandb_ctx: dict[str, Any] | None) -> None:
    if schedule_cfg is None:
        return
    if hasattr(agent, "scheduler"):
        agent.scheduler = None

    base_lr = float(schedule_cfg["base_lr"])
    _set_optimizer_lr(agent, base_lr)
    original_post = agent.post_interaction

    def _post(self, *, timestep: int, timesteps: int) -> None:
        step_offset = int((wandb_ctx or {}).get("_wandb_step_offset", 0) or 0)
        absolute_step = step_offset + int(timestep) + 1
        lr = _scheduled_lr(absolute_step=absolute_step, schedule_cfg=schedule_cfg)
        _set_optimizer_lr(self, lr)
        self.track_data("Learning / Learning rate", lr)
        original_post(timestep=timestep, timesteps=timesteps)

    agent.post_interaction = MethodType(_post, agent)


def _maybe_run_wandb_gif_eval(*, agent: Any, wandb_ctx: dict[str, Any], ts1: int) -> None:
    """Run GIF subprocess when trainer step ``ts1`` crosses a ``wandb_eval_interval`` band (every step, rank 0)."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    if not wandb_ctx or not wandb_ctx.get("wandb_gif"):
        return
    interval = int(wandb_ctx.get("wandb_eval_interval", 0) or 0)
    if interval <= 0:
        return
    wandb_run = wandb_ctx.get("run")
    if wandb_run is None:
        return
    step_offset = int(wandb_ctx.get("_wandb_step_offset", 0) or 0)
    absolute_step = step_offset + int(ts1)
    milestone = absolute_step // interval
    last_m = int(wandb_ctx.get("_wandb_gif_milestone", 0))
    if ts1 <= 0 or absolute_step <= 0 or milestone <= last_m:
        return
    wandb_ctx["_wandb_gif_milestone"] = milestone
    label_step = milestone * interval
    eval_cuda = os.environ.get("SG2RL_WANDB_EVAL_CUDA", "").strip()
    if not eval_cuda:
        # Training is paused here (subprocess.run below is synchronous/blocking).
        # In DDP, rank-0 blocks → rank-1 stalls at next NCCL barrier → both GPUs
        # have no active kernels.  Default to GPU 0 so eval always runs.
        # Override with SG2RL_WANDB_EVAL_CUDA=<id> to use a specific GPU.
        eval_cuda = "0"
        if not wandb_ctx.get("_wandb_gif_cuda_warned"):
            print(
                "[sg2_rl] wandb GIF eval: SG2RL_WANDB_EVAL_CUDA not set — defaulting to GPU 0 "
                "(training is paused during eval so GPU kernels are idle; VRAM is shared).",
                flush=True,
            )
            wandb_ctx["_wandb_gif_cuda_warned"] = True
    ckpt = (
        Path(wandb_ctx["repo_root"])
        / "artifacts"
        / "wandb_eval_ckpt"
        / f"milestone_{label_step}_trainer_{absolute_step}.pt"
    )
    print(
        f"[sg2_rl] wandb: starting GIF eval (milestone={milestone}, trainer_step={absolute_step}, local_step={ts1}, label_step={label_step})",
        flush=True,
    )
    if not wandb_utils.save_agent_checkpoint(agent, ckpt):
        return
    output_dir = wandb_utils.gif_eval_output_dir(
        repo_root=Path(wandb_ctx["repo_root"]),
        global_step=int(label_step),
        trainer_step=int(absolute_step),
    )
    rc, output_dir, log_path = wandb_utils.launch_gif_eval_subprocess(
        repo_root=Path(wandb_ctx["repo_root"]),
        python=sys.executable,
        task=str(wandb_ctx["task"]),
        skrl_cfg=str(wandb_ctx["skrl_cfg"]),
        checkpoint=ckpt,
        episodes=int(wandb_ctx["wandb_eval_episodes"]),
        steps=int(wandb_ctx["wandb_eval_steps"]),
        wandb_project=str(wandb_ctx["wandb_project"]),
        wandb_entity=(str(wandb_ctx["wandb_entity"]).strip() or None),
        eval_cuda=eval_cuda,
        seed_base=int(wandb_ctx.get("seed", 0)),
        global_step=int(label_step),
        trainer_step=int(absolute_step),
        output_dir=output_dir,
    )
    if rc != 0:
        print(f"[sg2_rl] wandb GIF eval subprocess exited with code {rc} (check /tmp/sg2rl_wandb_gif_eval_step*.log)", flush=True)
        return
    uploaded = wandb_utils.log_gif_directory(wandb_run, output_dir=output_dir, step=int(label_step))
    if uploaded <= 0:
        print(f"[sg2_rl] wandb: no GIFs found in {output_dir} after eval (check {log_path})", flush=True)
        return
    print(f"[sg2_rl] wandb: uploaded {uploaded} GIF(s) from {output_dir}", flush=True)


def _install_wandb_gif_post_hook(*, agent: Any, wandb_ctx: dict[str, Any] | None) -> None:
    """Fire GIF eval on schedule every env step, independent of SKRL write_interval / TensorBoard."""
    if wandb_ctx is None or not wandb_ctx.get("wandb_gif"):
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    _orig = agent.post_interaction

    def _post(self, *, timestep: int, timesteps: int) -> None:
        _orig(timestep=timestep, timesteps=timesteps)
        ts1 = int(timestep) + 1
        _maybe_run_wandb_gif_eval(agent=self, wandb_ctx=wandb_ctx, ts1=ts1)

    agent.post_interaction = MethodType(_post, agent)


def _install_console_reporter(*, runner: Any, env: Any, wandb_ctx: dict[str, Any] | None = None) -> None:
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
        if learning_rate is None:
            learning_rate = _get_optimizer_lr(self)
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

        wandb_run = wandb_ctx.get("run") if wandb_ctx else None
        if wandb_run is not None:
            step_offset = int((wandb_ctx or {}).get("_wandb_step_offset", 0) or 0)
            wandb_utils.log_metrics(wandb_run, dict(self.tracking_data), step=step_offset + int(timestep))

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

    _ag = runner.agent
    print(
        f"[sg2_rl] diag: write_interval={getattr(_ag, 'write_interval', 'n/a')} "
        f"checkpoint_interval={getattr(_ag, 'checkpoint_interval', 'n/a')} "
        f"RANK={os.environ.get('RANK', '')} LOCAL_RANK={os.environ.get('LOCAL_RANK', '')} "
        f"WORLD_SIZE={os.environ.get('WORLD_SIZE', '')} "
        f"SG2RL_WANDB_EVAL_CUDA={os.environ.get('SG2RL_WANDB_EVAL_CUDA', '')}",
        flush=True,
    )

    wandb_run = wandb_utils.init_wandb(
        enabled=bool(args_cli.wandb),
        project=str(args_cli.wandb_project),
        entity=(str(args_cli.wandb_entity).strip() or None),
        group=(str(args_cli.wandb_group).strip() or None),
        name=(str(args_cli.wandb_name).strip() or None),
        config={
            "task": args_cli.task,
            "num_envs": int(args_cli.num_envs),
            "seed": int(args_cli.seed),
            "skrl_cfg": str(args_cli.skrl_cfg),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
    )
    wandb_ctx: dict[str, Any] | None = None
    if wandb_run is not None:
        wandb_ctx = {
            "run": wandb_run,
            "wandb_gif": bool(args_cli.wandb_gif),
            "wandb_eval_interval": int(args_cli.wandb_eval_interval),
            "wandb_eval_episodes": int(args_cli.wandb_eval_episodes),
            "wandb_eval_steps": int(args_cli.wandb_eval_steps),
            "repo_root": str(_REPO_ROOT),
            "task": str(args_cli.task),
            "skrl_cfg": str(Path(args_cli.skrl_cfg).resolve()),
            "wandb_project": str(args_cli.wandb_project),
            "wandb_entity": str(args_cli.wandb_entity),
            "seed": int(args_cli.seed),
            "_wandb_gif_milestone": 0,
            "_wandb_step_offset": 0,
            "_wandb_run_step": int(getattr(wandb_run, "step", 0) or 0),
        }

    total_timesteps = int(
        (cfg.get("trainer", {}) or {}).get("timesteps", 0)
        or getattr(runner.trainer, "timesteps", 0)
        or 0
    )
    lr_schedule_cfg = _resolve_lr_schedule(cfg, total_timesteps=total_timesteps)
    _install_lr_controller(agent=runner.agent, schedule_cfg=lr_schedule_cfg, wandb_ctx=wandb_ctx)
    _install_console_reporter(runner=runner, env=env, wandb_ctx=wandb_ctx)
    _install_wandb_gif_post_hook(agent=runner.agent, wandb_ctx=wandb_ctx)

    resume_ckpt = str(args_cli.resume).strip()
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {ckpt_path}")
        print(f"[sg2_rl] Resuming (weights-only) from checkpoint: {ckpt_path}", flush=True)
        runner.agent.load(str(ckpt_path))
        # Weights-only resume: the new run starts the trainer at step 0 of a fresh
        # budget. Do NOT touch ``_wandb_step_offset`` or ``trainer.initial_timestep``
        # so the LR schedule (and wandb plots) are indexed on this new run's local
        # trainer step -- matching the YAML ``sg2rl.lr_schedule`` boundaries.

    runner.trainer.train()

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
