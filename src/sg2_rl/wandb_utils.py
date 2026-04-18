"""Optional Weights & Biases helpers for training (rank-0 only).

GIF eval runs in a subprocess with its own Isaac process; set ``SG2RL_WANDB_EVAL_CUDA``
to a GPU id that is not fully saturated by training when using large ``num_envs``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

_RUN: Any | None = None


def is_rank0() -> bool:
    return int(os.environ.get("RANK", "0") or 0) == 0 and int(os.environ.get("LOCAL_RANK", "0") or 0) == 0


def wandb_available() -> bool:
    try:
        import wandb  # noqa: F401

        return True
    except Exception:
        return False


def init_wandb(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    group: str | None,
    name: str | None,
    config: dict[str, Any],
) -> Any | None:
    """Initialize wandb on rank 0. Returns run or None."""
    global _RUN
    if _RUN is not None:
        return _RUN
    if not enabled or not is_rank0():
        return None
    if not wandb_available():
        print("[sg2_rl] wandb: package not installed (pip install wandb). Skipping.", flush=True)
        return None
    import wandb

    proj = (project or "").strip() or os.environ.get("WANDB_PROJECT", "sg2-rl") or "sg2-rl"
    ent = (entity or "").strip() or None
    grp = (group or "").strip() or None
    nm = (name or "").strip() or None

    _RUN = wandb.init(
        project=proj,
        entity=ent,
        group=grp,
        name=nm,
        config=config,
    )
    return _RUN


def flatten_metrics(metrics: dict[str, Any], *, prefix: str = "train") -> dict[str, Any]:
    """Flatten SKRL ``tracking_data`` keys for wandb (scalar floats only)."""
    out: dict[str, Any] = {}
    for raw_key, value in metrics.items():
        key = raw_key.strip()
        key = re.sub(r"\s*/\s*", "/", key)
        key = re.sub(r"[^a-zA-Z0-9_/]+", "_", key).strip("_").lower()
        key = f"{prefix}/{key}"
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            try:
                import numpy as np

                value = float(np.mean(value))
            except Exception:
                continue
        elif hasattr(value, "item"):
            try:
                value = float(value.item())
            except Exception:
                continue
        elif isinstance(value, (int, float)):
            value = float(value)
        else:
            continue
        if key not in out:
            out[key] = value
    return out


def log_metrics(run: Any | None, metrics: dict[str, Any], *, step: int) -> None:
    if run is None:
        return
    flat = flatten_metrics(metrics, prefix="train")
    if flat:
        run.log(flat, step=int(step))


def save_agent_checkpoint(agent: Any, path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(agent, "save"):
            agent.save(str(path))
            return path.is_file()
    except Exception as exc:
        print(f"[sg2_rl] wandb: agent.save failed: {exc}", flush=True)
    return False


def gif_eval_output_dir(*, repo_root: Path, global_step: int, trainer_step: int) -> Path:
    """Return a deterministic output directory for GIF eval artifacts."""
    return (
        repo_root
        / "artifacts"
        / "wandb_gif_eval"
        / f"step_{int(global_step):09d}_trainer_{int(trainer_step):09d}"
    )


def collect_gif_paths(output_dir: Path) -> list[Path]:
    """Return episode GIFs in deterministic order."""
    if not output_dir.is_dir():
        return []
    return sorted(
        [path for path in output_dir.glob("episode_*.gif") if path.is_file()],
        key=lambda path: path.name,
    )


def log_gif_directory(run: Any | None, *, output_dir: Path, step: int) -> int:
    """Upload rendered GIF files to an existing wandb run from the parent process."""
    if run is None:
        return 0
    gif_paths = collect_gif_paths(output_dir)
    if not gif_paths:
        return 0
    try:
        import wandb
    except Exception as exc:
        print(f"[sg2_rl] wandb: unable to import wandb for GIF upload: {exc}", flush=True)
        return 0

    payload: dict[str, Any] = {}
    for gif_path in gif_paths:
        stem = gif_path.stem
        suffix = stem.split("_")[-1]
        try:
            episode_index = int(suffix)
        except Exception:
            episode_index = len(payload)
        # W&B only supports one level of nesting in log keys. Keep media keys flat enough
        # that the workspace media panels resolve reliably across resumed runs.
        payload[f"eval/front_cam_episode_{episode_index:02d}"] = wandb.Video(
            str(gif_path),
            caption=f"trainer_step={int(step)} episode={episode_index}",
            format="gif",
        )

    if not payload:
        return 0
    run.log(payload, step=int(step))
    return len(payload)


def launch_gif_eval_subprocess(
    *,
    repo_root: Path,
    python: str,
    task: str,
    skrl_cfg: str,
    checkpoint: Path,
    episodes: int,
    steps: int,
    wandb_project: str,
    wandb_entity: str | None,
    eval_cuda: str | None,
    seed_base: int = 0,
    global_step: int = 0,
    trainer_step: int = 0,
    output_dir: Path | None = None,
) -> tuple[int, Path, Path]:
    """Run ``scripts/wandb_gif_eval.py`` in a child process (separate Isaac app)."""
    env = os.environ.copy()
    env["WANDB_GLOBAL_STEP"] = str(int(global_step))
    if eval_cuda is not None and str(eval_cuda).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(eval_cuda).strip()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if wandb_project:
        env["WANDB_PROJECT"] = str(wandb_project)
    if wandb_entity:
        env["WANDB_ENTITY"] = str(wandb_entity)
    output_dir = output_dir or gif_eval_output_dir(repo_root=repo_root, global_step=global_step, trainer_step=trainer_step)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = repo_root / "scripts" / "wandb_gif_eval.py"
    cmd = [
        python,
        str(script),
        "--task",
        task,
        "--skrl_cfg",
        str(skrl_cfg),
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(int(episodes)),
        "--steps",
        str(int(steps)),
        "--seed_base",
        str(int(seed_base)),
        "--output_dir",
        str(output_dir),
        "--headless",
    ]
    log_path = Path(f"/tmp/sg2rl_wandb_gif_eval_step{int(global_step)}.log")
    print(
        f"[sg2_rl] wandb: launching GIF eval (CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'unset')}) "
        f"output_dir={output_dir} log={log_path}",
        flush=True,
    )
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, env=env, cwd=str(repo_root), check=False, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[sg2_rl] wandb: GIF eval finished code={r.returncode} (see {log_path})", flush=True)
        return int(r.returncode), output_dir, log_path
    except Exception as exc:
        print(f"[sg2_rl] wandb: GIF eval subprocess failed: {exc}", flush=True)
        return 1, output_dir, log_path
