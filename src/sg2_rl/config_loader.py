"""Simple config loader for gym-registered tasks.

Loads an env config class from the gym registry entry point string and returns it.
"""
from __future__ import annotations

import functools
import importlib
from typing import Callable, Sequence

import gymnasium as gym


def _load_class(entry_point: str):
    mod_path, cls_name = entry_point.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def load_task_cfg(task_name: str, agent_cfg_key: str = "skrl_cfg_entry_point"):
    spec = gym.spec(task_name)
    kwargs = spec.kwargs or {}
    env_cfg_ep = kwargs.get("env_cfg_entry_point", "")
    if not env_cfg_ep:
        raise ValueError(f"Task {task_name!r} has no env_cfg_entry_point")
    env_cfg_cls = _load_class(env_cfg_ep)
    env_cfg = env_cfg_cls()
    agent_cfg = kwargs.get(agent_cfg_key)
    return env_cfg, agent_cfg


def task_config(
    task_name: str,
    agent_cfg_key: str = "skrl_cfg_entry_point",
    _hydra_args: Sequence[str] | None = None,
) -> Callable:
    """Decorator that loads the task config and calls func(env_cfg, agent_cfg, ...).

    The _hydra_args parameter is accepted for call-site compatibility with the
    old hydra_task_compose(task, key, []) signature but is ignored.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            env_cfg, agent_cfg = load_task_cfg(task_name, agent_cfg_key)
            return func(env_cfg, agent_cfg, *args, **kwargs)
        return wrapper
    return decorator
