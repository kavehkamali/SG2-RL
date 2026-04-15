"""Simple config loader replacing uwlab_tasks.utils.hydra.hydra_task_compose.

Loads an env config class from the gym registry entry point string and returns it.
No Hydra dependency required.
"""
from __future__ import annotations

import functools
import importlib
from typing import Callable

import gymnasium as gym


def _load_class(entry_point: str):
    """Import 'module.path:ClassName' and return the class."""
    mod_path, cls_name = entry_point.rsplit(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def load_task_cfg(task_name: str, agent_cfg_key: str = "skrl_cfg_entry_point"):
    """Load env_cfg (and optional agent_cfg path) from a registered gym task.

    Returns (env_cfg_instance, agent_cfg) where agent_cfg is a file path string
    or None.
    """
    spec = gym.spec(task_name)
    kwargs = spec.kwargs or {}
    env_cfg_ep = kwargs.get("env_cfg_entry_point", "")
    if not env_cfg_ep:
        raise ValueError(f"Task {task_name!r} has no env_cfg_entry_point")
    env_cfg_cls = _load_class(env_cfg_ep)
    env_cfg = env_cfg_cls()
    agent_cfg = kwargs.get(agent_cfg_key)
    return env_cfg, agent_cfg


def task_config(task_name: str, agent_cfg_key: str = "skrl_cfg_entry_point") -> Callable:
    """Decorator that loads the task config and calls func(env_cfg, agent_cfg, ...).

    Drop-in replacement for uwlab_tasks.utils.hydra.hydra_task_compose with
    no Hydra overhead.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            env_cfg, agent_cfg = load_task_cfg(task_name, agent_cfg_key)
            return func(env_cfg, agent_cfg, *args, **kwargs)
        return wrapper
    return decorator
