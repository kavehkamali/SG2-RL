from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Absolute path to repository root (parent of `src/`)."""
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return repo_root() / "configs"
