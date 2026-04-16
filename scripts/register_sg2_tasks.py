#!/usr/bin/env python3
"""Register SG2-RL gym ids before training.
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from sg2_rl.gym_register import ensure_task_registered  # noqa: E402


def main() -> None:
    ensure_task_registered("FFWSG2-PegSmoke-v0")
    ensure_task_registered("FFWSG2-PegGraspLift-v0")
    ensure_task_registered("FFWSG2-PegInsert-v0")
    print("[register_sg2_tasks] OK", flush=True)


if __name__ == "__main__":
    main()
