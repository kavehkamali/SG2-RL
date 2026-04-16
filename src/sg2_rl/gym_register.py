"""Register FFW SG2 peg gym task IDs."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from sg2_rl.paths import configs_dir

_ENV_CFG_MOD = "sg2_rl.env_cfg"

REGISTERED_TASKS: dict[str, tuple[str, str]] = {
    "FFWSG2-PegSmoke-v0": (f"{_ENV_CFG_MOD}:FfwSg2PegPartialAssemblySmokeEnvCfg", "skrl_agent_placeholder.yaml"),
    "FFWSG2-PegGraspLift-v0": (f"{_ENV_CFG_MOD}:FfwSg2PegPpoStage1EnvCfg", "skrl_ppo_mlp_stage1_grasp_lift.yaml"),
    "FFWSG2-PegInsert-v0": (f"{_ENV_CFG_MOD}:FfwSg2PegPpoStage2EnvCfg", "skrl_ppo_mlp_stage2_insert.yaml"),
}


def ensure_task_registered(task_id: str, skrl_yaml_override: str = "") -> None:
    try:
        gym.spec(task_id)
        return
    except Exception:
        pass
    if task_id not in REGISTERED_TASKS:
        return
    env_ep, yaml_name = REGISTERED_TASKS[task_id]
    if str(skrl_yaml_override).strip():
        ypath = Path(str(skrl_yaml_override).strip()).expanduser()
    else:
        ypath = configs_dir() / yaml_name
    if not ypath.is_file():
        raise FileNotFoundError(
            f"Missing SKRL yaml for gym registration: {ypath}. "
            "Pass skrl_yaml_override=/abs/path.yaml or add configs/skrl_agent_placeholder.yaml."
        )
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point": env_ep,
            "skrl_cfg_entry_point": str(ypath.resolve()),
        },
        disable_env_checker=True,
    )
    print(f"[sg2_rl] Registered gym task {task_id!r} (skrl yaml: {ypath})", flush=True)
