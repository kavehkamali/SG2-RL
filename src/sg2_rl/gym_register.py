"""Register OmniReset FFWSG2 peg gym ids when `omnireset_sg2_config_init` was not imported."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from sg2_rl.paths import configs_dir

_FFW_PEG_ENV_MOD = (
    "uwlab_tasks.manager_based.manipulation.omnireset_sg2.config.ffw_sg2_peg_partial_smoke_env_cfg"
)

# Task id -> (env_cfg_entry_point, skrl yaml filename under configs/)
REGISTERED_TASKS: dict[str, tuple[str, str]] = {
    "OmniReset-FFWSG2-PegPartialAssemblySmoke-v0": (
        f"{_FFW_PEG_ENV_MOD}:FfwSg2PegPartialAssemblySmokeEnvCfg",
        "skrl_agent_placeholder.yaml",
    ),
    "OmniReset-FFWSG2-PegApproachOnly-A-v0": (
        f"{_FFW_PEG_ENV_MOD}:FfwSg2PegPartialAssemblyApproachOnlyAEnvCfg",
        "skrl_agent_placeholder.yaml",
    ),
    "OmniReset-FFWSG2-PegApproachOnly-B-v0": (
        f"{_FFW_PEG_ENV_MOD}:FfwSg2PegPartialAssemblyApproachOnlyBEnvCfg",
        "skrl_agent_placeholder.yaml",
    ),
    # Same smoke peg scene + vector obs (joints, peg/hole in robot frame); dedicated SKRL PPO yaml (MLP, long run).
    "OmniReset-FFWSG2-PegMLPGraspLift-v0": (
        f"{_FFW_PEG_ENV_MOD}:FfwSg2PegPartialAssemblySmokeEnvCfg",
        "skrl_ppo_mlp_grasp_lift_96k.yaml",
    ),
}


def ensure_task_registered(task_id: str, skrl_yaml_override: str = "") -> None:
    """Idempotent gym.register for known FFWSG2 peg tasks using absolute SKRL yaml path."""
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
