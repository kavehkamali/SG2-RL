"""Bind peg/hole prop colliders to shared physics material prims under /World/SG2RL_SharedMaterials.

UsdFileCfg does not support physics_material_path like CuboidCfg; we spawn global materials in the
scene and bind collision prims after sim.reset via a startup event (compatible with replicate_physics).
"""
from __future__ import annotations

import logging
import os

import torch

from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)

PEG_PHYS_PATH = "/World/SG2RL_SharedMaterials/PegPhys"
HOLE_PHYS_PATH = "/World/SG2RL_SharedMaterials/HolePhys"


def _ensure_shared_material_prims() -> None:
    """Create global physics material prims once (InteractiveScene cannot spawn RigidBodyMaterialCfg via AssetBaseCfg)."""
    import omni.usd

    import isaaclab.sim as sim_utils

    stage = omni.usd.get_context().get_stage()
    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.75, dynamic_friction=0.75, restitution=0.0,
    )
    if not stage.GetPrimAtPath(PEG_PHYS_PATH).IsValid():
        mat.func(PEG_PHYS_PATH, mat)
    if not stage.GetPrimAtPath(HOLE_PHYS_PATH).IsValid():
        mat.func(HOLE_PHYS_PATH, mat)


def _collision_prim_paths_under(stage, root_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    out: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            out.append(str(prim.GetPath()))
    return out


def bind_sg2rl_prop_shared_physics_materials(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> None:
    """Bind collision meshes under peg/hole rigid bodies to shared physics materials."""
    if os.environ.get("SG2_RL_SKIP_MATERIAL_BIND", "").strip() in ("1", "true", "yes"):
        logger.info("Skipping shared physics material bind (SG2_RL_SKIP_MATERIAL_BIND).")
        return

    import omni.usd
    from isaaclab.sim.utils import bind_physics_material

    _ensure_shared_material_prims()
    stage = omni.usd.get_context().get_stage()
    insertive = env.scene["insertive_object"]
    receptive = env.scene["receptive_object"]
    ins_paths = insertive.root_physx_view.prim_paths
    rec_paths = receptive.root_physx_view.prim_paths
    if ins_paths is None or len(ins_paths) == 0:
        logger.warning("bind_sg2rl_prop_shared_physics_materials: empty insertive prim_paths")
        return
    n = int(insertive.num_instances)
    if env_ids is None:
        idxs = list(range(n))
    else:
        if hasattr(env_ids, "view"):
            idxs = [int(x) for x in env_ids.view(-1).tolist()]
        else:
            idxs = [int(x) for x in env_ids]

    peg_bound = 0
    hole_bound = 0
    for i in idxs:
        if i < 0 or i >= n:
            continue
        ins_root = str(ins_paths[i])
        rec_root = str(rec_paths[i])
        for col in _collision_prim_paths_under(stage, ins_root):
            bind_physics_material(col, PEG_PHYS_PATH, stage=stage)
            peg_bound += 1
        for col in _collision_prim_paths_under(stage, rec_root):
            bind_physics_material(col, HOLE_PHYS_PATH, stage=stage)
            hole_bound += 1
    if peg_bound == 0 and hole_bound == 0:
        logger.warning(
            "bind_sg2rl_prop_shared_physics_materials: no CollisionAPI prims under peg/hole; "
            "shared materials may not apply. Check USD asset structure."
        )
