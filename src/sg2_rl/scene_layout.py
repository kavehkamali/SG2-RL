"""Tweaks to Hydra-resolved env configs before ``gym.make`` (recording / smoke scripts)."""

from __future__ import annotations

# Scene assets that share the tabletop cluster (FFW SG2 peg smoke).
_CLUSTER_ASSET_NAMES: tuple[str, ...] = ("receptive_object", "insertive_object", "work_surface")


def _pos_tuple(pos) -> tuple[float, float, float]:
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def offset_receptive_object_world_spawn(
    env_cfg,
    dx: float,
    dy: float,
    dz: float = 0.0,
) -> tuple[float, float, float]:
    """Add ``(dx,dy,dz)`` to ``scene.receptive_object`` initial world position (peg-hole only)."""
    init = env_cfg.scene.receptive_object.init_state
    x, y, z = _pos_tuple(init.pos)
    init.pos = (x + dx, y + dy, z + dz)
    return _pos_tuple(init.pos)


def offset_viewer_eye_and_lookat(env_cfg, dx: float, dy: float, dz: float = 0.0) -> None:
    """Shift default ``viewer.eye`` and ``viewer.lookat`` by the same delta (parallel camera move)."""
    v = env_cfg.viewer
    lx, ly, lz = _pos_tuple(v.lookat)
    ex, ey, ez = _pos_tuple(v.eye)
    v.lookat = (lx + dx, ly + dy, lz + dz)
    v.eye = (ex + dx, ey + dy, ez + dz)


def offset_receptive_and_viewer_for_world_shift(
    env_cfg,
    dx: float,
    dy: float,
    dz: float = 0.0,
) -> None:
    """Move the hole spawn and keep the packaged default viewer framing consistent."""
    offset_receptive_object_world_spawn(env_cfg, dx, dy, dz)
    offset_viewer_eye_and_lookat(env_cfg, dx, dy, dz)


def apply_peg_hole_workspace_shift(
    env_cfg,
    cluster_dx: float,
    cluster_dy: float,
    cluster_dz: float,
    *,
    peg_offset_x_from_hole: float = -0.10,
    peg_offset_y_from_hole: float = 0.0,
    shift_viewer: bool = True,
) -> None:
    """Shift peg, hole, and table together, then place the pin relative to the hole.

    - **Cluster shift** moves ``receptive_object``, ``insertive_object``, and ``work_surface``
      together so the pin stays on the table.
    - **peg_offset_x_from_hole** / **peg_offset_y_from_hole**: after the cluster shift, the
      insertive root XY is set from the receptive root XY plus these offsets. Peg **Z** is
      kept from the post-cluster insertive pose (table height).

    Stock defaults put the pin at ``+0.10`` m in world **X** from the hole (pin away from the
    robot). Use ``peg_offset_x_from_hole=-0.10`` for the same spacing with the pin **toward**
    the robot (−X in that layout). Magnitude ``0.10`` matches the stock lateral offset.
    """
    scene = env_cfg.scene
    for name in _CLUSTER_ASSET_NAMES:
        asset = getattr(scene, name, None)
        if asset is None:
            continue
        init = asset.init_state
        x, y, z = _pos_tuple(init.pos)
        init.pos = (x + cluster_dx, y + cluster_dy, z + cluster_dz)

    rec = scene.receptive_object.init_state
    ins = scene.insertive_object.init_state
    hx, hy, _ = _pos_tuple(rec.pos)
    _, _, pz = _pos_tuple(ins.pos)
    ins.pos = (hx + peg_offset_x_from_hole, hy + peg_offset_y_from_hole, pz)

    if shift_viewer:
        offset_viewer_eye_and_lookat(env_cfg, cluster_dx, cluster_dy, cluster_dz)
