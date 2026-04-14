"""Tweaks to Hydra-resolved env configs before ``gym.make`` (recording / smoke scripts)."""

from __future__ import annotations


def _pos_tuple(pos) -> tuple[float, float, float]:
    return (float(pos[0]), float(pos[1]), float(pos[2]))


def offset_receptive_object_world_spawn(
    env_cfg,
    dx: float,
    dy: float,
    dz: float = 0.0,
) -> tuple[float, float, float]:
    """Add ``(dx,dy,dz)`` to ``scene.receptive_object`` initial world position (peg-hole only).

    The peg (``insertive_object``) and table stay fixed so the pin is no longer against the hole.
    """
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
