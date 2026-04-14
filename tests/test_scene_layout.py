"""Tests for scene layout helpers (no Isaac)."""

from __future__ import annotations

from types import SimpleNamespace

from sg2_rl.scene_layout import offset_receptive_and_viewer_for_world_shift, offset_receptive_object_world_spawn


def test_offset_receptive_only():
    init = SimpleNamespace(pos=(0.62, 0.0, 0.96))
    receptive = SimpleNamespace(init_state=init)
    scene = SimpleNamespace(receptive_object=receptive)
    env_cfg = SimpleNamespace(scene=scene)

    offset_receptive_object_world_spawn(env_cfg, -0.14, 0.02, -0.01)
    assert init.pos == (0.62 - 0.14, 0.02, 0.96 - 0.01)


def test_offset_receptive_and_viewer():
    init = SimpleNamespace(pos=(1.0, 0.0, 0.9))
    receptive = SimpleNamespace(init_state=init)
    scene = SimpleNamespace(receptive_object=receptive)
    viewer = SimpleNamespace(lookat=(1.0, 0.0, 0.82), eye=(2.5, 0.0, 1.1))
    env_cfg = SimpleNamespace(scene=scene, viewer=viewer)

    offset_receptive_and_viewer_for_world_shift(env_cfg, -0.1, 0.05, 0.0)
    assert init.pos == (0.9, 0.05, 0.9)
    assert viewer.lookat == (0.9, 0.05, 0.82)
    assert viewer.eye == (2.4, 0.05, 1.1)
