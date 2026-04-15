"""Rendering settings for headless video recording.

Leave the headless kit renderer defaults untouched (clean rasterized look).
Only set resolution.
"""
from __future__ import annotations


def enable_high_quality(resolution: tuple[int, int] = (1920, 1080)) -> None:
    import carb
    s = carb.settings.get_settings()
    s.set("/app/renderer/resolution/width", int(resolution[0]))
    s.set("/app/renderer/resolution/height", int(resolution[1]))
    print(f"[sg2_rl] Rendering: {resolution[0]}x{resolution[1]}", flush=True)
