"""Rendering settings for headless video recording.

Uses standard rasterization (RaytracedLighting) — no path tracing, no noise.
"""
from __future__ import annotations


def enable_high_quality(resolution: tuple[int, int] = (1920, 1080)) -> None:
    import carb

    s = carb.settings.get_settings()

    # Force rasterization renderer (no ray/path tracing noise)
    s.set("/rtx/rendermode", "RaytracedLighting")

    # Full dome light (not approximate) so HDRI sky shows on floor
    s.set("/rtx/domeLight/upperLowerStrategy", 0)

    # Resolution
    s.set("/app/renderer/resolution/width", int(resolution[0]))
    s.set("/app/renderer/resolution/height", int(resolution[1]))

    print(
        f"[sg2_rl] Rendering: rasterization, dome light full — {resolution[0]}x{resolution[1]}",
        flush=True,
    )
