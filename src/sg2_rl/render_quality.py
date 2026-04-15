"""Minimal rendering fixes for headless video recording.

The headless kit disables reflections for performance. We re-enable just
reflections so the grid floor shows the sky, without turning on expensive
path-tracing features that cause dotted / noisy frames.
"""
from __future__ import annotations


def enable_high_quality(resolution: tuple[int, int] = (1920, 1080)) -> None:
    """Enable reflections and set resolution for clean recording."""
    import carb

    s = carb.settings.get_settings()

    # Re-enable reflections (needed for grid floor to reflect the sky dome)
    s.set("/rtx/reflections/enabled", True)
    s.set("/rtx/reflections/denoiser/enabled", True)

    # Full dome light evaluation (not approximate) so HDRI sky is visible
    s.set("/rtx/domeLight/upperLowerStrategy", 0)

    # Ambient occlusion for depth
    s.set("/rtx/ambientOcclusion/enabled", True)

    # DLSS quality
    s.set("/rtx/post/dlss/execMode", 2)

    # Resolution
    s.set("/app/renderer/resolution/width", int(resolution[0]))
    s.set("/app/renderer/resolution/height", int(resolution[1]))

    print(
        f"[sg2_rl] Rendering: reflections ON, dome light full, AO ON — {resolution[0]}x{resolution[1]}",
        flush=True,
    )
