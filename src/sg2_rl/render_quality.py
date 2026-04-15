"""Override headless kit RTX settings for high-quality video recording.

The default isaaclab.python.headless.rendering.kit disables reflections,
ambient occlusion, indirect diffuse, and translucency for performance.
Call enable_high_quality() after AppLauncher to turn them back on.
"""
from __future__ import annotations


def enable_high_quality(resolution: tuple[int, int] = (1920, 1080)) -> None:
    """Enable RTX features for publication-quality renders."""
    import carb

    s = carb.settings.get_settings()

    # --- Re-enable features disabled by the headless kit ---
    s.set("/rtx/reflections/enabled", True)
    s.set("/rtx/indirectDiffuse/enabled", True)
    s.set("/rtx/ambientOcclusion/enabled", True)
    s.set("/rtx/translucency/enabled", True)
    s.set("/rtx/directLighting/sampledLighting/enabled", True)

    # --- Denoisers (smooth noise from ray tracing) ---
    s.set("/rtx/reflections/denoiser/enabled", True)
    s.set("/rtx/indirectDiffuse/denoiser/enabled", True)
    s.set("/rtx/ambientOcclusion/denoiserMode", 1)
    s.set("/rtx-transient/dldenoiser/enabled", True)

    # --- Dome light: full spherical (not approximate) ---
    s.set("/rtx/domeLight/upperLowerStrategy", 0)

    # --- Shadows ---
    s.set("/rtx/shadows/enabled", True)

    # --- DLSS quality mode ---
    s.set("/rtx/post/dlss/execMode", 2)  # 0=Performance, 1=Balanced, 2=Quality

    # --- Subpixel anti-aliasing ---
    s.set("/rtx/raytracing/subpixel/mode", 0)

    # --- Path tracing samples (higher = cleaner but slower) ---
    s.set("/rtx/pathtracing/maxSamplesPerLaunch", 1000000)

    # --- Resolution override for the render product ---
    s.set("/app/renderer/resolution/width", int(resolution[0]))
    s.set("/app/renderer/resolution/height", int(resolution[1]))

    print(
        f"[sg2_rl] High-quality rendering enabled: reflections, AO, indirect diffuse, "
        f"translucency, denoisers ON — resolution {resolution[0]}x{resolution[1]}",
        flush=True,
    )
