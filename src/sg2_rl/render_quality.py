"""Resolution-only override. All RTX settings left at headless kit defaults (clean rasterized look)."""
from __future__ import annotations


def enable_high_quality(resolution: tuple[int, int] = (1920, 1080)) -> None:
    import carb
    s = carb.settings.get_settings()
    s.set("/app/renderer/resolution/width", int(resolution[0]))
    s.set("/app/renderer/resolution/height", int(resolution[1]))
    print(f"[sg2_rl] Render resolution: {resolution[0]}x{resolution[1]}", flush=True)


def warm_up_renderer(sim, num_steps: int = 0) -> None:
    for _ in range(num_steps):
        sim.render()
