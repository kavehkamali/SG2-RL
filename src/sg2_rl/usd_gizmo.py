"""RGB world-frame axes (R≈+X, G≈+Y, B≈+Z) as USD BasisCurves for in-render debug."""

from __future__ import annotations

from typing import Sequence


def _axis_points(cx: float, cy: float, cz: float, L: float) -> list[tuple[float, float, float]]:
    return [
        (cx, cy, cz),
        (cx + L, cy, cz),
        (cx, cy, cz),
        (cx, cy + L, cz),
        (cx, cy, cz),
        (cx, cy, cz + L),
    ]


def _ensure_parent_xforms(stage, prim_path: str) -> None:
    from pxr import UsdGeom

    parent = prim_path.rsplit("/", 1)[0]
    if not parent or parent == "/":
        return
    cur = ""
    for seg in parent.strip("/").split("/"):
        cur += "/" + seg
        if not stage.GetPrimAtPath(cur).IsValid():
            UsdGeom.Xform.Define(stage, cur)


def ensure_rgb_axes(
    stage,
    prim_path: str,
    center_xyz: Sequence[float],
    axis_length: float = 0.12,
    line_width: float = 0.006,
) -> None:
    """Create or update RGB-colored axis segments at ``center_xyz`` (world frame)."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    _ensure_parent_xforms(stage, prim_path)

    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    L = float(axis_length)
    vec_pts = Vt.Vec3fArray([Gf.Vec3f(*p) for p in _axis_points(cx, cy, cz, L)])
    counts = Vt.IntArray([2, 2, 2])

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        axes = UsdGeom.BasisCurves.Define(stage, prim_path)
        axes.CreateTypeAttr("linear")
        axes.CreateWrapAttr("nonperiodic")
        axes.CreateCurveVertexCountsAttr(counts)
        axes.CreateWidthsAttr(Vt.FloatArray([line_width]))
        colors = Vt.Vec3fArray(
            [
                Gf.Vec3f(1.0, 0.05, 0.05),
                Gf.Vec3f(1.0, 0.05, 0.05),
                Gf.Vec3f(0.05, 1.0, 0.05),
                Gf.Vec3f(0.05, 1.0, 0.05),
                Gf.Vec3f(0.1, 0.35, 1.0),
                Gf.Vec3f(0.1, 0.35, 1.0),
            ]
        )
        pv = UsdGeom.PrimvarsAPI(axes.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, "vertex"
        )
        pv.Set(colors)
        axes.CreatePointsAttr(vec_pts)
    else:
        UsdGeom.BasisCurves(prim).GetPointsAttr().Set(vec_pts)
