"""Single-color USD polyline for planned paths (headless-safe)."""

from __future__ import annotations

from typing import Sequence


def draw_planned_path_polyline(stage, prim_path: str, points_xyz: Sequence[Sequence[float]], width: float = 0.005) -> None:
    """Create or update a linear ``BasisCurves`` polyline in world space."""
    from pxr import Gf, Sdf, UsdGeom, Vt

    if len(points_xyz) < 2:
        return

    parent = prim_path.rsplit("/", 1)[0]
    cur = ""
    for seg in parent.strip("/").split("/"):
        cur += "/" + seg
        if not stage.GetPrimAtPath(cur).IsValid():
            UsdGeom.Xform.Define(stage, cur)

    pts = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points_xyz])
    counts = Vt.IntArray([len(points_xyz)])

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        path = UsdGeom.BasisCurves.Define(stage, prim_path)
        path.CreateTypeAttr("linear")
        path.CreateWrapAttr("nonperiodic")
        path.CreateCurveVertexCountsAttr(counts)
        path.CreateWidthsAttr(Vt.FloatArray([width]))
        path.CreatePointsAttr(pts)
        pc = UsdGeom.PrimvarsAPI(path.GetPrim()).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, "constant"
        )
        pc.Set(Vt.Vec3fArray([Gf.Vec3f(0.95, 0.85, 0.15)]))  # amber / planned path
    else:
        crv = UsdGeom.BasisCurves(prim)
        crv.GetPointsAttr().Set(pts)
        crv.GetCurveVertexCountsAttr().Set(counts)
