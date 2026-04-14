"""Distance / height checks for peg grasp–lift recording (no Isaac import required in tests)."""

from __future__ import annotations


def streak_update(condition: bool, streak: int) -> int:
    """Increment streak when ``condition`` holds, else reset to 0."""
    return streak + 1 if condition else 0


def pin_lifted(
    peg_z: float,
    peg_z0: float,
    *,
    table_z: float,
    dz_min: float = 0.035,
    z_clear_above_table: float = 0.10,
) -> bool:
    """True if the peg has risen enough vs reset or clears the table band."""
    if peg_z - peg_z0 >= dz_min:
        return True
    if peg_z - table_z >= z_clear_above_table:
        return True
    return False
