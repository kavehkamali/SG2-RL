from sg2_rl.peg_grasp_monitor import pin_lifted, streak_update


def test_streak_update():
    s = 0
    s = streak_update(True, s)
    s = streak_update(True, s)
    assert s == 2
    s = streak_update(False, s)
    assert s == 0


def test_pin_lifted_dz():
    assert pin_lifted(0.90, 0.86, table_z=0.82, dz_min=0.035, z_clear_above_table=0.20) is True
    assert pin_lifted(0.87, 0.86, table_z=0.82, dz_min=0.035, z_clear_above_table=0.20) is False


def test_pin_lifted_table_clear():
    assert pin_lifted(0.93, 0.86, table_z=0.82, dz_min=0.10, z_clear_above_table=0.10) is True
