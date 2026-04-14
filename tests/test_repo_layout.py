from __future__ import annotations

from pathlib import Path


def test_repo_has_core_files():
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / "configs" / "skrl_agent_placeholder.yaml").is_file()
    assert (root / "src" / "sg2_rl" / "gym_register.py").is_file()
    assert (root / "scripts" / "record_orbit_pin_wrist_gizmos.py").is_file()
    assert (root / "scripts" / "record_path_apf_visual_only.py").is_file()
    assert (root / "scripts" / "record_path_apf_follow_gripper.py").is_file()
    assert (root / "src" / "sg2_rl" / "apf_path.py").is_file()
    assert (root / "src" / "sg2_rl" / "orbit_camera.py").is_file()
