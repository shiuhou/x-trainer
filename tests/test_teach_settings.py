"""The working teleop settings are sufficient; metadata/second configs are unnecessary."""

from dataclasses import replace

import pytest

from teach_config import RepeatConfig, TeachError, load_settings
from teleop_config import (
    DEFAULT_GRIPPER_CONFIG,
    DEFAULT_NOVA_CONFIG,
    DEFAULT_TELEOP_CONFIG,
)


def working_config(**kwargs):
    return RepeatConfig.from_teleop(
        replace(DEFAULT_NOVA_CONFIG, ip="fixture"),
        replace(DEFAULT_GRIPPER_CONFIG, port="fixture"),
        DEFAULT_TELEOP_CONFIG,
        **kwargs,
    )


def test_existing_settings_need_no_model_firmware_or_json():
    config = working_config()
    assert config.data["nova"]["host"] == "fixture"
    assert config.gripper["open_position"] == DEFAULT_GRIPPER_CONFIG.open_position
    assert config.data["limits"]["max_speed_deg_s"] == list(
        DEFAULT_TELEOP_CONFIG.max_speed_deg_s
    )
    assert config.data["limits"]["joint_limits_deg"] is None
    assert "identity" not in config.profile()
    assert "model_number" not in config.gripper
    assert RepeatConfig.from_profile(config.profile()).profile() == config.profile()


def test_monitor_checks_can_be_overridden_in_existing_settings():
    config = working_config(checks={"joint_start_tolerance_deg": [0.5] * 6})
    assert config.checks["joint_start_tolerance_deg"] == [0.5] * 6
    with pytest.raises(TeachError, match="unknown"):
        working_config(checks={"typo": 1})
    with pytest.raises(TeachError, match="finite"):
        working_config(checks={"max_lateness_s": float("nan")})


def test_trusted_python_settings_load_without_hardware(tmp_path):
    path = tmp_path / "teleop_config.py"
    path.write_text(
        "from dataclasses import replace\n"
        "from teleop_config import *\n"
        "DEFAULT_NOVA_CONFIG = replace(DEFAULT_NOVA_CONFIG, ip='fixture')\n"
    )
    settings = load_settings(path)
    assert settings.DEFAULT_NOVA_CONFIG.ip == "fixture"
    with pytest.raises(TeachError, match="Cannot load"):
        load_settings(tmp_path / "missing.py")


def test_placeholder_endpoints_still_rejected():
    with pytest.raises(TeachError, match="explicitly configured"):
        RepeatConfig.from_teleop(
            DEFAULT_NOVA_CONFIG,
            DEFAULT_GRIPPER_CONFIG,
            DEFAULT_TELEOP_CONFIG,
        )
