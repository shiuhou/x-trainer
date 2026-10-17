"""Reuse existing teleop settings; validate episodes without hardware imports."""

from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


class TeachError(ValueError):
    """Invalid configuration, recording, or execution condition."""


def number(value: Any, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TeachError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise TeachError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return float(value)


def integer(value: Any, name: str, low: int = 0, high: int | None = None) -> int:
    if type(value) is not int or value < low or (high is not None and value > high):
        raise TeachError(f"{name} is outside its integer range")
    return int(value)


def vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 6:
        raise TeachError(f"{name} requires six finite numbers")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
        raise TeachError(f"{name} requires six finite numbers")
    return [float(v) for v in value]


def object_keys(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise TeachError(f"{name} requires exactly: {', '.join(sorted(keys))}")
    return dict(value)


def load_settings(path: Path | None = None) -> ModuleType:
    """Load a trusted local teleop_config.py, never import a hardware driver."""
    if path is None:
        local = Path(__file__).with_name("teleop_config.local.py")
        if not local.exists():
            import teleop_config

            return teleop_config
        path = local
    spec = importlib.util.spec_from_file_location("_teach_local_settings", path)
    if spec is None or spec.loader is None:
        raise TeachError(f"Cannot load settings: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their defining module.
    try:
        spec.loader.exec_module(module)
    except (OSError, ValueError) as exc:
        raise TeachError(f"Cannot load settings {path}: {exc}") from exc
    return module


GRIPPER_KEYS = {
    "port",
    "baudrate",
    "servo_id",
    "open_position",
    "closed_position",
    "command_speed",
    "command_acceleration",
    "serial_timeout_s",
}
CHECK_KEYS = {
    "joint_start_tolerance_deg",
    "joint_tracking_tolerance_deg",
    "gripper_start_tolerance",
    "gripper_tracking_tolerance",
    "feedback_timeout_s",
    "gripper_feedback_timeout_s",
    "stationary_window_s",
    "stationary_joint_tolerance_deg",
    "stationary_gripper_tolerance",
    "monitor_period_s",
    "max_lateness_s",
    "completion_timeout_s",
}


@dataclass(frozen=True)
class RepeatConfig:
    data: dict[str, Any]

    @classmethod
    def from_teleop(
        cls,
        nova: Any,
        gripper: Any,
        teleop: Any,
        checks: dict[str, Any] | None = None,
    ) -> RepeatConfig:
        """Reuse connection, raw gripper commands and limits from working teleop.

        Checks are software abort thresholds, not a robot safety envelope.
        Override individual values with REPEAT_CHECKS in the same settings file.
        """
        thresholds = {
            "joint_start_tolerance_deg": [2.0] * 6,
            "joint_tracking_tolerance_deg": [10.0] * 6,
            "gripper_start_tolerance": 30,
            # A grasp can stop short of the commanded endpoint. Compare actual
            # demonstrated start/end positions separately from command tracking.
            "gripper_tracking_tolerance": abs(
                gripper.closed_position - gripper.open_position
            ),
            "feedback_timeout_s": teleop.stale_timeout_s,
            "gripper_feedback_timeout_s": 0.3,
            "stationary_window_s": 0.3,
            "stationary_joint_tolerance_deg": 0.2,
            "stationary_gripper_tolerance": 5,
            "monitor_period_s": 0.01,
            "max_lateness_s": 0.01,
            "completion_timeout_s": 5.0,
        }
        if checks is not None:
            if not isinstance(checks, dict) or set(checks) - CHECK_KEYS:
                raise TeachError("REPEAT_CHECKS contains unknown check names")
            thresholds.update(checks)
        g = asdict(gripper)
        return cls.parse(
            {
                "nova": {
                    "host": nova.ip,
                    "dashboard_port": nova.dashboard_port,
                    "feedback_port": 30004,
                    "socket_timeout_s": nova.socket_timeout_s,
                    "stop_timeout_s": nova.stop_timeout_s,
                },
                "gripper": {
                    **{k: v for k, v in g.items() if k in GRIPPER_KEYS},
                    "serial_timeout_s": g.get(
                        "serial_timeout_s", nova.socket_timeout_s
                    ),
                },
                "limits": {
                    "joint_limits_deg": (
                        [list(pair) for pair in teleop.joint_limits_deg]
                        if teleop.joint_limits_deg is not None
                        else None
                    ),
                    "max_speed_deg_s": list(teleop.max_speed_deg_s),
                },
                "checks": thresholds,
            }
        )

    @classmethod
    def parse(cls, raw: Any) -> RepeatConfig:
        data = object_keys(
            raw, "configuration", {"nova", "gripper", "limits", "checks"}
        )
        nova = object_keys(
            data["nova"],
            "nova",
            {
                "host",
                "dashboard_port",
                "feedback_port",
                "socket_timeout_s",
                "stop_timeout_s",
            },
        )
        if (
            not isinstance(nova["host"], str)
            or not nova["host"].strip()
            or "SET_" in nova["host"]
        ):
            raise TeachError("nova.host must be explicitly configured")
        for key in ("dashboard_port", "feedback_port"):
            integer(nova[key], key, 1, 65535)
        for key in ("socket_timeout_s", "stop_timeout_s"):
            number(nova[key], key, positive=True)
        gripper = object_keys(data["gripper"], "gripper", GRIPPER_KEYS)
        if (
            not isinstance(gripper["port"], str)
            or not gripper["port"].strip()
            or "SET_" in gripper["port"]
        ):
            raise TeachError("gripper.port must be explicitly configured")
        integer(gripper["baudrate"], "baudrate", 1)
        integer(gripper["servo_id"], "servo_id", 0, 253)
        for key in ("open_position", "closed_position"):
            integer(gripper[key], key, 0, 4095)
        if gripper["open_position"] == gripper["closed_position"]:
            raise TeachError("gripper endpoints must differ")
        integer(gripper["command_speed"], "command_speed", 0, 4095)
        integer(gripper["command_acceleration"], "command_acceleration", 0, 255)
        number(gripper["serial_timeout_s"], "serial_timeout_s", positive=True)
        limits = object_keys(
            data["limits"], "limits", {"joint_limits_deg", "max_speed_deg_s"}
        )
        pairs = limits["joint_limits_deg"]
        if pairs is not None and (not isinstance(pairs, list) or len(pairs) != 6):
            raise TeachError("joint_limits_deg requires six lower/upper pairs")
        for pair in pairs or []:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in pair
                )
                or pair[0] >= pair[1]
            ):
                raise TeachError("invalid joint limit pair")
        for v in vector(limits["max_speed_deg_s"], "max_speed_deg_s"):
            number(v, "max_speed_deg_s", positive=True)
        checks = object_keys(data["checks"], "checks", CHECK_KEYS)
        for key in ("joint_start_tolerance_deg", "joint_tracking_tolerance_deg"):
            for v in vector(checks[key], key):
                number(v, key, positive=True)
        for key in CHECK_KEYS - {
            "joint_start_tolerance_deg",
            "joint_tracking_tolerance_deg",
        }:
            number(checks[key], key, positive=True)
        if checks["monitor_period_s"] > min(
            checks["feedback_timeout_s"], checks["gripper_feedback_timeout_s"]
        ):
            raise TeachError("monitor period exceeds feedback freshness limit")
        if checks["completion_timeout_s"] <= checks["stationary_window_s"]:
            raise TeachError("completion timeout must exceed stationary window")
        return cls(data)

    @property
    def checks(self) -> dict[str, Any]:
        return dict(self.data["checks"])

    @property
    def gripper(self) -> dict[str, Any]:
        return dict(self.data["gripper"])

    def profile(self) -> dict[str, Any]:
        """Store raw command settings, never local endpoints or credentials."""
        return {
            "gripper": {
                k: v
                for k, v in self.gripper.items()
                if k not in {"port", "serial_timeout_s"}
            },
            "limits": self.data["limits"],
            "checks": self.checks,
        }

    @classmethod
    def from_profile(cls, profile: Any) -> RepeatConfig:
        p = object_keys(profile, "profile", {"gripper", "limits", "checks"})
        g = object_keys(
            p["gripper"], "profile.gripper", GRIPPER_KEYS - {"port", "serial_timeout_s"}
        )
        return cls.parse(
            {
                **p,
                "gripper": {**g, "port": "offline", "serial_timeout_s": 1},
                "nova": {
                    "host": "offline",
                    "dashboard_port": 29999,
                    "feedback_port": 30004,
                    "socket_timeout_s": 1,
                    "stop_timeout_s": 1,
                },
            }
        )

    def compatible(self, profile: dict[str, Any]) -> None:
        recorded = self.from_profile(profile).profile()
        if self.profile()["gripper"] != recorded["gripper"]:
            raise TeachError("recorded gripper does not match current configuration")

    def joints_in_range(self, joints: list[float]) -> None:
        vector(joints, "joints")
        limits = self.data["limits"]["joint_limits_deg"]
        if limits is None:
            return  # Preserve teleop's unspecified limits; never invent model limits.
        for value, (low, high) in zip(joints, limits):
            if not low <= value <= high:
                raise TeachError("joint target outside configured limits")
