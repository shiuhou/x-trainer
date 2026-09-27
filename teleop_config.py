"""Configuration for the lightweight Nova teleoperation runtime."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


JointLimits = Optional[Tuple[Tuple[float, float], ...]]


@dataclass(frozen=True)
class LeaderConfig:
    port: str
    baudrate: int
    joint_ids: Tuple[int, ...]
    append_id: int
    trigger_id: int
    position_address: int
    position_length: int
    torque_address: int
    offsets_rad: Tuple[float, ...]
    signs: Tuple[int, ...]
    # Measured HAND_RIGHT trigger endpoints from the 2026-09-26 trace.
    # The trigger position decreases when pressed/closed.
    trigger_open_deg: float = 205.3
    trigger_closed_deg: float = 169.5
    # The measured read transaction is about 5.8 ms. 150 Hz keeps a fresh
    # sample available for the 33 Hz control loop without busy-spinning.
    sample_hz: float = 150.0

    def __post_init__(self):
        if len(self.joint_ids) != 6:
            raise ValueError("Leader must contain six joint IDs")
        if len(self.offsets_rad) != len(self.joint_ids):
            raise ValueError("Leader offsets must match joint IDs")
        if len(self.signs) != len(self.joint_ids):
            raise ValueError("Leader signs must match joint IDs")
        if any(sign not in (-1, 1) for sign in self.signs):
            raise ValueError("Leader signs must be -1 or 1")
        if self.baudrate <= 0 or self.sample_hz <= 0:
            raise ValueError("Leader baudrate and sample rate must be positive")
        if self.trigger_open_deg == self.trigger_closed_deg:
            raise ValueError("Leader trigger endpoints must differ")

    @property
    def torque_ids(self) -> Tuple[int, ...]:
        return self.joint_ids + (self.append_id, self.trigger_id)

    @property
    def read_ids(self) -> Tuple[int, ...]:
        return self.joint_ids + (self.trigger_id,)


@dataclass(frozen=True)
class NovaConfig:
    ip: str = "SET_NOVA_IP"
    dashboard_port: int = 29999
    socket_timeout_s: float = 0.20
    stop_timeout_s: float = 2.0
    required_start_mode: int = 5


@dataclass(frozen=True)
class GripperConfig:
    port: str = "/dev/serial/by-id/SET_GRIPPER_RIGHT_SERIAL"
    baudrate: int = 115200
    servo_id: int = 1
    open_position: int = 1470
    closed_position: int = 2490
    command_speed: int = 500
    command_acceleration: int = 0
    position_deadband: int = 2
    input_deadband: float = 0.03
    command_period_s: float = 0.04

    def __post_init__(self):
        if self.baudrate <= 0:
            raise ValueError("Gripper baudrate must be positive")
        if not 0 <= self.servo_id <= 253:
            raise ValueError("Gripper servo ID must be in [0, 253]")
        if self.open_position >= self.closed_position:
            raise ValueError("Gripper open position must be below closed position")
        if not 0 <= self.command_speed <= 4095:
            raise ValueError("Gripper command speed must be in [0, 4095]")
        if not 0 <= self.command_acceleration <= 255:
            raise ValueError("Gripper acceleration must be in [0, 255]")
        if (
            self.position_deadband < 0
            or not 0.0 <= self.input_deadband < 0.5
            or self.command_period_s <= 0
        ):
            raise ValueError("Invalid gripper command filtering")


@dataclass(frozen=True)
class GripperDiagnosticConfig:
    """Candidate-only gripper state thresholds; never used for control."""

    endpoint_tolerance: int = 20
    tracking_error_threshold: int = 30
    # Empty-close baseline peaked at 2; the object-hold trace remained at
    # roughly 9-12. Keep margin for that measured separation while retaining
    # the persistence requirement below.
    candidate_current_threshold: int = 8
    candidate_persistence_s: float = 0.25
    target_change_threshold: int = 5

    def __post_init__(self):
        if self.endpoint_tolerance < 0 or self.tracking_error_threshold < 0:
            raise ValueError("Gripper diagnostic position thresholds must be nonnegative")
        if self.candidate_current_threshold < 0:
            raise ValueError("Gripper diagnostic current threshold must be nonnegative")
        if self.candidate_persistence_s <= 0 or self.target_change_threshold < 0:
            raise ValueError("Invalid gripper diagnostic timing or target threshold")


@dataclass(frozen=True)
class TeleopConfig:
    dt_s: float = 0.03
    max_offset_deg: Tuple[float, ...] = (20.0, 20.0, 20.0, 20.0, 20.0, 20.0)
    # Baseline traces showed a stable ~33 Hz loop and 5.8 ms ServoJ RTT.
    # The previous 15 deg/s cap caused visible intentional lag, so the
    # shoulder/elbow and wrist caps are now split for a more responsive feel.
    max_speed_deg_s: Tuple[float, ...] = (30.0, 30.0, 30.0, 40.0, 40.0, 40.0)
    map_sign: Tuple[int, ...] = (1, -1, -1, 1, -1, 1)
    scale: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    deadband_deg: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    joint_limits_deg: JointLimits = None
    stale_timeout_s: float = 0.20
    servo_t: float = 0.1
    servo_aheadtime: float = 50.0
    servo_gain: float = 500.0
    # Upstream X-Trainer key packet is (A, B, sensor). None disables clutch
    # until a physical button has been verified by the operator.
    clutch_key: Optional[str] = None

    def __post_init__(self):
        lengths = {
            len(self.max_offset_deg),
            len(self.max_speed_deg_s),
            len(self.map_sign),
            len(self.scale),
            len(self.deadband_deg),
        }
        if lengths != {6}:
            raise ValueError("Teleop joint settings must contain six values")
        if any(sign not in (-1, 1) for sign in self.map_sign):
            raise ValueError("Mapping signs must be -1 or 1")
        if any(value < 0 for value in self.max_offset_deg):
            raise ValueError("Maximum offsets must be nonnegative")
        if any(value <= 0 for value in self.max_speed_deg_s):
            raise ValueError("Maximum speeds must be positive")
        if any(value < 0 for value in self.deadband_deg):
            raise ValueError("Deadbands must be nonnegative")
        if self.dt_s <= 0 or self.stale_timeout_s <= 0:
            raise ValueError("Control period and stale timeout must be positive")
        if not 0.004 <= self.servo_t <= 3600.0:
            raise ValueError("ServoJ t is outside the Nova range")
        if not 20.0 <= self.servo_aheadtime <= 100.0:
            raise ValueError("ServoJ aheadtime is outside the Nova range")
        if not 200.0 <= self.servo_gain <= 1000.0:
            raise ValueError("ServoJ gain is outside the Nova range")
        if self.clutch_key not in (None, "a", "b"):
            raise ValueError("clutch_key must be None, 'a', or 'b'")
        if self.joint_limits_deg is not None:
            if len(self.joint_limits_deg) != 6:
                raise ValueError("Joint limits must contain six pairs")
            for lower, upper in self.joint_limits_deg:
                if lower > upper:
                    raise ValueError("Joint limit lower bound exceeds upper bound")


DEFAULT_LEADER_CONFIG = LeaderConfig(
    port=(
        "/dev/serial/by-id/"
        "SET_LEADER_RIGHT_SERIAL"
    ),
    baudrate=2_000_000,
    joint_ids=(11, 12, 14, 15, 16, 17),
    append_id=13,
    trigger_id=18,
    position_address=140,
    position_length=4,
    torque_address=64,
    offsets_rad=(0.8, 2.36, 1.6, 3.88, 5.48, 1.46),
    signs=(1, 1, -1, 1, 1, 1),
)

DEFAULT_NOVA_CONFIG = NovaConfig()
DEFAULT_GRIPPER_CONFIG = GripperConfig()
DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG = GripperDiagnosticConfig()
DEFAULT_TELEOP_CONFIG = TeleopConfig()


def validate_vector(name: str, values: Sequence[float], length: int = 6):
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{name} must contain finite values")
