"""Explicit Feetech/SCServo interface for the Nova end-effector servo.

This client is deliberately separate from the historical LeIsaac
``DobotGripper`` class. Opening the port is read-only. Torque is changed only
when the caller explicitly invokes ``enable_torque`` or ``disable_torque``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from leisaac.xtrainer_utils.third_party.feetech.scservo_sdk import (
    COMM_SUCCESS,
    PortHandler,
    sms_sts,
)
from leisaac.xtrainer_utils.third_party.feetech.scservo_sdk.sms_sts import (
    SMS_STS_PRESENT_POSITION_L,
    SMS_STS_TORQUE_ENABLE,
)


@dataclass(frozen=True)
class GripperStatus:
    model_number: int
    servo_id: int
    position: int
    torque_enabled: bool
    moving: bool
    load: int
    current: int
    voltage_raw: int
    temperature_raw: int


class GripperProtocolError(RuntimeError):
    """Raised when the Feetech device rejects or fails a request."""


class NovaGripper:
    """Read or explicitly command the Feetech servo used by Nova's gripper."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        servo_id: int = 1,
        open_position: int = 1470,
        closed_position: int = 2490,
        command_speed: int = 500,
        command_acceleration: int = 0,
        position_deadband: int = 2,
        input_deadband: float = 0.03,
        command_period_s: float = 0.04,
    ):
        if baudrate <= 0:
            raise ValueError("Gripper baudrate must be positive")
        if not 0 <= servo_id <= 253:
            raise ValueError("Feetech servo ID must be in [0, 253]")
        if open_position >= closed_position:
            raise ValueError("open_position must be below closed_position")
        if not 0 <= command_speed <= 4095:
            raise ValueError("Feetech command speed must be in [0, 4095]")
        if not 0 <= command_acceleration <= 255:
            raise ValueError("Feetech acceleration must be in [0, 255]")
        if position_deadband < 0:
            raise ValueError("position_deadband must be nonnegative")
        if not 0.0 <= input_deadband < 0.5:
            raise ValueError("input_deadband must be in [0, 0.5)")
        if command_period_s <= 0:
            raise ValueError("command_period_s must be positive")

        self.port_name = port
        self.baudrate = baudrate
        self.servo_id = servo_id
        self.open_position = open_position
        self.closed_position = closed_position
        self.command_speed = command_speed
        self.command_acceleration = command_acceleration
        self.position_deadband = position_deadband
        self.input_deadband = input_deadband
        self.command_period_s = command_period_s
        self._port: Optional[PortHandler] = None
        self._servo: Optional[sms_sts] = None
        self._last_command_position: Optional[int] = None
        self._last_command_time = 0.0

    @property
    def is_open(self) -> bool:
        return self._port is not None and self._servo is not None

    def open(self) -> GripperStatus:
        """Open and identify the servo without writing any register."""
        if self.is_open:
            return self.read_status()

        port = PortHandler(self.port_name)
        if not port.openPort():
            raise GripperProtocolError(f"Cannot open gripper port: {self.port_name}")
        if not port.setBaudRate(self.baudrate):
            port.closePort()
            raise GripperProtocolError(
                f"Cannot set gripper baudrate: {self.baudrate}"
            )

        servo = sms_sts(port)
        self._port = port
        self._servo = servo
        try:
            return self.read_status()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._port is not None:
            self._port.closePort()
        self._port = None
        self._servo = None
        self._last_command_position = None
        self._last_command_time = 0.0

    def read_position(self) -> int:
        servo = self._require_open()
        position, result, error = servo.ReadPos(self.servo_id)
        self._require_success("ReadPos", result, error)
        return int(position)

    def read_status(self) -> GripperStatus:
        servo = self._require_open()
        model, result, error = servo.ping(self.servo_id)
        self._require_success("Ping", result, error)
        torque, result, error = servo.read1ByteTxRx(
            self.servo_id, SMS_STS_TORQUE_ENABLE
        )
        self._require_success("ReadTorqueEnable", result, error)
        data, result, error = servo.readTxRx(
            self.servo_id, SMS_STS_PRESENT_POSITION_L, 15
        )
        self._require_success("ReadPresentStatus", result, error)
        if len(data) != 15:
            raise GripperProtocolError(
                f"ReadPresentStatus returned {len(data)} bytes, expected 15"
            )
        word = servo.scs_makeword
        signed = servo.scs_tohost
        position = signed(word(data[0], data[1]), 15)
        load = word(data[4], data[5])
        voltage = data[6]
        temperature = data[7]
        moving = data[10]
        current = signed(word(data[13], data[14]), 15)
        return GripperStatus(
            model_number=int(model),
            servo_id=self.servo_id,
            position=position,
            torque_enabled=bool(torque),
            moving=bool(moving),
            # Load is retained as the raw protocol register; current is signed.
            load=int(load),
            current=int(current),
            voltage_raw=int(voltage),
            temperature_raw=int(temperature),
        )

    def enable_torque(self) -> None:
        self._write_torque(True)

    def disable_torque(self) -> None:
        self._write_torque(False)

    def normalized_to_position(self, normalized: float) -> int:
        """Map the continuous trigger range [0, 1] to a servo position."""
        if not math.isfinite(normalized):
            raise ValueError("Gripper normalized command must be finite")
        normalized = max(0.0, min(1.0, normalized))
        return int(
            round(
                self.open_position
                + normalized * (self.closed_position - self.open_position)
            )
        )

    def command_normalized(
        self,
        normalized: float,
        timing: Optional[dict] = None,
    ) -> Optional[int]:
        """Command 0=open through 1=closed; returns None if rate-skipped."""
        position = self.normalized_to_position(normalized)
        now = time.monotonic()
        if self._last_command_position is not None:
            if abs(position - self._last_command_position) < self.position_deadband:
                return None
            if now - self._last_command_time < self.command_period_s:
                return None
        self.command_position(position, timing=timing)
        return position

    def command_position(self, position: int, timing: Optional[dict] = None) -> None:
        """Write one Feetech target position; torque must be enabled by caller."""
        servo = self._require_open()
        if not self.open_position <= position <= self.closed_position:
            raise ValueError(
                f"Gripper position {position} outside configured range "
                f"[{self.open_position}, {self.closed_position}]"
            )
        if timing is not None:
            timing["t_gripper_send"] = time.monotonic_ns()
        result, error = servo.WritePosEx(
            self.servo_id,
            int(position),
            self.command_speed,
            self.command_acceleration,
        )
        if timing is not None:
            timing["t_gripper_response"] = time.monotonic_ns()
        self._require_success("WritePosEx", result, error)
        self._last_command_position = int(position)
        self._last_command_time = time.monotonic()

    def _write_torque(self, enabled: bool) -> None:
        servo = self._require_open()
        result, error = servo.write1ByteTxRx(
            self.servo_id,
            SMS_STS_TORQUE_ENABLE,
            1 if enabled else 0,
        )
        self._require_success(
            "EnableTorque" if enabled else "DisableTorque", result, error
        )

    def _require_open(self) -> sms_sts:
        if self._servo is None:
            raise RuntimeError("NovaGripper is not open")
        return self._servo

    @staticmethod
    def _require_success(operation: str, result: int, error: int) -> None:
        if result != COMM_SUCCESS or error != 0:
            raise GripperProtocolError(
                f"{operation} failed: result={result}, error={error}"
            )
