"""Small, explicit Nova dashboard client for the teleoperation runtime."""

import math
import re
import socket
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from teleop_config import NovaConfig


class NovaProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class NovaResponse:
    error_id: int
    values: tuple[float, ...]
    raw: str
    command: str


class NovaClient:
    """Dashboard client with no implicit enable, clear, or control requests."""

    def __init__(self, config: NovaConfig):
        self.config = config
        self._socket: Optional[socket.socket] = None
        self._receive_buffer = b""

    def connect(self):
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.config.socket_timeout_s)
        try:
            sock.connect((self.config.ip, self.config.dashboard_port))
        except Exception:
            sock.close()
            raise
        self._socket = sock
        self._receive_buffer = b""

    def close(self):
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._receive_buffer = b""

    def robot_mode(self) -> int:
        response = self._request("RobotMode()")
        self._raise_for_error(response)
        values = self._parse_values(response)
        if len(values) != 1:
            raise NovaProtocolError(f"Invalid RobotMode response: {response.raw}")
        return int(values[0])

    def require_mode(self, expected: Optional[int] = None):
        expected = (
            self.config.required_start_mode
            if expected is None
            else expected
        )
        mode = self.robot_mode()
        if mode != expected:
            raise NovaProtocolError(
                f"Nova RobotMode must be {expected}, got {mode}"
            )

    def get_angle(self) -> list[float]:
        response = self._request("GetAngle()")
        self._raise_for_error(response)
        values = self._parse_values(response)
        if len(values) != 6:
            raise NovaProtocolError(f"Expected six Nova angles: {response.raw}")
        return values

    def get_pose(self, user: int = 0, tool: int = 0) -> list[float]:
        """Read the current Cartesian pose in explicit user/tool frames."""
        if type(user) is not int or not 0 <= user <= 50:
            raise ValueError("user must be an integer from 0 to 50")
        if type(tool) is not int or not 0 <= tool <= 50:
            raise ValueError("tool must be an integer from 0 to 50")
        response = self._request(f"GetPose(user={user},tool={tool})")
        self._raise_for_error(response)
        values = self._parse_values(response)
        if len(values) != 6:
            raise NovaProtocolError(f"Expected six Nova pose values: {response.raw}")
        return values

    def get_error_id(self) -> int:
        return self._request("GetErrorID()").error_id

    def check_joint_path(
        self,
        start_deg: Sequence[float],
        target_deg: Sequence[float],
        acceleration: int = 5,
        velocity: int = 5,
    ) -> int:
        """Return CheckOddMovJ's path result (zero means reachable)."""
        start = self._joint_argument(start_deg)
        target = self._joint_argument(target_deg)
        self._motion_percentage("acceleration", acceleration)
        self._motion_percentage("velocity", velocity)
        response = self._request(
            f"CheckOddMovJ(joint={start},joint={target},"
            f"a={acceleration},v={velocity},cp=0)"
        )
        self._raise_for_error(response)
        if len(response.values) != 1 or not response.values[0].is_integer():
            raise NovaProtocolError(f"Invalid CheckOddMovJ response: {response.raw}")
        return int(response.values[0])

    def mov_j(
        self,
        target_deg: Sequence[float],
        acceleration: int = 5,
        velocity: int = 5,
    ) -> NovaResponse:
        """Queue an explicit joint-space movement; never called by connect()."""
        target = self._joint_argument(target_deg)
        self._motion_percentage("acceleration", acceleration)
        self._motion_percentage("velocity", velocity)
        response = self._request(
            f"MovJ(joint={target},a={acceleration},v={velocity},cp=0)"
        )
        self._raise_for_error(response)
        if len(response.values) != 1 or not response.values[0].is_integer():
            raise NovaProtocolError(f"Invalid MovJ response: {response.raw}")
        return response

    def get_current_command_id(self) -> int:
        response = self._request("GetCurrentCommandID()")
        self._raise_for_error(response)
        if len(response.values) != 1 or not response.values[0].is_integer():
            raise NovaProtocolError(
                f"Invalid GetCurrentCommandID response: {response.raw}"
            )
        return int(response.values[0])

    @staticmethod
    def _joint_argument(joints_deg: Sequence[float]) -> str:
        if len(joints_deg) != 6:
            raise ValueError("Joint movement requires six angles")
        values = [float(value) for value in joints_deg]
        if any(not math.isfinite(value) for value in values):
            raise ValueError("Joint angles must be finite")
        return "{" + ",".join(f"{value:.6f}" for value in values) + "}"

    @staticmethod
    def _motion_percentage(name: str, value: int):
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError(f"{name} must be an integer from 1 to 100")

    def servo_j(
        self,
        joints_deg: Sequence[float],
        t: Optional[float] = None,
        aheadtime: Optional[float] = None,
        gain: Optional[float] = None,
        timing: Optional[dict] = None,
    ) -> NovaResponse:
        if len(joints_deg) != 6:
            raise ValueError("ServoJ requires six joint angles")
        if any(not math.isfinite(float(value)) for value in joints_deg):
            raise ValueError("ServoJ joint angles must be finite")

        t = 0.1 if t is None else float(t)
        aheadtime = 50.0 if aheadtime is None else float(aheadtime)
        gain = 500.0 if gain is None else float(gain)
        if not 0.004 <= t <= 3600.0:
            raise ValueError("ServoJ t is outside the Nova range")
        if not 20.0 <= aheadtime <= 100.0:
            raise ValueError("ServoJ aheadtime is outside the Nova range")
        if not 200.0 <= gain <= 1000.0:
            raise ValueError("ServoJ gain is outside the Nova range")

        command = (
            "ServoJ("
            + ",".join(f"{float(value):.6f}" for value in joints_deg)
            + f",t={t:g},aheadtime={aheadtime:g},gain={gain:g})"
        )
        return self._request(command, timing=timing)

    def stop(self) -> NovaResponse:
        if self._socket is None:
            raise RuntimeError("NovaClient is not connected")
        # Stop is the safety-path command. A busy controller can take longer
        # to acknowledge it than the short 29999 control-loop timeout.
        previous_timeout = self._socket.gettimeout()
        self._socket.settimeout(self.config.stop_timeout_s)
        try:
            return self._request("Stop()")
        finally:
            self._socket.settimeout(previous_timeout)

    @staticmethod
    def _raise_for_error(response: NovaResponse):
        if response.error_id != 0:
            raise NovaProtocolError(response.raw)

    def _request(self, command: str, timing: Optional[dict] = None) -> NovaResponse:
        if self._socket is None:
            raise RuntimeError("NovaClient is not connected")
        if not command or ";" in command or "\n" in command:
            raise ValueError("Invalid Nova command")
        if timing is not None:
            timing["t_servoj_send"] = time.monotonic_ns()
        self._socket.sendall(command.encode("ascii"))
        raw = self._receive_response()
        if timing is not None:
            timing["t_servoj_response"] = time.monotonic_ns()
        match = re.match(r"\s*(-?\d+)\s*,", raw)
        if match is None:
            raise NovaProtocolError(f"Cannot parse Nova ErrorID: {raw}")
        return NovaResponse(
            error_id=int(match.group(1)),
            values=tuple(self._parse_brace_numbers(raw)),
            raw=raw,
            command=command,
        )

    def _receive_response(self) -> str:
        while b";" not in self._receive_buffer:
            chunk = self._socket.recv(4096)
            if not chunk:
                raise RuntimeError("Nova TCP connection closed")
            self._receive_buffer += chunk

        raw, self._receive_buffer = self._receive_buffer.split(b";", 1)
        return raw.decode("ascii", errors="replace") + ";"

    @staticmethod
    def _parse_brace_numbers(response: str) -> list[float]:
        match = re.search(r"\{([^}]*)\}", response)
        if match is None:
            raise NovaProtocolError(f"Cannot parse Nova values: {response}")
        content = match.group(1).strip()
        if not content:
            return []
        try:
            return [float(value.strip()) for value in content.split(",")]
        except ValueError as exc:
            raise NovaProtocolError(
                f"Cannot parse Nova values: {response}"
            ) from exc

    @classmethod
    def _parse_values(cls, response: NovaResponse) -> list[float]:
        return list(response.values)
