"""Dobot Nova 30004 realtime feedback reader.

The Nova controller sends fixed 1440-byte little-endian frames about every
8 ms. Only fields needed by teleoperation diagnostics are decoded here.
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional


FEEDBACK_PORT = 30004
FRAME_SIZE = 1440
ROBOT_MODE_OFFSET = 24
CONTROLLER_TIMESTAMP_OFFSET = 32
Q_TARGET_OFFSET = 192
Q_ACTUAL_OFFSET = 432
JOINT_COUNT = 6
_U64 = struct.Struct("<Q")
_JOINTS = struct.Struct("<6d")


@dataclass(frozen=True)
class NovaFeedbackSample:
    robot_mode: int
    controller_timestamp_ms: int
    q_target_deg: tuple[float, ...]
    q_actual_deg: tuple[float, ...]
    host_timestamp_ns: int

    @property
    def tracking_error_deg(self) -> tuple[float, ...]:
        return tuple(
            target - actual
            for target, actual in zip(self.q_target_deg, self.q_actual_deg)
        )


def decode_feedback(frame: bytes) -> NovaFeedbackSample:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"Expected {FRAME_SIZE} feedback bytes, got {len(frame)}")
    robot_mode = _U64.unpack_from(frame, ROBOT_MODE_OFFSET)[0]
    controller_timestamp_ms = _U64.unpack_from(
        frame,
        CONTROLLER_TIMESTAMP_OFFSET,
    )[0]
    # Dobot's 30004 QTarget/QActual fields are joint angles in degrees.  The
    # TCP ServoJ/Dashboard API uses the same unit; do not apply a radian
    # conversion here.
    q_target_deg = _JOINTS.unpack_from(frame, Q_TARGET_OFFSET)
    q_actual_deg = _JOINTS.unpack_from(frame, Q_ACTUAL_OFFSET)
    return NovaFeedbackSample(
        robot_mode=robot_mode,
        controller_timestamp_ms=controller_timestamp_ms,
        q_target_deg=tuple(q_target_deg),
        q_actual_deg=tuple(q_actual_deg),
        host_timestamp_ns=time.monotonic_ns(),
    )


class NovaFeedbackReader:
    def __init__(self, ip: str, port: int = FEEDBACK_PORT, timeout_s: float = 0.20):
        self.ip = ip
        self.port = port
        self.timeout_s = timeout_s
        self._socket: Optional[socket.socket] = None
        self._latest: Optional[NovaFeedbackSample] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None

    def connect(self):
        if self._socket is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        try:
            sock.connect((self.ip, self.port))
        except Exception:
            sock.close()
            raise
        self._socket = sock

    def start(self):
        if self._socket is None:
            raise RuntimeError("NovaFeedbackReader is not connected")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="nova-30004-feedback",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = self._socket.recv(FRAME_SIZE - len(buffer))
                if not chunk:
                    raise RuntimeError("Nova 30004 connection closed")
                buffer += chunk
                if len(buffer) < FRAME_SIZE:
                    continue
                frame, buffer = buffer[:FRAME_SIZE], buffer[FRAME_SIZE:]
                sample = decode_feedback(frame)
                with self._lock:
                    self._latest = sample
                    self._last_error = None
            except socket.timeout:
                continue
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                if self._stop.wait(0.02):
                    break

    def latest(self, max_age_s: Optional[float] = None) -> Optional[NovaFeedbackSample]:
        with self._lock:
            sample = self._latest
        if sample is None or max_age_s is None:
            return sample
        age_s = (time.monotonic_ns() - sample.host_timestamp_ns) / 1_000_000_000.0
        return sample if age_s <= max_age_s else None

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
