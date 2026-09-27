"""Pure Dobot V4.6.5 30004 framing and trusted joint fields.

Only fields documented by the project audit are decoded.  The packet evidence
is shape evidence; it is not a firmware compatibility claim.

``host_timestamp_ns`` is sampled with the local monotonic clock when a frame
is received.  ``controller_timestamp_ms`` is the controller's separate clock;
the two clocks must not be subtracted directly. QTarget/QActual are retained
in the native joint-degree representation observed in the verified frame;
they are not converted through radians. Wall-clock timestamps belong
in the recorder/session layer when an operator needs cross-process correlation.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

FRAME_SIZE = 1440
FEEDBACK_PORT = 30004
TEST_VALUE = 0x0123456789ABCDEF
ROBOT_MODE_OFFSET = 24
CONTROLLER_TIMESTAMP_OFFSET = 32
Q_TARGET_OFFSET = 192
Q_ACTUAL_OFFSET = 432
JOINT_COUNT = 6
VALID_ROBOT_MODES = frozenset(range(1, 12))
_U16 = struct.Struct("<H")
_U64 = struct.Struct("<Q")
_JOINTS = struct.Struct("<6d")


@dataclass(frozen=True)
class FeedbackSample:
    robot_mode: int
    controller_timestamp_ms: int
    q_target_deg: tuple[float, ...]
    q_actual_deg: tuple[float, ...]
    host_timestamp_ns: int

    def age_ms(self, now_ns: int) -> float:
        return (now_ns - self.host_timestamp_ns) / 1_000_000.0


def _mode(value: int) -> int:
    if value not in VALID_ROBOT_MODES:
        raise ValueError(f"invalid RobotMode {value}")
    return value


def decode_frame(frame: bytes, *, host_timestamp_ns: int) -> FeedbackSample:
    if len(frame) != FRAME_SIZE:
        raise ValueError(f"expected {FRAME_SIZE} bytes, got {len(frame)}")
    if _U16.unpack_from(frame, 0)[0] != FRAME_SIZE:
        raise ValueError("invalid 30004 MessageSize")
    if _U64.unpack_from(frame, 48)[0] != TEST_VALUE:
        raise ValueError("invalid 30004 TestValue sentinel")
    robot_mode = _mode(_U64.unpack_from(frame, ROBOT_MODE_OFFSET)[0])
    # The captured V4.6.5 frame and the existing Nova reader agree that these
    # six values are native joint degrees.  Do not apply a radians conversion:
    # doing so turns a valid pose such as J2=-38.79 into -2222.5 degrees.
    q_target = _JOINTS.unpack_from(frame, Q_TARGET_OFFSET)
    q_actual = _JOINTS.unpack_from(frame, Q_ACTUAL_OFFSET)
    if not all(math.isfinite(v) for v in (*q_target, *q_actual)):
        raise ValueError("non-finite joint feedback")
    return FeedbackSample(
        robot_mode=robot_mode,
        controller_timestamp_ms=_U64.unpack_from(frame, CONTROLLER_TIMESTAMP_OFFSET)[0],
        q_target_deg=q_target,
        q_actual_deg=q_actual,
        host_timestamp_ns=host_timestamp_ns,
    )


class FeedbackFramer:
    """Fixed-size stream framer; invalid alignment fails closed."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def feed(self, data: bytes, *, host_timestamp_ns: int) -> list[FeedbackSample]:
        samples: list[FeedbackSample] = []
        offset = 0
        while offset < len(data):
            take = min(FRAME_SIZE - len(self._buffer), len(data) - offset)
            self._buffer.extend(data[offset : offset + take])
            offset += take
            if (
                len(self._buffer) >= 2
                and _U16.unpack_from(self._buffer, 0)[0] != FRAME_SIZE
            ):
                self._buffer.clear()
                raise ValueError("invalid 30004 frame alignment")
            if len(self._buffer) == FRAME_SIZE:
                samples.append(
                    decode_frame(
                        bytes(self._buffer), host_timestamp_ns=host_timestamp_ns
                    )
                )
                self._buffer.clear()
        return samples

    def finish(self) -> None:
        if self._buffer:
            raise ValueError("truncated 30004 frame")
