"""Read-only X-Trainer Leader access with optional background sampling."""

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional

from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_read import (
    GroupSyncRead,
)
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import (
    PacketHandler,
)
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import (
    PortHandler,
)
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
)

from teleop_config import LeaderConfig


@dataclass(frozen=True)
class LeaderSample:
    joints_deg: tuple[float, ...]
    trigger_raw: int
    trigger_deg: float
    trigger_normalized: float
    keys: Optional[tuple[int, int, int]]
    trigger_rad: float
    read_start_ns: int
    timestamp_ns: int

    @property
    def read_ms(self) -> float:
        return (self.timestamp_ns - self.read_start_ns) / 1_000_000.0


def raw_to_signed(raw: int) -> int:
    return raw - 2**32 if raw >= 2**31 else raw


def raw_to_rad(raw: int) -> float:
    return raw_to_signed(raw) / 2048.0 * math.pi


class LeaderReader:
    """Dynamixel reader that never enables torque or writes PID values."""

    def __init__(self, config: LeaderConfig):
        self.config = config
        self._port: Optional[PortHandler] = None
        self._packet: Optional[PacketHandler] = None
        self._reader: Optional[GroupSyncRead] = None
        self._state_lock = threading.Lock()
        self._latest: Optional[LeaderSample] = None
        self._last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def open(self):
        if self._port is not None:
            return

        port = PortHandler(self.config.port)
        packet = PacketHandler(2.0)
        if not port.openPort():
            raise RuntimeError(f"Cannot open Leader: {self.config.port}")
        if not port.setBaudRate(self.config.baudrate):
            port.closePort()
            raise RuntimeError(
                f"Cannot set Leader baudrate: {self.config.baudrate}"
            )

        try:
            for motor_id in self.config.torque_ids:
                result, error = packet.write1ByteTxRx(
                    port,
                    motor_id,
                    self.config.torque_address,
                    0,
                )
                if result != COMM_SUCCESS or error != 0:
                    raise RuntimeError(
                        f"Cannot disable Leader torque for ID {motor_id}: "
                        f"result={result}, error={error}"
                    )

            reader = GroupSyncRead(
                port,
                packet,
                self.config.position_address,
                self.config.position_length,
            )
            for motor_id in self.config.read_ids:
                if not reader.addParam(motor_id):
                    raise RuntimeError(
                        f"Cannot add Leader ID {motor_id} to GroupSyncRead"
                    )
        except Exception:
            port.closePort()
            raise

        self._port = port
        self._packet = packet
        self._reader = reader

    def read_once(self) -> Optional[LeaderSample]:
        if self._reader is None:
            raise RuntimeError("LeaderReader is not open")

        read_start_ns = time.monotonic_ns()
        result = self._reader.txRxPacket()
        if result != COMM_SUCCESS:
            self._set_error(f"Leader GroupSyncRead failed: {result}")
            return None

        values = []
        for motor_id in self.config.read_ids:
            if not self._reader.isAvailable(
                motor_id,
                self.config.position_address,
                self.config.position_length,
            ):
                self._set_error(f"Leader position unavailable for ID {motor_id}")
                return None
            values.append(
                self._reader.getData(
                    motor_id,
                    self.config.position_address,
                    self.config.position_length,
                )
            )

        joints_deg = tuple(
            math.degrees(
                (raw_to_rad(values[index]) - self.config.offsets_rad[index])
                * self.config.signs[index]
            )
            for index in range(6)
        )
        trigger_rad = raw_to_rad(values[6])
        trigger_deg = math.degrees(trigger_rad)
        trigger_normalized = (
            (trigger_deg - self.config.trigger_open_deg)
            / (self.config.trigger_closed_deg - self.config.trigger_open_deg)
        )
        trigger_normalized = max(0.0, min(1.0, trigger_normalized))
        keys = self._read_keys()
        timestamp_ns = time.monotonic_ns()
        self._set_error(None)
        return LeaderSample(
            joints_deg=joints_deg,
            trigger_raw=values[6],
            trigger_deg=trigger_deg,
            trigger_normalized=trigger_normalized,
            keys=keys,
            trigger_rad=trigger_rad,
            read_start_ns=read_start_ns,
            timestamp_ns=timestamp_ns,
        )

    def _read_keys(self) -> Optional[tuple[int, int, int]]:
        """Read the X-Trainer two-byte button packet used by the SDK."""
        if self._port is None:
            return None
        try:
            self._port.writePort(bytes((0xAA, 0x55, 0xAA)))
            self._port.setPacketTimeout(2)
            packet = bytearray()
            while len(packet) < 2:
                packet.extend(self._port.readPort(2 - len(packet)))
                if self._port.isPacketTimeout():
                    return None
            first, second = packet[:2]
            if (first | second) != 0xFF or (first & second) != 0x00:
                return None
            return ((first >> 4) & 0x0F, first & 0x0F, 0)
        except Exception:
            return None

    def start(self):
        if self._reader is None:
            raise RuntimeError("LeaderReader is not open")
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="xtrainer-leader-reader",
            daemon=True,
        )
        self._thread.start()

    def _sampling_loop(self):
        period = 1.0 / self.config.sample_hz
        next_sample = time.monotonic()
        while not self._stop.is_set():
            try:
                sample = self.read_once()
            except Exception as exc:
                self._set_error(str(exc))
                sample = None
            if sample is not None:
                with self._state_lock:
                    self._latest = sample

            next_sample += period
            self._stop.wait(max(0.0, next_sample - time.monotonic()))

    def latest(self, max_age_s: Optional[float] = None) -> Optional[LeaderSample]:
        with self._state_lock:
            sample = self._latest
        if sample is None or max_age_s is None:
            return sample
        age_s = (time.monotonic_ns() - sample.timestamp_ns) / 1_000_000_000.0
        return sample if age_s <= max_age_s else None

    @property
    def last_error(self) -> Optional[str]:
        with self._state_lock:
            return self._last_error

    def _set_error(self, message: Optional[str]):
        with self._state_lock:
            self._last_error = message

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._port is not None:
            self._port.closePort()
            self._port = None
            self._packet = None
            self._reader = None
