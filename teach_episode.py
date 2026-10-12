"""Versioned, complete command episodes and non-blocking R/E recording."""

from __future__ import annotations

import hashlib
import json
import queue
import re
import threading
from collections import deque
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO

from teach_config import RepeatConfig, TeachError, integer, number, object_keys, vector


@dataclass(frozen=True)
class RobotState:
    time_ns: int
    feedback_ns: int
    joints_deg: list[float]
    target_deg: list[float]
    robot_mode: int
    gripper_ns: int
    gripper_position: int
    gripper_moving: bool
    gripper_torque: bool

    @classmethod
    def parse(cls, raw: Any) -> RobotState:
        d = object_keys(raw, "state", set(cls.__dataclass_fields__))
        for k in ("time_ns", "feedback_ns", "gripper_ns"):
            integer(d[k], k)
        if d["feedback_ns"] > d["time_ns"] or d["gripper_ns"] > d["time_ns"]:
            raise TeachError("feedback timestamp is in the future")
        vector(d["joints_deg"], "actual joints")
        vector(d["target_deg"], "feedback target")
        integer(d["robot_mode"], "robot_mode", 1, 11)
        integer(d["gripper_position"], "gripper_position", 0, 4095)
        for k in ("gripper_moving", "gripper_torque"):
            if type(d[k]) is not bool:
                raise TeachError(f"{k} must be boolean")
        return cls(**d)

    def fresh(self, config: RepeatConfig, *, now_ns: int | None = None) -> None:
        vector(self.joints_deg, "actual joints")
        vector(self.target_deg, "feedback target")
        now = self.time_ns if now_ns is None else now_ns
        for timestamp, limit in (
            (self.feedback_ns, config.checks["feedback_timeout_s"]),
            (self.gripper_ns, config.checks["gripper_feedback_timeout_s"]),
        ):
            if not 0 <= now - timestamp <= limit * 1e9:
                raise TeachError("feedback is stale or in the future")
        if self.robot_mode not in (5, 7, 8):
            raise TeachError(f"unexpected RobotMode {self.robot_mode}")
        if not self.gripper_torque:
            raise TeachError("gripper torque must already be enabled")
        config.joints_in_range(self.joints_deg)
        config.joints_in_range(self.target_deg)
        low, high = sorted(
            (config.gripper["open_position"], config.gripper["closed_position"])
        )
        if not low <= self.gripper_position <= high:
            raise TeachError("gripper feedback outside configured endpoints")


def within(actual: list[float], target: list[float], tolerances: list[float]) -> bool:
    return all(abs(a - b) <= t for a, b, t in zip(actual, target, tolerances))


class StationaryWindow:
    def __init__(self, config: RepeatConfig):
        self.config = config
        self.states: deque[RobotState] = deque()

    def observe(self, state: RobotState) -> None:
        if self.states and state.time_ns <= self.states[-1].time_ns:
            self.states.clear()
        self.states.append(state)
        cutoff = state.time_ns - self.config.checks["stationary_window_s"] * 1e9
        while len(self.states) > 2 and self.states[1].time_ns <= cutoff:
            self.states.popleft()

    def require(self, state: RobotState, commanded: list[float]) -> None:
        state.fresh(self.config)
        c = self.config.checks
        if (
            not self.states
            or self.states[-1] != state
            or state.time_ns - self.states[0].time_ns < c["stationary_window_s"] * 1e9
        ):
            raise TeachError("waiting for a complete stationary window")
        for old in self.states:
            old.fresh(self.config)
            if old.gripper_moving:
                raise TeachError("gripper is still moving")
        if any(
            b.time_ns - a.time_ns
            > min(c["feedback_timeout_s"], c["gripper_feedback_timeout_s"]) * 1e9
            for a, b in zip(self.states, list(self.states)[1:])
        ):
            raise TeachError("stationary observations have a gap")
        for i in range(6):
            values = [s.joints_deg[i] for s in self.states]
            if max(values) - min(values) > c["stationary_joint_tolerance_deg"]:
                raise TeachError("arm is still moving")
        positions = [s.gripper_position for s in self.states]
        if max(positions) - min(positions) > c["stationary_gripper_tolerance"]:
            raise TeachError("gripper position is still changing")
        if not within(state.joints_deg, commanded, c["joint_start_tolerance_deg"]):
            raise TeachError("arm has not reached the held target")


class EpisodeWriter:
    """Bounded writer. Only a healthy, normally closed file gets a valid footer."""

    def __init__(self, path: Path, header: dict[str, Any], max_queue: int = 512):
        if max_queue <= 0:
            raise TeachError("queue size must be positive")
        if path.exists():
            raise TeachError("episode file already exists")
        self.path = path
        self.header = header
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self.done = threading.Event()
        self.error: str | None = None
        self.count = 0
        self._footer: dict[str, Any] | None = None
        self._thread = threading.Thread(
            target=self._run, name="teach-writer", daemon=True
        )
        self._thread.start()

    def submit(self, record: dict[str, Any]) -> None:
        if self.error or self._stop.is_set():
            raise TeachError(self.error or "recording is closed")
        try:
            self._queue.put_nowait({**record, "seq": self.count})
            self.count += 1
        except queue.Full as exc:
            self.invalidate("recording queue full; episode is invalid")
            raise TeachError(self.error) from exc

    def finish(self, end_ns: int, state: RobotState) -> None:
        if self.error or self._stop.is_set():
            raise TeachError(self.error or "recording is closed")
        self._footer = {
            "record_type": "teach_end",
            "valid": True,
            "records": self.count,
            "end_ns": end_ns,
            "final_state": asdict(state),
        }
        self._stop.set()

    def invalidate(self, reason: str) -> None:
        self.error = self.error or reason
        self._stop.set()

    def close(self) -> None:
        if not self._stop.is_set():
            self.invalidate("recording interrupted before E")
        self._thread.join(timeout=2)
        if self._thread.is_alive():
            self.invalidate("writer shutdown timed out")

    @staticmethod
    def _write(stream: TextIO, record: dict[str, Any]) -> None:
        stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x", encoding="utf-8") as stream:
                self._write(stream, self.header)
                written = 0
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        record = self._queue.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    self._write(stream, record)
                    written += 1
                    if written % 32 == 0:
                        stream.flush()
                if (
                    self.error is None
                    and self._footer is not None
                    and written == self.count
                ):
                    self._write(stream, self._footer)
                stream.flush()
        except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
            self.invalidate(f"episode write failed: {exc}")
        finally:
            self.done.set()


@dataclass(frozen=True)
class Episode:
    sha256: str
    header: dict[str, Any]
    events: list[dict[str, Any]]
    observations: list[RobotState]
    footer: dict[str, Any]

    @property
    def duration_ns(self) -> int:
        return int(self.footer["end_ns"] - self.header["start_ns"])

    @property
    def initial(self) -> RobotState:
        return RobotState.parse(self.header["initial_state"])

    @property
    def final(self) -> RobotState:
        return RobotState.parse(self.footer["final_state"])

    def validate_limits(self, config: RepeatConfig) -> None:
        config.compatible(self.header["profile"])
        c = config.checks
        previous = self.initial.target_deg
        previous_ns = self.header["start_ns"]
        for state in [self.initial, *self.observations, self.final]:
            state.fresh(config)
        for event in self.events:
            if event["kind"] == "arm":
                target = vector(event["joints_deg"], "target")
                config.joints_in_range(target)
                dt = (event["send_ns"] - previous_ns) / 1e9
                if dt <= 0:
                    if target != previous:
                        raise TeachError("changed arm target has no elapsed time")
                elif any(
                    abs(a - b) > speed * dt + 1e-6
                    for a, b, speed in zip(
                        target, previous, config.data["limits"]["max_speed_deg_s"]
                    )
                ):
                    raise TeachError("recorded arm speed exceeds current limits")
                previous, previous_ns = target, event["send_ns"]
            else:
                low, high = sorted(
                    (config.gripper["open_position"], config.gripper["closed_position"])
                )
                if not low <= event["position"] <= high:
                    raise TeachError("gripper target outside configured range")
        if not within(self.final.joints_deg, previous, c["joint_start_tolerance_deg"]):
            raise TeachError("recording ends before arm reaches its final target")
        last_gripper = next(
            (e["position"] for e in reversed(self.events) if e["kind"] == "gripper"),
            self.initial.gripper_position,
        )
        if (
            abs(self.final.gripper_position - last_gripper)
            > c["gripper_tracking_tolerance"]
            or self.final.gripper_moving
        ):
            raise TeachError("recording ends before gripper reaches its final target")


def validate_event(event: dict[str, Any]) -> None:
    common = {"record_type", "seq", "kind", "send_ns", "reply_ns", "error_id", "reply"}
    if event.get("kind") == "arm":
        object_keys(
            event,
            "arm event",
            common | {"joints_deg", "t", "aheadtime", "gain", "command"},
        )
        q = vector(event["joints_deg"], "command joints")
        t, ahead, gain = (
            number(event[k], k, positive=True) for k in ("t", "aheadtime", "gain")
        )
        if not (0.004 <= t <= 3600 and 20 <= ahead <= 100 and 200 <= gain <= 1000):
            raise TeachError("invalid ServoJ parameters")
        command = (
            "ServoJ("
            + ",".join(f"{v:.6f}" for v in q)
            + f",t={t:g},aheadtime={ahead:g},gain={gain:g})"
        )
        if (
            event["command"] != command
            or not isinstance(event["reply"], str)
            or re.match(r"\s*0\s*,", event["reply"]) is None
            or not event["reply"].rstrip().endswith(command + ";")
        ):
            raise TeachError("invalid ServoJ command or response")
    elif event.get("kind") == "gripper":
        object_keys(
            event, "gripper event", common | {"position", "speed", "acceleration"}
        )
        integer(event["position"], "position", 0, 4095)
        integer(event["speed"], "speed", 0, 4095)
        integer(event["acceleration"], "acceleration", 0, 255)
        if event["reply"] != "COMM_SUCCESS":
            raise TeachError("invalid gripper response")
    else:
        raise TeachError("unknown command kind")
    integer(event["seq"], "seq")
    integer(event["send_ns"], "send_ns")
    integer(event["reply_ns"], "reply_ns")
    if (
        type(event["error_id"]) is not int
        or event["error_id"] != 0
        or event["reply_ns"] < event["send_ns"]
    ):
        raise TeachError("failed command or invalid response timing")


def load_episode(path: Path, config: RepeatConfig | None = None) -> Episode:
    try:
        content = path.read_bytes()

        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            d: dict[str, Any] = {}
            for k, v in items:
                if k in d:
                    raise TeachError("duplicate JSON key")
                d[k] = v
            return d

        rows = [
            json.loads(line, object_pairs_hook=pairs) for line in content.splitlines()
        ]
        if len(rows) < 4 or not content.endswith(b"\n"):
            raise TeachError("incomplete episode")
        header = object_keys(
            rows[0],
            "header",
            {"record_type", "schema_version", "start_ns", "profile", "initial_state"},
        )
        if (
            header["record_type"] != "teach_metadata"
            or type(header["schema_version"]) is not int
            or header["schema_version"] != 1
        ):
            raise TeachError(
                "not a Teach/Repeat v1 episode; legacy traces are not executable"
            )
        start = integer(header["start_ns"], "start_ns")
        recorded = RepeatConfig.from_profile(header["profile"])
        initial = RobotState.parse(header["initial_state"])
        if initial.time_ns != start:
            raise TeachError("initial state timestamp differs from episode start")
        footer = object_keys(
            rows[-1],
            "footer",
            {"record_type", "valid", "records", "end_ns", "final_state"},
        )
        if footer["record_type"] != "teach_end" or footer["valid"] is not True:
            raise TeachError("missing valid completion marker")
        end = integer(footer["end_ns"], "end_ns")
        if integer(footer["records"], "records") != len(rows) - 2 or end <= start:
            raise TeachError("invalid record count or episode duration")
        final = RobotState.parse(footer["final_state"])
        if final.time_ns != end:
            raise TeachError("final state timestamp differs from episode end")
        events: list[dict[str, Any]] = []
        observations = [initial]
        last_send, last_reply, last_record_time = start - 1, start, start
        for seq, row in enumerate(rows[1:-1]):
            if (
                not isinstance(row, dict)
                or row.get("seq") != seq
                or type(row.get("seq")) is not int
            ):
                raise TeachError("non-contiguous record sequence")
            if row.get("record_type") == "command":
                validate_event(row)
                send, reply = row["send_ns"], row["reply_ns"]
                if (
                    not start <= send <= reply <= end
                    or send <= last_send
                    or send < last_reply
                    or send < last_record_time
                ):
                    raise TeachError("non-monotonic or overlapping command timestamps")
                if row["kind"] == "gripper" and (
                    row["speed"] != recorded.gripper["command_speed"]
                    or row["acceleration"] != recorded.gripper["command_acceleration"]
                ):
                    raise TeachError("gripper command differs from recorded profile")
                last_send, last_reply, last_record_time = send, reply, send
                events.append(row)
            elif row.get("record_type") == "observation":
                object_keys(row, "observation", {"record_type", "seq", "state"})
                state = RobotState.parse(row["state"])
                if (
                    not last_record_time <= state.time_ns <= end
                    or state.time_ns < last_reply
                    or state.time_ns <= observations[-1].time_ns
                ):
                    raise TeachError("non-monotonic observation timestamp")
                last_record_time = state.time_ns
                observations.append(state)
            else:
                raise TeachError("unknown episode record")
        if not events or not any(e["kind"] == "arm" for e in events):
            raise TeachError("episode contains no arm commands")
        observations.append(final)
        if any(
            b.time_ns - a.time_ns > recorded.checks["feedback_timeout_s"] * 1e9
            for a, b in pairwise(observations)
        ):
            raise TeachError("episode observations have missing intervals")
        episode = Episode(
            hashlib.sha256(content).hexdigest(), header, events, observations, footer
        )
        episode.validate_limits(recorded)
        if config is not None:
            episode.validate_limits(config)
        return episode
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TeachError(f"Invalid episode {path}: {exc}") from exc
