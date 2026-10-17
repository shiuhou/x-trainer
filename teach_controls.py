"""Cycle-boundary recording controls and terminal restoration."""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

from teach_config import RepeatConfig, TeachError
from teach_episode import EpisodeWriter, RobotState, StationaryWindow


class TerminalKeys:
    def __init__(self, stream: TextIO = sys.stdin):
        self.stream = stream
        self.saved: list[Any] | None = None

    def open(self) -> None:
        if not self.stream.isatty():
            raise TeachError("R/E recording requires an interactive terminal")
        self.saved = termios.tcgetattr(self.stream.fileno())
        try:
            tty.setcbreak(self.stream.fileno())
        except BaseException:
            self.close()
            raise

    def poll(self) -> list[str]:
        keys = []
        while select.select([self.stream], [], [], 0)[0]:
            # Read the descriptor rather than buffered stdin (which may hide
            # pending bytes from select). ISIG remains enabled for Ctrl+C.
            import os

            value = os.read(self.stream.fileno(), 1)
            if not value:
                raise TeachError("terminal input closed")
            keys.append(value.decode("ascii", errors="ignore").lower())
        return keys

    def close(self) -> None:
        if self.saved is not None:
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self.saved)
            self.saved = None


class TeachSession:
    def __init__(self, directory: Path, config: RepeatConfig):
        self.directory = directory
        self.config = config
        self.window = StationaryWindow(config)
        self.writer: EpisodeWriter | None = None
        self.writers: list[EpisodeWriter] = []
        self.status = "待錄製"
        self.last_gripper_target: int | None = None
        self.gripper_status: Any = None
        self.gripper_feedback_ns: int | None = None

    @property
    def active(self) -> bool:
        return self.status == "錄製中"

    def boundary(
        self,
        keys: list[str],
        state: RobotState,
        arm_target: list[float],
        gripper_target: int | None,
    ) -> list[str]:
        self.window.observe(state)
        messages: list[str] = []
        if self.writer is not None and self.writer.error:
            self.status = "無效"
            raise TeachError(self.writer.error)
        if (
            self.status == "保存中"
            and self.writer is not None
            and self.writer.done.is_set()
        ):
            self.status = "已保存"
            messages.append(f"已保存: {self.writer.path}")
        if self.active:
            state.fresh(self.config)
            assert self.writer is not None
            self.writer.submit({"record_type": "observation", "state": asdict(state)})
        for key in keys:
            if key not in ("r", "e"):
                continue
            try:
                self.window.require(state, arm_target)
                target = self.last_gripper_target if self.active else gripper_target
                if (
                    target is not None
                    and abs(state.gripper_position - target)
                    > self.config.checks["gripper_tracking_tolerance"]
                ):
                    raise TeachError("gripper has not reached the held target")
                if key == "r":
                    if self.active or self.status == "保存中":
                        raise TeachError("finish the current recording first")
                    path = (
                        self.directory
                        / f"teach_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex}.jsonl"
                    )
                    header = {
                        "record_type": "teach_metadata",
                        "schema_version": 2,
                        "start_ns": state.time_ns,
                        "profile": self.config.profile(),
                        "initial_state": asdict(state),
                    }
                    self.writer = EpisodeWriter(path, header)
                    self.writers.append(self.writer)
                    self.last_gripper_target = gripper_target
                    self.status = "錄製中"
                    messages.append(f"錄製中: {path}")
                elif self.active:
                    assert self.writer is not None
                    self.writer.finish(state.time_ns, state)
                    self.status = "保存中"
                    messages.append("錄製結束，保存中；後續遙操不納入此段")
                else:
                    raise TeachError("no recording is active")
            except TeachError as exc:
                messages.append(f"{key.upper()} 未接受: {exc}")
        return messages

    def poll_hardware(
        self,
        keys: TerminalKeys,
        feedback: Any,
        gripper: Any,
        arm_target: list[float],
        gripper_target: int | None,
    ) -> None:
        """Sample feedback and apply R/E once at the control-cycle boundary."""
        if (
            self.gripper_feedback_ns is None
            or time.monotonic_ns() - self.gripper_feedback_ns
            > self.config.checks["gripper_feedback_timeout_s"] * 1e9 / 3
        ):
            self.gripper_status = gripper.read_status()
            self.gripper_feedback_ns = time.monotonic_ns()
        if self.gripper_status.servo_id != self.config.gripper["servo_id"]:
            raise TeachError("gripper ID changed")
        # Refresh arm cache after the serial read; it can have blocked.
        arm = feedback.latest(self.config.checks["feedback_timeout_s"])
        if arm is None:
            raise TeachError("Nova feedback stale")
        assert self.gripper_feedback_ns is not None
        state = RobotState(
            time.monotonic_ns(),
            arm.host_timestamp_ns,
            list(arm.q_actual_deg),
            list(arm.q_target_deg),
            arm.robot_mode,
            self.gripper_feedback_ns,
            self.gripper_status.position,
            self.gripper_status.moving,
            self.gripper_status.torque_enabled,
        )
        for message in self.boundary(keys.poll(), state, arm_target, gripper_target):
            print(message)

    def arm_command(
        self,
        joints: Any,
        servo: Any,
        response: Any,
        timing: dict[str, Any],
    ) -> None:
        """Record exactly the values serialized by the existing NovaClient."""
        self.command(
            {
                "kind": "arm",
                "send_ns": timing["t_servoj_send"],
                "reply_ns": timing["t_servoj_response"],
                "error_id": response.error_id,
                "reply": response.raw,
                "command": response.command,
                "joints_deg": [float(f"{v:.6f}") for v in joints],
                "t": float(f"{servo.servo_t:g}"),
                "aheadtime": float(f"{servo.servo_aheadtime:g}"),
                "gain": float(f"{servo.servo_gain:g}"),
            }
        )

    def gripper_command(self, position: int, timing: dict[str, Any]) -> None:
        """Call only after a successful physical position write (never a hold)."""
        self.command(
            {
                "kind": "gripper",
                "send_ns": timing["t_gripper_send"],
                "reply_ns": timing["t_gripper_response"],
                "error_id": 0,
                "reply": "COMM_SUCCESS",
                "position": position,
                "speed": self.config.gripper["command_speed"],
                "acceleration": self.config.gripper["command_acceleration"],
            }
        )

    def command(self, event: dict[str, Any]) -> None:
        if self.active:
            assert self.writer is not None
            if event["error_id"] != 0:
                self.abort("command rejected during recording")
                raise TeachError("command rejected during recording")
            self.writer.submit({"record_type": "command", **event})
            if event["kind"] == "gripper":
                self.last_gripper_target = event["position"]

    def abort(self, reason: str) -> None:
        if self.active and self.writer is not None:
            self.writer.invalidate(reason)
            self.status = "無效"

    def close(self) -> None:
        self.abort("teleoperation ended before E")
        for writer in self.writers:
            writer.close()
            if writer.error:
                print(f"無效: {writer.path}: {writer.error}")
