"""Absolute-time replay with injectable clock and transports for offline tests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from teach_config import RepeatConfig, TeachError
from teach_episode import Episode, RobotState, StationaryWindow, within


class Clock(Protocol):
    def now_ns(self) -> int: ...
    def sleep(self, seconds: float) -> None: ...


class MonotonicClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


@dataclass(frozen=True)
class SendResult:
    send_ns: int
    reply_ns: int
    error_id: int
    reply: str


class RepeatTransport(Protocol):
    def snapshot(self) -> RobotState: ...
    def send(
        self, event: dict[str, Any], deadline_ns: int, latest_ns: int
    ) -> SendResult: ...
    def stop(self) -> str: ...


def check_start(
    state: RobotState, episode: Episode, config: RepeatConfig, now_ns: int
) -> None:
    state.fresh(config, now_ns=now_ns)
    if state.robot_mode != 5:
        raise TeachError("start requires RobotMode 5")
    c = config.checks
    if not within(
        state.joints_deg, episode.initial.joints_deg, c["joint_start_tolerance_deg"]
    ):
        raise TeachError("arm is not at the recorded start")
    if (
        abs(state.gripper_position - episode.initial.gripper_position)
        > c["gripper_start_tolerance"]
    ):
        raise TeachError("gripper is not at the recorded start")
    if state.gripper_moving:
        raise TeachError("gripper is moving at start")


class RepeatRunner:
    def __init__(
        self,
        episode: Episode,
        config: RepeatConfig,
        transport: RepeatTransport,
        clock: Clock | None = None,
    ):
        self.episode, self.config, self.transport = episode, config, transport
        self.clock = clock or MonotonicClock()
        self.arm_target = episode.initial.joints_deg
        self.gripper_target = episode.initial.gripper_position
        self.origin = 0
        self._reference_index = 0
        self.report: dict[str, Any] = {
            "schema_version": 1,
            "source_sha256": episode.sha256,
            "status": "aborted",
            "stop_reason": None,
            "stop_result": None,
            "commands_expected": len(episode.events),
            "commands_sent": 0,
            "commands_attempted": 0,
            "unacknowledged_event_seq": None,
            "events": [],
            "observations": [],
        }

    def _monitor(self) -> RobotState:
        state = self.transport.snapshot()
        now = self.clock.now_ns()
        state.fresh(self.config, now_ns=now)
        c = self.config.checks
        if not within(
            state.joints_deg, self.arm_target, c["joint_tracking_tolerance_deg"]
        ) or not within(
            state.joints_deg, state.target_deg, c["joint_tracking_tolerance_deg"]
        ):
            raise TeachError("arm tracking error exceeds configured tolerance")
        if (
            abs(state.gripper_position - self.gripper_target)
            > c["gripper_tracking_tolerance"]
        ):
            raise TeachError("gripper tracking error exceeds configured tolerance")
        reference_time = self.episode.header["start_ns"] + max(0, now - self.origin)
        samples = self.episode.observations
        while (
            self._reference_index + 1 < len(samples)
            and samples[self._reference_index + 1].time_ns <= reference_time
        ):
            self._reference_index += 1
        a = samples[self._reference_index]
        b = samples[min(self._reference_index + 1, len(samples) - 1)]
        fraction = min(
            1.0, max(0.0, (reference_time - a.time_ns) / max(1, b.time_ns - a.time_ns))
        )
        reference = [x + (y - x) * fraction for x, y in zip(a.joints_deg, b.joints_deg)]
        self.report["observations"].append(
            {
                "elapsed_ns": now - self.origin,
                "actual_deg": state.joints_deg,
                "tracking_error_deg": [
                    t - q for t, q in zip(self.arm_target, state.joints_deg)
                ],
                "demonstration_error_deg": [
                    q - r for q, r in zip(state.joints_deg, reference)
                ],
                "gripper_position": state.gripper_position,
                "gripper_tracking_error": self.gripper_target - state.gripper_position,
            }
        )
        return state

    def _wait_until(self, deadline_ns: int) -> None:
        while True:
            self._monitor()
            remaining = deadline_ns - self.clock.now_ns()
            if remaining <= 0:
                return
            self.clock.sleep(
                min(remaining / 1e9, self.config.checks["monitor_period_s"])
            )

    def require_start(self) -> None:
        window = StationaryWindow(self.config)
        deadline = self.clock.now_ns() + int(
            self.config.checks["completion_timeout_s"] * 1e9
        )
        while True:
            state = self.transport.snapshot()
            check_start(state, self.episode, self.config, self.clock.now_ns())
            window.observe(state)
            try:
                window.require(state, state.target_deg)
                return
            except TeachError:
                if self.clock.now_ns() >= deadline:
                    raise TeachError(
                        "start did not become stationary before timeout"
                    ) from None
                self.clock.sleep(self.config.checks["monitor_period_s"])

    def _complete(self) -> None:
        deadline = self.clock.now_ns() + int(
            self.config.checks["completion_timeout_s"] * 1e9
        )
        window = StationaryWindow(self.config)
        c = self.config.checks
        while True:
            state = self._monitor()
            window.observe(state)
            try:
                window.require(state, self.arm_target)
                c = self.config.checks
                if not within(
                    state.joints_deg,
                    self.episode.final.joints_deg,
                    c["joint_start_tolerance_deg"],
                ):
                    raise TeachError("arm differs from recorded final state")
                if (
                    abs(state.gripper_position - self.episode.final.gripper_position)
                    > c["gripper_start_tolerance"]
                ):
                    raise TeachError("gripper differs from recorded final state")
                return
            except TeachError:
                if self.clock.now_ns() >= deadline:
                    raise TeachError(
                        "final state did not settle before timeout"
                    ) from None
                self.clock.sleep(c["monitor_period_s"])

    def run(self) -> dict[str, Any]:
        armed = False
        try:
            self.episode.validate_limits(self.config)
            self.require_start()  # Recheck after the operator's REPLAY input.
            armed = True
            self.origin = self.clock.now_ns()
            self.report["started_ns"] = self.origin
            start = self.episode.header["start_ns"]
            events = self.episode.events
            for index, event in enumerate(events):
                deadline = self.origin + event["send_ns"] - start
                next_deadline = (
                    self.origin + events[index + 1]["send_ns"] - start
                    if index + 1 < len(events)
                    else self.origin + self.episode.duration_ns
                )
                self._wait_until(deadline)
                latest = min(
                    deadline + int(self.config.checks["max_lateness_s"] * 1e9),
                    next_deadline - 1,
                )
                if self.clock.now_ns() > latest:
                    raise TeachError("command deadline missed; no catch-up permitted")
                self.report["commands_attempted"] += 1
                self.report["unacknowledged_event_seq"] = event["seq"]
                result = self.transport.send(event, deadline, latest)
                self.report["unacknowledged_event_seq"] = None
                self.report["commands_sent"] += 1
                self.report["events"].append(
                    {
                        "seq": event["seq"],
                        "kind": event["kind"],
                        "deadline_ns": deadline,
                        "send_ns": result.send_ns,
                        "reply_ns": result.reply_ns,
                        "lateness_ns": result.send_ns - deadline,
                        "error_id": result.error_id,
                        "reply": result.reply,
                    }
                )
                if (
                    not deadline <= result.send_ns <= latest
                    or result.reply_ns < result.send_ns
                ):
                    raise TeachError("transport violated command deadline")
                if result.error_id != 0:
                    raise TeachError("command rejected")
                if event["kind"] == "arm":
                    self.arm_target = event["joints_deg"]
                else:
                    self.gripper_target = event["position"]
                if (
                    result.reply_ns >= next_deadline
                    or self.clock.now_ns() >= next_deadline
                ):
                    raise TeachError(
                        "next command became due while previous command was in progress"
                    )
                self._monitor()
            self._wait_until(self.origin + self.episode.duration_ns)
            self._complete()
            self.report["status"] = "completed"
            self.report["stop_reason"] = (
                "recorded tail elapsed and final state verified"
            )
        except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - operator abort and I/O failure share stop handling
            self.report["stop_reason"] = (
                "Ctrl+C" if isinstance(exc, KeyboardInterrupt) else str(exc)
            )
        finally:
            if armed:
                try:
                    self.report["stop_result"] = self.transport.stop()
                except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
                    self.report["status"] = "aborted"
                    self.report["stop_error"] = str(exc)
            self.report["finished_ns"] = self.clock.now_ns()
        return self.report
