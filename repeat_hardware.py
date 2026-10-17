"""Opt-in hardware adapter. Import only after --execute/config validation."""

from __future__ import annotations

import threading
import time
from typing import Any

from nova_client import NovaClient
from nova_gripper import GripperStatus, NovaGripper
from nova_realtime import NovaRealtimeReader
from repeat_runtime import SendResult
from teach_config import RepeatConfig, TeachError
from teach_episode import RobotState
from teleop_config import NovaConfig


class GripperFeedback:
    """Serialize reads/writes; cached reads keep disk/network out of waits."""

    def __init__(self, gripper: NovaGripper, config: RepeatConfig):
        self.gripper, self.config = gripper, config
        self.lock = threading.Lock()
        self.latest: tuple[GripperStatus, int] | None = None
        self.error: str | None = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="repeat-gripper-feedback"
        )

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            period = min(0.1, self.config.checks["gripper_feedback_timeout_s"] / 3)
            while not self.stopping.is_set():
                with self.lock:
                    status = self.gripper.read_status()
                    if status.servo_id != self.config.gripper["servo_id"]:
                        raise TeachError("gripper identity differs from configuration")
                    self.latest = (status, time.monotonic_ns())
                self.stopping.wait(period)
        except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
            self.error = str(exc)

    def state(self) -> tuple[GripperStatus, int]:
        if self.error:
            raise TeachError(self.error)
        if self.latest is None:
            raise TeachError("waiting for gripper feedback")
        return self.latest

    def close(self) -> None:
        self.stopping.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise TeachError("gripper feedback shutdown timed out")


class HardwareTransport:
    def __init__(self, config: RepeatConfig):
        self.config = config
        n = config.data["nova"]
        self.nova = NovaClient(
            NovaConfig(
                ip=n["host"],
                dashboard_port=n["dashboard_port"],
                socket_timeout_s=n["socket_timeout_s"],
                stop_timeout_s=n["stop_timeout_s"],
            )
        )
        self.feedback = NovaRealtimeReader(
            n["host"], n["feedback_port"], n["socket_timeout_s"]
        )
        self.gripper = NovaGripper(**config.gripper)
        self.gripper_feedback = GripperFeedback(self.gripper, config)

    def open(self, *, enable_gripper_torque: bool = False) -> None:
        self.nova.connect()
        self.nova.require_mode(5)
        status = self.gripper.open()
        if status.servo_id != self.config.gripper["servo_id"]:
            raise TeachError("gripper identity differs from configuration")
        if not status.torque_enabled:
            if not enable_gripper_torque:
                raise TeachError(
                    "Gripper torque is off; explicitly use --gripper-enable-torque"
                )
            self.gripper.enable_torque()
            if not self.gripper.read_status().torque_enabled:
                raise TeachError("gripper torque did not enable")
        self.feedback.connect()
        self.feedback.start()
        self.gripper_feedback.start()
        deadline = time.monotonic() + self.config.checks["completion_timeout_s"]
        while (
            self.feedback.state().latest is None or self.gripper_feedback.latest is None
        ):
            if self.feedback.state().error or self.gripper_feedback.error:
                raise TeachError(
                    self.feedback.state().error or self.gripper_feedback.error
                )
            if time.monotonic() >= deadline:
                raise TeachError("initial feedback timeout")
            time.sleep(self.config.checks["monitor_period_s"])

    def snapshot(self) -> RobotState:
        arm = self.feedback.latest(self.config.checks["feedback_timeout_s"])
        grip, grip_ns = self.gripper_feedback.state()
        return RobotState(
            time.monotonic_ns(),
            arm.host_timestamp_ns,
            list(arm.q_actual_deg),
            list(arm.q_target_deg),
            arm.robot_mode,
            grip_ns,
            grip.position,
            grip.moving,
            grip.torque_enabled,
        )

    def send(
        self, event: dict[str, Any], deadline_ns: int, latest_ns: int
    ) -> SendResult:
        self.snapshot().fresh(self.config, now_ns=time.monotonic_ns())
        timing: dict[str, Any] = {}
        if event["kind"] == "arm":
            response = self.nova.servo_j(
                event["joints_deg"],
                t=event["t"],
                aheadtime=event["aheadtime"],
                gain=event["gain"],
                timing=timing,
                send_not_after_ns=latest_ns,
            )
            return SendResult(
                timing["t_servoj_send"],
                timing["t_servoj_response"],
                response.error_id,
                response.raw,
            )
        with self.gripper_feedback.lock:
            self.snapshot().fresh(self.config, now_ns=time.monotonic_ns())
            self.gripper.command_position(
                event["position"], timing, send_not_after_ns=latest_ns
            )
        return SendResult(
            timing["t_gripper_send"], timing["t_gripper_response"], 0, "COMM_SUCCESS"
        )

    def stop(self) -> str:
        response = self.nova.stop()
        if response.error_id != 0:
            raise TeachError(f"Stop rejected: {response.raw}")
        return response.raw

    def close(self) -> None:
        self.feedback.close()
        try:
            self.gripper_feedback.close()
        finally:
            if not self.gripper_feedback.thread.is_alive():
                self.gripper.close()
            self.nova.close()
