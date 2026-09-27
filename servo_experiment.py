"""Hardware-independent experiment records and bounded asynchronous writer."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMetadata:
    experiment_id: str
    experiment_name: str
    created_at: str
    robot_host: str
    robot_model: str
    joint: str
    amplitude_deg: float
    rate_limit_deg_s: float
    servo_t: float
    servo_aheadtime: float
    servo_gain: float
    requested_loop_period_s: float
    initial_q_deg: list[float]
    safety_envelope_deg: float = 10.0
    experiment_status: str = "planned"
    abort_reason: str | None = None
    software_version: str | None = None


class ExperimentRecorder:
    def __init__(
        self, output: Path, metadata: ExperimentMetadata, max_queue: int = 512
    ):
        self.output = output
        self.metadata = metadata
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="servo-record-writer"
        )
        self._error: str | None = None
        self.submitted = self.written = self.dropped = self.high_water = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def submit(self, sample: dict[str, Any]) -> bool:
        self.submitted += 1
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            self.dropped += 1
            return False
        self.high_water = max(self.high_water, self._queue.qsize())
        return True

    @property
    def error(self) -> str | None:
        return self._error

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        try:
            with self.output.open("w", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "record_type": "metadata",
                            "schema_version": 1,
                            "metadata": asdict(self.metadata),
                        }
                    )
                    + "\n"
                )
                stream.flush()
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    stream.write(
                        json.dumps(
                            {"record_type": "sample", **item},
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    self.written += 1
                stream.write(
                    json.dumps(
                        {
                            "record_type": "final_metadata",
                            "metadata": asdict(self.metadata),
                            "submitted": self.submitted,
                            "written": self.written,
                            "dropped": self.dropped,
                            "queue_high_water": self.high_water,
                        },
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                stream.flush()
        except Exception as exc:
            self._error = repr(exc)


def sample_record(
    *,
    t_monotonic_ns: int,
    experiment_time_s: float,
    q_command_deg: list[float],
    q_target_feedback_deg: list[float],
    q_actual_deg: list[float],
    servo_send_ns: int | None,
    servo_reply_ns: int | None,
    feedback_host_timestamp_ns: int,
    feedback_controller_timestamp: int,
    feedback_age_ms: float,
    robot_mode: int,
) -> dict[str, Any]:
    return {
        "t_monotonic_ns": t_monotonic_ns,
        "experiment_time_s": experiment_time_s,
        "q_command_deg": q_command_deg,
        "q_target_feedback_deg": q_target_feedback_deg,
        "q_actual_deg": q_actual_deg,
        "servo_send_ns": servo_send_ns,
        "servo_reply_ns": servo_reply_ns,
        "servo_rtt_ms": (
            (servo_reply_ns - servo_send_ns) / 1_000_000
            if servo_send_ns and servo_reply_ns
            else None
        ),
        "feedback_host_timestamp_ns": feedback_host_timestamp_ns,
        "feedback_controller_timestamp": feedback_controller_timestamp,
        "feedback_age_ms": feedback_age_ms,
        "robot_mode": robot_mode,
    }
