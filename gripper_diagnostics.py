"""Candidate-only read-only state classification for the Nova gripper."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from nova_gripper import GripperStatus
from teleop_config import GripperDiagnosticConfig


@dataclass
class GripperDiagnostic:
    """Classify observed gripper behavior without issuing any command.

    The candidate states intentionally describe evidence only. They do not
    stop motion, change torque, or alter the commanded target.
    """

    open_position: int
    closed_position: int
    config: GripperDiagnosticConfig
    state: str = "UNKNOWN"
    _candidate_state: Optional[str] = None
    _candidate_since: Optional[float] = None
    _last_target_position: Optional[int] = None
    _last_actual_position: Optional[int] = None

    def update(
        self,
        target_position: Optional[int],
        status: Optional[GripperStatus],
        now: Optional[float] = None,
    ) -> str:
        """Return the current diagnostic state from the latest observations."""
        now = time.monotonic() if now is None else float(now)
        if target_position is None or status is None:
            self._reset_candidate()
            self.state = "UNKNOWN"
            return self.state

        target = int(target_position)
        actual = int(status.position)
        target_changed = (
            self._last_target_position is not None
            and abs(target - self._last_target_position)
            >= self.config.target_change_threshold
        )
        self._last_target_position = target
        self._last_actual_position = actual
        error = target - actual
        endpoint = self.config.endpoint_tolerance
        tracking_error = self.config.tracking_error_threshold

        if (
            abs(target - self.open_position) <= endpoint
            and abs(actual - self.open_position) <= endpoint
        ):
            base_state = "OPEN"
        elif (
            abs(target - self.closed_position) <= endpoint
            and abs(actual - self.closed_position) <= endpoint
        ):
            base_state = "CLOSED"
        elif error < -tracking_error:
            base_state = "OPENING"
        elif error > tracking_error:
            high_effort = abs(int(status.current)) >= self.config.candidate_current_threshold
            if high_effort:
                base_state = (
                    "CLOSING_BLOCKED_CANDIDATE"
                    if target_changed or bool(status.moving)
                    else "HOLDING_CANDIDATE"
                )
            else:
                base_state = "CLOSING"
        elif abs(int(status.current)) >= self.config.candidate_current_threshold:
            base_state = "HOLDING_CANDIDATE"
        else:
            base_state = "HOLDING"

        if not base_state.endswith("_CANDIDATE"):
            self._reset_candidate()
            self.state = base_state
            return self.state

        if self._candidate_state != base_state:
            self._candidate_state = base_state
            self._candidate_since = now
        if (
            self._candidate_since is not None
            and now - self._candidate_since >= self.config.candidate_persistence_s
        ):
            self.state = base_state
        elif self.state == base_state:
            # Keep a previously confirmed candidate while the same evidence
            # remains present. A different candidate type must re-arm its
            # own persistence timer.
            self.state = base_state
        else:
            self.state = "CLOSING" if error > 0 else "HOLDING"
        return self.state

    def _reset_candidate(self) -> None:
        self._candidate_state = None
        self._candidate_since = None
