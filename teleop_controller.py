"""Pure relative joint teleoperation state and mapping logic."""

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from teleop_config import TeleopConfig, validate_vector


@dataclass(frozen=True)
class ControllerOutput:
    delta_deg: tuple[float, ...]
    desired_deg: tuple[float, ...]
    commanded_deg: tuple[float, ...]
    actual_dt_s: float
    clutch_active: bool
    rate_limited: bool


class TeleopController:
    """Relative Leader to Nova mapping with anchor and clutch state."""

    def __init__(self, config: TeleopConfig):
        self.config = config
        self._leader_anchor: Optional[tuple[float, ...]] = None
        self._nova_anchor: Optional[tuple[float, ...]] = None
        self._commanded: Optional[list[float]] = None
        self._clutch_active = False
        self._last_update = time.monotonic()

    @property
    def clutch_active(self) -> bool:
        return self._clutch_active

    @property
    def commanded_deg(self) -> Optional[tuple[float, ...]]:
        if self._commanded is None:
            return None
        return tuple(self._commanded)

    @property
    def leader_anchor_deg(self) -> Optional[tuple[float, ...]]:
        return self._leader_anchor

    @property
    def nova_anchor_deg(self) -> Optional[tuple[float, ...]]:
        return self._nova_anchor

    def anchor(
        self,
        leader_deg: Sequence[float],
        nova_deg: Sequence[float],
    ):
        validate_vector("leader_deg", leader_deg)
        validate_vector("nova_deg", nova_deg)
        self._leader_anchor = tuple(float(value) for value in leader_deg)
        self._nova_anchor = tuple(float(value) for value in nova_deg)
        self._commanded = list(self._nova_anchor)
        self._last_update = time.monotonic()

    def set_clutch(
        self,
        pressed: bool,
        leader_deg: Optional[Sequence[float]] = None,
        nova_reference_deg: Optional[Sequence[float]] = None,
    ):
        pressed = bool(pressed)
        if pressed == self._clutch_active:
            return

        if pressed:
            self._clutch_active = True
            return

        if leader_deg is None:
            raise ValueError("Leader state is required when releasing clutch")
        if nova_reference_deg is None:
            nova_reference_deg = self.commanded_deg
        if nova_reference_deg is None:
            raise RuntimeError("Controller has no Nova reference for clutch release")

        self.anchor(leader_deg, nova_reference_deg)
        self._clutch_active = False

    def update(
        self,
        leader_deg: Sequence[float],
        actual_dt_s: Optional[float] = None,
    ) -> ControllerOutput:
        if self._leader_anchor is None or self._nova_anchor is None:
            raise RuntimeError("TeleopController is not anchored")
        validate_vector("leader_deg", leader_deg)

        now = time.monotonic()
        if actual_dt_s is None:
            actual_dt_s = now - self._last_update
        actual_dt_s = min(
            max(float(actual_dt_s), 0.001),
            max(2.0 * self.config.dt_s, 0.001),
        )
        self._last_update = now

        if self._clutch_active:
            commanded = tuple(self._commanded)
            return ControllerOutput(
                delta_deg=(0.0,) * 6,
                desired_deg=commanded,
                commanded_deg=commanded,
                actual_dt_s=actual_dt_s,
                clutch_active=True,
                rate_limited=False,
            )

        delta = []
        desired = []
        for index in range(6):
            mapped = (
                (float(leader_deg[index]) - self._leader_anchor[index])
                * self.config.map_sign[index]
                * self.config.scale[index]
            )
            if abs(mapped) <= self.config.deadband_deg[index]:
                mapped = 0.0
            mapped = max(
                -self.config.max_offset_deg[index],
                min(self.config.max_offset_deg[index], mapped),
            )
            target = self._nova_anchor[index] + mapped
            if self.config.joint_limits_deg is not None:
                lower, upper = self.config.joint_limits_deg[index]
                target = max(lower, min(upper, target))
            delta.append(mapped)
            desired.append(target)

        rate_limited = False
        for index in range(6):
            error = desired[index] - self._commanded[index]
            max_step = self.config.max_speed_deg_s[index] * actual_dt_s
            step = max(-max_step, min(max_step, error))
            if not math.isclose(step, error, rel_tol=0.0, abs_tol=1e-12):
                rate_limited = True
            self._commanded[index] += step

        return ControllerOutput(
            delta_deg=tuple(delta),
            desired_deg=tuple(desired),
            commanded_deg=tuple(self._commanded),
            actual_dt_s=actual_dt_s,
            clutch_active=False,
            rate_limited=rate_limited,
        )
