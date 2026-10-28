"""Small, simulator-independent checks at the model/action boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

CAMERAS = {
    "front_rgb": (480, 640, 3),
    "right_rgb": (480, 640, 3),
    "wrist_rgb": (720, 1280, 3),
}
LABELS = ("x", "y", "z", "yaw", "pitch", "roll", "gripper")


@dataclass(frozen=True)
class Limits:
    # Right-arm base frame, metres; includes upstream cube/plate work surface.
    low: tuple[float, float, float] = (-0.75, -0.75, -0.05)
    high: tuple[float, float, float] = (0.75, 0.75, 0.9)
    waypoint_m: float = 0.02
    chunk_path_m: float = 0.30
    max_waypoints: int = 64

    def validate(self, rows: NDArray, current: NDArray) -> None:
        if (
            rows.ndim != 2
            or rows.shape[1] != 7
            or not 1 <= len(rows) <= self.max_waypoints
        ):
            raise ValueError("invalid_action: expected a bounded Nx7 chunk")
        if not np.isfinite(rows).all():
            raise ValueError("invalid_action: nonfinite value")
        if np.any(np.abs(rows[:, 3:6]) > 1e-8):
            raise ValueError(
                "unsupported_orientation: only reset orientation is supported"
            )
        if np.any(rows[:, :3] < self.low) or np.any(rows[:, :3] > self.high):
            raise ValueError("workspace_reject")
        if np.any(rows[:, 6] < 0) or np.any(rows[:, 6] > 1):
            raise ValueError("invalid_gripper")
        distances = np.linalg.norm(
            np.diff(np.vstack((current[:3], rows[:, :3])), axis=0), axis=1
        )
        if np.any(distances > self.waypoint_m + 1e-7):
            raise ValueError("waypoint_displacement_exceeded")
        if float(distances.sum()) > self.chunk_path_m + 1e-7:
            raise ValueError("chunk_path_exceeded")


def aperture_to_joints(
    open_fraction: float, travel: float = 0.04
) -> tuple[float, float]:
    """Upstream authored joints are zero open, +/-travel closed."""
    if not np.isfinite(open_fraction) or not 0 <= open_fraction <= 1:
        raise ValueError("invalid_gripper")
    closure = (1.0 - open_fraction) * travel
    return -closure, closure


def measured_aperture(left: float, right: float, travel: float = 0.04) -> float:
    """Normalized measured opening; do not replace contact-limited position by command."""
    return 1.0 - (right - left) / (2 * travel)


def measured_motion(
    previous_joints: NDArray,
    joints: NDArray,
    previous_pose: NDArray,
    pose: NDArray,
    dt: float,
) -> NDArray:
    """Actual substep motion: joint, translation and quaternion angular speeds.

    PhysX split-impulse velocity fields are not the transform-integration
    velocities under contact. Never substitute a control-rate average: sample
    every physics substep so opposing excursions cannot cancel.
    """
    if (
        not np.isfinite(dt)
        or dt <= 0
        or previous_joints.shape != joints.shape
        or joints.ndim != 1
        or joints.size == 0
        or previous_pose.shape != (7,)
        or pose.shape != (7,)
        or not all(
            np.isfinite(a).all() for a in (previous_joints, joints, previous_pose, pose)
        )
    ):
        raise ValueError("invalid_motion_sample")
    q0, q1 = previous_pose[3:].astype(float), pose[3:].astype(float)
    norms = np.array([np.linalg.norm(q0), np.linalg.norm(q1)])
    if np.any(np.abs(norms - 1) > 0.001):
        raise ValueError("invalid_motion_quaternion")
    q0, q1 = q0 / norms[0], q1 / norms[1]
    vector = q0[0] * q1[1:] - q1[0] * q0[1:] + np.cross(q0[1:], q1[1:])
    angle = 2 * np.arctan2(np.linalg.norm(vector), abs(np.dot(q0, q1)))
    return np.array(
        [
            np.max(np.abs(joints - previous_joints)) / dt,
            np.linalg.norm(pose[:3] - previous_pose[:3]) / dt,
            angle / dt,
        ]
    )


def settling_sample_valid(
    measured: NDArray, reported: NDArray, *, released: bool
) -> bool:
    """Keep the speed limits and additionally require native velocities on release."""
    if measured.shape != (3,) or reported.shape != (3,):
        raise ValueError("invalid_settling_sample")
    if not np.isfinite(measured).all() or not np.isfinite(reported).all():
        return False
    limits = np.array([0.02, 0.01, 0.1])
    return bool(
        np.all(measured < limits) and (not released or np.all(reported < limits))
    )


class Freshness:
    """Accept only new coherent captures; static pixels alone are not stale."""

    def __init__(self) -> None:
        self.generation = 0
        self.time = -1.0
        self.frames: dict[str, int] = {}

    def accept(
        self,
        generation: int,
        sim_time: float,
        images: dict[str, NDArray],
        frames: dict[str, int],
    ) -> None:
        if (
            generation <= self.generation
            or not np.isfinite(sim_time)
            or sim_time <= self.time
        ):
            raise ValueError("stale_observation")
        if set(images) != set(CAMERAS):
            raise ValueError("missing_camera")
        if set(frames) != set(CAMERAS) or any(
            type(frame) is not int or frame <= self.frames.get(name, -1)
            for name, frame in frames.items()
        ):
            raise ValueError("stale_camera_frame")
        for name, shape in CAMERAS.items():
            if images[name].shape != shape or images[name].dtype != np.uint8:
                raise ValueError("invalid_camera_tensor")
        self.generation, self.time = generation, sim_time
        self.frames = frames.copy()
