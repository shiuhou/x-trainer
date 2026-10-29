"""Official Inspect Robots contract over the upstream X-Trainer Isaac scene."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from inspect_robots import (
    ActionSemantics,
    Box,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    StateField,
    StateSpec,
    CameraSpec,
)

from .contracts import (
    CAMERAS,
    LABELS,
    TRACKING_POSITION_M,
    Freshness,
    Limits,
    aperture_to_joints,
    measured_aperture,
    measured_motion,
    settling_sample_valid,
    validate_closure_sequence,
)


class XTrainerEmbodiment:
    def __init__(self, env: Any, output: Path, *, open_only: bool = False):
        import torch
        from isaaclab.utils import math as math_utils
        from .task import RIGHT_BASE_IN_ROOT, GRASP_IN_WRIST, GRIPPING_REGION_WRIST_Z

        self.torch, self.math, self.env = torch, math_utils, env
        self.grip_midpoint_wrist_z = sum(GRIPPING_REGION_WRIST_Z) / 2
        self.robot = env.scene["robot"]
        self.output, self.open_only = output, open_only
        self.limits = Limits(
            chunk_path_m=0.01 if open_only else 0.30,
            max_waypoints=32 if open_only else 64,
        )
        self.device = env.device
        self.base = torch.tensor([RIGHT_BASE_IN_ROOT], device=self.device)
        self.offset = torch.tensor([GRASP_IN_WRIST], device=self.device)
        self.wrist_id = self.robot.find_bodies("J2_6")[0][0]
        self.left_ids = self.robot.find_joints(
            [f"J1_{i}" for i in range(1, 9)], preserve_order=True
        )[0]
        self.right_ids = self.robot.find_joints(
            [f"J2_{i}" for i in range(1, 7)], preserve_order=True
        )[0]
        self.finger_ids = self.robot.find_joints(["J2_7", "J2_8"], preserve_order=True)[
            0
        ]
        self.left_initial = self.robot.data.joint_pos[:, self.left_ids].clone()
        self.reset_pos, self.reset_quat = self.pose()
        self.reset_pos, self.reset_quat = (
            self.reset_pos.clone(),
            self.reset_quat.clone(),
        )
        self.generation, self.ticks = 0, 0
        self.freshness = Freshness()
        self.observation_trace: list[dict[str, Any]] = []
        self.action_trace: list[dict[str, Any]] = []
        self.pre_check_rejections: list[str] = []
        self.timing = {
            key: 0.0 for key in ("render", "validation", "execute", "settle")
        }
        self.current: Observation | None = None
        self.consumed = False
        self.instruction = ""
        self.step_count = 0
        self.last_gripper_command = 1.0
        self.closed_seen = False
        self.grasp_seen = False
        self.max_left_drift = 0.0
        self.initial_cube = self.env.scene["cube"].data.root_pos_w.clone()
        self.max_lift = 0.0
        self.diagnostic_hook = None
        self.raw_wrist_rgb: np.ndarray | None = None
        self.visual_reference: dict[str, Any] = {}
        self.configure_views()
        self.info = EmbodimentInfo(
            name="xtrainer_isaac_right_simulation",
            action_space=self.action_space(),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(name, shape[0], shape[1])
                    for name, shape in CAMERAS.items()
                ),
                state=StateSpec((StateField("eef_state", (7,), "m,rad,normalized"),)),
            ),
            control_hz=10.0,
            is_simulated=True,
            capabilities=frozenset({"resettable", "renderable"}),
            docs=(
                "Simulation only. Control ONLY the right arm; left arm remains at rest. "
                "move_to is absolute grasp-center position in metres in the right arm base frame. "
                "Here grasp-center denotes the upstream wrist +Z160 mm reference, not the physical contact midpoint. "
                f"The narrow finger gripping region spans wrist Z={GRIPPING_REGION_WRIST_Z} metres; "
                "its midpoint is approximately 22.646 mm below the reported reference in base Z at the fixed orientation. "
                "yaw,pitch,roll are reset-relative radians and must all be zero. "
                "Gripper 0 closes, 1 opens; always explicitly specify gripper for every move. "
                "For grasping, move to the approach/grasp position with gripper=1, "
                "Align X/Y above the object while maintaining clear height before descending; "
                "avoid diagonal insertion near the object. Check wrist_rgb for centering and "
                "right_rgb for clearance; front_rgb alone can hide a finger/object overlap. "
                "Near the object, descend vertically in separate move_to calls of 0.005 to 0.010 m "
                "or less, inspecting a fresh observation after each call. Small interpolated "
                "waypoints inside one long chunk do not provide intermediate policy images. "
                "If either finger overlaps the object, correct alignment while clear before descending. "
                "Then wait for the fresh settled observation. Close in a separate move_to "
                "with only gripper specified (omit XYZ). Moving while newly closing is rejected. "
                "After closure, lift/transport may keep the same closed gripper command. "
                "State gripper is measured position, which can differ from a closed command under contact. "
                "front_rgb and right_rgb are fixed workspace views; wrist_rgb follows the right wrist. "
                "Wrist RGB has a yellow ring projecting the geometric finger contact midpoint, "
                "with +base-X green and +base-Y cyan endpoints 20 mm from that point in its plane. "
                "These robot-only annotations are NOT detected object targets. The image center "
                "is NOT the grip center because the wrist camera is offset and tilted. "
                "Compare object location with the grip-plane ring and axis directions, allowing "
                "for depth/parallax; verify clearance in the other views before descent. "
                f"Maximum Cartesian path per decision {self.limits.chunk_path_m} m, "
                f"maximum {self.limits.max_waypoints} interpolated waypoints. "
                "A failed IK/tracking check aborts execution. After each chunk, physics settles and fresh images arrive. "
                f"Initial rest grasp-center in base metres: {self.reset_pos[0].cpu().tolist()}. "
                "For the cube-to-plate task return to the initial rest pose with open gripper after placing. "
                + (
                    "This connectivity check requires gripper=1 throughout."
                    if open_only
                    else ""
                )
            ),
        )

    def action_space(self) -> Box:
        """Pin open-only commands in the official tool's interpolation space.

        Measured aperture stays measured. The tool clips intermediate commands
        to this fixed bound, so tiny contact/float residuals cannot command a
        partial close during an open-only connectivity check.
        """
        return Box(
            shape=(7,),
            low=np.array([*self.limits.low, 0, 0, 0, 1 if self.open_only else 0]),
            high=np.array([*self.limits.high, 0, 0, 0, 1]),
            semantics=ActionSemantics(
                control_mode="eef_abs_pose",
                rotation_repr="none",
                frame="base",
                gripper="continuous",
                dim_labels=LABELS,
                max_step=(
                    0.005,
                    0.005,
                    0.005,
                    None,
                    None,
                    None,
                    None if self.open_only else 0.2,
                ),
            ),
        )

    def configure_views(self) -> None:
        from .task import FIXED_VIEWS

        t = self.torch
        for name, (eye, lookat) in FIXED_VIEWS.items():
            self.env.scene[name].set_world_poses_from_view(
                t.tensor([eye], device=self.device),
                t.tensor([lookat], device=self.device),
            )

    def pose(self) -> tuple[Any, Any]:
        p, q = self.math.subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.body_pos_w[:, self.wrist_id],
            self.robot.data.body_quat_w[:, self.wrist_id],
        )
        return p + self.math.quat_apply(q, self.offset) - self.base, q

    def state(self) -> np.ndarray:
        p, q = self.pose()
        relative = self.math.quat_mul(self.math.quat_inv(self.reset_quat), q)
        roll, pitch, yaw = self.math.euler_xyz_from_quat(relative)
        rotation = np.array([yaw.item(), pitch.item(), roll.item()])
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi
        fingers = self.robot.data.joint_pos[0, self.finger_ids].cpu().numpy()
        return np.array(
            [*p[0].cpu().numpy(), *rotation, measured_aperture(*fingers)],
            dtype=np.float64,
        )

    def annotate_wrist(self, rgb: np.ndarray) -> np.ndarray:
        """Project only own-link geometry and base axes through the actual camera."""
        from .visual_reference import draw_reference, project_reference

        t = self.torch
        sensor = self.env.scene["wrist_rgb"]
        wrist_p = self.robot.data.body_pos_w[0, self.wrist_id]
        wrist_q = self.robot.data.body_quat_w[0, self.wrist_id]
        midpoint = self.grip_midpoint_wrist_z
        local = t.tensor([0.0, 0.0, midpoint], device=self.device)
        center = wrist_p + self.math.quat_apply(wrist_q, local)
        offsets = t.tensor(
            [[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], device=self.device
        )
        world = center + self.math.quat_apply(
            self.robot.data.root_quat_w.expand(3, -1), offsets
        )
        # TiledCamera in this pinned Isaac version caches world pose at reset.
        # Compose the live parent FK and authored camera offset instead.
        if sensor.cfg.offset.convention != "ros":
            raise RuntimeError("wrist_reference_requires_ros_camera_offset")
        camera_p, camera_q = self.math.combine_frame_transforms(
            wrist_p.reshape(1, 3),
            wrist_q.reshape(1, 4),
            t.tensor([sensor.cfg.offset.pos], device=self.device),
            t.tensor([sensor.cfg.offset.rot], device=self.device),
        )
        camera_points = (
            self.math.quat_apply(
                self.math.quat_inv(camera_q).expand(3, -1),
                world - camera_p,
            )
            .cpu()
            .numpy()
        )
        intrinsics = sensor.data.intrinsic_matrices[0].cpu().numpy()
        pixels = project_reference(camera_points, intrinsics)
        self.visual_reference = {
            "source": "robot_geometry_and_camera_only",
            "plane": "finger_contact_midpoint",
            "pixels": pixels.tolist(),
            "axis_length_m": 0.02,
            "object_state_used": False,
        }
        return draw_reference(rgb, pixels)

    def capture_images(self) -> tuple[dict, dict]:
        """Render current physics state, without advancing simulation."""
        self.env.sim.render()
        images = {}
        frames = {}
        for name in CAMERAS:
            sensor = self.env.scene[name]
            sensor.update(0.0, force_recompute=True)
            images[name] = sensor.data.output["rgb"][0, ..., :3].cpu().numpy().copy()
            frames[name] = int(sensor.frame[0].item())
        self.raw_wrist_rgb = images["wrist_rgb"].copy()
        images["wrist_rgb"] = self.annotate_wrist(images["wrist_rgb"])
        return images, frames

    def diagnostic_capture(self, label: str) -> dict[str, Any]:
        """Terminal evidence only; never publish a policy observation/generation."""
        started = time.monotonic()
        sim_time = float(self.env.sim.current_time)
        images, frames = self.capture_images()
        if self.raw_wrist_rgb is not None:
            Image.fromarray(self.raw_wrist_rgb).save(
                self.output / f"{label}_wrist_raw.png"
            )
        if float(self.env.sim.current_time) != sim_time:
            raise RuntimeError("diagnostic_capture_advanced_physics")
        montage = Image.new("RGB", (1920, 480))
        for index, (name, pixels) in enumerate(images.items()):
            im = Image.fromarray(pixels)
            im.save(self.output / f"{label}_{name}.png")
            montage.paste(im.resize((640, 480)), (index * 640, 0))
        montage.save(self.output / f"{label}_montage.png")
        self.timing["render"] += time.monotonic() - started
        return {
            "kind": "diagnostic_only",
            "policy_usable": False,
            "sim_time": sim_time,
            "ticks": self.ticks,
            "last_policy_generation": self.generation,
            "sensor_frames": frames,
        }

    def observe(self, label: str) -> Observation:
        started = time.monotonic()
        images, frames = self.capture_images()
        if self.raw_wrist_rgb is not None:
            Image.fromarray(self.raw_wrist_rgb).save(
                self.output / f"{label}_wrist_raw.png"
            )
        generation = self.generation + 1
        sim_time = float(self.env.sim.current_time)
        self.freshness.accept(generation, sim_time, images, frames)
        self.generation = generation
        obs = Observation(
            images=images,
            state={"eef_state": self.state()},
            instruction=self.instruction,
            image_times={n: sim_time for n in images},
            state_time=sim_time,
        )
        self.current, self.consumed = obs, False
        self.observation_trace.append(
            {
                "generation": generation,
                "sim_time": sim_time,
                "host_monotonic": time.monotonic(),
                "visual_reference": self.visual_reference,
                "sensor_frames": frames,
                "eef_state": obs.state["eef_state"].tolist(),
                "images": {
                    n: hashlib.sha256(im.tobytes()).hexdigest()
                    for n, im in images.items()
                },
            }
        )
        thumbs = []
        for name, im in images.items():
            image = Image.fromarray(im)
            if label in ("initial", "final"):
                image.save(self.output / f"{label}_{name}.png")
            thumbs.append(image.resize((640, 480)))
        montage = Image.new("RGB", (1920, 480))
        for index, image in enumerate(thumbs):
            montage.paste(image, (640 * index, 0))
        montage.save(self.output / f"{label}_montage.png")
        self.timing["render"] += time.monotonic() - started
        return obs

    def pre_check(self, rows: np.ndarray) -> str | None:
        started = time.monotonic()
        try:
            if self.current is None or self.consumed:
                raise ValueError("stale_observation")
            rows = np.asarray(rows)
            current = self.state()
            self.limits.validate(rows, current)
            validate_closure_sequence(rows, current, self.last_gripper_command)
            if self.open_only and np.any(np.abs(rows[:, 6] - 1) > 1e-8):
                raise ValueError("connectivity_requires_open_gripper")
        except ValueError as exc:
            self.pre_check_rejections.append(str(exc))
            return str(exc)
        finally:
            self.timing["validation"] += time.monotonic() - started
        return None

    def physics(self, row: np.ndarray) -> None:
        t = self.torch
        target = t.tensor(row[:3], device=self.device, dtype=t.float32).reshape(1, 3)
        wrist = target + self.base - self.math.quat_apply(self.reset_quat, self.offset)
        fingers = t.tensor([aperture_to_joints(float(row[6]))], device=self.device)
        command = t.cat((self.left_initial, wrist, self.reset_quat, fingers), dim=-1)
        self.step_command(command)
        self.last_gripper_command = float(row[6])

    def step_command(self, command: Any) -> None:
        """Normal physics and invariant checks, also used by recorded-action replay."""
        t = self.torch
        # Check every native substep, not an averaged before/after control tick.
        # Preserve native velocity fields for diagnostics and released-state checks.
        indices = self.right_ids + self.finger_ids

        def sample():
            return (
                self.robot.data.joint_pos[0, indices].cpu().numpy().copy(),
                self.env.scene["cube"].data.root_state_w[0, :7].cpu().numpy().copy(),
            )

        previous = sample()
        original_update = self.env.scene.update
        speeds = []

        def update(dt):
            nonlocal previous
            original_update(dt)
            current = sample()
            speeds.append(
                measured_motion(previous[0], current[0], previous[1], current[1], dt)
            )
            previous = current

        self.env.scene.update = update
        try:
            self.env.step(command)
        finally:
            self.env.scene.update = original_update
        if len(speeds) != self.env.cfg.decimation:
            raise RuntimeError("incomplete_physics_motion_samples")
        self.last_motion_speeds = np.max(speeds, axis=0)
        self.ticks += 1
        if not t.isfinite(self.robot.data.joint_pos).all():
            raise RuntimeError("execution_nonfinite")
        drift = float(
            t.max(
                t.abs(self.robot.data.joint_pos[:, self.left_ids] - self.left_initial)
            )
        )
        self.max_left_drift = max(self.max_left_drift, drift)
        if drift > 0.02:
            raise RuntimeError("left_arm_drift")
        limits = self.robot.data.soft_joint_pos_limits[0]
        joints = self.robot.data.joint_pos[0]
        if bool(
            t.any(joints < limits[:, 0] - 0.01) | t.any(joints > limits[:, 1] + 0.01)
        ):
            raise RuntimeError("joint_limit_reject")
        lift = float((self.env.scene["cube"].data.root_pos_w - self.initial_cube)[0, 2])
        self.max_lift = max(self.max_lift, lift)
        if self.diagnostic_hook is not None:
            self.diagnostic_hook(command)

    def initialize_settled_scene(self) -> None:
        """Hold the authored rest target while initial contacts settle."""
        target = self.state().copy()
        target[3:6], target[6] = 0, 1
        self.settle(target)
        self.initial_cube = self.env.scene["cube"].data.root_pos_w.clone()
        self.max_lift = 0.0

    def execute(
        self, chunk: Any, observation: Observation, *, minimum_ticks: int = 1
    ) -> Observation:
        if minimum_ticks not in (1, 2):
            raise ValueError("invalid_minimum_ticks")
        if observation is not self.current or self.consumed:
            raise ValueError("stale_observation")
        rows = np.array([action.data for action in chunk.actions], dtype=np.float64)
        rejection = self.pre_check(rows)
        if rejection:
            raise ValueError(rejection)
        self.consumed = True
        for row in rows:
            start = time.monotonic()
            tracking_samples = []
            try:
                for tick in range(30):
                    self.physics(row)
                    state = self.state()
                    tracking_samples.append(
                        {
                            "control_tick": tick + 1,
                            "position_error_m": float(
                                np.linalg.norm(state[:3] - row[:3])
                            ),
                            "orientation_error_rad": float(np.linalg.norm(state[3:6])),
                            "actual": state.tolist(),
                        }
                    )
                    if (
                        tick + 1 >= minimum_ticks
                        and tracking_samples[-1]["position_error_m"]
                        < TRACKING_POSITION_M
                        and tracking_samples[-1]["orientation_error_rad"] < 0.035
                    ):
                        break
                else:
                    self.action_trace.append(
                        {
                            "status": "IK_tracking_failure",
                            "target": row.tolist(),
                            "actual": state.tolist(),
                            "tracking_samples": tracking_samples,
                            "right_joints": self.robot.data.joint_pos[0, self.right_ids]
                            .cpu()
                            .tolist(),
                            "recovery": "aborted_no_automatic_motion",
                        }
                    )
                    raise RuntimeError("IK_tracking_failure")
            finally:
                self.timing["execute"] += time.monotonic() - start
            self.step_count += 1
            self.closed_seen |= bool(row[6] < 0.2)
            self.action_trace.append(
                {
                    "waypoint": self.step_count,
                    "target": row.tolist(),
                    "actual": self.state().tolist(),
                }
            )
        self.settle(rows[-1])
        return self.observe(f"decision_{len(self.observation_trace):02d}")

    def settle(self, target: np.ndarray) -> None:
        start = time.monotonic()
        try:
            self._settle(target)
        finally:
            self.timing["settle"] += time.monotonic() - start

    def _settle(self, target: np.ndarray) -> None:
        # Hold target while fingers reach their setpoint/contact equilibrium.
        stable = 0
        for _ in range(120):
            self.physics(target)
            speed = float(
                self.robot.data.joint_vel[:, self.right_ids + self.finger_ids]
                .abs()
                .max()
            )
            cube = self.env.scene["cube"].data
            # An open/released scene must also satisfy native velocity checks.
            # Do not accept a slowly changing pose as proof of safe release.
            reported = np.array(
                [
                    speed,
                    float(cube.root_lin_vel_w.norm()),
                    float(cube.root_ang_vel_w.norm()),
                ]
            )
            still = settling_sample_valid(
                self.last_motion_speeds, reported, released=target[6] > 0.95
            )
            stable = stable + 1 if still else 0
            if stable >= 10:
                break
        else:
            raise RuntimeError("settle_timeout")
        if np.linalg.norm(self.state()[:3] - target[:3]) > 0.005:
            raise RuntimeError("post_settle_tracking_failure")

    def metrics(self) -> dict[str, Any]:
        from isaaclab.managers import SceneEntityCfg
        from leisaac.tasks.pick_cube.mdp.terminations import task_done
        from leisaac.tasks.pick_cube.mdp.observations import (
            object_grasped,
            put_cube_to_plate,
        )

        kwargs = {
            "object_cfg": SceneEntityCfg("cube"),
            "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
        }
        cube = self.env.scene["cube"].data
        success = bool(
            task_done(self.env, [SceneEntityCfg("cube")], SceneEntityCfg("Plate"))[0]
        )
        settled = (
            float(cube.root_lin_vel_w.norm()) < 0.01
            and float(cube.root_ang_vel_w.norm()) < 0.1
        )
        released = float(self.state()[6]) > 0.95
        grasped = bool(object_grasped(self.env, **kwargs)[0])
        self.grasp_seen |= grasped
        return {
            "upstream_success": success,
            "success": success and released and settled,
            "grasped_now": grasped,
            "grasp_seen": self.grasp_seen,
            "upstream_placed": bool(
                put_cube_to_plate(
                    self.env, plate_cfg=SceneEntityCfg("Plate"), **kwargs
                )[0]
            ),
            "max_lift_m": self.max_lift,
            "closed_command_seen": self.closed_seen,
            "released": released,
            "settled": settled,
            "max_left_joint_drift_rad": self.max_left_drift,
            "cube_pose": cube.root_state_w[0, :7].cpu().tolist(),
            "plate_pose": self.env.scene["Plate"]
            .data.root_state_w[0, :7]
            .cpu()
            .tolist(),
        }
