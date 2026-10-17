"""Run the modular six-axis Leader to Nova teleoperation path."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from episode_recorder import CameraRecorder, CameraSpec
from nova_client import NovaClient
from nova_realtime import NovaRealtimeReader
from teach_config import RepeatConfig, TeachError, load_settings
from teach_controls import TeachSession, TerminalKeys
from teleop_config import (
    DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG,
    DEFAULT_TELEOP_CONFIG,
)
from teleop_controller import TeleopController
from teleop_telemetry import TelemetryWriter, make_trace_path


def read_leader_with_deadline(reader, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = reader.read_once()
        if sample is not None:
            return sample
        time.sleep(0.01)
    raise RuntimeError(f"Cannot read Leader: {reader.last_error}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="explicitly allow hardware teleoperation"
    )
    parser.add_argument("--settings", type=Path, help="existing local teleop_config.py")
    parser.add_argument("--teach-dir", type=Path, help="enable R/E episode recording")
    vector = ("J1", "J2", "J3", "J4", "J5", "J6")
    parser.add_argument(
        "--max-offset",
        type=float,
        nargs=6,
        metavar=vector,
        default=None,
        help="maximum mapped offset in degrees",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        nargs=6,
        metavar=vector,
        default=None,
        help="maximum command speed in degrees/second",
    )
    parser.add_argument(
        "--scale",
        type=float,
        nargs=6,
        metavar=vector,
        default=None,
        help="relative Leader-to-Nova scale",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        nargs=6,
        metavar=vector,
        default=None,
        help="mapped joint deadband in degrees",
    )
    parser.add_argument(
        "--clutch-key",
        choices=("none", "a", "b"),
        default="none",
        help="hold this verified Leader button for clutch",
    )
    parser.add_argument(
        "--dt", type=float, default=None, help="ServoJ loop period in seconds"
    )
    parser.add_argument(
        "--response-profile",
        choices=("standard", "responsive"),
        default="standard",
        help=(
            "rate-limit profile; responsive is an explicit live-test option "
            "and keeps the same joint offset safety limit"
        ),
    )
    parser.add_argument(
        "--servo-t",
        type=float,
        default=None,
        help="ServoJ point time in seconds (default comes from teleop_config)",
    )
    parser.add_argument(
        "--servo-aheadtime",
        type=float,
        default=None,
        help="ServoJ aheadtime parameter (20-100)",
    )
    parser.add_argument(
        "--servo-gain",
        type=float,
        default=None,
        help="ServoJ gain parameter (200-1000)",
    )
    parser.add_argument(
        "--gripper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="connect the locally configured Feetech gripper",
    )
    parser.add_argument(
        "--gripper-enable-torque",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable gripper torque after Enter (explicit opt-in); "
            "--no-gripper-enable-torque selects read-only operation"
        ),
    )
    parser.add_argument(
        "--episode-label",
        default=None,
        help="optional label saved in the telemetry metadata (for example empty-close)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "record a camera into the episode; repeat for multiple cameras "
            "(for example front=/dev/v4l/by-id/...)"
        ),
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="optional requested camera width",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="optional requested camera height",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=30.0,
        help="camera capture rate (default: 30)",
    )
    parser.add_argument(
        "--camera-jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality from 1 to 100 (default: 90)",
    )
    args = parser.parse_args(argv)
    if args.gripper_enable_torque and not args.gripper:
        parser.error("--gripper-enable-torque requires --gripper")
    if args.teach_dir and not args.gripper:
        parser.error("--teach-dir requires --gripper")
    return args


def parse_camera_specs(args) -> list[CameraSpec]:
    """Parse repeated NAME=PATH options without touching hardware."""
    specs = []
    names = set()
    for value in args.camera:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--camera must use NAME=PATH, got: {value!r}")
        if name in names:
            raise ValueError(f"duplicate camera name: {name}")
        names.add(name)
        specs.append(
            CameraSpec(
                name=name,
                path=path,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                jpeg_quality=args.camera_jpeg_quality,
            )
        )
    return specs


def make_teleop_config(args, base=DEFAULT_TELEOP_CONFIG):
    values: dict[str, Any] = {
        "clutch_key": None if args.clutch_key == "none" else args.clutch_key,
    }
    if args.response_profile == "responsive":
        # This is deliberately an opt-in profile.  The offset clamp stays at
        # the standard value until the Nova joint limits are measured.
        values["max_speed_deg_s"] = (35.0, 35.0, 35.0, 45.0, 45.0, 45.0)
    if args.max_offset is not None:
        values["max_offset_deg"] = tuple(args.max_offset)
    if args.max_speed is not None:
        values["max_speed_deg_s"] = tuple(args.max_speed)
    if args.scale is not None:
        values["scale"] = tuple(args.scale)
    if args.deadband is not None:
        values["deadband_deg"] = tuple(args.deadband)
    if args.dt is not None:
        values["dt_s"] = args.dt
    if args.servo_t is not None:
        values["servo_t"] = args.servo_t
    if args.servo_aheadtime is not None:
        values["servo_aheadtime"] = args.servo_aheadtime
    if args.servo_gain is not None:
        values["servo_gain"] = args.servo_gain
    return replace(base, **values)


def clutch_state(sample, clutch_key):
    """Return button state, or None when the raw key packet is unavailable.

    The upstream Dynamixel driver interprets a released button as 1 and a
    pressed button as 0. This is only applied when A/B is explicitly selected.
    """
    if clutch_key is None or sample.keys is None or len(sample.keys) < 2:
        return None
    index = 0 if clutch_key == "a" else 1
    return sample.keys[index] == 0


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    camera_specs = parse_camera_specs(args)
    settings = load_settings(args.settings)
    teleop_config = make_teleop_config(args, settings.DEFAULT_TELEOP_CONFIG)
    if not args.execute:
        print(
            "Offline options validated. Hardware requires --execute and configured teleop settings."
        )
        return 0
    if not sys.stdin.isatty():
        raise TeachError("--execute requires an interactive terminal")
    nova_config = settings.DEFAULT_NOVA_CONFIG
    leader_config = settings.DEFAULT_LEADER_CONFIG
    local_config = RepeatConfig.from_teleop(
        nova_config,
        settings.DEFAULT_GRIPPER_CONFIG,
        teleop_config,
        getattr(settings, "REPEAT_CHECKS", None),
    )
    if not leader_config.port or "SET_" in leader_config.port:
        raise TeachError("Set the Leader port in your existing teleop_config.py")
    n = local_config.data["nova"]
    gripper_config = asdict(settings.DEFAULT_GRIPPER_CONFIG)
    from gripper_diagnostics import GripperDiagnostic
    from leader_reader import LeaderReader
    from nova_gripper import NovaGripper

    leader = LeaderReader(leader_config)
    nova = NovaClient(nova_config)
    controller = TeleopController(teleop_config)
    gripper = NovaGripper(**gripper_config) if args.gripper else None
    gripper_diagnostic = (
        GripperDiagnostic(
            gripper_config["open_position"],
            gripper_config["closed_position"],
            DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG,
        )
        if gripper is not None
        else None
    )
    gripper_torque_enabled_by_us = False
    motion_started = False
    telemetry = None
    camera_recorders = []
    feedback = NovaRealtimeReader(
        nova_config.ip, n["feedback_port"], n["socket_timeout_s"]
    )
    teach = TeachSession(args.teach_dir, local_config) if args.teach_dir else None
    keys = TerminalKeys() if teach else None

    try:
        leader.open()
        if gripper is not None:
            gripper_status = gripper.open()
            print(
                "Gripper detected: "
                f"ID={gripper_status.servo_id} "
                f"model={gripper_status.model_number} "
                f"position={gripper_status.position} "
                f"torque={'on' if gripper_status.torque_enabled else 'off'}"
            )
            if not args.gripper_enable_torque:
                print(
                    "Gripper command writes are disabled. "
                    "Use --gripper-enable-torque for trigger control."
                )
        nova.connect()
        nova.require_mode()

        nova.get_angle()
        read_leader_with_deadline(leader, 2.0)

        print()
        print("Modular Nova teleoperation preflight")
        print("RobotMode is 5 (ENABLE idle).")
        print("No EnableRobot() or ClearError() will be sent.")
        print("Keep the Nova work area clear and the emergency stop reachable.")
        print(f"Clutch button: {args.clutch_key}")
        if gripper is not None and args.gripper_enable_torque:
            print("Enter also authorizes gripper torque and trigger control.")
        print("Press Enter to anchor and start; Ctrl+C stops.")
        input()

        # Recheck both endpoints immediately before allowing motion.
        nova.require_mode()
        leader_anchor = read_leader_with_deadline(leader, 1.0)
        nova_anchor = nova.get_angle()
        gripper_arm_trigger = (
            leader_anchor.trigger_normalized if gripper is not None else None
        )
        gripper_trigger_armed = False
        controller.anchor(leader_anchor.joints_deg, nova_anchor)
        trace_path = make_trace_path()
        episode_dir = Path(trace_path).with_suffix("")
        telemetry = TelemetryWriter(
            trace_path,
            metadata={
                "episode": {
                    "label": args.episode_label,
                    "trace_path": trace_path,
                    "episode_dir": str(episode_dir),
                },
                "leader_port": leader_config.port,
                "nova_ip": nova_config.ip,
                "nova_port": nova_config.dashboard_port,
                "leader_zero_deg": list(leader_anchor.joints_deg),
                "nova_zero_deg": list(nova_anchor),
                "teleop_config": {
                    "dt_s": teleop_config.dt_s,
                    "max_offset_deg": list(teleop_config.max_offset_deg),
                    "max_speed_deg_s": list(teleop_config.max_speed_deg_s),
                    "scale": list(teleop_config.scale),
                    "deadband_deg": list(teleop_config.deadband_deg),
                    "clutch_key": teleop_config.clutch_key,
                    "leader_sample_hz": leader_config.sample_hz,
                    "response_profile": args.response_profile,
                    "servo_t": teleop_config.servo_t,
                    "servo_aheadtime": teleop_config.servo_aheadtime,
                    "servo_gain": teleop_config.servo_gain,
                },
                "gripper": (
                    {
                        "port": gripper_config["port"],
                        "baudrate": gripper_config["baudrate"],
                        "servo_id": gripper_config["servo_id"],
                        "open_position": gripper_config["open_position"],
                        "closed_position": gripper_config["closed_position"],
                        "input_deadband": gripper_config["input_deadband"],
                        "torque_requested": args.gripper_enable_torque,
                        "episode_label": args.episode_label,
                        "diagnostic": {
                            "endpoint_tolerance": DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG.endpoint_tolerance,
                            "tracking_error_threshold": DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG.tracking_error_threshold,
                            "candidate_current_threshold": DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG.candidate_current_threshold,
                            "candidate_persistence_s": DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG.candidate_persistence_s,
                            "target_change_threshold": DEFAULT_GRIPPER_DIAGNOSTIC_CONFIG.target_change_threshold,
                            "control_action": "none",
                        },
                    }
                    if gripper is not None
                    else None
                ),
                "cameras": [spec.metadata() for spec in camera_specs],
            },
        )
        print(f"Telemetry trace: {trace_path}")

        for spec in camera_specs:
            recorder = CameraRecorder(
                spec,
                episode_dir,
                telemetry.submit,
            )
            recorder.start()
            camera_recorders.append(recorder)
            print(
                f"Camera recording active: {spec.name} -> "
                f"{episode_dir / 'cameras' / spec.name}"
            )

        feedback.connect()
        feedback.start()
        feedback_deadline = time.monotonic() + 1.0
        while feedback.state().latest is None and time.monotonic() < feedback_deadline:
            time.sleep(0.01)
        if feedback.state().latest is None:
            raise RuntimeError(
                f"Nova 30004 feedback unavailable: {feedback.state().error}"
            )
        leader.start()
        if keys is not None:
            keys.open()
            print("待錄製: 靜止後按 R 開始、E 結束；Ctrl+C 停止遙操")
        motion_started = True
        last_cycle = time.monotonic()
        last_leader_ok = last_cycle
        # A released button is the safe initial state. If the operator is
        # already holding the selected button, the first valid sample changes
        # this to True and enters clutch without moving Nova.
        clutch_pressed_state = False
        key_missing_since = None
        last_gripper_feedback = 0.0
        last_gripper_target_position = None
        last_gripper_status = None
        last_gripper_feedback_ns = None
        last_gripper_feedback_error = None
        if gripper is not None and args.gripper_enable_torque:
            if not gripper_status.torque_enabled:
                gripper.enable_torque()
                gripper_torque_enabled_by_us = True
            print(
                "Gripper torque enabled explicitly. Move the trigger by "
                f">={gripper_config['input_deadband']:.2f} to arm it; "
                "this prevents a startup jump."
            )
            print(
                "Gripper mapping is continuous: "
                f"0.0 -> {gripper_config['open_position']}, "
                f"0.5 -> "
                f"{(gripper_config['open_position'] + gripper_config['closed_position']) // 2}, "
                f"1.0 -> {gripper_config['closed_position']}."
            )
        print("Teleoperation active.")
        print(
            "Response settings: "
            f"speed={list(teleop_config.max_speed_deg_s)}, "
            f"t={teleop_config.servo_t:g}, "
            f"aheadtime={teleop_config.servo_aheadtime:g}, "
            f"gain={teleop_config.servo_gain:g}"
        )
        if teleop_config.clutch_key is None:
            print("Clutch disabled (no physical button selected).")
        else:
            print(
                f"Clutch enabled: hold Leader button {teleop_config.clutch_key.upper()} "
                "(pressed=0 according to upstream packet convention)."
            )

        while True:
            cycle_start = time.monotonic()
            cycle_start_ns = time.monotonic_ns()
            camera_error = next(
                (
                    f"{recorder.spec.name}: {recorder.error}"
                    for recorder in camera_recorders
                    if recorder.error is not None
                ),
                None,
            )
            if camera_error is not None:
                print(f"Camera recorder failed; stopping Nova: {camera_error}")
                break
            sample = leader.latest(teleop_config.stale_timeout_s)

            if sample is None:
                if time.monotonic() - last_leader_ok > teleop_config.stale_timeout_s:
                    print("Leader stale; stopping Nova.")
                    cycle_end_ns = time.monotonic_ns()
                    telemetry.submit(
                        {
                            "record_type": "sample",
                            "wall_time_ns": time.time_ns(),
                            "t_cycle_start": cycle_start_ns,
                            "t_leader_read_start": None,
                            "t_leader_read_done": None,
                            "t_target_compute_done": None,
                            "t_servoj_send": None,
                            "t_servoj_response": None,
                            "t_cycle_end": cycle_end_ns,
                            "status": "leader_stale",
                            "whole_loop_ms": (cycle_end_ns - cycle_start_ns)
                            / 1_000_000,
                        }
                    )
                    break
                time.sleep(0.005)
                continue

            last_leader_ok = time.monotonic()
            feedback_sample = feedback.latest(teleop_config.stale_timeout_s)
            if feedback_sample is None:
                print("Nova feedback stale; stopping Nova.")
                break
            if feedback_sample.robot_mode not in (5, 7, 8):
                print(
                    "Nova entered unexpected RobotMode "
                    f"{feedback_sample.robot_mode}; stopping Nova."
                )
                break

            if teach is not None:
                assert gripper is not None and keys is not None
                assert controller.commanded_deg is not None
                teach.poll_hardware(
                    keys,
                    feedback,
                    gripper,
                    list(controller.commanded_deg),
                    last_gripper_target_position,
                )

            observed_clutch = clutch_state(sample, teleop_config.clutch_key)
            if teleop_config.clutch_key is not None:
                if sample.keys is None:
                    if key_missing_since is None:
                        key_missing_since = time.monotonic()
                    elif (
                        time.monotonic() - key_missing_since
                        > teleop_config.stale_timeout_s
                    ):
                        print("Leader clutch button data stale; stopping Nova.")
                        break
                else:
                    key_missing_since = None
            if observed_clutch is not None and observed_clutch != clutch_pressed_state:
                controller.set_clutch(
                    observed_clutch,
                    leader_deg=sample.joints_deg,
                    nova_reference_deg=controller.commanded_deg,
                )
                clutch_pressed_state = observed_clutch
                print(
                    "Clutch "
                    + (
                        "pressed; holding Nova."
                        if observed_clutch
                        else "released; re-anchored."
                    )
                )
            actual_dt = max(last_leader_ok - last_cycle, 0.001)
            output = controller.update(
                sample.joints_deg,
                actual_dt_s=actual_dt,
            )
            last_cycle = last_leader_ok
            target_compute_ns = time.monotonic_ns()

            local_config.joints_in_range(list(output.commanded_deg))
            servo_timing: dict[str, Any] = {}
            response = nova.servo_j(
                output.commanded_deg,
                t=teleop_config.servo_t,
                aheadtime=teleop_config.servo_aheadtime,
                gain=teleop_config.servo_gain,
                timing=servo_timing,
            )
            status = "ok" if response.error_id == 0 else "servoj_rejected"
            if response.error_id != 0:
                print(f"ServoJ rejected: {response.raw}")

            if teach is not None:
                teach.arm_command(
                    output.commanded_deg, teleop_config, response, servo_timing
                )
            if response.error_id != 0:
                break
            gripper_timing: dict[str, Any] = {}
            gripper_target_position = None
            gripper_diagnostic_state = None
            diagnostic_target = last_gripper_target_position
            if gripper is not None:
                if args.gripper_enable_torque:
                    if (
                        not gripper_trigger_armed
                        and gripper_arm_trigger is not None
                        and abs(sample.trigger_normalized - gripper_arm_trigger)
                        >= gripper_config["input_deadband"]
                    ):
                        gripper_trigger_armed = True
                        print("Gripper trigger armed.")
                    if gripper_trigger_armed:
                        gripper_target_position = gripper.command_normalized(
                            sample.trigger_normalized, timing=gripper_timing
                        )
                        if gripper_target_position is not None:
                            last_gripper_target_position = gripper_target_position
                            diagnostic_target = gripper_target_position
                            if teach is not None:
                                teach.gripper_command(
                                    gripper_target_position, gripper_timing
                                )
                if time.monotonic() - last_gripper_feedback >= 0.20:
                    try:
                        last_gripper_status = gripper.read_status()
                        last_gripper_feedback_ns = time.monotonic_ns()
                        last_gripper_feedback_error = None
                    except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
                        # Keep the control trace useful if an optional status
                        # register is unavailable; command errors still abort.
                        last_gripper_feedback_error = repr(exc)
                    last_gripper_feedback = time.monotonic()

                if gripper_diagnostic is not None:
                    gripper_diagnostic_state = gripper_diagnostic.update(
                        diagnostic_target,
                        last_gripper_status,
                    )

            remaining = teleop_config.dt_s - (time.monotonic() - cycle_start)
            if remaining > 0:
                time.sleep(remaining)
            cycle_end_ns = time.monotonic_ns()
            telemetry.submit(
                {
                    "record_type": "sample",
                    "wall_time_ns": time.time_ns(),
                    "t_cycle_start": cycle_start_ns,
                    "t_leader_read_start": sample.read_start_ns,
                    "t_leader_read_done": sample.timestamp_ns,
                    "t_target_compute_done": target_compute_ns,
                    "t_servoj_send": servo_timing.get("t_servoj_send"),
                    "t_servoj_response": servo_timing.get("t_servoj_response"),
                    "t_cycle_end": cycle_end_ns,
                    "leader_deg": list(sample.joints_deg),
                    "leader_trigger_raw": sample.trigger_raw,
                    "leader_trigger_deg": sample.trigger_deg,
                    "leader_trigger_normalized": sample.trigger_normalized,
                    "leader_keys": list(sample.keys)
                    if sample.keys is not None
                    else None,
                    "delta_deg": list(output.delta_deg),
                    "desired_deg": list(output.desired_deg),
                    "commanded_deg": list(output.commanded_deg),
                    "servoj_command": response.command,
                    "leader_read_ms": sample.read_ms,
                    "leader_age_ms": (cycle_start_ns - sample.timestamp_ns) / 1_000_000,
                    "compute_ms": (target_compute_ns - cycle_start_ns) / 1_000_000,
                    "servoj_rtt_ms": (
                        (
                            servo_timing["t_servoj_response"]
                            - servo_timing["t_servoj_send"]
                        )
                        / 1_000_000
                        if "t_servoj_response" in servo_timing
                        else None
                    ),
                    "gripper_trigger_normalized": sample.trigger_normalized,
                    "gripper_target_position": (diagnostic_target),
                    "gripper_actual_position": (
                        last_gripper_status.position
                        if last_gripper_status is not None
                        else None
                    ),
                    "gripper_feedback_mono_ns": last_gripper_feedback_ns,
                    "gripper_feedback_age_ms": (
                        (cycle_end_ns - last_gripper_feedback_ns) / 1_000_000
                        if last_gripper_feedback_ns is not None
                        else None
                    ),
                    "gripper_tracking_error": (
                        diagnostic_target - last_gripper_status.position
                        if diagnostic_target is not None
                        and last_gripper_status is not None
                        else None
                    ),
                    "gripper_diagnostic_state": gripper_diagnostic_state,
                    "gripper_status": (
                        {
                            "position": last_gripper_status.position,
                            "torque_enabled": last_gripper_status.torque_enabled,
                            "moving": last_gripper_status.moving,
                            "load": last_gripper_status.load,
                            "current": last_gripper_status.current,
                            "voltage_raw": last_gripper_status.voltage_raw,
                            "temperature_raw": last_gripper_status.temperature_raw,
                        }
                        if last_gripper_status is not None
                        else None
                    ),
                    "gripper_feedback_error": last_gripper_feedback_error,
                    "gripper_rtt_ms": (
                        (
                            gripper_timing["t_gripper_response"]
                            - gripper_timing["t_gripper_send"]
                        )
                        / 1_000_000
                        if "t_gripper_response" in gripper_timing
                        else None
                    ),
                    "whole_loop_ms": (cycle_end_ns - cycle_start_ns) / 1_000_000,
                    "actual_loop_hz": 1_000_000_000
                    / max(cycle_end_ns - cycle_start_ns, 1),
                    "actual_dt_s": output.actual_dt_s,
                    "clutch_active": output.clutch_active,
                    "rate_limited": output.rate_limited,
                    "status": status,
                    "nova_feedback": (
                        {
                            "robot_mode": feedback_sample.robot_mode,
                            "controller_timestamp_ms": feedback_sample.controller_timestamp_ms,
                            "q_target_deg": list(feedback_sample.q_target_deg),
                            "q_actual_deg": list(feedback_sample.q_actual_deg),
                            "tracking_error_deg": [
                                t - q
                                for t, q in zip(
                                    feedback_sample.q_target_deg,
                                    feedback_sample.q_actual_deg,
                                )
                            ],
                            "host_age_ms": (
                                cycle_end_ns - feedback_sample.host_timestamp_ns
                            )
                            / 1_000_000,
                        }
                        if feedback_sample is not None
                        else None
                    ),
                    "nova_target_deg": list(output.commanded_deg),
                    "nova_actual_deg": (
                        list(feedback_sample.q_actual_deg)
                        if feedback_sample is not None
                        else None
                    ),
                    "nova_feedback_mono_ns": (
                        feedback_sample.host_timestamp_ns
                        if feedback_sample is not None
                        else None
                    ),
                    "nova_feedback_age_ms": (
                        (cycle_end_ns - feedback_sample.host_timestamp_ns) / 1_000_000
                        if feedback_sample is not None
                        else None
                    ),
                }
            )
            if response.error_id != 0:
                break

    except KeyboardInterrupt:
        print("\nCtrl+C received.")
    finally:
        if teach is not None:
            teach.abort("teleoperation stopped before E")
        if motion_started:
            try:
                print("Sending Stop() ...")
                print(nova.stop().raw)
            except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
                print(f"Stop failed: {exc}")
        if keys is not None:
            try:
                keys.close()
            except Exception as exc:  # noqa: BLE001 - continue cleanup after terminal failure
                print(f"Terminal restore failed: {exc}")
        if teach is not None:
            teach.close()
        for recorder in camera_recorders:
            try:
                summary = recorder.stop()
                if telemetry is not None:
                    telemetry.submit(summary)
                print(
                    f"Camera {recorder.spec.name}: "
                    f"captured={summary['frames_captured']} "
                    f"written={summary['frames_written']} "
                    f"dropped={summary['frames_dropped']}"
                )
            except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
                print(f"Camera {recorder.spec.name} shutdown failed: {exc}")
        if telemetry is not None:
            telemetry.stop()
        feedback.close()
        if gripper is not None:
            if gripper_torque_enabled_by_us:
                try:
                    gripper.disable_torque()
                    print("Gripper torque disabled.")
                except Exception as exc:  # noqa: BLE001 - any I/O failure must terminate/record the run
                    print(f"Gripper torque disable failed: {exc}")
            gripper.close()
        leader.close()
        nova.close()
        print("Connections closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
