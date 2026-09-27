"""Preflight and supervised J1 ServoJ step identification.

This tool never enables, clears, recovers, or reconnects an active feedback
experiment. Motion requires both --allow-motion and an interactive CONFIRM.
"""

from __future__ import annotations

import argparse
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from nova_client import NovaClient
from nova_realtime import NovaRealtimeReader
from servo_experiment import ExperimentMetadata, ExperimentRecorder, sample_record
from teleop_config import DEFAULT_NOVA_CONFIG


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", choices=("J1",), default="J1")
    parser.add_argument("--amplitude-deg", type=float, default=3.0)
    parser.add_argument("--rate-limit-deg-s", type=float, default=3.0)
    parser.add_argument("--hold-s", type=float, default=1.0)
    parser.add_argument("--return-hold-s", type=float, default=1.0)
    parser.add_argument("--loop-period-s", type=float, default=0.03)
    parser.add_argument("--servo-t", type=float, default=0.1)
    parser.add_argument("--servo-aheadtime", type=float, default=50.0)
    parser.add_argument("--servo-gain", type=float, default=500.0)
    parser.add_argument("--feedback-max-age-s", type=float, default=0.20)
    parser.add_argument(
        "--safety-envelope-deg",
        type=float,
        default=10.0,
        help="abort if feedback leaves this envelope around the initial pose",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/servo_identification/j1_step.jsonl"),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-motion", action="store_true")
    return parser.parse_args(argv)


def _planned(args, initial):
    print("Planned Nova ServoJ identification")
    print(f"  joint={args.joint} amplitude={args.amplitude_deg:g} deg")
    print(
        f"  trajectory=anchor -> +{args.amplitude_deg:g} deg -> hold -> anchor -> hold"
    )
    print(f"  rate_limit={args.rate_limit_deg_s:g} deg/s loop={args.loop_period_s:g} s")
    print(
        f"  ServoJ t={args.servo_t:g} aheadtime={args.servo_aheadtime:g} gain={args.servo_gain:g}"
    )
    print(f"  safety_envelope={args.safety_envelope_deg:g} deg around initial pose")
    print(f"  initial_q={initial}")


def _target(anchor, elapsed, args):
    q = list(anchor)
    a = args.amplitude_deg
    r = args.rate_limit_deg_s
    rise = a / r
    if elapsed < rise:
        q[0] += r * elapsed
    elif elapsed < rise + args.hold_s:
        q[0] += a
    elif elapsed < rise + args.hold_s + rise:
        q[0] += a - r * (elapsed - rise - args.hold_s)
    return q


def main(argv=None) -> int:
    args = parse_args(argv)
    if (
        args.amplitude_deg <= 0
        or args.rate_limit_deg_s <= 0
        or args.loop_period_s <= 0
        or args.safety_envelope_deg <= 0
    ):
        raise SystemExit(
            "amplitude, rate limit, loop period, and safety envelope must be positive"
        )
    nova = NovaClient(DEFAULT_NOVA_CONFIG)
    feedback = NovaRealtimeReader(DEFAULT_NOVA_CONFIG.ip)
    recorder = None
    motion_started = False
    metadata = None
    try:
        nova.connect()
        dashboard_start = time.monotonic_ns()
        mode = nova.robot_mode()
        error_id = nova.get_error_id()
        dashboard_rtt_ms = (time.monotonic_ns() - dashboard_start) / 1_000_000
        if mode != DEFAULT_NOVA_CONFIG.required_start_mode:
            raise RuntimeError(f"preflight requires RobotMode 5, got {mode}")
        if error_id != 0:
            raise RuntimeError(f"preflight found GetErrorID={error_id}")
        initial = nova.get_angle()
        feedback.connect()
        feedback.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                current = feedback.latest(args.feedback_max_age_s)
                break
            except (RuntimeError, TimeoutError):
                time.sleep(0.01)
        else:
            raise RuntimeError("no fresh 30004 feedback during preflight")
        delta = [a - b for a, b in zip(initial, current.q_actual_deg)]
        print(f"RobotMode={mode} GetErrorID={error_id}")
        print(f"Dashboard GetAngle={initial}")
        print(f"30004 q_actual={list(current.q_actual_deg)}")
        print(f"GetAngle - q_actual delta_deg={delta}")
        state = feedback.state()
        elapsed_feedback_s = max(
            (time.monotonic_ns() - state.started_ns) / 1_000_000_000, 1e-9
        )
        print(
            f"feedback_age_ms={current.age_ms(time.monotonic_ns()):.2f} frames={state.frames} estimated_hz={state.frames / elapsed_feedback_s:.1f}"
        )
        print(f"dashboard_preflight_rtt_ms={dashboard_rtt_ms:.2f}")
        _planned(args, initial)
        metadata = ExperimentMetadata(
            experiment_id=uuid.uuid4().hex,
            experiment_name="j1_step",
            created_at=datetime.now(timezone.utc).isoformat(),
            robot_host=DEFAULT_NOVA_CONFIG.ip,
            robot_model="Dobot Nova2",
            joint=args.joint,
            amplitude_deg=args.amplitude_deg,
            rate_limit_deg_s=args.rate_limit_deg_s,
            servo_t=args.servo_t,
            servo_aheadtime=args.servo_aheadtime,
            servo_gain=args.servo_gain,
            requested_loop_period_s=args.loop_period_s,
            initial_q_deg=list(initial),
            safety_envelope_deg=args.safety_envelope_deg,
        )
        if args.preflight_only:
            print("Preflight passed. No ServoJ command was sent.")
            return 0
        if not args.allow_motion:
            raise SystemExit(
                "motion requires --allow-motion; no ServoJ command was sent"
            )
        if input("Type CONFIRM to begin supervised motion: ").strip() != "CONFIRM":
            print("Motion not confirmed; no ServoJ command was sent.")
            return 0
        metadata.experiment_status = "running"
        recorder = ExperimentRecorder(args.output, metadata)
        motion_started = True
        start = time.monotonic()
        total = (
            args.amplitude_deg / args.rate_limit_deg_s
            + args.hold_s
            + args.amplitude_deg / args.rate_limit_deg_s
            + args.return_hold_s
        )
        while time.monotonic() - start <= total:
            loop_start = time.monotonic()
            elapsed = loop_start - start
            sample = feedback.latest(args.feedback_max_age_s)
            if sample.robot_mode not in (5, 7, 8):
                raise RuntimeError(f"unexpected RobotMode {sample.robot_mode}")
            deviations = [
                actual - anchor for actual, anchor in zip(sample.q_actual_deg, initial)
            ]
            if any(abs(value) > args.safety_envelope_deg for value in deviations):
                raise RuntimeError(
                    "q_actual exceeded safety envelope: "
                    f"deviation_deg={deviations}, limit={args.safety_envelope_deg:g}"
                )
            command = _target(initial, elapsed, args)
            timing = {}
            response = nova.servo_j(
                command,
                t=args.servo_t,
                aheadtime=args.servo_aheadtime,
                gain=args.servo_gain,
                timing=timing,
            )
            if response.error_id != 0:
                raise RuntimeError(f"ServoJ rejected ErrorID={response.error_id}")
            loop_start_ns = time.monotonic_ns()
            if not recorder.submit(
                sample_record(
                    t_monotonic_ns=loop_start_ns,
                    experiment_time_s=elapsed,
                    q_command_deg=command,
                    q_target_feedback_deg=list(sample.q_target_deg),
                    q_actual_deg=list(sample.q_actual_deg),
                    servo_send_ns=timing.get("t_servoj_send"),
                    servo_reply_ns=timing.get("t_servoj_response"),
                    feedback_host_timestamp_ns=sample.host_timestamp_ns,
                    feedback_controller_timestamp=sample.controller_timestamp_ms,
                    feedback_age_ms=sample.age_ms(loop_start_ns),
                    robot_mode=sample.robot_mode,
                )
            ):
                raise RuntimeError("experiment recorder queue full")
            if recorder.error is not None:
                raise RuntimeError(f"experiment recorder failed: {recorder.error}")
            time.sleep(max(0.0, args.loop_period_s - (time.monotonic() - loop_start)))
        metadata.experiment_status = "completed"
        return 0
    except KeyboardInterrupt:
        if metadata:
            metadata.experiment_status, metadata.abort_reason = "aborted", "Ctrl+C"
        return 130
    except Exception as exc:
        if metadata:
            metadata.experiment_status, metadata.abort_reason = "aborted", repr(exc)
        print(f"Experiment aborted: {exc}")
        return 1
    finally:
        if motion_started:
            try:
                nova.stop()
            except Exception as exc:
                print(f"Stop() cleanup failed: {exc}")
        if recorder is not None:
            recorder.close()
        feedback.close()
        nova.close()


if __name__ == "__main__":
    raise SystemExit(main())
