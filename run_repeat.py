"""Validate/preflight a new Teach episode; --execute enables one live replay."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from repeat_runtime import RepeatRunner
from teach_config import RepeatConfig, TeachError, load_settings
from teach_episode import load_episode


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--settings", type=Path, help="existing local teleop_config.py")
    parser.add_argument(
        "--gripper-enable-torque",
        action="store_true",
        help="explicitly enable gripper torque during live preflight if off",
    )
    parser.add_argument(
        "--preview-events", type=int, default=10, help="maximum offline event rows"
    )
    parser.add_argument("--report-dir", type=Path, default=Path("repeat_reports"))
    args = parser.parse_args(argv)
    try:
        if args.preview_events < 0:
            raise TeachError("--preview-events must be nonnegative")
        if args.gripper_enable_torque and not args.execute:
            raise TeachError("--gripper-enable-torque requires --execute")
        config = None
        if args.execute or args.settings:
            settings = load_settings(args.settings)
            config = RepeatConfig.from_teleop(
                settings.DEFAULT_NOVA_CONFIG,
                settings.DEFAULT_GRIPPER_CONFIG,
                settings.DEFAULT_TELEOP_CONFIG,
                getattr(settings, "REPEAT_CHECKS", None),
            )
        episode = load_episode(args.episode, config)
        print(
            json.dumps(
                {
                    "source_sha256": episode.sha256,
                    "commands": len(episode.events),
                    "duration_s": episode.duration_ns / 1e9,
                    "start_joints_deg": episode.initial.joints_deg,
                    "start_gripper_position": episode.initial.gripper_position,
                    "mode": "execute" if args.execute else "offline preview",
                },
                indent=2,
            )
        )
        if not args.execute:
            # Offline time/event table; no transport construction or hardware imports.
            for event in episode.events[: args.preview_events]:
                print(
                    json.dumps(
                        {
                            "seq": event["seq"],
                            "at_s": (event["send_ns"] - episode.header["start_ns"])
                            / 1e9,
                            "kind": event["kind"],
                            "target": event.get("joints_deg", event.get("position")),
                        }
                    )
                )
            if len(episode.events) > args.preview_events:
                print(
                    f"... {len(episode.events) - args.preview_events} further commands validated"
                )
            return 0
        assert config is not None
        if not sys.stdin.isatty():
            raise TeachError("--execute requires an interactive terminal")
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.report_dir / f"repeat_{uuid.uuid4().hex}.json"
        # Reserve a writable report before opening any hardware.
        with report_path.open("x", encoding="utf-8") as report_stream:
            transport = None
            report = {
                "schema_version": 1,
                "source_sha256": episode.sha256,
                "status": "aborted",
                "stop_reason": "preflight not completed",
            }
            try:
                from repeat_hardware import HardwareTransport

                transport = HardwareTransport(config)
                runner = RepeatRunner(episode, config, transport)
                report = runner.report
                transport.open(enable_gripper_torque=args.gripper_enable_torque)
                runner.require_start()
                print("Using your existing teleop connection and gripper settings.")
                print(
                    "Start verified. Restore objects, clear the work area, keep E-stop reachable."
                )
                print("Type REPLAY to reproduce this episode once:")
                if input().strip() != "REPLAY":
                    raise TeachError("replay cancelled")
                report = runner.run()
            except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - all preflight failures are reported
                report["stop_reason"] = str(exc) or "Ctrl+C"
            finally:
                if transport is not None:
                    try:
                        transport.close()
                    except Exception as exc:  # noqa: BLE001 - cleanup failures are reported
                        report["close_error"] = str(exc)
                        report["status"] = "aborted"
                json.dump(report, report_stream, allow_nan=False, indent=2)
                report_stream.write("\n")
            print(f"{report['status']}: {report['stop_reason']}; report: {report_path}")
            return 0 if report["status"] == "completed" else 1
    except (TeachError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
