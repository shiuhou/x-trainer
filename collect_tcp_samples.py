#!/usr/bin/env python3
"""Collect read-only Nova Tool 0 pose samples for offline TCP calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from nova_client import NovaClient
from teleop_config import DEFAULT_NOVA_CONFIG, NovaConfig


def write_samples(path: Path, user: int, tool: int, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"user": user, "tool": tool, "samples": samples}
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ip", default=DEFAULT_NOVA_CONFIG.ip)
    parser.add_argument("--port", type=int, default=DEFAULT_NOVA_CONFIG.dashboard_port)
    parser.add_argument("--user", type=int, default=0)
    parser.add_argument("--tool", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose a new filename")
    client = NovaClient(
        NovaConfig(
            ip=args.ip,
            dashboard_port=args.port,
            socket_timeout_s=DEFAULT_NOVA_CONFIG.socket_timeout_s,
            required_start_mode=DEFAULT_NOVA_CONFIG.required_start_mode,
        )
    )
    samples = []
    try:
        client.connect()
        mode = client.robot_mode()
        print(f"RobotMode={mode}; read-only collection, no motion command will be sent.")
        print(
            "Touch the same fixed reference point with the fingertip in at least "
            "six different orientations. Keep the robot still before each capture."
        )
        while True:
            try:
                input(f"Press Enter to capture sample {len(samples) + 1} (Ctrl+C to finish): ")
            except EOFError:
                break
            pose = client.get_pose(user=args.user, tool=args.tool)
            samples.append({"pose": pose, "host_time_ns": time.time_ns()})
            print(
                f"GetPose(user={args.user},tool={args.tool}): "
                + json.dumps(pose)
            )
            write_samples(args.output, args.user, args.tool, samples)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
    if len(samples) < 4:
        raise SystemExit("Need at least four captured samples; no output was written.")
    write_samples(args.output, args.user, args.tool, samples)
    print(f"Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
