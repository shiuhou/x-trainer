#!/usr/bin/env python3
"""Safely move the Nova to the software joint-zero pose.

This utility only performs a reset when the operator confirms at the prompt.
It never calls PowerOn(), EnableRobot(), ClearError(), or RequestControl().
The controller must already be in TCP control mode and RobotMode()==5.

Run from this directory:

    python3 reset.py

The target is deliberately explicit and can be changed only in the source or
by using the command-line options below.  The default is the Nova software
joint-zero pose: {0, 0, 0, 0, 0, 0} degrees.
"""

from __future__ import annotations

import argparse
import re
import socket
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Sequence


ROBOT_IP = "SET_NOVA_IP"
PORT = 29999
DEFAULT_TARGET_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_ACCELERATION = 5
DEFAULT_VELOCITY = 5
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_TOLERANCE_DEG = 0.5


class NovaProtocolError(RuntimeError):
    """Raised when a Dashboard response is malformed or reports an error."""


@dataclass(frozen=True)
class Response:
    error_id: int
    values: tuple[float, ...]
    raw: str


def parse_response(raw: str) -> Response:
    match = re.match(r"\s*(-?\d+)\s*,", raw)
    if match is None:
        raise NovaProtocolError(f"Cannot parse ErrorID: {raw!r}")

    values_match = re.search(r"\{([^}]*)\}", raw)
    if values_match is None:
        raise NovaProtocolError(f"Cannot parse response values: {raw!r}")

    content = values_match.group(1).strip()
    if not content:
        values: tuple[float, ...] = ()
    else:
        try:
            values = tuple(float(part.strip()) for part in content.split(","))
        except ValueError as exc:
            raise NovaProtocolError(f"Cannot parse response values: {raw!r}") from exc

    return Response(int(match.group(1)), values, raw)


def receive_response(sock: socket.socket) -> str:
    data = bytearray()
    while b";" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Nova closed the Dashboard connection")
        data.extend(chunk)
    raw, _remainder = bytes(data).split(b";", 1)
    return raw.decode("ascii", errors="replace") + ";"


def command(sock: socket.socket, text: str) -> Response:
    if not text or ";" in text or "\n" in text or "\r" in text:
        raise ValueError("Invalid Dashboard command")
    print(f">> {text}")
    sock.sendall(text.encode("ascii"))
    response = parse_response(receive_response(sock))
    print(f"<< {response.raw.strip()}")
    return response


def require_ok(response: Response, operation: str) -> Response:
    if response.error_id != 0:
        raise NovaProtocolError(
            f"{operation} failed with ErrorID={response.error_id}: {response.raw.strip()}"
        )
    return response


def format_joint(values: Iterable[float]) -> str:
    return "{" + ",".join(f"{float(value):.6f}" for value in values) + "}"


def parse_joint(values: Sequence[float], name: str) -> tuple[float, ...]:
    if len(values) != 6:
        raise ValueError(f"{name} must contain exactly six values")
    result = tuple(float(value) for value in values)
    if any(value != value or abs(value) == float("inf") for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def wait_until_complete(
    sock: socket.socket,
    result_id: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        mode_response = require_ok(command(sock, "RobotMode()"), "RobotMode")
        if len(mode_response.values) != 1:
            raise NovaProtocolError(f"Invalid RobotMode response: {mode_response.raw}")
        mode = int(mode_response.values[0])

        if mode in (9, 11):
            raise NovaProtocolError(f"Nova entered fault mode {mode}")

        current_response = require_ok(
            command(sock, "GetCurrentCommandID()"), "GetCurrentCommandID"
        )
        if len(current_response.values) != 1:
            raise NovaProtocolError(
                f"Invalid GetCurrentCommandID response: {current_response.raw}"
            )
        current_id = int(current_response.values[0])

        print(f"Waiting: RobotMode={mode}, command={current_id}/{result_id}")
        if mode == 5 and current_id == result_id:
            return
        time.sleep(0.2)

    raise TimeoutError(
        f"Timed out after {timeout_s:.1f}s waiting for command {result_id}"
    )


def final_angle_check(
    sock: socket.socket,
    target_deg: Sequence[float],
    tolerance_deg: float,
) -> bool:
    response = require_ok(command(sock, "GetAngle()"), "GetAngle")
    if len(response.values) != 6:
        raise NovaProtocolError(f"Invalid GetAngle response: {response.raw}")
    actual = response.values
    errors = [abs(actual[i] - target_deg[i]) for i in range(6)]
    print("Final angle:", " ".join(f"{value: .3f}" for value in actual))
    print("Absolute error:", " ".join(f"{value: .3f}" for value in errors))
    return max(errors) <= tolerance_deg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=ROBOT_IP, help=f"Nova IP (default: {ROBOT_IP})")
    parser.add_argument("--port", type=int, default=PORT, help=f"Dashboard port (default: {PORT})")
    parser.add_argument("--acceleration", type=int, default=DEFAULT_ACCELERATION, choices=range(1, 101))
    parser.add_argument("--velocity", type=int, default=DEFAULT_VELOCITY, choices=range(1, 101))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_DEG)
    parser.add_argument(
        "--target",
        type=float,
        nargs=6,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        default=DEFAULT_TARGET_DEG,
        help="six target joint angles in degrees (default: all zero)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.timeout <= 0 or args.tolerance < 0:
        raise ValueError("timeout must be positive and tolerance nonnegative")
    target_deg = parse_joint(args.target, "target")

    sock: socket.socket | None = None
    motion_started = False
    try:
        print(f"Connecting to Nova Dashboard {args.ip}:{args.port} ...")
        sock = socket.create_connection((args.ip, args.port), timeout=1.0)
        sock.settimeout(2.0)

        mode = require_ok(command(sock, "RobotMode()"), "RobotMode")
        if len(mode.values) != 1 or int(mode.values[0]) != 5:
            raise NovaProtocolError(
                f"Nova must already be RobotMode 5 (ENABLE idle), got {mode.raw.strip()}"
            )

        error = command(sock, "GetErrorID()")
        require_ok(error, "GetErrorID")
        if error.values and any(int(value) != 0 for value in error.values):
            raise NovaProtocolError(f"Nova reports an error: {error.raw.strip()}")

        current = require_ok(command(sock, "GetAngle()"), "GetAngle")
        if len(current.values) != 6:
            raise NovaProtocolError(f"Invalid GetAngle response: {current.raw}")
        print("Current angle:", " ".join(f"{value: .3f}" for value in current.values))
        print("Target angle: ", " ".join(f"{value: .3f}" for value in target_deg))

        path = command(
            sock,
            "CheckOddMovJ("
            f"joint={format_joint(current.values)},"
            f"joint={format_joint(target_deg)},"
            f"a={args.acceleration},v={args.velocity},cp=0)",
        )
        require_ok(path, "CheckOddMovJ")
        if len(path.values) != 1 or int(path.values[0]) != 0:
            raise NovaProtocolError(
                f"CheckOddMovJ rejected the path, result={path.values}: {path.raw.strip()}"
            )

        print()
        print("The path check passed.")
        print("This will move the real Nova to the requested joint angles.")
        print("Keep the workspace clear and keep the emergency stop reachable.")
        input("Press Enter to send MovJ, or Ctrl+C to cancel ... ")

        move = require_ok(
            command(
                sock,
                "MovJ("
                f"joint={format_joint(target_deg)},"
                f"a={args.acceleration},v={args.velocity},cp=0)",
            ),
            "MovJ",
        )
        if len(move.values) != 1 or not move.values[0].is_integer():
            raise NovaProtocolError(f"Invalid MovJ response: {move.raw}")
        result_id = int(move.values[0])
        motion_started = True
        print(f"MovJ accepted with ResultID={result_id}.")
        wait_until_complete(sock, result_id, args.timeout)
        motion_started = False
        print("Nova reports the move complete.")
        if not final_angle_check(sock, target_deg, args.tolerance):
            print("WARNING: final angle is outside the requested tolerance.")
            return 2
        print("Reset complete.")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled by operator.", file=sys.stderr)
        return 130
    except (OSError, NovaProtocolError, TimeoutError, ValueError) as exc:
        print(f"Reset failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if motion_started and sock is not None:
            try:
                print("Attempting Stop() ...", file=sys.stderr)
                command(sock, "Stop()")
            except Exception as exc:  # best effort only
                print(f"Stop failed: {exc}", file=sys.stderr)
        if sock is not None:
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
