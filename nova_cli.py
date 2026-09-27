#!/usr/bin/env python3
"""Small manual Nova Dashboard terminal.

This tool never sends a command on startup. It only sends an allow-listed
command after the operator types it at the ``Nova>`` prompt.
"""

from __future__ import annotations

import argparse
import socket


DEFAULT_IP = "SET_NOVA_IP"
DEFAULT_PORT = 29999
ALLOWED_COMMANDS = {
    "RobotMode()",
    "GetAngle()",
    "GetErrorID()",
    "ClearError()",
    "StartDrag()",
    "StopDrag()",
}

# These commands can change controller state. They still require the exact
# command at the prompt, and the extra confirmation prevents an accidental
# paste from immediately changing a live robot.
CONFIRMATION_REQUIRED = {
    "ClearError()": "CLEARERROR",
}


def receive_response(sock: socket.socket) -> str:
    data = bytearray()
    while b";" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Nova closed the Dashboard connection")
        data.extend(chunk)
    response, _remainder = bytes(data).split(b";", 1)
    return response.decode("ascii", errors="replace") + ";"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default=DEFAULT_IP)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Connecting to Nova Dashboard {args.ip}:{args.port} ...")
    sock = socket.create_connection((args.ip, args.port), timeout=2.0)
    sock.settimeout(2.0)
    print("Connected. No command was sent automatically.")
    print("Allowed: " + ", ".join(sorted(ALLOWED_COMMANDS)))
    print("Type quit or exit to close.")

    try:
        while True:
            try:
                text = input("Nova> ").strip()
            except EOFError:
                print()
                break
            if text.lower() in {"quit", "exit"}:
                break
            if text not in ALLOWED_COMMANDS:
                print("Command not allowed. Use one of:")
                print("  " + "\n  ".join(sorted(ALLOWED_COMMANDS)))
                continue
            confirmation_word = CONFIRMATION_REQUIRED.get(text)
            if confirmation_word is not None:
                print(
                    "ClearError() changes controller state. Only continue after "
                    "the collision cause is removed and the work area is safe."
                )
                confirmation = input(
                    f"Type {confirmation_word} to send {text}, or anything else to cancel: "
                ).strip()
                if confirmation != confirmation_word:
                    print("Cancelled; no command sent.")
                    continue
            sock.sendall(text.encode("ascii"))
            print("<=" , receive_response(sock))
    except KeyboardInterrupt:
        print("\nClosed.")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
