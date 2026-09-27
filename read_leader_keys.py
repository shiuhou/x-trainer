#!/usr/bin/env python3
"""Print the verified X-Trainer HAND_RIGHT button packet.

This is a Leader-only diagnostic. It never connects to Nova and never sends
robot motion commands. The upstream packet is reported as ``(A, B, sensor)``;
for the current non-sensor HAND_RIGHT configuration the third value is 0.
"""

import time

from leader_reader import LeaderReader
from teleop_config import DEFAULT_LEADER_CONFIG


def main():
    reader = LeaderReader(DEFAULT_LEADER_CONFIG)
    try:
        reader.open()
        print("Leader opened; torque is OFF.")
        print("Press/release the physical A and B buttons; Ctrl+C exits.")
        last = None
        while True:
            sample = reader.read_once()
            if sample is None:
                print(f"\rread failed: {reader.last_error}", end="", flush=True)
            elif sample.keys != last:
                print(
                    f"keys(A,B,sensor)={sample.keys} "
                    f"trigger={sample.trigger_normalized:.3f}"
                )
                last = sample.keys
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
