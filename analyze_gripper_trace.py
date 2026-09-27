#!/usr/bin/env python3
"""Summarize read-only gripper evidence from a run_teleop JSONL trace."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def status_values(rows, key):
    values = []
    for row in rows:
        status = row.get("gripper_status")
        value = status.get(key) if isinstance(status, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def extrema(values):
    return f"min={min(values):.1f} max={max(values):.1f}" if values else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, nargs="+")
    args = parser.parse_args()

    for trace in args.trace:
        metadata = None
        rows = []
        with trace.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("record_type") == "metadata":
                    metadata = record
                elif record.get("record_type") == "sample":
                    rows.append(record)

        session = (metadata or {}).get("session", {})
        gripper_metadata = session.get("gripper") or {}
        print(f"trace: {trace}")
        print(f"episode_label: {gripper_metadata.get('episode_label', 'n/a')}")
        print(f"samples: {len(rows)}")
        print(f"trigger_normalized: {extrema(finite_values(rows, 'gripper_trigger_normalized'))}")
        print(f"target_position: {extrema(finite_values(rows, 'gripper_target_position'))}")
        print(f"actual_position: {extrema(finite_values(rows, 'gripper_actual_position'))}")
        print(f"present_load: {extrema(status_values(rows, 'load'))}")
        print(f"present_current: {extrema(status_values(rows, 'current'))}")
        print(f"voltage_raw: {extrema(status_values(rows, 'voltage_raw'))}")
        print(f"temperature_raw: {extrema(status_values(rows, 'temperature_raw'))}")
        states = Counter(
            row.get("gripper_diagnostic_state")
            for row in rows
            if row.get("gripper_diagnostic_state")
        )
        if states:
            print("diagnostic_states: " + ", ".join(
                f"{state}={count}" for state, count in sorted(states.items())
            ))

        errors = sorted({
            row["gripper_feedback_error"]
            for row in rows
            if row.get("gripper_feedback_error")
        })
        if errors:
            print("feedback_errors:")
            for error in errors:
                print(f"  {error}")


if __name__ == "__main__":
    main()
