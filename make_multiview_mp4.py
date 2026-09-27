#!/usr/bin/env python3
"""Compose timestamp-aligned three-camera JPEG frames into one MP4."""

from __future__ import annotations

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class FrameRef:
    timestamp_ns: int
    path: Path


def load_frame_refs(trace_path: Path) -> dict[str, list[FrameRef]]:
    episode_dir = trace_path.with_suffix("")
    cameras: dict[str, list[FrameRef]] = {}
    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "camera_frame":
                continue
            camera = record["camera_name"]
            frame_path = episode_dir / record["frame_path"]
            if not frame_path.is_file():
                raise FileNotFoundError(f"Missing frame: {frame_path}")
            cameras.setdefault(camera, []).append(
                FrameRef(record["capture_mono_ns"], frame_path)
            )
    for camera, refs in cameras.items():
        refs.sort(key=lambda ref: ref.timestamp_ns)
        if not refs:
            raise ValueError(f"Camera {camera!r} has no frames")
    return cameras


def fit_frame(cv2, frame, width: int, height: int):
    """Resize without distortion and letterbox into a fixed pane."""
    import numpy as np

    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=resized.dtype)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def choose_ref(refs: list[FrameRef], timestamps: list[int], target_ns: int) -> FrameRef:
    index = bisect.bisect_right(timestamps, target_ns) - 1
    return refs[max(0, index)]


def compose(args) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("MP4 composition requires opencv-python") from exc

    trace_path = args.trace.resolve()
    cameras = load_frame_refs(trace_path)
    required = (args.top, args.bottom_left, args.bottom_right)
    missing = [name for name in required if name not in cameras]
    if missing:
        available = ", ".join(sorted(cameras)) or "none"
        raise ValueError(f"Missing cameras: {', '.join(missing)}; available: {available}")

    selected = {name: cameras[name] for name in required}
    start_ns = max(refs[0].timestamp_ns for refs in selected.values())
    end_ns = min(refs[-1].timestamp_ns for refs in selected.values())
    if end_ns <= start_ns:
        raise ValueError("Camera timestamp ranges do not overlap")

    top_width = args.width
    bottom_width = args.width // 2
    bottom_height = args.height // 3
    top_height = args.height - bottom_height
    if top_width % 2 or args.height <= bottom_height:
        raise ValueError("width must be even and height must leave a top pane")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open MP4 output: {args.output}")

    timestamps = {
        name: [ref.timestamp_ns for ref in refs]
        for name, refs in selected.items()
    }
    frame_period_ns = round(1_000_000_000 / args.fps)
    frame_count = 0
    try:
        target_ns = start_ns
        while target_ns <= end_ns:
            panes = []
            for name in required:
                ref = choose_ref(selected[name], timestamps[name], target_ns)
                frame = cv2.imread(str(ref.path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Cannot decode frame: {ref.path}")
                panes.append(frame)

            top = fit_frame(cv2, panes[0], top_width, top_height)
            bottom_left = fit_frame(cv2, panes[1], bottom_width, bottom_height)
            bottom_right = fit_frame(cv2, panes[2], bottom_width, bottom_height)
            bottom = cv2.hconcat((bottom_left, bottom_right))
            output_frame = cv2.vconcat((top, bottom))
            writer.write(output_frame)
            frame_count += 1
            target_ns += frame_period_ns
    finally:
        writer.release()

    duration_s = (end_ns - start_ns) / 1_000_000_000
    print(f"output: {args.output}")
    print(f"frames: {frame_count}")
    print(f"fps: {args.fps:g}")
    print(f"duration_s: {duration_s:.3f}")
    print(f"layout: {args.top} top; {args.bottom_left} + {args.bottom_right} bottom")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="teleop JSONL trace")
    parser.add_argument("--output", type=Path, required=True, help="output MP4 path")
    parser.add_argument("--top", default="front", help="camera spanning the top pane")
    parser.add_argument("--bottom-left", default="left", help="camera in the bottom-left pane")
    parser.add_argument("--bottom-right", default="right", help="camera in the bottom-right pane")
    parser.add_argument("--fps", type=float, default=15.0, help="output video FPS")
    parser.add_argument("--width", type=int, default=960, help="output width")
    parser.add_argument("--height", type=int, default=1080, help="output height")
    args = parser.parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("fps, width and height must be positive")
    return args


if __name__ == "__main__":
    compose(build_parser())
