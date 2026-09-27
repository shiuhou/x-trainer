#!/usr/bin/env python3
"""Non-blocking camera capture for synchronized teleoperation episodes."""

from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class CameraSpec:
    name: str
    path: str
    width: Optional[int] = None
    height: Optional[int] = None
    fps: float = 30.0
    jpeg_quality: int = 90

    def __post_init__(self):
        if not self.name or re.fullmatch(r"[A-Za-z0-9_.-]+", self.name) is None:
            raise ValueError("camera name must contain only letters, numbers, _, ., or -")
        if not self.path:
            raise ValueError("camera path must not be empty")
        if self.width is not None and self.width <= 0:
            raise ValueError("camera width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("camera height must be positive")
        if self.fps <= 0:
            raise ValueError("camera fps must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("JPEG quality must be from 1 to 100")

    def metadata(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "width_requested": self.width,
            "height_requested": self.height,
            "fps_requested": self.fps,
            "jpeg_quality": self.jpeg_quality,
        }


class CameraRecorder:
    """Capture and encode frames off the ServoJ control thread.

    Camera records are submitted to the same TelemetryWriter queue as control
    samples. The shared monotonic timestamps make frame/state joins possible
    without making the control loop wait for a camera or disk write.
    """

    def __init__(
        self,
        spec: CameraSpec,
        episode_dir: Path,
        submit: Callable[[dict], None],
        queue_size: int = 64,
    ):
        self.spec = spec
        self.episode_dir = Path(episode_dir)
        self.frame_dir = self.episode_dir / "cameras" / spec.name
        self._submit = submit
        self._queue: queue.Queue[tuple[int, int, int, object]] = queue.Queue(
            maxsize=queue_size
        )
        self._stop = threading.Event()
        self._capture = None
        self._capture_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._capture_error: Optional[str] = None
        self._writer_error: Optional[str] = None
        self._dropped = 0
        self._dropped_total = 0
        self._dropped_lock = threading.Lock()
        self._frames_captured = 0
        self._frames_written = 0
        self._first_capture_mono_ns: Optional[int] = None
        self._last_capture_mono_ns: Optional[int] = None

    def start(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Camera recording requires the opencv-python package") from exc

        if self._capture_thread is not None and self._capture_thread.is_alive():
            raise RuntimeError(f"Camera {self.spec.name} is already running")

        self.frame_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(self.spec.path, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Cannot open camera {self.spec.name}: {self.spec.path}")
        if self.spec.width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.spec.width)
        if self.spec.height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.spec.height)
        if self.spec.fps:
            capture.set(cv2.CAP_PROP_FPS, self.spec.fps)
        self._capture = capture
        self._stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-capture-{self.spec.name}",
            daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"camera-writer-{self.spec.name}",
            daemon=True,
        )
        self._capture_thread.start()
        self._writer_thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._capture is not None:
            self._capture.release()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        if self._writer_thread is not None:
            # Shutdown is outside the control loop. Give the writer enough
            # time to drain queued frames so the episode tail is complete.
            self._writer_thread.join(timeout=5.0)
        duration_s = None
        actual_fps = None
        if (
            self._first_capture_mono_ns is not None
            and self._last_capture_mono_ns is not None
        ):
            duration_s = (
                self._last_capture_mono_ns - self._first_capture_mono_ns
            ) / 1_000_000_000
            if duration_s > 0:
                actual_fps = self._frames_captured / duration_s
        return {
            "record_type": "camera_summary",
            "camera_name": self.spec.name,
            "frames_captured": self._frames_captured,
            "frames_written": self._frames_written,
            "frames_dropped": self._dropped_total,
            "capture_duration_s": duration_s,
            "actual_fps": actual_fps,
            "capture_error": self._capture_error,
            "writer_error": self._writer_error,
            "writer_alive": bool(
                self._writer_thread is not None and self._writer_thread.is_alive()
            ),
        }

    @property
    def error(self) -> Optional[str]:
        """Return a capture/write error that should stop a live episode."""
        return self._capture_error or self._writer_error

    def _capture_loop(self) -> None:
        import cv2

        interval_ns = int(1_000_000_000 / self.spec.fps)
        next_allowed_ns = 0
        sequence = 0
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            capture_mono_ns = time.monotonic_ns()
            capture_wall_ns = time.time_ns()
            if not ok:
                if not self._stop.is_set():
                    self._capture_error = "camera read returned no frame"
                break
            if capture_mono_ns < next_allowed_ns:
                continue
            next_allowed_ns = capture_mono_ns + interval_ns
            sequence += 1
            self._frames_captured += 1
            if self._first_capture_mono_ns is None:
                self._first_capture_mono_ns = capture_mono_ns
            self._last_capture_mono_ns = capture_mono_ns
            try:
                self._queue.put_nowait((sequence, capture_mono_ns, capture_wall_ns, frame))
            except queue.Full:
                with self._dropped_lock:
                    self._dropped += 1
                    self._dropped_total += 1
        if self._capture is not None:
            self._capture.release()

    def _writer_loop(self) -> None:
        import cv2

        params = [cv2.IMWRITE_JPEG_QUALITY, self.spec.jpeg_quality]
        while not self._stop.is_set() or not self._queue.empty():
            try:
                sequence, mono_ns, wall_ns, frame = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                frame_path = self.frame_dir / f"frame_{sequence:08d}.jpg"
                if not cv2.imwrite(str(frame_path), frame, params):
                    raise RuntimeError("cv2.imwrite returned false")
                height, width = frame.shape[:2]
                relative_path = frame_path.relative_to(self.episode_dir).as_posix()
                with self._dropped_lock:
                    dropped = self._dropped
                    self._dropped = 0
                self._submit(
                    {
                        "record_type": "camera_frame",
                        "camera_name": self.spec.name,
                        "frame_seq": sequence,
                        "capture_mono_ns": mono_ns,
                        "capture_wall_time_ns": wall_ns,
                        "frame_path": relative_path,
                        "width": width,
                        "height": height,
                        "dropped_before": dropped,
                    }
                )
                self._frames_written += 1
            except Exception as exc:
                self._writer_error = repr(exc)
                self._stop.set()
            finally:
                self._queue.task_done()
