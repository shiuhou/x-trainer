"""Asynchronous JSONL telemetry for the modular teleoperation runtime."""

import json
import os
import queue
import statistics
import threading
import time
from typing import Optional


TRACE_DIR = "teleop_traces"
TRACE_QUEUE_SIZE = 512
TRACE_FLUSH_PERIOD_S = 0.5
SUMMARY_PERIOD_S = 2.0


def make_trace_path(directory: str = TRACE_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    suffix = time.time_ns() % 1_000_000_000
    return os.path.join(
        directory,
        f"teleop_{time.strftime('%Y%m%d_%H%M%S')}_{suffix:09d}.jsonl",
    )


class TelemetryWriter:
    METRICS = (
        "leader_read_ms",
        "leader_age_ms",
        "compute_ms",
        "servoj_rtt_ms",
        "whole_loop_ms",
        "actual_loop_hz",
    )

    def __init__(self, trace_path: str, metadata: Optional[dict] = None):
        self.trace_path = trace_path
        self._file = open(trace_path, "w", encoding="utf-8")
        self._queue = queue.Queue(maxsize=TRACE_QUEUE_SIZE)
        self._stop = threading.Event()
        self._dropped = 0
        self._drop_lock = threading.Lock()
        self._written = 0
        self._trace_error = None
        self._last_flush = time.monotonic()
        self._file.write(
            json.dumps(
                {
                    "record_type": "metadata",
                    "schema_version": 2,
                    "wall_time_ns": time.time_ns(),
                    "session": metadata or {},
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._file.flush()
        self._thread = threading.Thread(
            target=self._run,
            name="teleop-telemetry-writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, sample: dict):
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            with self._drop_lock:
                self._dropped += 1

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        if not self._thread.is_alive():
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass

    def _take_dropped(self):
        with self._drop_lock:
            dropped = self._dropped
            self._dropped = 0
        return dropped

    def _write(self, sample):
        if self._trace_error is not None:
            return
        try:
            self._file.write(
                json.dumps(sample, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
            self._written += 1
            now = time.monotonic()
            if self._written % 32 == 0 or now - self._last_flush >= TRACE_FLUSH_PERIOD_S:
                self._file.flush()
                self._last_flush = now
        except Exception as exc:
            self._trace_error = repr(exc)

    @staticmethod
    def _p95(values):
        values = sorted(values)
        return values[int(0.95 * (len(values) - 1))]

    def _summary(self, samples, dropped):
        if not samples and not dropped:
            return
        print(f"[telemetry] samples={len(samples)} dropped={dropped} trace={self._written}")
        if self._trace_error is not None:
            print(f"  trace_error: {self._trace_error}")
        for metric in self.METRICS:
            values = [
                sample[metric]
                for sample in samples
                if isinstance(sample.get(metric), (int, float))
            ]
            if not values:
                continue
            print(
                f"  {metric}: mean={statistics.fmean(values):.3f} "
                f"median={statistics.median(values):.3f} "
                f"p95={self._p95(values):.3f} max={max(values):.3f}"
            )

    def _run(self):
        samples = []
        window_start = time.monotonic()
        while not self._stop.is_set() or not self._queue.empty():
            try:
                sample = self._queue.get(timeout=0.05)
                self._write(sample)
                if sample.get("record_type", "sample") == "sample":
                    samples.append(sample)
                self._queue.task_done()
            except queue.Empty:
                pass
            if time.monotonic() - window_start >= SUMMARY_PERIOD_S:
                self._summary(samples, self._take_dropped())
                samples = []
                window_start = time.monotonic()
        self._summary(samples, self._take_dropped())
