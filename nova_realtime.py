"""Receive-only 30004 reader for experiments.

The reader never reconnects. A disconnect or parser failure is terminal for
the active experiment; a new preflight must create a new reader.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass

from nova_feedback_protocol import (
    FEEDBACK_PORT,
    FeedbackFramer,
    FeedbackSample,
    FRAME_SIZE,
)


@dataclass(frozen=True)
class FeedbackState:
    latest: FeedbackSample | None
    error: str | None
    frames: int
    started_ns: int


class NovaRealtimeReader:
    def __init__(self, host: str, port: int = FEEDBACK_PORT, timeout_s: float = 0.2):
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest: FeedbackSample | None = None
        self._error: str | None = None
        self._frames = 0
        self._started_ns = 0

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._sock = sock

    def start(self) -> None:
        if self._sock is None:
            raise RuntimeError("connect before start")
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._started_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._run, name="nova-30004-reader", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        framer = FeedbackFramer()
        try:
            while not self._stop.is_set():
                chunk = self._sock.recv(FRAME_SIZE)  # type: ignore[union-attr]
                if not chunk:
                    raise ConnectionError("30004 socket closed")
                samples = framer.feed(chunk, host_timestamp_ns=time.monotonic_ns())
                if samples:
                    with self._lock:
                        self._latest = samples[-1]
                        self._frames += len(samples)
        except socket.timeout:
            with self._lock:
                self._error = "30004 receive timeout"
        except Exception as exc:
            with self._lock:
                if not self._stop.is_set():
                    self._error = repr(exc)

    def state(self) -> FeedbackState:
        with self._lock:
            return FeedbackState(
                self._latest, self._error, self._frames, self._started_ns
            )

    def latest(self, max_age_s: float) -> FeedbackSample:
        state = self.state()
        if state.error:
            raise RuntimeError(state.error)
        if (
            state.latest is None
            or state.latest.age_ms(time.monotonic_ns()) > max_age_s * 1000
        ):
            raise TimeoutError("30004 feedback is stale")
        return state.latest

    def close(self) -> None:
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
