"""HTTP phase timing with an opt-in, episode-bounded read-timeout experiment."""

import time
from collections.abc import Callable

import httpx


class PhaseTraceTransport(httpx.BaseTransport):
    def __init__(
        self,
        inner: httpx.BaseTransport,
        *,
        read_timeout_s: float | None = None,
        remaining: Callable[[], float] | None = None,
    ):
        if read_timeout_s is not None and (read_timeout_s != 180 or remaining is None):
            raise ValueError("read_timeout_override_requires_180_and_episode_budget")
        self.read_timeout_s = read_timeout_s
        self.remaining = remaining
        self.inner = inner
        self.events: list[dict] = []
        self.requests = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if self.read_timeout_s is not None:
            assert self.remaining is not None
            timeouts = dict(request.extensions["timeout"])
            timeouts["read"] = min(self.read_timeout_s, self.remaining())
            request.extensions["timeout"] = timeouts
        self.requests += 1
        attempt = self.requests
        started = time.monotonic()
        previous = request.extensions.get("trace")

        def trace(name: str, info: dict) -> None:
            if (
                any(
                    phase in name
                    for phase in (
                        "connect_tcp.",
                        "start_tls.",
                        "send_request_headers.",
                        "send_request_body.",
                        "receive_response_headers.",
                        "receive_response_body.",
                    )
                )
                and len(self.events) < 1024
            ):
                event = {
                    "attempt": attempt,
                    "event": name,
                    "elapsed_s": time.monotonic() - started,
                }
                self.events.append(event)
            if previous is not None:
                previous(name, info)

        request.extensions["trace"] = trace
        try:
            return self.inner.handle_request(request)
        except httpx.TimeoutException as exc:
            self.events.append(
                {
                    "attempt": attempt,
                    "event": "transport.timeout",
                    "timeout_type": type(exc).__name__,
                    "elapsed_s": time.monotonic() - started,
                }
            )
            raise

    def close(self) -> None:
        self.inner.close()
