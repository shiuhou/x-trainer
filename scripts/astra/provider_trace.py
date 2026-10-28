"""Metadata-only HTTP phase timing; leaves routing, deadlines and retries intact."""

import time

import httpx


class PhaseTraceTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport):
        self.inner = inner
        self.events: list[dict] = []
        self.requests = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
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
