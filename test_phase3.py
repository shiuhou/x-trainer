from __future__ import annotations

import struct
import time

import pytest

from nova_feedback_protocol import FRAME_SIZE, TEST_VALUE, FeedbackFramer, decode_frame
from nova_realtime import NovaRealtimeReader
from servo_experiment import ExperimentMetadata, ExperimentRecorder


def packet(mode=5):
    data = bytearray(FRAME_SIZE)
    struct.pack_into("<H", data, 0, FRAME_SIZE)
    struct.pack_into("<Q", data, 24, mode)
    struct.pack_into("<Q", data, 32, 1234)
    struct.pack_into("<Q", data, 48, TEST_VALUE)
    struct.pack_into("<6d", data, 192, *(0.01 * i for i in range(6)))
    struct.pack_into("<6d", data, 432, *(0.02 * i for i in range(6)))
    return bytes(data)


def test_valid_and_fragmented_and_coalesced():
    p = packet()
    assert decode_frame(p, host_timestamp_ns=10).q_actual_deg[1] == pytest.approx(0.02)
    framer = FeedbackFramer()
    assert framer.feed(p[:100], host_timestamp_ns=10) == []
    assert len(framer.feed(p[100:], host_timestamp_ns=10)) == 1
    assert len(FeedbackFramer().feed(p + p, host_timestamp_ns=10)) == 2


def test_recorded_phase2b_frame_uses_native_joint_degrees():
    path = __import__("pathlib").Path(
        "/home/dsa/project/dobot/robocurve/evidence/phase2b/phase2b_feedback.bin"
    )
    if not path.exists():
        pytest.skip("workspace evidence is not present")
    sample = decode_frame(path.read_bytes(), host_timestamp_ns=10)
    assert sample.q_actual_deg == pytest.approx(
        (
            -0.0356506333,
            -38.7906913757,
            140.7869415283,
            -40.7154960632,
            -99.22099304199,
            -97.76510620117,
        )
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda b: struct.pack_into("<H", b, 0, 1),
        lambda b: struct.pack_into("<Q", b, 48, 0),
        lambda b: struct.pack_into("<Q", b, 24, 99),
    ],
)
def test_invalid_packets_fail_closed(mutator):
    data = bytearray(packet())
    mutator(data)
    with pytest.raises(ValueError):
        decode_frame(bytes(data), host_timestamp_ns=1)


def test_truncated_frame():
    with pytest.raises(ValueError):
        decode_frame(packet()[:-1], host_timestamp_ns=1)
    framer = FeedbackFramer()
    framer.feed(packet()[:20], host_timestamp_ns=1)
    with pytest.raises(ValueError):
        framer.finish()


def test_feedback_age_is_explicit():
    sample = decode_frame(packet(), host_timestamp_ns=1_000_000_000)
    assert sample.age_ms(1_025_000_000) == pytest.approx(25.0)


def test_realtime_socket_close_is_terminal(monkeypatch):
    class ClosedSocket:
        def settimeout(self, _timeout):
            pass

        def recv(self, _size):
            return b""

        def close(self):
            pass

    monkeypatch.setattr(
        "nova_realtime.socket.create_connection",
        lambda *_args, **_kwargs: ClosedSocket(),
    )
    reader = NovaRealtimeReader("fixture", timeout_s=0.01)
    reader.connect()
    reader.start()
    deadline = time.monotonic() + 1.0
    while reader.state().error is None and time.monotonic() < deadline:
        time.sleep(0.005)
    assert "socket closed" in reader.state().error
    with pytest.raises(RuntimeError, match="socket closed"):
        reader.latest(0.2)
    reader.close()


def test_recorder_writes_metadata_and_sample(tmp_path):
    metadata = ExperimentMetadata(
        "id", "test", "now", "host", "Nova2", "J1", 3, 3, 0.1, 50, 500, 0.03, [0] * 6
    )
    output = tmp_path / "exp.jsonl"
    recorder = ExperimentRecorder(output, metadata)
    assert recorder.submit({"t_monotonic_ns": 1})
    recorder.close()
    lines = output.read_text().splitlines()
    assert '"record_type": "metadata"' in lines[0]
    assert '"record_type":"sample"' in lines[1]
    assert '"record_type":"final_metadata"' in lines[2]
    assert '"experiment_status":"planned"' in lines[2]
