"""Hardware-independent recording, scheduling and fail-closed acceptance tests."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
from dataclasses import asdict, replace

import pytest

from repeat_runtime import RepeatRunner, SendResult
from run_repeat import run
from teach_config import RepeatConfig, TeachError
from teach_controls import TeachSession, TerminalKeys
from teach_episode import EpisodeWriter, RobotState, StationaryWindow, load_episode


def config_data():
    return {
        "identity": {
            "robot_id": "fixture-robot",
            "model": "fixture",
            "firmware": "fixture",
            "calibration_id": "fixture-calibration",
        },
        "nova": {
            "host": "fixture",
            "dashboard_port": 29999,
            "feedback_port": 30004,
            "socket_timeout_s": 0.1,
            "stop_timeout_s": 0.2,
        },
        "gripper": {
            "port": "fixture",
            "baudrate": 115200,
            "servo_id": 1,
            "model_number": 42,
            "open_position": 2490,
            "closed_position": 1470,
            "command_speed": 500,
            "command_acceleration": 0,
            "serial_timeout_s": 0.1,
        },
        "limits": {"joint_limits_deg": [[-180, 180]] * 6, "max_speed_deg_s": [30] * 6},
        "checks": {
            "joint_start_tolerance_deg": [0.1] * 6,
            "joint_tracking_tolerance_deg": [5] * 6,
            "gripper_start_tolerance": 10,
            "gripper_tracking_tolerance": 1200,
            "feedback_timeout_s": 0.2,
            "gripper_feedback_timeout_s": 0.2,
            "stationary_window_s": 0.02,
            "stationary_joint_tolerance_deg": 0.01,
            "stationary_gripper_tolerance": 2,
            "monitor_period_s": 0.01,
            "max_lateness_s": 0.005,
            "completion_timeout_s": 0.5,
        },
    }


@pytest.fixture
def config():
    return RepeatConfig.parse(config_data())


def state(ns, q=0.0, grip=2490):
    return RobotState(ns, ns, [q] * 6, [q] * 6, 5, ns, grip, False, True)


def arm_event(ns, q=0.0):
    command = (
        "ServoJ("
        + ",".join(f"{q:.6f}" for _ in range(6))
        + ",t=0.1,aheadtime=50,gain=500)"
    )
    return {
        "record_type": "command",
        "kind": "arm",
        "send_ns": ns,
        "reply_ns": ns + 1_000_000,
        "error_id": 0,
        "reply": f"0,{{1}},{command};",
        "command": command,
        "joints_deg": [q] * 6,
        "t": 0.1,
        "aheadtime": 50,
        "gain": 500,
    }


def grip_event(ns, position):
    return {
        "record_type": "command",
        "kind": "gripper",
        "send_ns": ns,
        "reply_ns": ns + 1_000_000,
        "error_id": 0,
        "reply": "COMM_SUCCESS",
        "position": position,
        "speed": 500,
        "acceleration": 0,
    }


def make_episode(path, config):
    start = 1_000_000_000
    writer = EpisodeWriter(
        path,
        {
            "record_type": "teach_metadata",
            "schema_version": 1,
            "start_ns": start,
            "initial_state": asdict(state(start)),
            "profile": config.profile(),
        },
    )
    writer.submit(arm_event(start + 30_000_000))
    writer.submit(grip_event(start + 35_000_000, 1470))
    writer.submit(arm_event(start + 60_000_000, 0.3))
    writer.submit(
        {
            "record_type": "observation",
            "state": asdict(state(start + 65_000_000, 0.3, 1470)),
        }
    )
    writer.submit(arm_event(start + 90_000_000, 0.3))
    writer.submit(grip_event(start + 95_000_000, 2490))
    writer.submit(
        {
            "record_type": "observation",
            "state": asdict(state(start + 100_000_000, 0.3, 2490)),
        }
    )
    writer.finish(start + 150_000_000, state(start + 150_000_000, 0.3, 2490))
    writer.close()
    assert writer.error is None
    return load_episode(path, config)


class FakeClock:
    def __init__(self):
        self.ns = 10_000_000_000

    def now_ns(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += round(seconds * 1e9)


class MemoryTransport:
    def __init__(self, clock):
        self.clock = clock
        self.q = [0.0] * 6
        self.grip = 2490
        self.sent = []
        self.stops = 0
        self.delay_ns = 1_000_000
        self.error_id = 0
        self.fault = None
        self.fail_stop = False

    def snapshot(self):
        s = state(self.clock.ns, self.q[0], self.grip)
        if self.fault and self.sent:
            return self.fault(s)
        return s

    def send(self, event, deadline_ns, latest_ns):
        self.sent.append(dict(event))
        ns = self.clock.ns
        self.clock.ns += self.delay_ns
        if event["kind"] == "arm":
            self.q = event["joints_deg"]
        else:
            self.grip = event["position"]
        return SendResult(ns, self.clock.ns, self.error_id, "fixture response")

    def stop(self):
        self.stops += 1
        if self.fail_stop:
            raise ConnectionError("stop failed")
        return "0,{},Stop();"


def test_replay_original_order_targets_timing_and_tail(tmp_path, config):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "completed", report
    assert transport.sent == episode.events
    assert transport.stops == 1
    origin = report["started_ns"]
    assert [e["send_ns"] - origin for e in report["events"]] == [
        30_000_000,
        35_000_000,
        60_000_000,
        90_000_000,
        95_000_000,
    ]
    assert report["finished_ns"] - origin >= episode.duration_ns
    assert all(e["lateness_ns"] == 0 for e in report["events"])
    assert report["source_sha256"] == episode.sha256
    assert report["observations"]


def test_blocked_command_aborts_without_catchup(tmp_path, config):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    transport.delay_ns = 6_000_000
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "aborted"
    assert len(transport.sent) == 1 and transport.stops == 1
    assert "next command" in report["stop_reason"]


@pytest.mark.parametrize(
    "fault",
    [
        lambda s: replace(s, feedback_ns=s.time_ns - 300_000_000),
        lambda s: replace(s, gripper_ns=s.time_ns - 300_000_000),
        lambda s: replace(s, robot_mode=4),
        lambda s: replace(s, gripper_torque=False),
        lambda s: replace(s, joints_deg=[10.0] * 6),
        lambda s: replace(s, gripper_position=1300),
    ],
)
def test_feedback_fault_stops_no_further_commands(tmp_path, config, fault):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    transport.fault = fault
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "aborted"
    assert len(transport.sent) == 1 and transport.stops == 1


@pytest.mark.parametrize("q,grip", [(1.0, 2490), (0.0, 1470)])
def test_start_mismatch_never_sends(tmp_path, config, q, grip):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    transport.q, transport.grip = [q] * 6, grip
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "aborted"
    assert transport.sent == [] and transport.stops == 0


def test_rejected_and_failed_stop_report(tmp_path, config):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    transport.error_id, transport.fail_stop = 1, True
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "aborted"
    assert report["stop_error"] == "stop failed"
    assert len(transport.sent) == 1


@pytest.mark.parametrize(
    "exception", [ConnectionError("TCP closed"), KeyboardInterrupt()]
)
def test_exception_or_interrupt_attempts_stop(tmp_path, config, exception):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)

    def fail(*args):
        raise exception

    transport.send = fail
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "aborted" and transport.stops == 1


def test_late_monitor_aborts_before_send(tmp_path, config):
    episode = make_episode(tmp_path / "episode.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    original = transport.snapshot
    calls = 0

    def delayed():
        nonlocal calls
        calls += 1
        if calls == 7:
            clock.ns += 40_000_000
        return original()

    transport.snapshot = delayed
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "aborted"
    assert len(transport.sent) < len(episode.events)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop(),
        lambda r: r.__setitem__(2, {**r[2], "seq": 99}),
        lambda r: r[1].__setitem__("joints_deg", [float("nan")] * 6),
        lambda r: r[2].__setitem__("send_ns", r[1]["send_ns"]),
        lambda r: r[1].__setitem__("error_id", 1),
        lambda r: r[-1].__setitem__("records", 1),
        lambda r: r[0].__setitem__("schema_version", 2),
        lambda r: r[1].__setitem__("command", "PowerOn()"),
    ],
)
def test_invalid_episode_rejected(tmp_path, config, change):
    path = tmp_path / "episode.jsonl"
    make_episode(path, config)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(TeachError):
        load_episode(path)


def test_profile_mismatch_and_speed_limits(tmp_path, config):
    path = tmp_path / "episode.jsonl"
    make_episode(path, config)
    raw = config_data()
    raw["identity"]["calibration_id"] = "different"
    with pytest.raises(TeachError, match="identity"):
        load_episode(path, RepeatConfig.parse(raw))
    raw = config_data()
    raw["limits"]["max_speed_deg_s"] = [1] * 6
    with pytest.raises(TeachError, match="speed"):
        load_episode(path, RepeatConfig.parse(raw))


def test_default_cli_zero_hardware(tmp_path, config, monkeypatch):
    path = tmp_path / "episode.jsonl"
    make_episode(path, config)

    def denied(*args, **kwargs):
        raise AssertionError("hardware/network construction forbidden")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    assert run([str(path)]) == 0
    assert "repeat_hardware" not in sys.modules
    assert run([str(path), "--execute"]) == 1


def test_stationary_window_rejects_moving_stale_gaps(config):
    w = StationaryWindow(config)
    for ns, q in [(0, 0.0), (10_000_000, 0.1), (20_000_000, 0.0)]:
        w.observe(state(ns, q))
    with pytest.raises(TeachError, match="moving"):
        w.require(state(20_000_000), [0.0] * 6)
    w = StationaryWindow(config)
    s = state(0)
    w.observe(s)
    with pytest.raises(TeachError, match="window"):
        w.require(s, [0.0] * 6)


def test_r_e_boundary_multiple_episodes_and_only_real_writes(tmp_path, config):
    session = TeachSession(tmp_path, config)
    for ns in [0, 10_000_000, 20_000_000]:
        messages = session.boundary(
            ["r"] if ns == 20_000_000 else [], state(ns), [0.0] * 6, None
        )
    assert session.active and "錄製中" in messages[0]
    session.command(arm_event(21_000_000))
    session.boundary([], state(30_000_000), [0.0] * 6, None)
    session.command(arm_event(31_000_000))
    session.boundary(["e"], state(40_000_000), [0.0] * 6, None)
    first = session.writer
    assert first.done.wait(2)
    session.boundary([], state(50_000_000), [0.0] * 6, None)
    assert session.status == "已保存"
    session.boundary(["r"], state(60_000_000), [0.0] * 6, None)
    session.command(arm_event(61_000_000))
    session.boundary(["e"], state(70_000_000), [0.0] * 6, None)
    session.close()
    paths = list(tmp_path.glob("*.jsonl"))
    assert len(paths) == 2
    for path in paths:
        episode = load_episode(path, config)
        assert all(e["kind"] == "arm" for e in episode.events)


def test_interrupted_recording_no_valid_footer(tmp_path, config):
    writer = EpisodeWriter(tmp_path / "incomplete.jsonl", {})
    writer.close()
    with pytest.raises(TeachError):
        load_episode(writer.path)


def test_queue_full_invalidates(tmp_path, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    original = EpisodeWriter._write

    def paused(stream, record):
        entered.set()
        assert release.wait(2)
        original(stream, record)

    monkeypatch.setattr(EpisodeWriter, "_write", staticmethod(paused))
    writer = EpisodeWriter(tmp_path / "queue.jsonl", {}, max_queue=1)
    assert entered.wait(2)
    writer.submit({"record_type": "fixture"})
    with pytest.raises(TeachError, match="queue full"):
        writer.submit({"record_type": "fixture"})
    release.set()
    writer.close()
    assert writer.error
    assert "teach_end" not in writer.path.read_text()


def test_disk_failure_and_no_overwrite(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(EpisodeWriter, "_write", staticmethod(fail))
    writer = EpisodeWriter(tmp_path / "disk.jsonl", {})
    assert writer.done.wait(2)
    assert "disk full" in writer.error
    with pytest.raises(TeachError, match="already exists"):
        EpisodeWriter(writer.path, {})


def test_terminal_keys_and_restoration():
    import termios
    import time

    master, slave = os.openpty()
    with os.fdopen(slave, "r") as stream:
        saved = termios.tcgetattr(stream.fileno())
        keys = TerminalKeys(stream)
        try:
            keys.open()
            os.write(master, b"RE")
            deadline = time.monotonic() + 1
            got = []
            while len(got) < 2 and time.monotonic() < deadline:
                got += keys.poll()
            assert got == ["r", "e"]
        finally:
            keys.close()
        assert termios.tcgetattr(stream.fileno()) == saved
        os.close(master)


def test_config_requires_explicit_values():
    raw = config_data()
    raw["checks"].pop("max_lateness_s")
    with pytest.raises(TeachError):
        RepeatConfig.parse(raw)


def test_held_object_compares_recorded_actual_gripper_end(tmp_path, config):
    path = tmp_path / "held.jsonl"
    make_episode(path, config)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # A physical object stops at 1800 even though the final close target is 1470.
    rows[6]["position"] = 1470
    rows[7]["state"]["gripper_position"] = 1800
    rows[-1]["final_state"]["gripper_position"] = 1800
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    episode = load_episode(path, config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    original = transport.send

    def with_object(event, deadline, latest):
        result = original(event, deadline, latest)
        if (
            event["kind"] == "gripper"
            and event["send_ns"] == episode.events[-1]["send_ns"]
        ):
            transport.grip = 1800
        return result

    transport.send = with_object
    report = RepeatRunner(episode, config, transport, clock).run()
    assert report["status"] == "completed", report
    assert transport.grip == 1800


def test_preflight_rechecked_after_confirmation(tmp_path, config):
    from repeat_runtime import check_start

    episode = make_episode(tmp_path / "start.jsonl", config)
    clock = FakeClock()
    transport = MemoryTransport(clock)
    runner = RepeatRunner(episode, config, transport, clock)
    runner.require_start()
    transport.q = [1.0] * 6
    report = runner.run()
    assert report["status"] == "aborted"
    assert transport.sent == []
    with pytest.raises(TeachError):
        check_start(replace(state(clock.ns), robot_mode=7), episode, config, clock.ns)


def test_end_key_rejects_motion_without_closing_recording(tmp_path, config):
    session = TeachSession(tmp_path, config)
    for ns in [0, 10_000_000, 20_000_000]:
        session.boundary(["r"] if ns == 20_000_000 else [], state(ns), [0.0] * 6, None)
    session.command(arm_event(21_000_000))
    messages = session.boundary(["e"], state(30_000_000, 0.2), [0.0] * 6, None)
    assert session.active
    assert "未接受" in messages[0]
    session.close()
    with pytest.raises(TeachError):
        load_episode(session.writer.path)
