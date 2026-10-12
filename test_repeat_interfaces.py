"""Protocol deadlines, reversed gripper polarity and integrated teleop recording."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from nova_client import NovaClient, NovaProtocolError
from teach_config import RepeatConfig
from teach_episode import load_episode
from teleop_config import NovaConfig
from test_teach_repeat import FakeClock, config_data


class SocketFixture:
    def __init__(self, response):
        self.response = response
        self.writes = []

    def sendall(self, command):
        self.writes.append(command)

    def recv(self, size):
        response, self.response = self.response, b""
        return response


def test_servoj_deadline_checked_before_socket_write():
    client = NovaClient(NovaConfig())
    sock = SocketFixture(b"")
    client._socket = sock
    with pytest.raises(TimeoutError, match="deadline"):
        client.servo_j([0] * 6, send_not_after_ns=0)
    assert sock.writes == []


def test_response_echo_cannot_acknowledge_another_command():
    client = NovaClient(NovaConfig())
    client._socket = SocketFixture(b"0,{},Stop();")
    with pytest.raises(NovaProtocolError, match="match"):
        client.servo_j([0] * 6)


def test_gripper_reversed_polarity_and_explicit_deadline(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "source/leisaac"))
    from gripper_diagnostics import GripperDiagnostic
    from nova_gripper import GripperStatus, NovaGripper
    from teleop_config import GripperDiagnosticConfig

    gripper = NovaGripper("fixture", open_position=2490, closed_position=1470)
    writes = []
    gripper._servo = SimpleNamespace(
        WritePosEx=lambda *args: writes.append(args) or (0, 0)
    )
    assert gripper.normalized_to_position(0) == 2490
    assert gripper.normalized_to_position(1) == 1470
    gripper.command_position(1470)
    gripper.command_position(2490)
    assert [w[1] for w in writes] == [1470, 2490]
    with pytest.raises(TimeoutError):
        gripper.command_position(1800, send_not_after_ns=0)
    assert len(writes) == 2
    with pytest.raises(ValueError):
        gripper.command_position(1400)
    diag = GripperDiagnostic(2490, 1470, GripperDiagnosticConfig())
    status = GripperStatus(42, 1, 2000, True, True, 0, 0, 120, 25)
    assert diag.update(1470, status) == "CLOSING"
    assert diag.update(2490, status) == "OPENING"


def test_default_teleop_is_offline(monkeypatch):
    import run_teleop

    def denied(*args, **kwargs):
        raise AssertionError("must not construct hardware")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    assert run_teleop.run([]) == 0


def test_teleop_r_e_records_actual_writes_and_keeps_running(tmp_path, monkeypatch):
    import run_teleop as rt

    clock = FakeClock()
    time = SimpleNamespace(
        monotonic=lambda: clock.ns / 1e9,
        monotonic_ns=clock.now_ns,
        time_ns=clock.now_ns,
        sleep=clock.sleep,
    )
    monkeypatch.setattr(rt, "time", time)
    monkeypatch.setattr(
        rt, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True))
    )
    monkeypatch.syspath_prepend(str(Path(__file__).parent / "source/leisaac"))
    import leader_reader
    import nova_gripper

    class Leader:
        def __init__(self, *args):
            self.cycle = 0

        def open(self):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def read_once(self):
            return self.sample()

        def sample(self):
            trigger = 1.0 if self.cycle in (4, 5) else 0.0
            return SimpleNamespace(
                joints_deg=(0.0,) * 6,
                trigger_normalized=trigger,
                trigger_raw=0,
                trigger_deg=0.0,
                keys=(1, 1, 0),
                read_start_ns=clock.ns,
                timestamp_ns=clock.ns,
                read_ms=0,
            )

        def latest(self, *args):
            self.cycle += 1
            if self.cycle == 10:
                raise KeyboardInterrupt()
            return self.sample()

    class Gripper:
        enabled = disabled = 0

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.position = 2490
            self.previous = None

        def status(self):
            return SimpleNamespace(
                servo_id=1,
                model_number=42,
                position=self.position,
                torque_enabled=True,
                moving=False,
                load=0,
                current=0,
                voltage_raw=0,
                temperature_raw=0,
            )

        def open(self):
            return self.status()

        def read_status(self):
            return self.status()

        def close(self):
            pass

        def enable_torque(self):
            Gripper.enabled += 1

        def disable_torque(self):
            Gripper.disabled += 1

        def command_normalized(self, value, timing):
            position = 1470 if value else 2490
            if position == self.previous:
                return None
            timing["t_gripper_send"] = clock.ns
            clock.ns += 1_000_000
            timing["t_gripper_response"] = clock.ns
            self.position = self.previous = position
            return position

    class Nova:
        stopped = 0

        def __init__(self, *args):
            pass

        def connect(self):
            pass

        def require_mode(self):
            pass

        def close(self):
            pass

        def get_angle(self):
            return [0.0] * 6

        def servo_j(self, joints, t, aheadtime, gain, timing):
            timing["t_servoj_send"] = clock.ns
            clock.ns += 1_000_000
            timing["t_servoj_response"] = clock.ns
            command = (
                "ServoJ("
                + ",".join(f"{v:.6f}" for v in joints)
                + f",t={t:g},aheadtime={aheadtime:g},gain={gain:g})"
            )
            return SimpleNamespace(
                error_id=0, command=command, raw=f"0,{{1}},{command};"
            )

        def stop(self):
            Nova.stopped += 1
            return SimpleNamespace(raw="0,{},Stop();")

    class Feedback:
        def __init__(self, *args):
            pass

        def connect(self):
            pass

        def start(self):
            pass

        def close(self):
            pass

        def state(self):
            return SimpleNamespace(latest=True, error=None)

        def latest(self, *args):
            return SimpleNamespace(
                host_timestamp_ns=clock.ns,
                robot_mode=5,
                q_actual_deg=(0.0,) * 6,
                q_target_deg=(0.0,) * 6,
                controller_timestamp_ms=0,
            )

    class Keys:
        restored = False

        def __init__(self):
            self.cycle = 0

        def open(self):
            pass

        def close(self):
            Keys.restored = True

        def poll(self):
            self.cycle += 1
            return ["r"] if self.cycle == 3 else ["e"] if self.cycle == 8 else []

    class Telemetry:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, *args):
            pass

        def stop(self):
            pass

    monkeypatch.setattr(leader_reader, "LeaderReader", Leader)
    monkeypatch.setattr(nova_gripper, "NovaGripper", Gripper)
    monkeypatch.setattr(rt, "NovaClient", Nova)
    monkeypatch.setattr(rt, "NovaRealtimeReader", Feedback)
    monkeypatch.setattr(rt, "TerminalKeys", Keys)
    monkeypatch.setattr(rt, "TelemetryWriter", Telemetry)
    monkeypatch.setattr(rt, "make_trace_path", lambda: str(tmp_path / "trace.jsonl"))
    monkeypatch.setattr("builtins.input", lambda: "")
    config_path = tmp_path / "repeat.local.json"
    config_path.write_text(json.dumps(config_data()))
    leader_path = tmp_path / "leader.local.json"
    leader_path.write_text(
        json.dumps(
            {
                "port": "fixture",
                "baudrate": 2000000,
                "joint_ids": [11, 12, 14, 15, 16, 17],
                "append_id": 13,
                "trigger_id": 18,
                "position_address": 140,
                "position_length": 4,
                "torque_address": 64,
                "offsets_rad": [0.0] * 6,
                "signs": [1] * 6,
                "trigger_open_deg": 200.0,
                "trigger_closed_deg": 160.0,
                "sample_hz": 150.0,
            }
        )
    )
    result = rt.run(
        [
            "--execute",
            "--config",
            str(config_path),
            "--leader-config",
            str(leader_path),
            "--gripper",
            "--gripper-enable-torque",
            "--max-speed",
            *(["30"] * 6),
            "--teach-dir",
            str(tmp_path / "episodes"),
        ]
    )
    assert result == 0 and Nova.stopped == 1 and Keys.restored
    paths = list((tmp_path / "episodes").glob("*.jsonl"))
    assert len(paths) == 1
    episode = load_episode(paths[0], RepeatConfig.load(config_path))
    assert [e["position"] for e in episode.events if e["kind"] == "gripper"] == [
        1470,
        2490,
    ]
    assert len([e for e in episode.events if e["kind"] == "arm"]) == 5
    assert Gripper.enabled == 0 and Gripper.disabled == 0
