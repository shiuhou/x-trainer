"""Focused policy-boundary tests; no Isaac, provider or hardware required."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/leisaac"))
from leisaac.astra.contracts import (
    CAMERAS,
    Freshness,
    Limits,
    aperture_to_joints,
    measured_aperture,
)


def test_gripper_polarity_and_contact_readback():
    assert aperture_to_joints(1) == (0, 0)
    assert aperture_to_joints(0) == (-0.04, 0.04)
    assert measured_aperture(-0.02, 0.02) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        aperture_to_joints(float("nan"))


def test_whole_chunk_checks_later_waypoints_before_execution():
    current = np.array([0, 0, 0.2, 0, 0, 0, 1.0])
    rows = np.tile(current, (2, 1))
    rows[0, 2] += 0.005
    rows[1, 2] += 0.1
    with pytest.raises(ValueError, match="waypoint_displacement"):
        Limits().validate(rows, current)


@pytest.mark.parametrize(
    "index,value,reason",
    [
        (0, np.nan, "nonfinite"),
        (0, 3, "workspace"),
        (3, 0.01, "orientation"),
        (6, 1.1, "gripper"),
    ],
)
def test_reject_invalid_action(index, value, reason):
    current = np.array([0, 0, 0.2, 0, 0, 0, 1.0])
    row = current.copy()
    row[index] = value
    with pytest.raises(ValueError, match=reason):
        Limits().validate(row[None], current)


def test_static_pixels_are_valid_but_reused_generation_or_time_is_not():
    images = {name: np.zeros(shape, dtype=np.uint8) for name, shape in CAMERAS.items()}
    f = Freshness()
    f.accept(1, 0.0, images, dict.fromkeys(CAMERAS, 1))
    f.accept(2, 0.1, images, dict.fromkeys(CAMERAS, 2))
    with pytest.raises(ValueError, match="stale"):
        f.accept(2, 0.2, images, dict.fromkeys(CAMERAS, 3))
    with pytest.raises(ValueError, match="stale"):
        f.accept(3, 0.1, images, dict.fromkeys(CAMERAS, 3))


def test_failed_capture_does_not_commit_generation():
    images = {name: np.zeros(shape, dtype=np.uint8) for name, shape in CAMERAS.items()}
    f = Freshness()
    del images["wrist_rgb"]
    with pytest.raises(ValueError, match="missing"):
        f.accept(1, 0.0, images, dict.fromkeys(CAMERAS, 1))
    assert f.generation == 0


def test_stale_sensor_rejected_despite_new_wrapper_generation():
    images = {name: np.zeros(shape, dtype=np.uint8) for name, shape in CAMERAS.items()}
    f = Freshness()
    f.accept(1, 0.0, images, dict.fromkeys(CAMERAS, 1))
    frames = dict.fromkeys(CAMERAS, 2)
    frames["wrist_rgb"] = 1
    with pytest.raises(ValueError, match="stale_camera"):
        f.accept(2, 0.1, images, frames)
    assert f.generation == 1


def test_official_request_budget_counts_internal_calls(monkeypatch):
    runtime = (
        Path(__file__).resolve().parents[1] / ".agents/runs/xtrainer-astra/runtime/s5"
    )
    monkeypatch.syspath_prepend(str(runtime))
    from s5_common import Budget, RunLimits, S5Error
    from s5_agent import BoundedTransport
    import httpx

    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"status": "completed"})

    budget = Budget(RunLimits(max_llm_calls=2))
    transport = BoundedTransport(httpx.MockTransport(respond), budget, live=False)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    transport.handle_request(request)
    transport.handle_request(request)
    with pytest.raises(S5Error, match="max_calls_reached"):
        transport.handle_request(request)
    assert len(calls) == 2
    transport.close()


def test_model_observation_contains_only_images_and_measured_state(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from leisaac.astra.embodiment import XTrainerEmbodiment

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        def __getitem__(self, index):
            return Tensor(self.array[index])

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def item(self):
            return self.array.item()

    class Camera:
        def __init__(self, shape):
            self.frame = Tensor([0])
            self.data = SimpleNamespace(
                output={"rgb": Tensor(np.zeros((1, *shape), dtype=np.uint8))}
            )

        def update(self, dt, force_recompute):
            self.frame = Tensor(self.frame.array + 1)

    emb = XTrainerEmbodiment.__new__(XTrainerEmbodiment)
    emb.env = SimpleNamespace(
        sim=SimpleNamespace(render=lambda: None, current_time=0.1),
        scene={name: Camera(shape) for name, shape in CAMERAS.items()},
    )
    emb.output = tmp_path
    emb.generation, emb.ticks = 0, 3
    emb.freshness = Freshness()
    emb.instruction = "Pick up the cube."
    emb.observation_trace = []
    emb.timing = {"render": 0.0}
    monkeypatch.setattr(emb, "state", lambda: np.array([0, 0, 0.2, 0, 0, 0, 1.0]))
    monkeypatch.setattr(
        emb,
        "metrics",
        lambda: (_ for _ in ()).throw(AssertionError("privileged evaluator accessed")),
    )
    obs = emb.observe("initial")
    assert set(obs.state) == {"eef_state"}
    assert set(obs.images) == set(CAMERAS)
    assert obs.state_time == 0.1
    assert obs.instruction == emb.instruction


@pytest.mark.parametrize("port", [7897, 18443])
def test_proxy_port_is_configured_and_guard_remains_scoped(monkeypatch, port):
    runtime = (
        Path(__file__).resolve().parents[1] / ".agents/runs/xtrainer-astra/runtime/s5"
    )
    monkeypatch.syspath_prepend(str(runtime))
    from s5_agent import SimulationIOGuard
    from s5_common import APPROVED_RELAY, S5Error

    guard = SimulationIOGuard(APPROVED_RELAY, proxy=f"http://127.0.0.1:{port}")
    with pytest.raises(S5Error, match="prohibited_io"):
        guard.check("socket.connect", (None, ("127.0.0.1", port)))
    with guard.provider_network():
        guard.check("socket.connect", (None, ("127.0.0.1", port)))
        for address in [
            ("127.0.0.1", port + 1),
            ("172.10.10.100", port),
            ("198.18.0.65", 443),
        ]:
            with pytest.raises(S5Error, match="prohibited_io"):
                guard.check("socket.connect", (None, address))
        with pytest.raises(S5Error, match="prohibited_io"):
            guard.check("socket.bind", (None, ("127.0.0.1", port)))
    assert not guard.active


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:65536",
        "http://127.0.0.1:0",
        "http://172.10.10.100:7897",
        "http://127.0.0.1:7897/other",
    ],
)
def test_proxy_rejects_ambiguous_or_unapproved_destination(monkeypatch, url):
    runtime = (
        Path(__file__).resolve().parents[1] / ".agents/runs/xtrainer-astra/runtime/s5"
    )
    monkeypatch.syspath_prepend(str(runtime))
    from s5_agent import SimulationIOGuard
    from s5_common import APPROVED_RELAY, S5Error

    with pytest.raises(S5Error, match="invalid_provider_configuration"):
        SimulationIOGuard(APPROVED_RELAY, proxy=url)


@pytest.fixture
def provider(monkeypatch):
    runtime = (
        Path(__file__).resolve().parents[1] / ".agents/runs/xtrainer-astra/runtime/s5"
    )
    monkeypatch.syspath_prepend(str(runtime))
    import s5_common

    return s5_common


@pytest.mark.parametrize("port", [1, 80, 443, 21349, 65535])
def test_any_valid_proxy_port_is_supported(provider, port):
    assert (
        provider.validate_provider_proxy(
            provider.APPROVED_RELAY, f"http://localhost:{port}"
        )
        == f"http://127.0.0.1:{port}"
    )


def test_auto_selects_working_desktop_proxy_after_stale_config(provider, monkeypatch):
    monkeypatch.setattr(
        provider, "desktop_provider_proxies", lambda: ["http://localhost:24138"]
    )
    attempts = []

    def preflight(base, proxy):
        attempts.append(proxy)
        return proxy.endswith(":24138")

    monkeypatch.setattr(provider, "provider_proxy_preflight", preflight)
    env = {"S5_PROVIDER_HTTPS_PROXY": "http://127.0.0.1:21349"}
    assert (
        provider.select_provider_proxy(provider.APPROVED_RELAY, env)
        == "http://127.0.0.1:24138"
    )
    assert attempts == ["http://127.0.0.1:21349", "http://127.0.0.1:24138"]


def test_auto_selects_standard_environment_and_deduplicates(provider, monkeypatch):
    monkeypatch.setattr(provider, "desktop_provider_proxies", lambda: [])
    attempts = []

    def preflight(base, proxy):
        attempts.append(proxy)
        return proxy.endswith(":24300")

    monkeypatch.setattr(provider, "provider_proxy_preflight", preflight)
    env = {
        "S5_PROVIDER_HTTPS_PROXY": "http://localhost:24301",
        "HTTPS_PROXY": "http://127.0.0.1:24301",
        "HTTP_PROXY": "http://localhost:24300",
    }
    assert provider.select_provider_proxy(provider.APPROVED_RELAY, env).endswith(
        ":24300"
    )
    assert len(attempts) == 2


def test_auto_never_probes_hardware_or_exposes_invalid_proxy_credentials(
    provider, monkeypatch
):
    monkeypatch.setattr(provider, "desktop_provider_proxies", lambda: [])
    monkeypatch.setattr(
        provider, "provider_proxy_preflight", lambda *a: pytest.fail("unsafe preflight")
    )
    with pytest.raises(provider.S5Error) as error:
        provider.select_provider_proxy(
            provider.APPROVED_RELAY,
            {
                "HTTPS_PROXY": "http://user:synthetic-secret@127.0.0.1:21349",
                "HTTP_PROXY": "http://172.10.10.100:29999",
            },
        )
    assert "synthetic-secret" not in str(error.value)
    assert error.value.category == "proxy_connect"


def test_explicit_or_direct_selection_does_not_discover_or_switch(
    provider, monkeypatch
):
    monkeypatch.setattr(
        provider,
        "desktop_provider_proxies",
        lambda: pytest.fail("unrequested discovery"),
    )
    assert (
        provider.select_provider_proxy(
            provider.APPROVED_RELAY, {}, "http://localhost:21349"
        )
        == "http://127.0.0.1:21349"
    )
    assert provider.select_provider_proxy(provider.APPROVED_RELAY, {}, "direct") is None


def test_preflight_is_credential_free_head_not_model_request(provider, monkeypatch):
    from types import SimpleNamespace

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="403 0")

    monkeypatch.setattr(provider.subprocess, "run", run)
    assert provider.provider_proxy_preflight(
        provider.APPROVED_RELAY, "http://localhost:21349"
    )
    command = commands[0]
    assert command[:2] == ["curl", "-q"]
    assert "--head" in command and "--insecure" not in command
    assert "--location" not in command
    assert command[-1] == provider.APPROVED_RELAY + "/"
    assert all(
        "authorization" not in word.lower() and "responses" not in word
        for word in command
    )


@pytest.mark.parametrize("result", [(1, "200 0"), (0, "000 0"), (0, "200 60")])
def test_failed_connect_or_tls_is_not_selected(provider, monkeypatch, result):
    from types import SimpleNamespace

    monkeypatch.setattr(
        provider.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=result[0], stdout=result[1]),
    )
    assert not provider.provider_proxy_preflight(
        provider.APPROVED_RELAY, "http://localhost:21349"
    )


@pytest.mark.parametrize("explicit_pose", [False, True])
def test_live_measured_aperture_does_not_reject_open_only_interpolation(
    monkeypatch, explicit_pose
):
    import json
    from types import SimpleNamespace
    from inspect_robots import Observation, ObservationSpace, StateSpec, StateField
    from inspect_robots_agent._tools import build_toolset
    from leisaac.astra.embodiment import XTrainerEmbodiment

    # Measured first live frame and its two model target variants. No simulator
    # ground truth or fabricated perfect gripper reading is supplied to the tool.
    state = np.array(
        [
            -0.3449961543083191,
            -0.11750148236751556,
            0.25565552711486816,
            1.7484556025237907e-7,
            8.931510819820687e-8,
            2.513934127534867e-8,
            0.9999999511267113,
        ]
    )
    original = state.copy()
    emb = XTrainerEmbodiment.__new__(XTrainerEmbodiment)
    emb.limits = Limits(chunk_path_m=0.01, max_waypoints=32)
    emb.current = Observation(state={"eef_state": state})
    emb.consumed = False
    emb.pre_check_rejections = []
    emb.timing = {"validation": 0.0}
    monkeypatch.setattr(emb, "state", lambda: state)
    obs_space = ObservationSpace(
        state=StateSpec((StateField("eef_state", (7,), "m,rad,normalized"),))
    )
    targets = (
        {
            "x": state[0],
            "y": -0.11750148981809616,
            "z": state[2] + 0.005,
            "yaw": 0,
            "pitch": 0,
            "roll": 0,
            "gripper": 1,
        }
        if explicit_pose
        else {"z": 0.2607, "gripper": 1}
    )
    call = SimpleNamespace(
        name="move_to",
        arguments=json.dumps(
            {"targets": targets, "note": "Offline reproduction of observed target."}
        ),
    )
    emb.open_only = False
    old_space = emb.action_space()
    emb.open_only = True
    old_tools = build_toolset(old_space, obs_space, 10.0, pre_check=emb.pre_check)
    assert (
        old_tools.execute(call, emb.current).error
        == "pre-check rejected this motion: connectivity_requires_open_gripper"
    )
    tools = build_toolset(emb.action_space(), obs_space, 10.0, pre_check=emb.pre_check)
    result = tools.execute(call, emb.current)
    assert result.error is None and result.chunk is not None
    assert all(action.data[6] == 1 for action in result.chunk.actions)
    np.testing.assert_array_equal(state, original)
    call.arguments = json.dumps(
        {"targets": {"gripper": 0.99}, "note": "Invalid close."}
    )
    assert "fixed" in tools.execute(call, emb.current).error
