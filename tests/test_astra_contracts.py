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
