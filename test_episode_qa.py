from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from episode_qa import EpisodeQAError, audit_episode, update_annotations, write_outputs


def _sample(reference_ns: int, *, nonmonotonic_wall: int | None = None) -> dict:
    return {
        "record_type": "sample",
        "wall_time_ns": nonmonotonic_wall
        if nonmonotonic_wall is not None
        else reference_ns + 1_000_000_000,
        "t_cycle_start": reference_ns - 1,
        "t_servoj_send": reference_ns,
        "t_cycle_end": reference_ns + 2,
        "leader_deg": [1, 2, 3, 4, 5, 6],
        "leader_trigger_normalized": 0.5,
        "leader_keys": [0, 0, 0],
        "nova_target_deg": [1, 2, 3, 4, 5, 6],
        "nova_actual_deg": [1, 2, 3, 4, 5, 6],
        "nova_feedback": {
            "controller_timestamp_ms": reference_ns // 1_000_000,
            "q_target_deg": [1, 2, 3, 4, 5, 6],
            "q_actual_deg": [1, 2, 3, 4, 5, 6],
            "robot_mode": 7,
        },
        "nova_feedback_age_ms": 3.0,
        "gripper_target_position": 2000,
        "gripper_actual_position": 1999,
        "gripper_feedback_mono_ns": reference_ns,
        "gripper_status": {"current": 1, "load": 2},
        "gripper_diagnostic_state": "UNKNOWN",
        "clutch_active": False,
        "actual_loop_hz": 30.0,
        "servoj_command": "ServoJ(1,2,3,4,5,6)",
        "status": "ok",
    }


def _episode(
    tmp_path: Path, *, second_frame_ns: int = 150, second_reference_ns: int = 200
) -> tuple[Path, Path]:
    episode_dir = tmp_path / "episode"
    camera_dir = episode_dir / "cameras" / "mystery"
    camera_dir.mkdir(parents=True)
    for index in (1, 2):
        Image.new("RGB", (4, 3), (index, 20, 30)).save(
            camera_dir / f"frame_{index:08d}.jpg", format="JPEG"
        )
    metadata = {
        "record_type": "metadata",
        "schema_version": 2,
        "wall_time_ns": 1,
        "session": {
            "episode": {"label": "synthetic", "trace_path": "episode.jsonl"},
            "cameras": [
                {
                    "name": "mystery",
                    "path": "/dev/video-mystery",
                    "width_requested": 4,
                    "height_requested": 3,
                    "fps_requested": 20.0,
                    "jpeg_quality": 90,
                }
            ],
            "leader_port": "synthetic",
            "nova_ip": "synthetic",
            "nova_port": 29999,
            "gripper": {},
        },
    }
    rows = [metadata, _sample(100), _sample(second_reference_ns)]
    rows.extend(
        [
            {
                "record_type": "camera_frame",
                "camera_name": "mystery",
                "frame_seq": 1,
                "capture_mono_ns": 90,
                "capture_wall_time_ns": 1_090,
                "frame_path": "cameras/mystery/frame_00000001.jpg",
                "width": 4,
                "height": 3,
                "dropped_before": 0,
            },
            {
                "record_type": "camera_frame",
                "camera_name": "mystery",
                "frame_seq": 2,
                "capture_mono_ns": second_frame_ns,
                "capture_wall_time_ns": 1_150,
                "frame_path": "cameras/mystery/frame_00000002.jpg",
                "width": 4,
                "height": 3,
                "dropped_before": 0,
            },
            {
                "record_type": "camera_summary",
                "camera_name": "mystery",
                "frames_captured": 2,
                "frames_written": 2,
                "frames_dropped": 0,
                "actual_fps": 20.0,
                "capture_duration_s": 0.06,
                "capture_error": None,
                "writer_error": None,
                "writer_alive": False,
            },
        ]
    )
    source = tmp_path / "episode.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    return source, episode_dir


def test_valid_episode_preserves_raw_names_and_causal_alignment(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "PASS_WITH_WARNINGS"
    assert result["camera_manifest"]["cameras"][0]["canonical_role"] is None
    assert result["alignment"][0]["cameras"]["mystery"]["frame_seq"] == 1
    assert result["alignment"][0]["cameras"]["mystery"]["camera_timestamp_ns"] <= 100
    assert result["alignment"][0]["cameras"]["mystery"]["frame_seq"] != 2


def test_malformed_json_is_fail(tmp_path: Path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text('{"record_type":"metadata"}\nnot-json\n', encoding="utf-8")
    result = audit_episode(source, episode_dir=tmp_path)
    assert result["qa_report"]["status"] == "FAIL"
    assert any("malformed JSON" in error for error in result["qa_report"]["errors"])


def test_non_monotonic_required_clock_is_fail(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    text = source.read_text(encoding="utf-8").replace(
        '"wall_time_ns": 1000000100', '"wall_time_ns": 1000000100', 1
    )
    rows = [json.loads(line) for line in text.splitlines()]
    rows[2]["wall_time_ns"] = rows[1]["wall_time_ns"] - 1
    source.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "FAIL"
    assert any(
        "wall_time_ns is non-monotonic" in error
        for error in result["qa_report"]["errors"]
    )


def test_missing_or_corrupt_jpeg_is_fail(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    (episode_dir / "cameras" / "mystery" / "frame_00000002.jpg").write_bytes(b"bad")
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "FAIL"
    assert any(
        "corrupt" in error or "cannot identify" in error
        for error in result["qa_report"]["errors"]
    )


def test_summary_mismatch_is_fail(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[-1]["frames_written"] = 1
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "FAIL"
    assert any(
        "summary counts disagree" in error for error in result["qa_report"]["errors"]
    )


def test_missing_camera_summary_is_fail(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows.pop()
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "FAIL"
    assert any(
        "missing camera_summary" in error for error in result["qa_report"]["errors"]
    )


def test_camera_path_mismatch_is_fail(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[4]["frame_path"] = "cameras/mystery/not-the-sequence.jpg"
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "FAIL"
    assert any(
        "does not match expected" in error for error in result["qa_report"]["errors"]
    )


def test_variable_fps_is_warning_not_failure(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path, second_frame_ns=190)
    result = audit_episode(source, episode_dir=episode_dir)
    assert result["qa_report"]["status"] == "PASS_WITH_WARNINGS"
    assert (
        result["camera_manifest"]["cameras"][0]["effective_fps_from_timestamps"] != 20.0
    )


def test_proven_camera_role_mapping(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["session"]["cameras"][0]["path"] = (
        "/dev/v4l/by-id/usb-CAMERA_SERIAL_1080P_USB_Camera-video-index0"
    )
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = audit_episode(source, episode_dir=episode_dir)
    camera = result["camera_manifest"]["cameras"][0]
    assert camera["canonical_role"] == "wrist_rgb"
    assert camera["mapping_status"] == "DOCUMENTED_STATIC"


def test_source_hash_and_annotation_sidecar_do_not_touch_raw(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "derived"
    result = audit_episode(source, episode_dir=episode_dir)
    write_outputs(result, output)
    update_annotations(output, success=True, notes="reviewed", segments=[])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert (
        json.loads((output / "annotations.json").read_text())["episode_success"] is True
    )
    manifest = json.loads((output / "episode_manifest.json").read_text())
    assert manifest["tool_geometry"]["calibration_status"] == "nominal_geometry"
    assert manifest["tool_geometry"]["flange_to_grasp_center_mm"] is None


def test_nonempty_derived_directory_requires_explicit_replace(tmp_path: Path) -> None:
    source, episode_dir = _episode(tmp_path)
    result = audit_episode(source, episode_dir=episode_dir)
    output = tmp_path / "derived"
    write_outputs(result, output)
    with pytest.raises(EpisodeQAError, match="not empty"):
        write_outputs(result, output)


def test_known_bad_trace_flag_is_explicit() -> None:
    from episode_qa import KNOWN_TRACE_FLAGS

    assert (
        KNOWN_TRACE_FLAGS["teleop_20260927_005504_476178737"]["training_status"]
        == "INVALID_FEEDBACK_LABELS"
    )
    assert (
        KNOWN_TRACE_FLAGS["teleop_20260927_011126_716641333"]["training_status"]
        == "NEEDS_SEGMENTATION"
    )
