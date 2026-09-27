"""Offline QA and canonical indexing for raw X-Trainer episodes.

This module deliberately has no robot, serial, camera, Isaac Sim, or model
runtime imports.  It reads the recorder's JSONL and JPEG files, writes a new
derived directory, and never edits the raw episode.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "dobot_episode_v1"
DEFAULT_REFERENCE_FIELD = "t_servoj_send"
CAMERA_AGE_WARNING_MS = 100.0

# These are evidence-backed mappings from the stable device paths used by the
# recorder and the Phase 5A camera contract.  Raw names remain unchanged.
CAMERA_ROLE_RULES = (
    ("CAMERA_SERIAL_1080P_USB_Camera", "wrist_rgb", "Phase 5A stable CAMERA_SERIAL path"),
    ("usb-0:3.1:1.0-video-index0", "front_rgb", "Phase 5A icSpring hub port 3.1"),
    ("usb-0:3.2:1.0-video-index0", "right_rgb", "Phase 5A icSpring hub port 3.2"),
)

KNOWN_TRACE_FLAGS = {
    "teleop_20260927_005504_476178737": {
        "training_status": "INVALID_FEEDBACK_LABELS",
        "reason": "30004 QTarget/QActual were converted with the historical radians bug",
        "scope": "nova_feedback.q_target_deg, nova_feedback.q_actual_deg, tracking errors",
    },
    "teleop_20260927_011126_716641333": {
        "training_status": "NEEDS_SEGMENTATION",
        "reason": "documentation records two manipulation tasks in one continuous episode",
        "scope": "episode-level task labels and training segmentation",
    },
    "teleop_20260927_015239_279634825": {
        "training_status": "ONE_TASK_CANDIDATE",
        "reason": "first formal one-task, three-camera recording; human success annotation remains separate",
        "scope": "episode-level candidate status",
    },
}


class EpisodeQAError(ValueError):
    """Raised for structural or data-integrity failures."""


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    index = int((len(ordered) - 1) * fraction)
    return ordered[index]


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": _percentile(ordered, 0.5),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _vector_ranges(
    values: list[list[float]], width: int
) -> list[dict[str, float | None]]:
    return [
        {
            "min": min((row[index] for row in values), default=None),
            "max": max((row[index] for row in values), default=None),
        }
        for index in range(width)
    ]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_repository_root(source: Path) -> Path:
    """Return the repository root used for provenance paths and commits.

    The recorder stores production traces below ``teleop_traces``.  Synthetic
    test episodes do not live in a Git checkout, so the source directory is a
    safe fallback for those inputs.
    """
    try:
        root = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if root:
            return Path(root).resolve()
    except (OSError, subprocess.CalledProcessError):
        pass
    if source.parent.name == "teleop_traces":
        return source.parent.parent
    return source.parent


def _is_strictly_increasing(values: list[int | float]) -> bool:
    return all(left < right for left, right in pairwise(values))


def _monotonic_report(values: list[int | float]) -> dict[str, Any]:
    violations = [
        index + 1
        for index, (left, right) in enumerate(pairwise(values))
        if right <= left
    ]
    return {
        "count": len(values),
        "strictly_increasing": not violations,
        "violations": violations[:20],
        "violation_count": len(violations),
    }


def _nondecreasing_report(values: list[int | float]) -> dict[str, Any]:
    violations = [
        index + 1
        for index, (left, right) in enumerate(pairwise(values))
        if right < left
    ]
    duplicates = sum(left == right for left, right in pairwise(values))
    return {
        "count": len(values),
        "nondecreasing": not violations,
        "violations": violations[:20],
        "violation_count": len(violations),
        "duplicate_count": duplicates,
    }


def _resolve_frame(episode_dir: Path, relative_path: str) -> Path:
    candidate = (episode_dir / relative_path).resolve()
    root = episode_dir.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise EpisodeQAError(
            f"camera frame escapes episode directory: {relative_path}"
        ) from exc
    return candidate


def _validate_jpeg(
    path: Path, expected_width: int | None, expected_height: int | None
) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "readable": False,
            "error": "missing",
        }
    if path.stat().st_size == 0:
        return {
            "path": str(path),
            "exists": True,
            "readable": False,
            "error": "zero_byte",
        }
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            image.convert("RGB").load()
        result = {
            "path": str(path),
            "exists": True,
            "readable": True,
            "format": "JPEG",
            "width": width,
            "height": height,
        }
        if (
            expected_width is not None
            and expected_height is not None
            and (width, height) != (expected_width, expected_height)
        ):
            result["error"] = (
                f"dimensions {width}x{height}, expected {expected_width}x{expected_height}"
            )
            result["readable"] = False
        return result
    except (ImportError, OSError, ValueError) as exc:
        return {
            "path": str(path),
            "exists": True,
            "readable": False,
            "error": repr(exc),
        }


def _camera_role(path: str) -> tuple[str | None, str, str | None]:
    for token, role, evidence in CAMERA_ROLE_RULES:
        if token in path:
            return role, "DOCUMENTED_STATIC", evidence
    return None, "UNRESOLVED", None


def _safe_float_list(value: Any, field: str, width: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or not all(_finite(item) for item in value)
    ):
        raise EpisodeQAError(f"{field} must contain {width} finite numeric values")
    return [float(item) for item in value]


def _candidate_events(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: str | None = None
    labels = {
        "HOLDING": "holding_candidate",
        "OPENING": "release_candidate",
        "CLOSING": "grasp_candidate",
    }
    for index, sample in enumerate(samples):
        state = sample.get("gripper_diagnostic_state")
        if state != previous and state in labels:
            events.append(
                {
                    "event": labels[state],
                    "state": state,
                    "sample_index": index,
                    "reference_host_monotonic_ns": sample[DEFAULT_REFERENCE_FIELD],
                    "source": "automatic_candidate",
                    "ground_truth": False,
                }
            )
        previous = state
    return events


def _trace_flag(source: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    return KNOWN_TRACE_FLAGS.get(
        source.stem,
        {
            "training_status": "UNCLASSIFIED",
            "reason": "no special historical compatibility flag is registered",
            "scope": "episode-level provenance",
        },
    )


def audit_episode(
    source_jsonl: str | Path,
    *,
    episode_dir: str | Path | None = None,
    validate_images: bool = True,
) -> dict[str, Any]:
    """Parse and QA one raw recorder episode without writing files."""
    source = Path(source_jsonl).resolve()
    if not source.exists():
        raise EpisodeQAError(f"source JSONL does not exist: {source}")
    root = (
        Path(episode_dir).resolve()
        if episode_dir is not None
        else source.parent / source.stem
    )
    errors: list[str] = []
    warnings: list[str] = []
    records: list[tuple[int, dict[str, Any]]] = []
    metadata_records: list[dict[str, Any]] = []
    samples: list[tuple[int, dict[str, Any]]] = []
    frame_records: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(
        list
    )
    summaries: dict[str, dict[str, Any]] = {}
    unknown_types: Counter[str] = Counter()

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                errors.append(f"line {line_number}: blank line")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: malformed JSON: {exc.msg}")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: record is not an object")
                continue
            record_type = record.get("record_type")
            records.append((line_number, record))
            if record_type == "metadata":
                metadata_records.append(record)
            elif record_type == "sample":
                samples.append((line_number, record))
            elif record_type == "camera_frame":
                name = record.get("camera_name")
                if not isinstance(name, str):
                    errors.append(
                        f"line {line_number}: camera_frame has no camera_name"
                    )
                else:
                    frame_records[name].append((line_number, record))
            elif record_type == "camera_summary":
                name = record.get("camera_name")
                if isinstance(name, str):
                    summaries[name] = record
                else:
                    errors.append(
                        f"line {line_number}: camera_summary has no camera_name"
                    )
            else:
                unknown_types[str(record_type)] += 1

    if len(metadata_records) != 1:
        errors.append(f"expected one metadata record, found {len(metadata_records)}")
    metadata = metadata_records[0] if metadata_records else {}
    session = (
        metadata.get("session", {})
        if isinstance(metadata.get("session", {}), dict)
        else {}
    )
    source_camera_configs = (
        session.get("cameras", [])
        if isinstance(session.get("cameras", []), list)
        else []
    )
    config_by_name = {
        item.get("name"): item
        for item in source_camera_configs
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }

    sample_values: list[dict[str, Any]] = [record for _, record in samples]
    required_sample_fields = (
        "wall_time_ns",
        DEFAULT_REFERENCE_FIELD,
        "t_cycle_start",
        "t_cycle_end",
        "leader_deg",
        "nova_target_deg",
        "nova_actual_deg",
        "nova_feedback",
    )
    for index, (line_number, sample) in enumerate(samples):
        for field in required_sample_fields:
            if field not in sample:
                errors.append(f"line {line_number}: sample missing {field}")
        for field in ("leader_deg", "nova_target_deg", "nova_actual_deg"):
            if field in sample:
                try:
                    _safe_float_list(sample[field], field, 6)
                except EpisodeQAError as exc:
                    errors.append(f"line {line_number}: {exc}")
        feedback = sample.get("nova_feedback")
        if isinstance(feedback, dict):
            for field in ("q_target_deg", "q_actual_deg"):
                try:
                    _safe_float_list(feedback[field], f"nova_feedback.{field}", 6)
                except (KeyError, EpisodeQAError) as exc:
                    errors.append(
                        f"line {line_number}: missing or invalid nova_feedback.{field}: {exc}"
                    )
        else:
            errors.append(f"line {line_number}: nova_feedback is not an object")

    monotonic_fields: dict[str, dict[str, Any]] = {}
    for field in (
        "t_cycle_start",
        DEFAULT_REFERENCE_FIELD,
        "t_cycle_end",
        "wall_time_ns",
    ):
        values = [
            sample[field] for sample in sample_values if _finite(sample.get(field))
        ]
        if len(values) != len(sample_values):
            errors.append(f"sample field {field} is missing or non-numeric")
        monotonic_fields[field] = _monotonic_report(values)
        if values and not monotonic_fields[field]["strictly_increasing"]:
            errors.append(f"sample field {field} is non-monotonic")
    for field in ("controller_timestamp_ms",):
        values = [
            sample.get("nova_feedback", {}).get(field)
            for sample in sample_values
            if _finite(sample.get("nova_feedback", {}).get(field))
        ]
        if values:
            monotonic_fields[field] = _monotonic_report(values)
            if not monotonic_fields[field]["strictly_increasing"]:
                warnings.append("controller_timestamp_ms is not strictly increasing")

    leader_values = [
        _safe_float_list(sample["leader_deg"], "leader_deg", 6)
        for sample in sample_values
        if isinstance(sample.get("leader_deg"), list)
        and len(sample["leader_deg"]) == 6
        and all(_finite(v) for v in sample["leader_deg"])
    ]
    target_values = [
        _safe_float_list(sample["nova_target_deg"], "nova_target_deg", 6)
        for sample in sample_values
        if isinstance(sample.get("nova_target_deg"), list)
        and len(sample["nova_target_deg"]) == 6
        and all(_finite(v) for v in sample["nova_target_deg"])
    ]
    actual_values = [
        _safe_float_list(sample["nova_actual_deg"], "nova_actual_deg", 6)
        for sample in sample_values
        if isinstance(sample.get("nova_actual_deg"), list)
        and len(sample["nova_actual_deg"]) == 6
        and all(_finite(v) for v in sample["nova_actual_deg"])
    ]
    q_target_values = [
        _safe_float_list(sample["nova_feedback"]["q_target_deg"], "q_target_deg", 6)
        for sample in sample_values
        if isinstance(sample.get("nova_feedback"), dict)
        and isinstance(sample["nova_feedback"].get("q_target_deg"), list)
        and len(sample["nova_feedback"]["q_target_deg"]) == 6
        and all(_finite(v) for v in sample["nova_feedback"]["q_target_deg"])
    ]
    q_actual_values = [
        _safe_float_list(sample["nova_feedback"]["q_actual_deg"], "q_actual_deg", 6)
        for sample in sample_values
        if isinstance(sample.get("nova_feedback"), dict)
        and isinstance(sample["nova_feedback"].get("q_actual_deg"), list)
        and len(sample["nova_feedback"]["q_actual_deg"]) == 6
        and all(_finite(v) for v in sample["nova_feedback"]["q_actual_deg"])
    ]
    tracking_values = [
        [abs(target - actual) for target, actual in zip(target, actual)]
        for target, actual in zip(q_target_values, q_actual_values)
    ]
    tracking_by_joint = (
        [[row[index] for row in tracking_values] for index in range(6)]
        if tracking_values
        else [[] for _ in range(6)]
    )

    reference_values = [
        sample[DEFAULT_REFERENCE_FIELD]
        for sample in sample_values
        if _finite(sample.get(DEFAULT_REFERENCE_FIELD))
    ]
    control_duration_s = (
        ((reference_values[-1] - reference_values[0]) / 1_000_000_000)
        if len(reference_values) >= 2
        else None
    )
    control_dts_ms = [
        (right - left) / 1_000_000 for left, right in pairwise(reference_values)
    ]

    camera_manifest: list[dict[str, Any]] = []
    camera_frame_indexes: dict[str, list[dict[str, Any]]] = {}
    camera_timestamp_lists: dict[str, list[int]] = {}
    camera_integrity_errors: list[str] = []
    global_frame_paths: set[str] = set()
    configured_names = set(config_by_name) | set(frame_records) | set(summaries)
    for name in sorted(configured_names):
        config = config_by_name.get(name, {})
        device_path = config.get("path")
        role, mapping_status, mapping_evidence = _camera_role(str(device_path or ""))
        rows = frame_records.get(name, [])
        frame_indexes: list[dict[str, Any]] = []
        mono_values: list[int] = []
        wall_values: list[int] = []
        sequences: list[int] = []
        seen_paths: set[str] = set()
        seen_sequences: set[int] = set()
        decoded_dimensions: set[tuple[int, int]] = set()
        for line_number, frame in rows:
            try:
                sequence = int(frame["frame_seq"])
                mono = int(frame["capture_mono_ns"])
                wall = int(frame["capture_wall_time_ns"])
                frame_path = str(frame["frame_path"])
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"line {line_number}: invalid camera frame fields: {exc}")
                continue
            if sequence in seen_sequences:
                errors.append(f"camera {name}: duplicate frame_seq {sequence}")
            if frame_path in seen_paths:
                errors.append(f"camera {name}: duplicate frame_path {frame_path}")
            if frame_path in global_frame_paths:
                errors.append(f"camera {name}: frame_path is reused: {frame_path}")
            if sequence < 1:
                errors.append(f"camera {name}: frame_seq must be positive")
            expected_path = f"cameras/{name}/frame_{sequence:08d}.jpg"
            if frame_path != expected_path:
                errors.append(
                    f"camera {name} line {line_number}: frame_path {frame_path!r} "
                    f"does not match expected {expected_path!r}"
                )
            seen_sequences.add(sequence)
            seen_paths.add(frame_path)
            global_frame_paths.add(frame_path)
            sequences.append(sequence)
            mono_values.append(mono)
            wall_values.append(wall)
            check = (
                _validate_jpeg(
                    _resolve_frame(root, frame_path),
                    frame.get("width"),
                    frame.get("height"),
                )
                if validate_images
                else {"exists": None, "readable": None}
            )
            if check.get("error"):
                camera_integrity_errors.append(
                    f"camera {name} line {line_number}: {check['error']}"
                )
            if check.get("readable"):
                decoded_dimensions.add((check["width"], check["height"]))
            frame_indexes.append(
                {
                    "frame_seq": sequence,
                    "frame_path": frame_path,
                    "capture_mono_ns": mono,
                    "capture_wall_time_ns": wall,
                    "width": frame.get("width"),
                    "height": frame.get("height"),
                    "dropped_before": frame.get("dropped_before", 0),
                    "integrity": check,
                }
            )
        camera_frame_indexes[name] = frame_indexes
        camera_timestamp_lists[name] = mono_values
        timestamp_report = _monotonic_report(mono_values)
        wall_report = _monotonic_report(wall_values)
        frame_dts_ms = [
            (right - left) / 1_000_000 for left, right in pairwise(mono_values)
        ]
        duration_s = (
            ((mono_values[-1] - mono_values[0]) / 1_000_000_000)
            if len(mono_values) >= 2
            else None
        )
        effective_fps = (
            ((len(mono_values) - 1) / duration_s)
            if duration_s and duration_s > 0
            else None
        )
        summary = summaries.get(name, {})
        dropped_total = sum(
            int(row.get("dropped_before") or 0) for row in frame_indexes
        )
        summary_consistent = bool(summary) and (
            summary.get("frames_captured") == len(frame_indexes) + dropped_total
            and summary.get("frames_written") == len(frame_indexes)
            and summary.get("frames_dropped") == dropped_total
        )
        if not summary:
            errors.append(f"camera {name}: missing camera_summary record")
        elif not summary_consistent:
            errors.append(f"camera {name}: summary counts disagree with frame records")
        if mapping_status == "UNRESOLVED":
            warnings.append(f"camera {name}: canonical role is unresolved")
        camera_manifest.append(
            {
                "raw_name": name,
                "canonical_role": role,
                "mapping_status": mapping_status,
                "mapping_evidence": mapping_evidence,
                "device_path": device_path,
                "filename_convention": "frame_%08d.jpg",
                "requested": {
                    key: config.get(key)
                    for key in (
                        "width_requested",
                        "height_requested",
                        "fps_requested",
                        "jpeg_quality",
                    )
                },
                "actual_dimensions": sorted(
                    decoded_dimensions
                    if validate_images
                    else {
                        (row.get("width"), row.get("height")) for row in frame_indexes
                    }
                ),
                "frame_count": len(frame_indexes),
                "first_timestamp": {
                    "capture_mono_ns": mono_values[0] if mono_values else None,
                    "capture_wall_time_ns": wall_values[0] if wall_values else None,
                },
                "last_timestamp": {
                    "capture_mono_ns": mono_values[-1] if mono_values else None,
                    "capture_wall_time_ns": wall_values[-1] if wall_values else None,
                },
                "effective_fps_from_timestamps": effective_fps,
                "recorder_summary": summary,
                "summary_consistent": summary_consistent,
                "timestamp_monotonicity": {
                    "capture_mono_ns": timestamp_report,
                    "capture_wall_time_ns": wall_report,
                },
                "inter_frame_dt_ms": _stats(frame_dts_ms),
                "dropped_before_total": dropped_total,
                "missing_or_corrupt_frames": sum(
                    1
                    for row in frame_indexes
                    if row.get("integrity", {}).get("readable") is False
                ),
            }
        )

    if camera_integrity_errors:
        errors.extend(camera_integrity_errors)
    if unknown_types:
        warnings.append(f"unknown record types present: {dict(unknown_types)}")
    if not sample_values:
        errors.append("no sample records")
    if (
        len(sample_values) >= 2
        and control_duration_s is not None
        and control_duration_s > 0
    ):
        effective_hz = (len(sample_values) - 1) / control_duration_s
    else:
        effective_hz = None

    reference_for_alignment = DEFAULT_REFERENCE_FIELD
    camera_times = {
        name: values for name, values in camera_timestamp_lists.items() if values
    }
    camera_age_values: defaultdict[str, list[float]] = defaultdict(list)
    camera_skew_values: list[float] = []
    no_recent_count = 0
    alignment_preview: list[dict[str, Any]] = []
    for sample_index, (line_number, sample) in enumerate(samples):
        reference_ns = sample.get(reference_for_alignment)
        associated: dict[str, Any] = {}
        latest_timestamps: list[int] = []
        if _finite(reference_ns):
            sample_has_gap = False
            for camera_name, timestamps in camera_times.items():
                position = bisect.bisect_right(timestamps, int(reference_ns)) - 1
                nearest_position = min(
                    range(len(timestamps)),
                    key=lambda i: abs(timestamps[i] - int(reference_ns)),
                )
                if position >= 0:
                    row = camera_frame_indexes[camera_name][position]
                    age_ms = (int(reference_ns) - timestamps[position]) / 1_000_000
                    camera_age_values[camera_name].append(age_ms)
                    latest_timestamps.append(timestamps[position])
                    associated[camera_name] = {
                        "frame_seq": row["frame_seq"],
                        "frame_path": row["frame_path"],
                        "camera_timestamp_ns": timestamps[position],
                        "reference_timestamp_ns": int(reference_ns),
                        "camera_age_ms": age_ms,
                        "causal": True,
                        "nearest_frame_seq": camera_frame_indexes[camera_name][
                            nearest_position
                        ]["frame_seq"],
                        "nearest_delta_ms": (
                            timestamps[nearest_position] - int(reference_ns)
                        )
                        / 1_000_000,
                    }
                else:
                    associated[camera_name] = {
                        "frame_seq": None,
                        "camera_timestamp_ns": None,
                        "reference_timestamp_ns": int(reference_ns),
                        "camera_age_ms": None,
                        "causal": True,
                        "nearest_frame_seq": camera_frame_indexes[camera_name][
                            nearest_position
                        ]["frame_seq"],
                        "nearest_delta_ms": (
                            timestamps[nearest_position] - int(reference_ns)
                        )
                        / 1_000_000,
                    }
                    sample_has_gap = True
            if len(latest_timestamps) == len(camera_times) and latest_timestamps:
                camera_skew_values.append(
                    (max(latest_timestamps) - min(latest_timestamps)) / 1_000_000
                )
            if any(
                value is None or value > CAMERA_AGE_WARNING_MS
                for value in (item.get("camera_age_ms") for item in associated.values())
            ):
                sample_has_gap = True
            if sample_has_gap:
                no_recent_count += 1
        alignment_preview.append(
            {
                "sample_index": sample_index,
                "raw_line_number": line_number,
                "reference_clock": "host_monotonic_ns",
                "reference_field": reference_for_alignment,
                "reference_host_monotonic_ns": reference_ns,
                "wall_time_ns": sample.get("wall_time_ns"),
                "leader": {
                    "leader_deg": sample.get("leader_deg"),
                    "leader_trigger_normalized": sample.get(
                        "leader_trigger_normalized"
                    ),
                    "leader_keys": sample.get("leader_keys"),
                },
                "action": {
                    "nova_target_deg": sample.get("nova_target_deg"),
                    "command": sample.get("servoj_command"),
                    "status": sample.get("status"),
                },
                "robot_actual": {
                    "nova_actual_deg": sample.get("nova_actual_deg"),
                    "q_actual_deg": (sample.get("nova_feedback") or {}).get(
                        "q_actual_deg"
                    ),
                    "controller_timestamp_ms": (sample.get("nova_feedback") or {}).get(
                        "controller_timestamp_ms"
                    ),
                    "robot_mode": (sample.get("nova_feedback") or {}).get("robot_mode"),
                },
                "gripper": {
                    "target_position": sample.get("gripper_target_position"),
                    "actual_position": sample.get("gripper_actual_position"),
                    "feedback_mono_ns": sample.get("gripper_feedback_mono_ns"),
                    "diagnostic_state": sample.get("gripper_diagnostic_state"),
                },
                "cameras": associated,
            }
        )

    trace_flag = _trace_flag(source, metadata)
    geometry_path = source.parent.parent / "gripper_geometry.yaml"
    geometry: dict[str, Any] = {}
    if geometry_path.exists():
        try:
            import yaml

            geometry = yaml.safe_load(geometry_path.read_text(encoding="utf-8")) or {}
        except (ImportError, OSError, TypeError, ValueError) as exc:
            warnings.append(f"could not read gripper_geometry.yaml: {exc}")
    else:
        warnings.append("gripper_geometry.yaml not found")
    nominal_translation = geometry.get("tcp_translation_mm")
    if not isinstance(nominal_translation, list):
        nominal_translation = None
        warnings.append("nominal tool geometry was not present in the source file")
    tool_geometry = {
        "flange_to_grasp_center_mm": nominal_translation,
        "flange_forward_axis": "+Z" if nominal_translation is not None else None,
        "max_opening_width_mm": (
            geometry.get("measurement_source", {}).get("max_opening_width_mm", {}) or {}
        ).get("value"),
        "closed_gap_mm": (
            geometry.get("measurement_source", {}).get("closed_gap_mm", {}) or {}
        ).get("value"),
        "left_right_symmetric": (
            geometry.get("measurement_source", {}).get("grasp_center", {}) or {}
        ).get("left_right_symmetric"),
        "calibration_status": "nominal_geometry",
        "measured_tcp": False,
        "source_path": "gripper_geometry.yaml",
        "source_sha256": _hash_file(geometry_path) if geometry_path.exists() else None,
    }
    if trace_flag["training_status"] == "ONE_TASK_CANDIDATE":
        warnings.append("episode success is unannotated")
    if no_recent_count:
        warnings.append(
            f"{no_recent_count} control samples have no sufficiently recent causal camera frame"
        )
    warnings.extend(
        (
            "camera FPS is timestamp-derived and variable; no fixed-FPS resampling is performed",
            "nominal +Z 197 mm geometry is not a precision TCP calibration",
            "derived nominal EEF pose is deferred; no measured EEF pose is claimed",
        )
    )
    qa_status = "FAIL" if errors else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    sample_duration = control_duration_s
    metrics = {
        "record_counts": {
            "total": len(records),
            "metadata": len(metadata_records),
            "samples": len(samples),
            "camera_frames": sum(len(rows) for rows in frame_records.values()),
            "camera_summaries": len(summaries),
        },
        "control": {
            "sample_count": len(samples),
            "duration_s": sample_duration,
            "effective_hz_from_reference_timestamps": effective_hz,
            "recorded_actual_loop_hz": _stats(
                [
                    sample.get("actual_loop_hz")
                    for sample in sample_values
                    if _finite(sample.get("actual_loop_hz"))
                ]
            ),
            "dt_ms": _stats(control_dts_ms),
            "timestamp_monotonicity": monotonic_fields,
        },
        "leader": {
            "sample_count": len(leader_values),
            "joint_ranges_deg": _vector_ranges(leader_values, 6),
            "trigger_normalized": _stats(
                [
                    sample.get("leader_trigger_normalized")
                    for sample in sample_values
                    if _finite(sample.get("leader_trigger_normalized"))
                ]
            ),
            "clutch_active_samples": sum(
                bool(sample.get("clutch_active")) for sample in sample_values
            ),
            "missing_values": len(sample_values) - len(leader_values),
        },
        "nova": {
            "target_ranges_deg": _vector_ranges(target_values, 6),
            "actual_ranges_deg": _vector_ranges(actual_values, 6),
            "q_target_ranges_deg": _vector_ranges(q_target_values, 6),
            "q_actual_ranges_deg": _vector_ranges(q_actual_values, 6),
            "tracking_error_abs_deg": {
                "all_joints": _stats(value for row in tracking_values for value in row),
                "per_joint": [_stats(values) for values in tracking_by_joint],
            },
            "feedback_missing_samples": len(sample_values) - len(q_actual_values),
            "robot_mode_counts": dict(
                Counter(
                    (sample.get("nova_feedback") or {}).get("robot_mode")
                    for sample in sample_values
                )
            ),
            "feedback_age_ms": _stats(
                [
                    sample.get("nova_feedback_age_ms")
                    for sample in sample_values
                    if _finite(sample.get("nova_feedback_age_ms"))
                ]
            ),
            "controller_timestamp_monotonicity": monotonic_fields.get(
                "controller_timestamp_ms"
            ),
        },
        "gripper": {
            "target_position": _stats(
                [
                    sample.get("gripper_target_position")
                    for sample in sample_values
                    if _finite(sample.get("gripper_target_position"))
                ]
            ),
            "actual_position": _stats(
                [
                    sample.get("gripper_actual_position")
                    for sample in sample_values
                    if _finite(sample.get("gripper_actual_position"))
                ]
            ),
            "current": _stats(
                [
                    sample.get("gripper_status", {}).get("current")
                    for sample in sample_values
                    if _finite((sample.get("gripper_status") or {}).get("current"))
                ]
            ),
            "load": _stats(
                [
                    sample.get("gripper_status", {}).get("load")
                    for sample in sample_values
                    if _finite((sample.get("gripper_status") or {}).get("load"))
                ]
            ),
            "diagnostic_state_counts": dict(
                Counter(
                    sample.get("gripper_diagnostic_state") for sample in sample_values
                )
            ),
            "feedback_timestamp_monotonicity": _nondecreasing_report(
                [
                    sample["gripper_feedback_mono_ns"]
                    for sample in sample_values
                    if _finite(sample.get("gripper_feedback_mono_ns"))
                ]
            ),
        },
        "cameras": camera_manifest,
        "cross_stream": {
            "reference_clock": "host_monotonic_ns",
            "reference_field": reference_for_alignment,
            "camera_age_ms": {
                name: _stats(values) for name, values in camera_age_values.items()
            },
            "camera_host_timestamp_skew_ms": _stats(camera_skew_values),
            "samples_without_recent_camera_le_100ms": no_recent_count,
        },
    }

    repository_root = _source_repository_root(source)
    try:
        source_relative_path = source.relative_to(repository_root).as_posix()
    except ValueError:
        source_relative_path = source.name
    repository_commit = _git_commit(repository_root)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": source.stem,
        "original_episode_label": session.get("episode", {}).get("label"),
        "source_jsonl": source.name,
        "source_jsonl_relative_path": source_relative_path,
        "source_jsonl_sha256": _hash_file(source),
        "source_episode_directory": root.name,
        "source_camera_directories": {
            name: f"cameras/{name}" for name in sorted(configured_names)
        },
        "source_jpeg_counts": {
            item["raw_name"]: item["frame_count"] for item in camera_manifest
        },
        "source_recorder_schema_version": metadata.get("schema_version"),
        "source_git_commit": repository_commit,
        "canonicalizer_git_commit": repository_commit,
        "canonicalizer_worktree_dirty": (
            repository_commit is not None
            and bool(
                subprocess.run(
                    ["git", "-C", str(repository_root), "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()
            )
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_status": "IMMUTABLE_SOURCE_AUDITED",
        "review_video": "cube_to_frame_002_multiview.mp4"
        if (root / "cube_to_frame_002_multiview.mp4").exists()
        else None,
        "time_domains": {
            "host_monotonic_ns": "process-local monotonic timestamps used for control/camera association",
            "wall_time_ns": "host wall clock timestamps retained for external chronology",
            "controller_timestamp_ms": "Nova 30004 controller clock; never subtracted from host clocks",
        },
        "compatibility": trace_flag,
    }
    annotations = {
        "schema_version": "dobot_episode_annotations_v1",
        "episode_success": None,
        "success_source": "unknown",
        "notes": None,
        "segments": [],
        "candidate_events": _candidate_events(sample_values),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provenance": provenance,
        "task": {
            "label": session.get("episode", {}).get("label"),
            "episode_candidate_status": trace_flag["training_status"],
            "raw_camera_names": sorted(configured_names),
        },
        "hardware": {
            "nova": {
                "ip": session.get("nova_ip"),
                "dashboard_port": session.get("nova_port"),
                "joint_units": "degrees",
                "feedback_source": "Nova 30004 QTarget/QActual",
                "zero_deg": session.get("nova_zero_deg"),
            },
            "leader": {
                "port": session.get("leader_port"),
                "joint_units": "degrees",
                "zero_deg": session.get("leader_zero_deg"),
            },
            "gripper": session.get("gripper"),
            "teleop_config": session.get("teleop_config"),
            "cameras": camera_manifest,
        },
        "tool_geometry": tool_geometry,
        "streams": {
            "leader": {
                "record_type": "sample",
                "fields": ["leader_deg", "leader_trigger_normalized", "leader_keys"],
            },
            "actions": {
                "record_type": "sample",
                "fields": ["nova_target_deg", "servoj_command", "status"],
            },
            "robot": {
                "record_type": "sample",
                "fields": [
                    "nova_actual_deg",
                    "nova_feedback.q_target_deg",
                    "nova_feedback.q_actual_deg",
                    "nova_feedback.controller_timestamp_ms",
                    "nova_feedback.robot_mode",
                ],
                "units": "native joint degrees",
            },
            "gripper": {
                "record_type": "sample",
                "fields": [
                    "gripper_target_position",
                    "gripper_actual_position",
                    "gripper_status",
                    "gripper_diagnostic_state",
                ],
            },
            "cameras": {
                "raw_names_preserved": True,
                "timestamp_field": "capture_mono_ns",
                "frame_path_field": "frame_path",
            },
        },
        "alignment": {
            "reference_clock": "host_monotonic_ns",
            "reference_field": reference_for_alignment,
            "method": "latest frame with camera capture_mono_ns <= reference timestamp (causal); nearest delta retained separately",
            "future_frames_used": False,
            "camera_age_warning_threshold_ms": CAMERA_AGE_WARNING_MS,
            "raw_timing_preserved": True,
        },
        "annotations": {"sidecar": "annotations.json", "success_is_unset": True},
        "qa": {
            "status": qa_status,
            "errors": errors,
            "warnings": warnings,
            "metrics_file": "qa_report.json",
        },
    }
    return {
        "manifest": manifest,
        "qa_report": {
            "schema_version": SCHEMA_VERSION,
            "status": qa_status,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
            "compatibility": trace_flag,
        },
        "camera_manifest": {
            "schema_version": SCHEMA_VERSION,
            "cameras": camera_manifest,
        },
        "annotations": annotations,
        "alignment": alignment_preview,
        "raw_audit": {
            "source": str(source),
            "record_counts": metrics["record_counts"],
            "record_types": dict(
                Counter(record.get("record_type") for _, record in records)
            ),
            "unknown_record_types": dict(unknown_types),
            "sample_fields": sorted(
                {key for sample in sample_values for key in sample}
            ),
            "metadata": metadata,
            "metrics": metrics,
            "errors": errors,
            "warnings": warnings,
        },
    }


def write_outputs(
    result: dict[str, Any], output_dir: str | Path, *, replace_derived: bool = False
) -> Path:
    destination = Path(output_dir)
    expected_files = {
        "episode_manifest.json",
        "qa_report.json",
        "camera_manifest.json",
        "annotations.json",
        "raw_audit.json",
        "alignment.jsonl",
        "README.md",
    }
    if destination.exists() and any(destination.iterdir()) and not replace_derived:
        raise EpisodeQAError(f"derived output directory is not empty: {destination}")
    if destination.exists() and any(
        path.name not in expected_files for path in destination.iterdir()
    ):
        raise EpisodeQAError(
            f"derived output directory contains an unexpected file: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "episode_manifest.json").write_text(
        _json_dump(result["manifest"]), encoding="utf-8"
    )
    (destination / "qa_report.json").write_text(
        _json_dump(result["qa_report"]), encoding="utf-8"
    )
    (destination / "camera_manifest.json").write_text(
        _json_dump(result["camera_manifest"]), encoding="utf-8"
    )
    (destination / "annotations.json").write_text(
        _json_dump(result["annotations"]), encoding="utf-8"
    )
    (destination / "raw_audit.json").write_text(
        _json_dump(result["raw_audit"]), encoding="utf-8"
    )
    with (destination / "alignment.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["alignment"]:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    readme = f"""# Phase 5D derived episode index\n\nThis directory is derived from the immutable raw JSONL and JPEG files.\n\n- Schema: `{result["manifest"]["schema_version"]}`\n- QA status: **{result["qa_report"]["status"]}**\n- Raw source: `{result["manifest"]["provenance"]["source_jsonl_relative_path"]}`\n- Alignment: causal latest camera frame with `capture_mono_ns <= t_servoj_send`; no image resampling\n- Raw camera names are preserved; canonical roles are evidence-backed or explicitly unresolved.\n- `annotations.json` is a sidecar. It does not modify the raw episode.\n- The MP4, when present, is human review only and is not used for timestamps.\n\nSee `episode_manifest.json`, `qa_report.json`, `camera_manifest.json`, `alignment.jsonl`, and `raw_audit.json`.\n"""
    (destination / "README.md").write_text(readme, encoding="utf-8")
    return destination


def update_annotations(
    output_dir: str | Path,
    *,
    success: bool | None,
    notes: str | None,
    segments: list[dict[str, Any]],
) -> Path:
    destination = Path(output_dir)
    annotation_path = destination / "annotations.json"
    if not annotation_path.exists():
        raise EpisodeQAError(f"annotation sidecar does not exist: {annotation_path}")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotations["episode_success"] = success
    annotations["success_source"] = "operator" if success is not None else "unknown"
    if notes is not None:
        annotations["notes"] = notes
    annotations.setdefault("segments", []).extend(segments)
    annotation_path.write_text(_json_dump(annotations), encoding="utf-8")
    return annotation_path


def _parse_segment(value: str) -> dict[str, Any]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "segment must be LABEL:START_MONOTONIC_NS:END_MONOTONIC_NS"
        )
    try:
        start, end = int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("segment timestamps must be integers") from exc
    if start > end:
        raise argparse.ArgumentTypeError("segment start must not exceed end")
    return {
        "label": parts[0],
        "start_timestamp": start,
        "end_timestamp": end,
        "source": "operator",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="audit a raw episode and write derived outputs"
    )
    validate.add_argument("--episode", required=True, type=Path)
    validate.add_argument("--episode-dir", type=Path)
    validate.add_argument("--output-dir", required=True, type=Path)
    validate.add_argument(
        "--skip-images",
        action="store_true",
        help="skip JPEG decode checks (not suitable for final QA)",
    )
    validate.add_argument(
        "--replace-derived",
        action="store_true",
        help="replace only existing files in a derived output directory",
    )
    annotate = subparsers.add_parser(
        "annotate", help="update the derived annotation sidecar"
    )
    annotate.add_argument("--output-dir", required=True, type=Path)
    annotate.add_argument(
        "--success", choices=("true", "false", "unknown"), default="unknown"
    )
    annotate.add_argument("--notes")
    annotate.add_argument("--segment", action="append", type=_parse_segment, default=[])
    args = parser.parse_args()
    try:
        if args.command == "validate":
            result = audit_episode(
                args.episode,
                episode_dir=args.episode_dir,
                validate_images=not args.skip_images,
            )
            write_outputs(result, args.output_dir, replace_derived=args.replace_derived)
            print(
                json.dumps(
                    {
                        "status": result["qa_report"]["status"],
                        "output_dir": str(args.output_dir),
                        "raw_source": str(args.episode),
                    },
                    indent=2,
                )
            )
            return 0 if result["qa_report"]["status"] != "FAIL" else 2
        success = {"true": True, "false": False, "unknown": None}[args.success]
        print(
            update_annotations(
                args.output_dir,
                success=success,
                notes=args.notes,
                segments=args.segment,
            )
        )
        return 0
    except (EpisodeQAError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
