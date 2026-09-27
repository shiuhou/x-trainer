# Canonical Episode Schema

Phase 5D defines `dobot_episode_v1` for the raw X-Trainer recorder output.
The schema is an audit/index layer. It does not replace or rewrite the raw
JSONL, JPEGs, or review MP4.

## Files

Each derived episode is written under the episode directory:

```text
derived/phase5d/
  episode_manifest.json
  qa_report.json
  camera_manifest.json
  alignment.jsonl
  annotations.json
  raw_audit.json
  README.md
```

`episode_manifest.json` is the stable description of provenance, streams,
tool geometry, and alignment semantics. `qa_report.json` contains calculated
metrics and the PASS/PASS_WITH_WARNINGS/FAIL decision. `alignment.jsonl` is a
reversible index: it points to raw sample lines and raw camera frame paths.
`annotations.json` is an editable sidecar. `raw_audit.json` preserves the
forensic record type and field inventory.

Provenance includes the source JSONL SHA256, source camera directories and
JPEG counts, recorder and repository commit identifiers, creation time, and a
`canonicalizer_worktree_dirty` marker. A dirty marker records repository state
at conversion time; it does not change the raw source identity.

## Raw stream semantics

The recorder's `sample` record is preserved as one control observation. It
contains the Leader state, the commanded Nova `ServoJ` target, Nova actual
feedback, and gripper diagnostics. Canonical indexing does not split these
fields into independently reconstructed streams.

- Leader and Nova joints are native **degrees**.
- `nova_target_deg` is the commanded ServoJ target.
- `nova_actual_deg` and `nova_feedback.q_actual_deg` are actual joint values.
- `nova_feedback.q_target_deg` is the 30004 target in native degrees.
- `controller_timestamp_ms` is the Nova controller clock.
- `t_cycle_*`, `t_servoj_*`, `*_mono_ns`, and camera `capture_mono_ns` are the
  host monotonic clock domain for this process/session.
- `wall_time_ns` and `capture_wall_time_ns` are host wall-clock timestamps.

Controller timestamps are never subtracted from host monotonic timestamps.
No exposure synchronization is claimed.

## Alignment

The alignment reference is `t_servoj_send` in the host monotonic domain. For
each control sample, each camera uses the latest frame satisfying:

```text
camera.capture_mono_ns <= sample.t_servoj_send
```

The index stores frame sequence, raw relative path, camera timestamp, reference
timestamp, and age in milliseconds. A nearest-frame delta is stored separately
for QA; it does not replace the causal association and may refer to a future
frame. Images are never interpolated, resampled, renamed, or extracted from
the MP4.

## Camera identity

Raw folder names are immutable. Canonical roles are assigned from the recorded
stable device path and the Phase 5A contract:

| Raw name | Evidence-backed role | Evidence |
| --- | --- | --- |
| `front` | `wrist_rgb` | CAMERA_SERIAL stable by-id path |
| `left` | `front_rgb` | icSpring hub port 3.1 by-path |
| `right` | `right_rgb` | icSpring hub port 3.2 by-path |

An unrecognized path remains `canonical_role: null` and
`mapping_status: UNRESOLVED`. A name alone is never sufficient.

## Tool and gripper

`gripper_geometry.yaml` contributes nominal flange-to-grasp-center geometry:
`[0, 0, 197]` mm along flange/tool `+Z`, symmetric left/right. The canonical
status is `calibration_status: nominal_geometry` and `measured_tcp: false`.
It is not a precision TCP calibration. Gripper positions, current, load,
diagnostic state, and torque status remain raw diagnostic fields; candidate
events are not ground-truth manipulation labels.

## Annotations and compatibility

`annotations.json` starts with `episode_success: null` and
`success_source: unknown`. The annotation CLI can add operator success, notes,
and timestamped segments without touching raw data. Automatically proposed
gripper transitions are marked `ground_truth: false`.

Known historical limitations are explicit in `provenance.compatibility`:
pilot_001 has invalid 30004 feedback labels because of the old radians bug,
pilot_002 contains two tasks and needs segmentation, and
`cube_to_frame_002` is a one-task candidate pending human success annotation.
