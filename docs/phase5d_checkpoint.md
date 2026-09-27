# Phase 5D Checkpoint: Episode Dataset QA + Canonical Schema

Date: 2026-09-27
Mode: offline only
Status: **FROZEN — PASS_WITH_WARNINGS for the target canonical candidate**

Local annotated tag: `phase5d-checkpoint`. Resolve the frozen commit with
`git rev-parse 'phase5d-checkpoint^{commit}'`.

## Scope and safety

The target raw JSONL, JPEGs, and review MP4 were read only. No Nova, Leader,
camera, gripper, network, serial port, Isaac Sim, Astra, training runtime, or
Robocurve motion path was opened or changed. No training-format conversion was
performed.

## Target result

Source:
`teleop_traces/teleop_20260927_015239_279634825.jsonl` (`cube_to_frame_002`)
Derived output:
`teleop_traces/teleop_20260927_015239_279634825/derived/phase5d/`

Raw SHA256 remained:
`87bdcd6ee68bcc1edb9e2ff79a764a34d239043f918032f3ad5d83d205082e9c`.
The source has 11,385 records: 1 metadata, 4,837 samples, 6,544 camera
frames, and 3 camera summaries. All 6,544 referenced JPEGs exist, decode as
RGB JPEGs, and are 640x480. Camera summary counts agree and all camera frame
timestamps and sequences are monotonic.

The control reference is host `t_servoj_send`, duration 145.545677031 s,
4,836 intervals, timestamp-derived rate 33.2266825 Hz. Recorder
`actual_loop_hz` median is 33.2713079 Hz. The derived alignment contains 4,837
records and uses the latest causal frame with
`capture_mono_ns <= t_servoj_send`; no future frame is used.

Raw camera names stay unchanged. Evidence-backed mappings are:

```text
front -> wrist_rgb  (CAMERA_SERIAL by-id path)
left  -> front_rgb  (icSpring USB hub port 3.1)
right -> right_rgb  (icSpring USB hub port 3.2)
```

The target has 2,185 / 2,182 / 2,177 frames, zero drops, and derived rates
14.8770 / 14.8999 / 14.8995 FPS. The rate is metadata; no fixed-FPS assumption
or resampling is introduced.

The canonical tool record stores flange-to-grasp-center `[0, 0, 197]` mm along
`+Z` as `calibration_status: nominal_geometry`, `measured_tcp: false`.
No nominal EEF pose was derived because the phase does not need to couple the
MuJoCo implementation into the QA converter.

Compatibility provenance is explicit: this trace is
`ONE_TASK_CANDIDATE`; pilot_001 is registered as invalid for 30004 feedback
training labels, and pilot_002 is registered as `NEEDS_SEGMENTATION`.
`annotations.json` has no fabricated success label; candidate gripper events
are marked `ground_truth: false`.

Warnings in the target report are: no human success annotation, ten initial
control samples without a sufficiently recent causal right-camera frame,
variable timestamp-derived FPS, nominal rather than precision TCP geometry,
and deferred nominal EEF pose derivation.

## Files changed in this phase

- `episode_qa.py`: offline parser, JPEG validator, QA metrics, compatibility
  flags, causal alignment, canonical manifest, strict camera identity checks,
  and annotation CLI.
- `test_episode_qa.py`: twelve synthetic integrity and provenance tests.
- `docs/episode_schema.md`: versioned canonical contract.
- `docs/phase5d_dataset_qa.md`: commands and PASS/WARNING/FAIL criteria.
- `docs/phase5d_raw_audit.md`: target forensic audit.
- `docs/phase5d_checkpoint.md`: this checkpoint.
- `docs/phase5d_freeze.json`: checksums binding the implementation, raw source,
  camera inventories, review video, geometry reference, and seven derived files.
- `README_NOVA_TELEOP.md`: offline QA entry point.

The derived episode directory is ignored by the existing raw-trace ignore
rule, but is present locally at the path above. Raw files were not rewritten.

## Validation

Targeted quality results:

```text
pytest test_phase3.py test_episode_qa.py dynamixel/test_driver.py: 26 passed
Ruff check (new files): passed
Ruff format --check (new files): passed
Python compile (new files): passed
git diff --check: passed
target validator: PASS_WITH_WARNINGS
```

The target manifest records the source path relative to the x-trainer Git
root and records that the canonicalizer worktree was dirty at generation time;
the source JSONL SHA256 is the immutable identity for later reproducibility.

The validator now fails closed for a missing camera summary, a frame path that
does not match its raw camera/sequence convention, or a frame path reused by
another camera. Decoded JPEG dimensions, rather than only recorder metadata,
are retained in `camera_manifest.json`.

The repo-wide Ruff/format scan still reports pre-existing violations across
the uncommitted teleop and vendored LeIsaac/Dynamixel tree. Full-tree compile
also encounters the existing Python 2 `DynamixelSDK/.../multi_port.py` example.
Those files were not modified for Phase 5D; the new files are clean under the
same tools.

Starting x-trainer HEAD: `5862c3ba4997ae0d4c41f69c73981353af3a8346`.
The operator requested a local commit and freeze after the implementation
checkpoint. The annotated tag `phase5d-checkpoint` identifies that commit.
Only the Phase 5D implementation, tests, documentation, freeze manifest, and
the existing `README_NOVA_TELEOP.md` documentation snapshot are included.
Other live runtime/configuration changes remain outside this commit.
The 26-test result includes 9 local Phase 3 tests, 12 Phase 5D tests and 5
existing FakeDynamixel tests; the Phase 3 test/runtime files remain local
untracked work, so that combined command requires this workspace.

Raw evidence and the derived output remain local and are not added to Git.
`phase5d_freeze.json` records their current byte identities, including a
SHA256 inventory over all 6,544 original JPEGs. The derived generation
metadata still correctly names the earlier HEAD and dirty worktree; the
freeze manifest binds the exact implementation without rewriting those
artifacts. This freeze is a version/checksum checkpoint, not a backup or
filesystem write lock. Subsequent annotations or validation outputs should
use a new derived directory to retain this checkpoint's original bytes.
No push or hardware operation is part of this freeze.

## Exit boundary

This phase freezes raw-to-canonical semantics only. Do not convert to LeRobot,
HDF5, RLDS, or WebDataset; do not downsample/resize/normalize; do not train or
start Astra; and do not collect another live episode until additional episodes
pass this validator and receive separate review.
