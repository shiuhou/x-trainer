# Phase 5D Dataset QA

The QA tool is `episode_qa.py`. It is offline-only and imports no Nova,
Leader, camera, gripper, Isaac Sim, or policy runtime. It validates raw JSONL
and every referenced JPEG, then writes a new derived directory.

## Commands

From `x-trainer/`:

```bash
python3 episode_qa.py validate \
  --episode teleop_traces/teleop_20260927_015239_279634825.jsonl \
  --output-dir teleop_traces/teleop_20260927_015239_279634825/derived/phase5d
```

The command refuses a non-empty destination unless the explicit
`--replace-derived` option is used. That option only replaces the seven known
derived artifact names; raw episode files are never candidates for replacement.

To add a human annotation without editing the raw JSONL:

```bash
python3 episode_qa.py annotate \
  --output-dir teleop_traces/teleop_20260927_015239_279634825/derived/phase5d \
  --success true \
  --notes "operator review note" \
  --segment approach:380225071928343:380225500000000
```

`--success unknown` keeps the value null. Segment timestamps are host
monotonic nanoseconds and the segment source is recorded as `operator`.

## Decision criteria

`FAIL` is reserved for malformed JSONL, missing required sample/action/state
fields, required clock non-monotonicity, missing or corrupt referenced JPEGs,
invalid decoded dimensions, duplicate or cross-camera frame identity, frame
paths that do not match the raw camera/sequence convention, a missing camera
summary, or inconsistent camera summary counts. A variable capture rate,
camera age gap, unresolved camera role, unannotated success, optional
telemetry omission, nominal tool geometry, and deferred EEF derivation are
warnings.

The target episode is `PASS_WITH_WARNINGS`: all 6544 referenced images decode
as RGB JPEGs at 640x480, all camera summaries agree, and required control and
controller clocks are strictly increasing. Its warnings are retained in
`qa_report.json` rather than hidden by a pass/fail shortcut.

## Safety and scope

The validator never opens a device or network socket and never imports the
live teleoperation clients. It does not infer task success from images, run
object detection/OCR, derive labels from the MP4, normalize joints, resize
images, resample timestamps, or convert to LeRobot/HDF5/RLDS/WebDataset.
Those operations remain later phases.
