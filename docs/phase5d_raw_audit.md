# Phase 5D Raw Forensic Audit

Source: `teleop_traces/teleop_20260927_015239_279634825.jsonl`
Episode label: `cube_to_frame_002`
Raw SHA256: `87bdcd6ee68bcc1edb9e2ff79a764a34d239043f918032f3ad5d83d205082e9c`

The source has 11,385 JSONL records:

| Record type | Count |
| --- | ---: |
| metadata | 1 |
| sample | 4,837 |
| camera_frame | 6,544 |
| camera_summary | 3 |

There is no separate shutdown/final-summary record. The three camera summary
records are the final records after the interleaved sample and frame records.

## Control, Leader, Nova, and gripper

The control reference is `t_servoj_send` in host monotonic nanoseconds. It spans
145.545677031 seconds and has 4,836 intervals. The timestamp-derived effective
rate is 33.2266825 Hz; the recorder's `actual_loop_hz` has median 33.2713079 Hz,
p95 33.2763753 Hz, and maximum 33.3276065 Hz. Reference dt is median
30.072981 ms, p95 30.093456 ms, maximum 71.055491 ms. All required control,
cycle, wall, and controller timestamps are strictly increasing.

All 4,837 samples include six Leader degrees, six commanded Nova degrees, six
actual Nova degrees, and 30004 feedback. Leader trigger normalized range is
`0.4661094623..0.9988543122`; 775 samples have `clutch_active=true`.
Nova `RobotMode` is 7 for 4,836 samples and 5 for one sample. No error field is
present in this recorder schema. Across all six joints, absolute target-vs-
actual error is median 0.0265274 degrees, p95 0.4536860 degrees, maximum
1.5986404 degrees. 30004 feedback age is median 34.093390 ms, p95 37.842843 ms,
maximum 72.372997 ms. Complete ranges are retained in `qa_report.json`.

Gripper target position is present in 4,406 samples, range 1945..2489. Actual
position is present in all 4,837 samples, range 1977..2488. Current range is
0..93 raw counts, load range is 0..1128 raw counts. Diagnostic states are:

```text
CLOSED 3659, OPENING 443, UNKNOWN 431, HOLDING 240, CLOSING 64
```

Repeated `gripper_feedback_mono_ns` values are expected because a control record
can reuse the latest gripper read; the field is non-decreasing with 4,231
duplicates and no backwards violations.

## Cameras

All three were configured at 640x480 and requested 20 FPS. Every referenced
JPEG exists, decodes as RGB JPEG, and is 640x480. Counts and timestamp-derived
rates are:

| Raw name | Stable device evidence | Canonical role | Frames | Derived FPS | Summary FPS | Drops |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `front` | CAMERA_SERIAL by-id | `wrist_rgb` | 2185 | 14.8770423 | 14.8838541 | 0 |
| `left` | icSpring hub 3.1 by-path | `front_rgb` | 2182 | 14.8999175 | 14.9067492 | 0 |
| `right` | icSpring hub 3.2 by-path | `right_rgb` | 2177 | 14.8995461 | 14.9063934 | 0 |

The recorder summary uses `frames_captured / duration`, while the derived rate
uses `(frame_count - 1) / (last_capture_mono_ns - first_capture_mono_ns)`.
The difference is reported, not forced away. All mono and wall capture
timestamps are strictly increasing, frame sequences run from 1 without gaps,
and no frame is missing or corrupt. The source names remain `front`, `left`,
and `right`; filenames follow `frame_%08d.jpg`. The role mapping comes from the
stable device paths and Phase 5A contract, not from the names.

## Alignment and review policy

`alignment.jsonl` associates each control record causally with the latest camera
frame whose `capture_mono_ns` is at or before `t_servoj_send`. No future frame
is used for that association. Camera age medians are approximately 33.56 ms
(`front`), 33.44 ms (`left`), and 33.60 ms (`right`); the latest-frame host
timestamp skew has median 38.54 ms and p95 57.46 ms. Ten initial control
samples have no sufficiently recent causal frame because the right camera starts
slightly later. These are host timestamp metrics, not exposure synchronization.

The multiview MP4 is human review only. Raw JPEG paths and their capture
timestamps remain authoritative.

## QA outcome

Overall status: **PASS_WITH_WARNINGS**. Warnings are: no human success
annotation, ten initial camera alignment gaps, variable timestamp-derived FPS,
nominal rather than precision TCP geometry, and deferred nominal EEF pose
derivation. The canonical output is the first formal one-task three-camera
candidate, not a training-ready dataset.
