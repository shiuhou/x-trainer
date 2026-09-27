# Nova and X-Trainer Leader Teleoperation

## Current field status (2026-09-27)

The direct Leader-to-Nova runtime has completed a real-hardware pilot with
6DoF relative ServoJ, B-button clutch/re-anchor, continuous Feetech gripper
trigger control, Nova 30004 actual-joint feedback and synchronized camera
episode recording. The latest pilot was
`teleop_traces/teleop_20260927_011126_716641333.jsonl`: `3400` control samples,
`1908/1908` camera frames written with no drops, approximately `33.26 Hz`
control rate, and a successful `0,{},Stop();` response. The operator reported
two small-cube-to-frame tasks in that one continuous trace.

The cross-project event history is in
[`../EVENT_LOG.md`](../EVENT_LOG.md). This direct runtime remains separate from
Isaac Sim and from the Robocurve live motion adapter. Pilot evidence is not a
general hardware safety certification; check the current `RobotMode == 5`, work
area and emergency stop before each live run.

This directory contains a lightweight Python teleoperation path for:

```text
X-Trainer HAND_RIGHT USB -> Python -> Dobot Nova TCP 29999
```

It does not require Isaac Sim at runtime. The scripts use the Dynamixel SDK
copied under `source/leisaac/leisaac/xtrainer_utils/` for Leader reads.

## Hardware and calibration

| Item | Value |
| --- | --- |
| Nova controller | `SET_NOVA_IP` |
| Nova dashboard port | `29999` |
| Leader port | `/dev/serial/by-id/SET_LEADER_RIGHT_SERIAL` |
| Leader baud rate | `2000000` |
| Leader joint IDs | `11, 12, 14, 15, 16, 17` |
| Leader append ID | `13` |
| Leader trigger ID | `18` |
| Position register | address `140`, length `4` |
| Torque register | address `64` |
| Measured Leader trigger open/closed | `205.3°` / `169.5°` |

The Nova end-effector servo was identified on September 25, 2026 through the
external CH340 USB-RS485 adapter. This is a separate serial bus from the
Leader:

| Gripper item | Verified value |
| --- | --- |
| Host port | `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0` (currently `/dev/ttyUSB0`) |
| Protocol | Feetech/SCServo protocol `0` |
| Baud rate | `115200` |
| Servo ID | `1` |
| Observed open range | approximately `1450` |
| Observed closed range | approximately `2507` |
| Direction | smaller position opens; larger position closes |

The historical `DobotGripper` class is not used by the modular runtime because
it hardcodes 1 Mbps and writes a torque-limit register during construction.
`nova_gripper.py` opens read-only and changes torque only after the explicit
`--gripper-enable-torque` opt-in.

The right-hand calibration is the one in
`source/leisaac/leisaac/xtrainer_utils/utils/dobot_config/dobot_settings.ini`.
The standalone scripts use the stable `serial/by-id` path directly, so a
changing `/dev/ttyACM*` number does not affect them.

The Leader trigger normalization uses the measured endpoints `205.3°`
(open) and `169.5°` (closed). Values between these endpoints remain continuous;
only values outside the measured range are clamped to 0 or 1.

The attached Nova gripper is confirmed as `Tool 0`; no `Tool 2` is present.
Confirmed finger geometry is recorded in `gripper_geometry.yaml`:
`95 mm` maximum opening, `0 mm` closed gap, `130 mm` single finger length,
`11 mm` fingertip width, and `9 mm` fingertip thickness. The grasp center is
physically confirmed `197 mm` in front of the flange, along Tool 0 `+Z`, and
left/right symmetric. The translation is recorded as
`tcp_translation_mm: [0, 0, 197]`, and the assembled hardware was also
physically verified with `tcp_rotation_deg: [0, 0, 0]`.

The first offline TCP calibration workflow is now available. The
`collect_tcp_samples.py` utility only reads `GetPose(user=0,tool=0)` and never
sends motion, enable, clear-error, or Tool-setting commands. Place the same
physical gripper reference point on a fixed reference point in at least six
different orientations, keep the robot still, and capture one sample per
orientation after confirming the endpoint manually:

```bash
python3 collect_tcp_samples.py \
  --output tcp_samples/gripper_tcp_samples.json
python3 tcp_calibration.py tcp_samples/gripper_tcp_samples.json
```

`tcp_calibration.py` solves only the flange-relative translation and reports
the fixed-point residual. It follows the documented Dobot fixed-axis
X->Y->Z Euler convention. Review the residual and the physical contact setup
before copying the printed candidate into `gripper_geometry.yaml`; TCP
rotation is intentionally still null and no Tool definition is written
automatically.

The first eight poses were recovered from the operator's terminal output into
`tcp_samples/gripper_tcp_samples.json` after the original save failed: this
repository's `checkpoints` is a dangling upstream symlink. The recovered
result has 7.0 mm RMS and 10.3 mm maximum fixed-point residual, so it is
evidence only, not an accepted TCP. The collector now saves after every
sample and refuses to overwrite an existing output file. Use a new filename
for another capture.

Live verification on 2026-09-26 succeeded with:

```bash
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --gripper --gripper-enable-torque --clutch-key b
```

This verified six-axis ServoJ, B-button clutch/re-anchor, continuous trigger
commands to the Feetech gripper, and shutdown torque disable.

The verified relative mapping is:

```text
Nova delta = Leader delta * [1, -1, -1, 1, -1, 1]
```

At startup the scripts capture both `leader_zero` and `nova_zero`. Leader
absolute angles are never sent directly to Nova.

## Phase 3 identification timestamps

`run_servoj_identification.py` records three deliberately separate time
domains. `t_monotonic_ns`, `servo_send_ns`, `servo_reply_ns`, and
`feedback_host_timestamp_ns` use the same local monotonic clock and are valid
for latency, freshness, and ordering calculations within one process. The
`feedback_controller_timestamp` field is Nova's controller timestamp in
milliseconds and is retained for diagnostics; it is not converted to host
time and must not be subtracted from a host monotonic value. Session and
teleoperation traces also retain `wall_time_ns` / `capture_wall_time_ns` for
human-readable cross-process correlation, but wall time is not used for stale
data decisions.

Existing `teleop_traces/*.jsonl` files are real task traces and can be used as
offline evidence or for timestamp/schema inspection. They are not ServoJ
identification experiments: they do not replace a dedicated
`run_servoj_identification.py` record with the known J1 step trajectory.

The trusted 30004 `QTarget` and `QActual` fields are decoded as native joint
degrees. This is supported by the captured Phase 2B 1440-byte frame and its
agreement with `GetAngle()`; no radians conversion is applied in the parser.

## Phase 5D offline episode QA

`episode_qa.py` audits a completed JSONL episode and every referenced JPEG
without importing or opening the live Nova, Leader, camera, or gripper clients.
It writes a versioned `dobot_episode_v1` derived index under
`teleop_traces/<episode>/derived/phase5d/`. Raw JSONL, JPEGs, and the review MP4
remain immutable; the MP4 is never used as a timestamp source.

For the formal `cube_to_frame_002` episode:

```bash
python3 episode_qa.py validate \
  --episode teleop_traces/teleop_20260927_015239_279634825.jsonl \
  --output-dir teleop_traces/teleop_20260927_015239_279634825/derived/phase5d
```

The alignment index uses the latest camera frame with
`capture_mono_ns <= t_servoj_send`; it preserves host monotonic, wall-clock,
and controller timestamp domains. Use `episode_qa.py annotate` to add human
success/segment metadata without editing the raw trace. See
`docs/episode_schema.md`, `docs/phase5d_dataset_qa.md`, and
`docs/phase5d_raw_audit.md` for the canonical contract and recomputed audit.

Read-only preflight for the first identification run:

```bash
cd /home/dsa/project/dobot/x-trainer
PYTHONPATH=source/leisaac python3 run_servoj_identification.py \
  --joint J1 --amplitude-deg 3 --preflight-only
```

The first motion command must be run by the operator beside the robot only
after reviewing the preflight output. It still requires the interactive
`CONFIRM` prompt:

```bash
PYTHONPATH=source/leisaac python3 run_servoj_identification.py \
  --joint J1 \
  --amplitude-deg 3 \
  --hold-s 1 \
  --return-hold-s 1 \
  --rate-limit-deg-s 3 \
  --safety-envelope-deg 10 \
  --servo-t 0.1 \
  --servo-aheadtime 50 \
  --servo-gain 500 \
  --output experiments/servo_identification/j1_step_001.jsonl \
  --allow-motion
```

The runner aborts if feedback becomes stale, framing fails, RobotMode leaves
the accepted set, actual joints leave the safety envelope, or the background
recorder reports an error. It does not reconnect during an active experiment.

## Scripts

| Script | Purpose | Nova motion |
| --- | --- | --- |
| `read_hand_right.py` | Read six joints and the trigger; torque off first | No |
| `read_leader_keys.py` | Read-only A/B/sensor button packet diagnostic | No |
| `leader_nova_dryrun.py` | Calculate and display relative six-axis targets | No |
| `leader_nova_servoj_j1.py` | Single-axis ServoJ test | Yes |
| `leader_nova_servoj_6dof.py` | Six-axis relative ServoJ teleoperation | Yes |
| `run_teleop.py` | Initial modular six-axis runtime | Yes |
| `reset.py` | Checked MovJ to a requested joint pose; defaults to all zero | Yes |
| `nova_cli.py` | Manual allow-listed Dashboard command terminal | Depends on typed command |

The modular runtime is split into:

| Module | Responsibility |
| --- | --- |
| `teleop_config.py` | Explicit Leader, Nova, and controller settings |
| `leader_reader.py` | Read-only torque-off Leader access, trigger normalization, button reads, and timestamped sampling |
| `nova_client.py` | Nova TCP request/response parsing and command validation |
| `nova_feedback.py` | Nova 30004 realtime `q_target` / `q_actual` reader |
| `collect_tcp_samples.py` | Read-only GetPose samples for Tool 0 pivot calibration |
| `tcp_calibration.py` | Offline flange-to-TCP translation solver; no hardware I/O |
| `teleop_controller.py` | Relative mapping, deadband, limits, rate limiting, and clutch state |
| `teleop_telemetry.py` | Asynchronous JSONL trace and timing summaries |
| `nova_gripper.py` | Explicit Feetech ping/read/torque/position interface for the Nova gripper |
| `gripper_diagnostics.py` | Read-only candidate state classifier; never changes control |
| `analyze_gripper_trace.py` | Read-only summary of gripper position/load/current evidence |
| `episode_recorder.py` | Bounded camera capture/JPEG writer for synchronized episode records |
| `make_multiview_mp4.py` | Timestamp-align three camera frame folders into one composite MP4 |
| `run_teleop.py` | Preflight, fresh anchor, control lifecycle, optional gripper, trace, and graceful stop |

Run from this directory. The `leisaac` package must be importable, normally
because the project is installed in the active environment.

```bash
cd /home/dsa/project/dobot/x-trainer
PYTHONPATH=source/leisaac python3 read_hand_right.py
PYTHONPATH=source/leisaac python3 read_leader_keys.py
PYTHONPATH=source/leisaac python3 leader_nova_dryrun.py
PYTHONPATH=source/leisaac python3 leader_nova_servoj_6dof.py
# Checks the path and waits for Enter before moving to the requested pose.
python3 reset.py
# Manual Dashboard terminal; it sends nothing until you type a command.
python3 nova_cli.py
# The modular runtime is available after the prototype baseline is understood.
PYTHONPATH=source/leisaac python3 run_teleop.py
# Six-axis teleop plus explicit trigger-to-gripper control.
# This enables gripper torque only after the Enter arm prompt.
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --gripper --gripper-enable-torque
```

## Synchronized episode recording

Camera recording is opt-in. Each `--camera NAME=PATH` starts a capture thread
and a separate JPEG writer thread; the ServoJ loop never reads a camera,
encodes JPEG, or writes image files. A camera that cannot be opened aborts
before the control loop starts, so a motion episode cannot run with a missing
observation stream.

The local USB camera can be selected by its stable V4L2 path:

```bash
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --response-profile responsive \
  --gripper --gripper-enable-torque --clutch-key b \
  --episode-label pilot_001 \
  --camera front=/dev/v4l/by-id/usb-CAMERA_SERIAL_1080P_USB_Camera_44434000_P030C01_CAMERA_SERIAL-video-index0
```

Repeat `--camera` for more cameras. Optional `--camera-width`,
`--camera-height`, `--camera-fps`, and `--camera-jpeg-quality` apply to every
camera in that run. The default capture rate is 30 FPS and JPEG quality is 90.

For a trace named `teleop_traces/teleop_<id>.jsonl`, images are stored under:

```text
teleop_traces/teleop_<id>/cameras/<camera_name>/frame_00000001.jpg
```

The JSONL contains one metadata record, control `sample` records, asynchronous
`camera_frame` records, and a final `camera_summary` for each camera. A camera
frame record contains `capture_mono_ns`, `capture_wall_time_ns`, image size,
relative `frame_path`, sequence number, and `dropped_before`. Control samples
contain Leader joints/trigger, mapped and commanded Nova targets, Nova 30004
actual joints and tracking error, gripper state plus its feedback timestamp,
clutch state, and cycle timing. `nova_feedback_mono_ns` identifies the host
time at which the corresponding 30004 sample was received.
`camera_summary` also records actual capture duration and FPS; the requested
FPS may not be achievable when the V4L2 device delivers frames more slowly.
Join camera frames to control state using the shared monotonic nanosecond clock;
wall-clock timestamps are retained for external dataset tooling.

The camera queue is bounded. If JPEG encoding or storage falls behind, frames
are dropped and counted in `dropped_before` and `camera_summary`; the control
loop continues at its configured period. On shutdown the runtime sends Nova
`Stop()`, drains the camera writer queue, writes camera summaries, and then
closes the telemetry file.

A camera read or JPEG write error is different from a queue drop: the control
loop reports the failing camera and exits through the normal Nova `Stop()` path.

To review a three-camera episode as one video, use the timestamp-aware
compositor. It places `front` across the top and `left`/`right` side by side
below it without changing the original JPEG/JSONL files:

```bash
python3 make_multiview_mp4.py \
  teleop_traces/teleop_20260927_015239_279634825.jsonl \
  --output teleop_traces/teleop_20260927_015239_279634825/cube_to_frame_002_multiview.mp4 \
  --top front --bottom-left left --bottom-right right \
  --fps 15 --width 960 --height 1080
```

The compositor uses each camera's `capture_mono_ns` and holds the most recent
frame at each output timestamp. The output FPS is a viewing rate; the original
per-camera timestamps and measured capture FPS remain authoritative for data
alignment.

`nova_cli.py` connects to `SET_NOVA_IP:29999` and accepts only the manually
typed `RobotMode()`, `GetAngle()`, `GetErrorID()`, `ClearError()`, `StartDrag()`,
and `StopDrag()` commands. `ClearError()` requires a second `CLEARERROR`
confirmation and must only be sent after the collision cause has been removed
and the work area has been checked. The CLI never sends `PowerOn()`,
`EnableRobot()`, or a movement command automatically.

`reset.py` requires the controller to already be in TCP control mode and
`RobotMode()==5`. By default it checks and sends
`MovJ(joint={0,0,0,0,0,0},a=5,v=5,cp=0)`. It does not call `PowerOn()`,
`EnableRobot()`, `ClearError()`, or `RequestControl()`. The utility waits for
the returned queue `ResultID` to become current while `RobotMode()` returns 5,
then verifies the final `GetAngle()` values. Use `--target J1 J2 J3 J4 J5 J6`
to check a different joint pose.

The modular runtime accepts experiment parameters without editing the source.
The standard profile remains the conservative default. For a faster but still
bounded response, use the explicit responsive profile:

```bash
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --response-profile responsive \
  --clutch-key b
```

The responsive profile uses `[35,35,35,45,45,45]` degrees/second while
keeping the `+/-20` degree offset clamp. You can override the rate caps with
`--max-speed J1 J2 J3 J4 J5 J6`. The loop period remains 30 ms.

ServoJ response parameters are independently selectable so that a live test
can change one value at a time:

```bash
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --response-profile responsive \
  --clutch-key b \
  --servo-aheadtime 35
```

`aheadtime` and `gain` have the expected tradeoff documented by Dobot:
smaller `aheadtime` or larger `gain` responds faster but can introduce
oscillation. Start with the rate profile, then test `--servo-aheadtime 35`,
then `--servo-gain 650`, and finally `--servo-t 0.06`, one change per run.
Every selected value is saved in the JSONL trace metadata.

The optional gripper is also disabled by default. With `--gripper`, the
program identifies the servo and records trigger/position telemetry without
writing a position. Add `--gripper-enable-torque` only when the operator is
ready to let the trigger command the verified servo. Torque is enabled only
after the Enter arm prompt, and the trigger must move by at least `0.03` from
its startup value before the first position command, preventing a startup jump.
The mapping is linear: normalized trigger `0.0 -> 1470` (open), `0.5 -> 1980`,
and `1.0 -> 2490` (closed), with a small endpoint margin. On shutdown,
torque is disabled if this run enabled it.

Each control sample also stores read-only Feetech feedback when available:
`position`, `moving`, `load`, `current`, `voltage_raw`, and
`temperature_raw`. Load is the raw protocol register and current is the signed
protocol value; they are evidence for comparison, not calibrated force or stall
thresholds. Label
separate experiments with `--episode-label`:

New traces also contain `gripper_diagnostic_state` and
`gripper_tracking_error`. The initial empty-close baseline peaked at 2 current
counts, while the object-hold trace stayed around 9-12. The candidate current
threshold is therefore 8 counts, with a 30-count tracking-error threshold and
250 ms persistence. These are diagnostic labels only: they do not stop or
change the gripper command.

```bash
# Empty close: no object between the fingers.
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --response-profile responsive --gripper --gripper-enable-torque \
  --clutch-key b --episode-label empty_close

# Object grasp and release: use the same object and the same approach each time.
PYTHONPATH=source/leisaac python3 run_teleop.py \
  --response-profile responsive --gripper --gripper-enable-torque \
  --clutch-key b --episode-label object_grasp_release
```

The program prints the trace path, for example
`teleop_traces/teleop_20260926_....jsonl`. Summarize it without changing any
hardware state:

```bash
python3 analyze_gripper_trace.py teleop_traces/teleop_*.jsonl
```

Keep both trace files. In the object episode, mark the close and open times
from the trigger/target-position columns. Compare `present_load`,
`present_current`, target position, and actual position around closing and
release. Do not set a stall threshold from one run; first repeat empty-close
and object-grasp trials with the same speed and object.

The optional clutch is disabled by default until the physical button mapping
has been verified. Run `python3 read_leader_keys.py`, press each physical
button, then select the verified button explicitly:

```bash
PYTHONPATH=source/leisaac python3 run_teleop.py --clutch-key a
```

The upstream X-Trainer packet is `(A, B, sensor)` and uses `0` for a pressed
button and `1` for a released button. While the selected button is held, Nova
holds its last commanded target. On release, the controller re-anchors the
current Leader pose to that target before sending new motion commands.

Use the dry run before any real motion test. The ServoJ script asks for an
explicit Enter confirmation after the initial state read.

## Safety contract

The current prototype intentionally follows these rules:

- Startup requires `RobotMode() == 5` (ENABLE idle).
- The scripts do not call `EnableRobot()` or `ClearError()`.
- Leader torque is disabled before position reads so the Leader can be moved by hand.
- Motion starts only after the user confirmation and a fresh relative anchor.
- Each mapped joint is currently limited to `+/-20` degrees from its Nova anchor.
- The standard target command is rate limited to
  `[30,30,30,40,40,40]` degrees/second per joint. The explicit responsive
  profile uses `[35,35,35,45,45,45]`.
- A missing Leader sample for more than `0.20` seconds causes a `Stop()` attempt.
- A nonzero ServoJ `ErrorID` stops further ServoJ commands.
- An unexpected Nova `RobotMode` from 30004 or stale configured clutch-button data stops the run.
- Ctrl+C and other exits attempt `Stop()` before connections are closed.

Keep the emergency stop reachable and keep the Nova work area clear. A TCP
failure can prevent a stop command from reaching the controller; the hardware
emergency stop remains the final stop mechanism.

## Phase B latency instrumentation

`leader_nova_servoj_6dof.py` now records timing samples without writing a log
file from the control loop. Samples are passed to a bounded queue and a daemon
thread both writes a JSONL trace and prints a summary every two seconds. If the
queue fills, the summary reports the number of dropped samples.

Each armed run creates a file under `teleop_traces/`, for example:

```text
teleop_traces/teleop_20260924_201530_123456789.jsonl
```

The first JSON object is session metadata. Following objects contain one
control-cycle sample with host monotonic timestamps, wall-clock timestamp,
Leader state, mapped target, commanded target, ServoJ text, status, and timing
metrics. When cameras are requested, the same JSONL also contains camera frame
records and camera summaries as described above. Nova 30004 feedback is stored
with each control sample.

The writer uses a bounded queue, flushes in batches, and never performs file
I/O in the control thread. A trace write error is reported in the diagnostic
summary without stopping the robot control loop.

The reported values are:

| Metric | Definition |
| --- | --- |
| `leader_read_ms` | From the start of `GroupSyncRead.txRxPacket()` until the read result is decoded |
| `compute_ms` | From completed Leader read through mapping, offset clamp, and rate limiting |
| `servoj_rtt_ms` | From immediately before TCP `sendall()` until a response ending in `;` is received |
| `whole_loop_ms` | From cycle start through the cycle sleep, including display and network work |
| `actual_loop_hz` | `1000 / whole_loop_ms` |

Every metric summary contains mean, median, p95, and maximum. Leader read
failures still contribute read and whole-loop samples; they do not create a
fake ServoJ RTT. The live command display is throttled to 10 Hz to reduce
terminal output pressure.

The current response baseline uses the measured stable loop and the split
joint rate caps:

```text
DT=0.03
MAX_OFFSET_DEG=[20, 20, 20, 20, 20, 20]
MAX_SPEED_DEG_S=[30, 30, 30, 40, 40, 40]
SERVO_T=0.1
SERVO_AHEADTIME=50
SERVO_GAIN=500
```

The optional responsive profile changes only `MAX_SPEED_DEG_S` to
`[35,35,35,45,45,45]`; it does not silently change `t`, `aheadtime`, `gain`,
or the offset clamp.

The modular `run_teleop.py` also opens Nova port `30004`. It decodes the 1440
byte little-endian realtime frame at the documented offsets for RobotMode,
controller timestamp, `QTarget`, and `QActual`. Every modular trace sample
contains the newest feedback snapshot and `target - actual` tracking error.
The feedback stream must produce a fresh sample before motion starts; a stale
feedback stream stops the run.

## Known prototype limits

- The modular `run_teleop.py` contains clutch/re-anchor logic. It is disabled
  by default and requires selecting the physically verified A or B key with
  `--clutch-key a` or `--clutch-key b`; the original
  `leader_nova_servoj_6dof.py` remains the older prototype without clutch.
- The modular `run_teleop.py` can now map trigger ID 18 to the verified Feetech
  gripper, but the original `leader_nova_servoj_6dof.py` remains unchanged and
  does not command the gripper.
- The six-axis script has no 30004 actual-joint feedback.
- The original `leader_nova_servoj_6dof.py` remains the validated prototype and
  does not use the new 30004 reader; `run_teleop.py` does.
- The physical A/B clutch packet still needs one live verification with
  `read_leader_keys.py` before selecting a button for motion tests.
- The relative mapping uses scale 1.0 and a default +/-20 degree offset limit.
  The current rate caps are `[30,30,30,40,40,40]` degrees/second, selected
  after the baseline showed a stable ~33 Hz loop and ~5.8 ms ServoJ RTT. The
  offset limit still prevents full-range one-to-one movement; raise it only
  after joint limits are measured on the real Nova.
- The current watchdog runs in the control thread, so a blocked serial or TCP
  call can delay its decision. This is a follow-up safety refactor.
- The two-second latency summary is diagnostic output; it must not be used as
  a substitute for the emergency stop.

Do not change `aheadtime`, `gain`, or the rate limiter based on feel alone.
Collect a baseline first, then change one control parameter per experiment.
