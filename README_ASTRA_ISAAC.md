# Astra / X-Trainer Isaac simulation

This integration uses the original X-Trainer USD robot, gripper, table, cube and
plate. Both arms remain in the asset; only the right arm is policy-controlled.
It does not connect to a robot controller, Leader, serial gripper or physical camera.
The frozen MuJoCo S5 experiments remain a separate historical baseline.

## Validated checkpoint

**Offline integration PASS; live connectivity BLOCKED; autonomous mission NOT RUN.**

The original PickCube scene renders with the complete upstream assembly. The
right-arm smoke moves up 4.989 mm and returns, closes and reopens the gripper,
and reads back an upstream HDF5 recording. The official pinned agent also runs
through two mock Responses requests, `move_to`, interpolation and Isaac execution:
six total smoke waypoints, 98 recorded physics steps, generations 1 through 6.
This is offline evidence, not a live Astra success.

The first live initialization stopped before transmission because package
metadata was missing. Installing the exact archived packages fixed it, and the
offline official-agent loop then passed. The subsequent live transport attempt
was refused by the configured local proxy (errno 111, no listener), before relay
HTTP transmission: **0 transmitted requests, 0 decisions, 0 model tokens reported**.
No automatic network retry or mission was performed. Resume requires restoring
the existing operator proxy and authorizing a new bounded connectivity run.

Only the new adapter checks were run: 11 focused tests, Ruff, formatting, mypy,
Python compilation and shell syntax passed. No MuJoCo regression campaign,
physical device access or push was performed. GUI keyboard launch is provided
but has not been exercised interactively.

Local evidence under `.agents/runs/xtrainer-astra/`:

- `original-driver580/`: original top and wrist captures.
- `offline-agent/`: final offline smoke, actions, frames, recording and metrics.
- `connect/`: preserved zero-request installation failure.
- `connect-live/`: preserved proxy failure, initial images and transport accounting.
- `smoke-tracking/`: uncompensated IK failure for comparison.

## Runtime

From this component directory, run `scripts/astra/setup.sh`. It prepares the
assets, exports reviewed provider sources, and builds `xtrainer-astra:isaac-2.1.0`
from the NVIDIA image digest in `docker/isaac-image.lock`. The base is the
upstream Docker recipe's Isaac Lab 2.1.0, not the optional pip extra for 2.3.0.
The validated host driver is Ubuntu's 580.178.04. Driver 595.99.02 crashed in
RTX Hydra initialization even with the unmodified official image. Driver changes
are operator work. The image still emits Warp UUID and torchvision warnings;
scene rendering, physics and the scoped offline loop passed despite them.
Shader caches are retained in two task-specific Docker volumes.

GPU access requires a working NVIDIA Docker runtime. The container receives GPU
devices only; no host serial/video devices and no privileged mode.

The workspace bootstrap reads the pinned local Inspect Robots reference and
frozen MuJoCo provider modules. It never copies dirty source or credentials.
The public code remains in this repository; `.agents/runs/xtrainer-astra/runtime`
is a generated local dependency directory, rebuildable with setup.

Assets use the complete `Dobot-robot/X-trainer` robot at
`02f008252001f654e28dee40d615c713ff79935a` and `dstx123/xtrainer-leisaac`
PickCube scene at `fb140bf9d4bf7de8b71818df59b93a1cb4c0564a`.
The robot is reused from the existing local official-asset download.
The scene repository's robot snapshot has unresolved collider references and is
not mixed into the official robot layers. Downloaded assets and the per-file
SHA256 manifest remain under ignored `assets/` paths.

## Use

```bash
# Original upstream keyboard GUI. Requires local DISPLAY/Xauthority.
scripts/astra/container.sh keyboard

# Unique output directories are required; prior evidence is never overwritten.
scripts/astra/container.sh original --output .agents/runs/xtrainer-astra/original
scripts/astra/container.sh smoke --output .agents/runs/xtrainer-astra/smoke

# Set ASTRA_CREDENTIAL_FILE to the existing operator env file in your shell.
# It is mounted read-only only for live modes; never source it or put its value
# into Docker environment variables, command arguments, images or evidence.
scripts/astra/container.sh connect --output .agents/runs/xtrainer-astra/connect
scripts/astra/container.sh mission --connect-result .agents/runs/xtrainer-astra/connect/result.json \
  --output .agents/runs/xtrainer-astra/mission
```

Live modes are paid. `connect` permits two requests/two decisions/180 seconds;
`mission` requires a prior connect PASS and permits twelve requests/twelve
decisions/600 seconds. A failed episode is not automatically retried. The
approved first experiment is one connect run followed, only on PASS, by one
mission. Re-running a live command is a new operator-authorized experiment.

## Observation and action contract

- `front_rgb`, `right_rgb`: 480x640x3 uint8 RGB, fixed workspace views.
- `wrist_rgb`: 720x1280x3 uint8 RGB, attached to the right wrist.
- `eef_state`: right-base grasp-center x/y/z in metres, measured reset-relative
  yaw/pitch/roll in radians, measured normalized opening (0 closed, 1 open).
- Right base is the USD right pedestal frame at shared-base X=1.06 m. Grasp center
  uses upstream PickCube's local wrist +Z160 mm convention, inside the authored
  finger contact region. This is simulation geometry, not a measured real TCP.
- `move_to` / official ActionChunk targets contain `[x,y,z,yaw,pitch,roll,gripper]`.
  Only zero orientation targets are supported; Isaac Differential IK holds the
  reset orientation. The authored opposite finger joint targets are derived from
  aperture, never copied from the MuJoCo 95 mm model. Empty closure measured
  0.073 and reopening 0.998 in the smoke; commanded zero is not substituted for
  measured contact-limited position. Full nominal travel is not claimed.
- Isaac Differential IK uses native PhysX gravity feedforward. Without it, the
  measured 5 mm target retained approximately 10 mm positional error; with it,
  the settled upward displacement was 4.989 mm. Authored masses, joint gains,
  friction, gravity and collision geometry are unchanged. Left-arm targets remain
  initial; measured maximum joint drift was 0.01248 rad, below the 0.02 rad abort
  bound. This is holding with finite compliance, not a welded/frozen left arm.
- Whole-chunk finite/bounds/step checks precede execution. Each waypoint is at
  most 20 mm; failures stop the chunk. Physics uses Isaac joint targets and contact
  simulation, never object teleport or action-driven joint-state writes.
- Observations are copied after physics and settle, with common simulation time,
  host monotonic time, sensor counters and increasing generation. Identical pixels
  in a static scene are legitimate; repeated capture identities are not.
- Evaluator object state stays out of policy inputs. Upstream cube-to-plate success
  includes returning to rest; additionally release and low object velocity are
  recorded and required by this runner. Model `done` alone is not success.

Provider construction, credentials, endpoint/proxy guard, request accounting and
metadata redaction directly reuse frozen S5 `s5_agent.py` / `s5_common.py`.
They load the official `inspect-robots-agent` at
`7e506e3b6f39530f18e75cf97b23c4287aa90f12`; no MuJoCo simulator or real-motion
backend is imported. Responses serialization, parsing and interpolation stay
inside the official agent. Wire capture and transcript persistence are off.

## Minimal verification

After setup, run `pytest -q tests/test_astra_contracts.py` in an environment with
NumPy, Pillow, httpx and the pinned Inspect Robots packages. The budget test reads
the generated frozen provider runtime. The camera-isolation test never imports
Isaac or touches hardware.
Ruff/format/mypy/compile checks target only `leisaac/astra`, `scripts/astra`, and
that test. Runtime validation is a short original-scene boot and one offline
movement/gripper/recording smoke, followed by the authorized bounded live runs.
There is no requirement for bit-exact GPU frames or a MuJoCo regression campaign.

Camera extrinsics and FOV are engineering placements, not measured calibration.
This is not a real-robot safety certificate or a sim-to-real success claim.
