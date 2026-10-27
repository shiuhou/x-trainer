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

The first live initialization lacked package metadata (fixed). A subsequent
attempt found no listener on the previous proxy, before transmission. After the
operator changed the proxy, the bounded connectivity run received **two real
HTTP 200 Responses, one returned policy decision, zero executed waypoints**.
Both model `move_to` calls were rejected by the open-only gripper check; the
agent then exhausted its two-call budget and returned a synthetic stop. This is
**live FAIL, not a transport failure or mission success**. The relay reported
`gpt-6-astra`; total usage was 59,687 tokens and episode time 29.421 seconds.
That failed episode was preserved without retry; the subsequently authorized
post-fix experiment is recorded below.

Offline reproduction identified the adapter defect: measured opening
`0.9999999511267113` produced interpolated commands slightly below 1, which
violated the connectivity-only exact-open check. The connectivity action space
now pins gripper low/high to 1; official interpolation therefore emits open
commands, while observation still reports measured aperture. Mission gripper
bounds remain 0..1. Neither validation tolerance nor the official agent changed.
Both original tool targets reproduce the rejection with the old space and pass
with the corrected space. Closing commands remain rejected in connectivity mode.

The updated Isaac smoke passed: two mock Responses calls through the open-only
contract, six total waypoints, generations 1..6, 98 recorded physics steps,
10.698 seconds. This is offline validation; it does not turn the failed live run
into a PASS. The operator subsequently authorized two additional connectivity
requests, raising the overall ceiling from 14 to 16, with one <=12-request
mission conditional on genuine connectivity PASS.

The authorized post-fix **live connectivity PASS** used implementation
`11fd66a1775d640f4adc3cf87f02bed5d3f172a4`, the same pinned official agent,
SciTrace Responses and `gpt-6-astra`. Automatic proxy selection passed CONNECT
and TLS checks. Two real requests produced two `move_to` decisions and three
official interpolated waypoints. Measured base Z went
`0.255655527 → 0.260686576 → 0.255710781 m`; all commanded gripper values were 1.
Physics/settle completed before each new observation: generations 1, 2, 3 at
simulation times 0.433333, 1.000000, 1.566667 s. The recorded episode has 46
physics steps. Model usage was 61,576 tokens; model time 19.379 s, render 0.934 s,
validation 0.003 s, execution 0.265 s, settle 1.717 s, total 28.230 s. These are
relay-reported token/model metadata; successful API routing does not attest
upstream model weights. Hardware/network guard violations were zero.

The single authorized **autonomous cube-to-plate mission FAILED** at
`execution / settle_timeout` on decision 6 (closing the gripper). There were six
real requests, six decisions, 129 executed waypoints, five completed post-action
observations (generations 1..6 including initial), and 636 recorded physics steps.
The original task, scene, cameras, IK, physics, workspace and evaluator were
unchanged. No second mission or automatic retry was run. This distinguishes a
working Astra/Isaac closed loop from an unsuccessful manipulation task.

The evaluator reported `success=false`, `upstream_success=false`,
`grasp_seen=false`, `grasped_now=false`, `upstream_placed=false`,
`released=false`, `settled=false`. Maximum measured cube lift after initial
settling was effectively zero (7.45e-9 m); transport and placement were not
reached. The model kept the gripper open for its five approach decisions, then
commanded closure. At the last completed waypoint the measured aperture was
0.5219; the subsequent bounded settle failed. Left-arm maximum joint drift was
0.012479 rad, within its existing abort bound.

Read-only analysis of the final 120 recorded physics ticks found cube linear
speed 0.0511–0.1598 m/s (last 0.0958), angular speed 2.480–10.049 rad/s
(last 8.531), and zero ticks meeting the existing object-still thresholds
(0.01 m/s and 0.1 rad/s). Therefore extending the time threshold is not an
established fix. Persistent motion during closure explains the settle rejection;
the underlying cause among approach/contact geometry/controller dynamics has
not been isolated. No physics parameter or success criterion was relaxed.

Mission usage was 194,297 tokens. Model time was 101.757 s, render 1.321 s,
validation 0.039 s, execution 14.680 s, completed settle 3.046 s, total 131.674 s:
the measured loop was predominantly model-bound. The failed settle's duration
is included in total time but not in the settle subtotal, since that accumulator
updates only on successful return. Runner total timings exclude Kit startup and
host proxy preflight. Both new episodes combined used 8 real requests and
255,873 tokens; including the earlier failed two-call connectivity run, 10 of
the newly authorized 16-request ceiling were consumed. Unused budget was not
used to retry the failed mission.

Evidence limitation: mission `final_*` images are the last successful policy
observation (generation 6), **before** the failed close/settle, not fresh images
of the terminal physics state. The exact terminal state and velocities are in
`mission-1/episode.hdf5`; `result.json` records the post-failure evaluator state.
The original failed result and recording are preserved. No private reasoning,
credential content, or physical-device access was introduced.

Scoped validation: 32 adapter tests, Ruff, formatting, mypy (six owned modules,
`--follow-imports=skip --ignore-missing-imports`), compilation and shell syntax
passed. No MuJoCo render/regression campaign or physical access was performed.
The shared S5 provider's focused suite separately passed 100 tests. GUI keyboard
launch is provided but has not been exercised interactively.

Local evidence under `.agents/runs/xtrainer-astra/`:

- `original-driver580/`: original top and wrist captures.
- `offline-agent/`: final offline smoke, actions, frames, recording and metrics.
- `connect/`: preserved zero-request installation failure.
- `connect-live/`: preserved proxy failure, initial images and transport accounting.
- `connect-configured-proxy/`: preserved two-response live failure, original measured state and targets.
- `offline-open-contract/`: corrected open-only official-agent smoke, recording and fresh frames.
- `connect-open-contract/`: authorized post-fix live connectivity PASS, actions, fresh images and recording.
- `mission-1/`: the single autonomous mission failure, action/observation traces, evaluator, usage and full physics recording.
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

# Set ASTRA_CREDENTIAL_FILE to the existing operator env-file path.
# Default proxy selection is automatic; no port number is embedded in this command.
scripts/astra/container.sh connect --output .agents/runs/xtrainer-astra/connect
scripts/astra/container.sh mission --connect-result .agents/runs/xtrainer-astra/connect/result.json \
  --output .agents/runs/xtrainer-astra/mission
```

`--provider-proxy auto` (default) selects a working local HTTP(S) proxy before
launching Isaac. It tries `S5_PROVIDER_HTTPS_PROXY`, then the standard
HTTPS/ALL/HTTP proxy environment settings, then GNOME's saved HTTPS/HTTP proxy
addresses. Desktop addresses are candidates even with the global desktop proxy
toggle off: this application's auto mode verifies them independently. Stale
candidates are skipped. There is no fixed port list and no scan of unrelated
listening services. CONNECT/TLS is checked with a bounded, credential-free HEAD
to the relay root, never a model request. Current host auto-selection and TLS
verification passed without supplying a port on the command line.

For an explicit override use `--provider-proxy http://localhost:<port>` (or an
IPv6 loopback URL); every valid TCP port is accepted. HTTP/HTTPS default ports
also work. Explicit URLs do not silently switch. `--provider-proxy direct`
selects direct routing with the existing public-address guard. The chosen route
is fixed for the episode; transport retries remain disabled. If no configured
candidate works, selection stops before paid requests.

The host resolves desktop configuration and passes only the selected address to
the container. `run.py` can also resolve auto mode from its own environment.
Proxy credentials and non-loopback proxy destinations are not supported. Socket
isolation still applies outside provider transport; it is not a fixed proxy-port
allowlist. The HTTP client still has `trust_env=False`: environment settings are
validated and selected explicitly, never blindly inherited by the client.
Credentials are parsed as data from a read-only mount, never sourced or copied
into Docker environment variables, command arguments, images or evidence.

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
metadata redaction reuse frozen S5 `s5_agent.py` / `s5_common.py` with the
reviewable `scripts/astra/provider-proxy.patch` applied by setup. The patch adds
the same configuration selection and loopback route validation as the local S5
provider code, without a fixed proxy port. Network audit
enforcement and frozen MuJoCo experiment evidence are preserved.
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
