# Astra / X-Trainer Isaac simulation

This integration uses the original X-Trainer USD robot, gripper, table, cube and
plate. Both arms remain in the asset; only the right arm is policy-controlled.
It does not connect to a robot controller, Leader, serial gripper or physical camera.
The frozen MuJoCo S5 experiments remain a separate historical baseline.

## Current grasp checkpoint

**Full privileged scripted pick/place PASS twice.** Original autonomous
`mission-1` remains FAIL. Both separately authorized post-fix Astra missions
(`mission-2-postfix` and `mission-3-postfix`) ended with **FAIL / model_timeout
on request 2**; scripted success is not autonomous-policy success. No automatic
model retry was run. The subsequently authorized `mission-4-postfix` received
four HTTP 200 responses but stopped at `IK_tracking_failure` during descent/closure;
autonomous pick/place remains FAIL.

The false closure timeout came from treating PhysX's reported velocity fields
as actual transform motion under contact. PhysX documents that split-impulse
position and velocity iterations produce different velocities; the reported
velocity-iteration output is not the velocity used to integrate transforms.
See [PhysX solver iterations](https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html#solver-iterations).
Our native-substep trace measured approximately 0.230 m/s reported speed while
position-derived speed was only 1.05e-6–2.32e-5 m/s in the same stationary grasp.

The adapter now measures joint, cube-translation and quaternion rotation changes
at **every physics substep**, retaining the maximum speed within each control
step. It requires ten consecutive control steps below the existing limits
(0.02 joint units/s, 0.01 m/s, 0.1 rad/s), within the existing 120-step bound.
This explicitly changes the contact-settle measurement, not just a tolerance.
Opposing substep excursions cannot cancel. Native velocity fields remain
recorded; open/released settling additionally requires the original native
velocity checks. Final task success and its released/settled checks are unchanged.
This is a simulator observation-semantic correction, not a physical safety claim.

The duplicate-collider and 240 Hz integration controls did not solve the failure;
both candidate configuration changes were retired. Authored collision meshes,
collision declarations, masses, friction, gains, solver and 60 Hz physics/30 Hz
control remain original. The scripted baseline approaches vertically from above;
the old diagonal approach caused contact/tracking failure in the finer-step
control. These privileged targets are never passed to the policy.

The public EEF remains the upstream +160 mm reference. Generic tool documentation
now distinguishes that reference from the USD narrow finger region
(+168.292..197.000 mm, midpoint +182.646 mm). No TCP rebasing or object coordinates
were added to policy input. Diagnostic clearance calculations now retain authored
body scale (cube 1.5, plate 0.0254) when transforming collision vertices.

Both full episodes used 176 official interpolated waypoints and 759 normal
physics control steps; recorded controls, joints and cube states match exactly.
They lifted at least 65.053 mm through the one-second hold, with at most 0.837 mm
cube/hand relative drift, transported 214.513 mm, released, and returned to rest.
Sampled transport clearance is 27.681 mm. Both finish with upstream `task_done`,
measured aperture >0.95 and native-velocity settled checks passing. Runtime was
37.917/38.128 seconds, excluding Kit startup. No object teleport or joint-state
playback was used after the single initial reset.

Evidence: `.agents/runs/xtrainer-astra/grasp-diagnosis/collider-control/`,
particularly `07-full-scripted-a/`, `08-full-scripted-b/` and `comparison.json`.
Use the credential-free `diagnose --diagnostic-case full` entry for this
**privileged scripted baseline**, not as autonomous-policy guidance.
Scoped validation: 60 tests, Ruff/format, mypy (five own source modules),
compilation and diff checks PASS. Historical evidence remains unchanged.

## Post-fix Astra mission

The one authorized trial used offline fix commit
`de3423c47adbcc94f578e7cea98c01371ae92f2e`, official agent revision
`7e506e3b6f39530f18e75cf97b23c4287aa90f12`, Responses, `gpt-6-astra`,
and the existing shared SciTrace provider/proxy configuration. Limits remained
12 requests, 12 decisions, 384 waypoints and 600 seconds, with transport retries
disabled. Evidence is preserved in `.agents/runs/xtrainer-astra/mission-2-postfix/`.

Two requests were transmitted. The first returned HTTP 200 and one permitted
`move_to`; its 25 official interpolated waypoints passed validation and executed
through Isaac IK/physics, settle and a fresh observation. Generations advanced
1 -> 2, simulation time 0.466667 -> 3.700000 seconds, with 110 recorded control
steps. The measured reference moved from approximately
`[-0.344996, -0.117501, 0.255656]` to `[-0.464981, -0.117498, 0.200001]` metres,
with the gripper open.

The second transmitted request returned no response before the existing shared
transport's 60-second request timeout. Exact failure:
`model_timeout: provider request timed out`. This is a provider/relay response
timeout; the evidence does not isolate which upstream service caused it. The
episode stopped before closure, grasp, lift or transport. Final evaluator:
`success=false`, `released=true`, `settled=true`. It therefore does not validate
autonomous grasping after the fix, nor demonstrate a new grasp failure.

Known returned usage is 29,909 tokens (29,687 input, 222 output) from request 1.
Request 2 returned no usage; its billed usage is unknown. Model/provider waiting
was 103.602 seconds (43.498 + 60.104), render 1.129, validation 0.014, execute
2.855, settle 1.311, and total 114.492 seconds excluding Kit startup/preflight.
The loop was dominated by provider waiting. The relay reported the requested
model identifier; this does not independently attest underlying model weights.

The failure montage renders the stopped state without another physics step;
it is diagnostic-only, not another policy generation. Raw recording and failure
metadata are retained. No hardware access, model retry or push occurred.

### Separately authorized isolated provider diagnostic

After that failed episode, the operator authorized one recorded-observation
diagnostic, without a simulator or action execution. Three credential-free HEAD
probes through the existing automatically selected proxy passed CONNECT, verified
TLS and returned HTTP 200 in 0.956–1.062 seconds. These only test relay reachability.

The single subsequent model request used the same official agent, Responses,
`gpt-6-astra`, shared provider/guard, 60-second request timeout and disabled
transport retries. It returned HTTP 200 in 22.668 seconds; the official agent
produced `move_to` and 16 interpolated waypoints, all **unexecuted**. TLS completed
at 0.697 seconds, the provider request body finished sending at 0.698 seconds,
response headers arrived at 22.615 seconds, and the response body finished at
22.667 seconds. Waiting from request-body completion to response headers was
21.917 seconds. Usage: 7,128 input + 80 output = 7,208 tokens. No timeout occurred.

This was a fresh agent session using the saved terminal diagnostic RGB images
and measured generation-2 EEF at the same stopped physics state. Those images
differ from the original policy pixels, and prior conversation history was not
reconstructed. It is **not an exact replay** or proof that the intermittent
timeout is fixed. The old failed request's coarse telemetry cannot identify
its precise timeout stage; neither test distinguishes relay waiting from
upstream model waiting. The subsequently authorized mission is recorded below.
Metadata-only stage events and result are in
`.agents/runs/xtrainer-astra/provider-diagnostic-1/result.json`; no request headers,
raw provider response or private reasoning were persisted. One container setup
failed to read the credential mount before client construction; restoring the
existing runner's DAC_OVERRIDE capability resolved access without changing the
credential file. That setup failure transmitted no model request.

## Renewed autonomous mission after isolated diagnostic

The operator authorized one further bounded mission after the isolated diagnostic.
`mission-3-postfix` ran from `d28155c7a7e98fbbacd6bb2937bafd655e44a424`, with the
same official agent, model, provider and 60-second request timeout. Limits stayed
12 requests / 12 decisions / 384 waypoints / 600 seconds; no automatic retry.

**FAIL / model_timeout: `provider request timed out` on transmitted request 2.**
Request 1 returned HTTP 200 in 50.569 seconds and a generic open-gripper `move_to`
to `[-0.345, -0.0175, 0.2557]` metres. Its 21 official interpolated waypoints
passed validation and executed through Isaac IK/physics, settle and fresh
observation, advancing generations 1 -> 2 with 89 recorded control steps.
Request 2 timed out after 60.103 seconds without a response or usage metadata.
There was one policy decision; no closure, grasp, lift or transport was reached.
Evaluator `success=false`, `grasp_seen=false`, `released=true`, `settled=true`.
The two full scripted successes remain valid; autonomous grasp is still unverified.

Returned usage is 7,231 input + 196 output = **7,427 known tokens**; request 2
usage/billing is unknown. Model waiting totalled 110.671 seconds, render 1.106,
validation 0.016, execute 2.178, settle 1.282; episode total **120.787 seconds**
excluding Kit startup/preflight. This run was dominated by provider waiting.
The precise timeout stage and whether relay or upstream processing caused it
are **not established**. An isolated successful call does not establish reliable
multi-turn service; history size or latency remain hypotheses, not root causes.

Primary evidence is local at `.agents/runs/xtrainer-astra/mission-3-postfix/`:
result, physics HDF5, actions, observations and terminal images are preserved.
A mistaken duplicate launch was rejected by the existing output-directory guard
before initialization or client construction, transmitting **zero extra requests**.
Its shell redirection replaced the adjacent verbose startup `.log`; that original
startup log is unavailable. The primary episode evidence was not overwritten.

The runner now records bounded metadata-only HTTP phase events (TCP, TLS,
request send, response receive and timeout class), chaining the existing transport
trace without changing routing, deadlines, retry policy or request contents.
It stores no headers, bodies or exception messages. This instrumentation was
added **after** mission 3 and validated offline only; it cannot retrospectively
identify that episode's timeout phase. Scoped checks: **62 adapter tests PASS**,
Ruff/format, mypy (six own sources), compilation, shell syntax and credential-content
audit PASS. No further provider call, physical hardware access or push occurred.

## Renewed mission with transport phase tracing

A separately authorized single `mission-4-postfix` ran from
`191931d523da807dbec127216e55f638eb54e0af`, using the unchanged task, official agent,
model, provider and action limits (12 requests / 12 decisions / 384 waypoints /
600 seconds). All **four requests returned HTTP 200**, in 19.909, 21.829, 28.199
and 16.138 seconds. Known usage totals **19,926 tokens**. No timeout or retry
occurred; this does not retrospectively identify the cause of earlier timeouts.
The metadata trace shows TLS completing in 0.549/0.530 seconds on the two new
connections. Provider response-header waiting dominates model latency.

The first three decisions executed and settled, with fresh generations 1 -> 2
-> 3 -> 4. Decision 4 requested simultaneous descent from approximately Z65 mm
to Z45 mm and closure to 0. The run completed 95 waypoints, then stopped at the
next waypoint after the existing 30-control-step tracking bound:
**FAIL / adapter_or_execution / `IK_tracking_failure`**.

Terminal reference target was `[-0.550, -0.003, 0.045]` m; measured position was
`[-0.552488, -0.001991, 0.048530]` m. Position error **4.434 mm** exceeded the
unchanged 3 mm convergence threshold; orientation error norm **0.009588 rad**
was within the 0.035 rad threshold. Measured aperture was 0.574489 at commanded
0. Contact-constrained tracking is a hypothesis, not a demonstrated root cause;
no gains, geometry, controller, timing or acceptance limits were changed.

The upstream grasp heuristic became true, but maximum lift was effectively zero
and there was no transport or release. `success=false`, `released=false`,
`settled=false`; this is **not a verified physical grasp or autonomous success**.
Total episode time was **107.244 s**, excluding Kit startup/preflight; model
transport total 86.075 s. Recorded render/validation/execute/settle times were
1.376/0.044/10.271/2.364 s. The existing execute accumulator excludes the final
failed waypoint's tracking time, so it is not a complete execution-time measure.

Local evidence: `.agents/runs/xtrainer-astra/mission-4-postfix/` contains result,
phase timings, action/observation traces, HDF5 and stopped-state failure images.
No duplicate launch, further model request, hardware access or push occurred.
Focused adapter tests: **62 PASS**; credential-content and diff audits PASS.
The next useful investigation is offline replay of this descent/closure and
comparison with the successful privileged baseline, without relaxing limits.

## Offline replay: mission 4 descent/closure tracking failure

Three bounded, credential-free, `--network none` episodes isolate the failure
without changing upstream physics, collisions, gains, IK or acceptance limits.
The original mission and successful scripted recordings remain read-only.

1. **Raw 17D replay:** all 389 controls, joint positions/velocities and cube
   poses/velocities exactly match the original (maximum absolute difference 0).
   Raw replay does not enforce Cartesian convergence; its historical stability
   label is not an acceptance result for this tracking failure.
2. **Recorded Cartesian replay with contact reporting:** official `move_to`
   interpolation reproduces the same 389 controls/states exactly, including
   `IK_tracking_failure` at the final descent/closure waypoint. Read-only contact
   reporting therefore did not change the trajectory.
3. **Separated descent/closure:** preserve all model target coordinates, but
   first descend with aperture=1 and settle, then close at that position. This
   passes the unchanged 3 mm / 0.035 rad / 30-tick tracking limits and original
   contact-settle checks: 101 waypoints, 385 controls. No lift was attempted.

| Measurement | Simultaneous descent + closure | Open descent, settle, then closure |
| --- | ---: | ---: |
| Final waypoint position error | 4.434 mm: FAIL | 0.0447 mm: PASS |
| Terminal position error after any settle | 4.434 mm | 0.0652 mm |
| Remaining descent near first bilateral contact | 16.757 mm | 0.005 mm |
| Terminal measured aperture | 0.57449 | 0.57296 |
| Last 20 substeps mean finger normal loads | 7.877 / 7.584 N | 7.701 / 7.683 N |
| Last 20 substeps mean cube/table normal load | 4.597 N | 0.0347 N |

Loads are sums of absolute contact-normal forces, not net forces or physical
force calibration. The sensor covers cube/finger and cube/table pairs, **not**
finger/table pairs. It does not prove a direct finger/table collision.

**Established cause for this episode:** closure establishes bilateral contact
while 16.8 mm of descent remains and the cube is supported by the table. Continued
downward tracking with the closed grasp loads the cube/table contact and settles
into an offset equilibrium. Over the final ten control steps, reference position
changes by at most 0.00163 mm, while the target residual stays 4.434 mm. This is
contact-constrained tracking, not evidence that the free-space IK target is
unreachable. Merely separating the same descent and closure removes the failure;
no longer timeout, looser tolerance or target-coordinate correction is needed
for this control case. It does not prove every coupled grasp is invalid.

The successful privileged baseline already uses open descent -> settle -> close.
Its geometry-derived reference target is `[-0.554470, -0.002233, 0.042761]` m;
the model target differs by `[+4.470, -0.767, +2.239]` mm. Those coordinate
errors were not necessary to cause this tracking failure: the unchanged model
target passes the separated control. Earlier scripted lift/transport PASS is
preserved, but the new separated case only establishes settled closure, **not**
secure lift or full task success.

The diagnostic CLI now supports `--diagnostic-case recorded` and `separated`
with `--diagnostic-source <episode-directory>`. It reads explicit recorded
`move_to` targets as JSON data, rather than using the old mission-1-specific
Cartesian list. Inputs require finite, explicit XYZ/aperture and fixed rotation;
normal embodiment validation still checks workspace, chunk and tracking limits.
Contact reporting and target splitting apply only to offline diagnosis; the live
policy, prompts and executor are unchanged. Raw result/physics/action/contact
records and a compact `comparison.json` are in
`.agents/runs/xtrainer-astra/grasp-diagnosis/mission4-tracking/`.

Validation: **67 focused tests PASS**, scoped Ruff/format, mypy, compilation and
diff checks PASS. No model requests, credentials, hardware or push. A defensible
next change would explicitly express open-descent/settle-before-close sequencing
in the policy interface and validate it offline; this investigation does not
silently rewrite live model actions or authorize another paid episode.

## Separate approach and closure contract

Following the mission-4 diagnosis, embodiment tool documentation now requires
open-gripper approach/descent, settled fresh observation, then a separate
`move_to` specifying only `gripper` (omit XYZ). This keeps object coordinates out
of the instruction and leaves the official tool/interpolator unchanged.

The production `pre_check` rejects a new closure if any waypoint leaves the
current position's existing 3 mm reached envelope. The same shared constant is
used by execution convergence; this does not loosen the old tracking threshold.
Tiny hold corrections within that envelope remain legal, so a previously reached
explicit position can still be held while closing. A rejection explains the two
required calls to the agent; no action is silently split or executed. Any agent
re-query still consumes the unchanged shared model-call budget.

Closure is compared with the **last executed gripper command**, not measured
aperture: a closed gripper holding a cube can measure about 0.57. Keeping its
existing command of 0 during lift/transport must remain legal. The measured
observation is never replaced with the command. Further tightening from a partial
closure during translation is rejected; pure closure and open approach pass.

Offline tests exercise the real official tool/pre-check path for the original
20 mm descent/closure, partial closure, gripper-only closure, residual hold
correction, open approach and contact-limited closed-grasp transport. A whole
chunk that moves then closes is also rejected before execution. **76 tests PASS**;
scoped Ruff/format, mypy and compilation PASS. No new Astra trial is authorized
or performed by this change. Historical failed recordings remain unchanged;
new recorded Cartesian diagnosis now encounters this stricter production gate.

One full privileged scripted regression with the final gate passes: 176 waypoints,
65.053 mm minimum held lift, 214.513 mm transport, release, return-to-rest and
upstream/evaluator success. Evidence: the existing `mission4-tracking/` directory's
`sequence-guard-full-corrected/`. An earlier `sequence-guard-full/` run rejected
open descent because its interpolation started below commanded aperture 1; that
failed candidate is retained. The final gate checks actual closing relative to
both measured starting aperture and previous command, rather than treating all
below-command intermediate samples as new closure. A measured-0.9998/open-target
regression covers this defect. These were zero-provider offline runs, not a new
autonomous success claim.

## Live test of separate-closure contract

One operator-authorized `mission-5-separate-close` ran from
`fd809c8affce7719394902871552e6f53a386e1a` with the new closure contract and unchanged
model/provider, 12-call/12-decision/384-waypoint/600-second limits. It stopped
without retry: **FAIL / provider/API_error**, exact returned error:

```text
provider HTTP status 502: {"type": "upstream_error", "message": "Upstream service temporarily unavailable"}
```

Two requests were transmitted. Request 1 returned HTTP 200 in 39.403 seconds;
one open-gripper decision completed 36 waypoints, settle and fresh generation
1 -> 2. Request 2 finished sending at 0.041 seconds and received the 502 headers
at 52.764 seconds (52.766 seconds total transport). This is an actual provider
HTTP error, not a local timeout or Isaac execution failure. The relay's error
does not isolate whether the relay itself or its upstream service failed.

No closure, lift or transport occurred. Evaluator success=false, grasp_seen=false,
released=true, settled=true. The new sequencing rule's autonomous grasp effect
remains **NOT_VERIFIED**; the prior full scripted PASS is unchanged. Known usage:
3,163 input + 126 output = **3,289 tokens**; request 2 returned no usage metadata.
Episode time **103.809 seconds**, excluding Kit startup/preflight; model transport
92.169 seconds, render 1.122, validation 0.010, execute 3.688, settle 1.279 seconds.

Evidence is retained in `.agents/runs/xtrainer-astra/mission-5-separate-close/`,
including result, phase timings, recording, action/observation traces and terminal
images. **76 focused tests PASS**, credential-content and diff audits PASS.
No further model request, hardware access or push occurred.

## Subsequent authorized separate-closure mission

`mission-6-separate-close` ran once from `3be420422b1642c3876225124422d22844ab9f6e`
with unchanged provider/model/agent and limits (12 requests, 12 decisions,
384 waypoints, 600 s). Result: **FAIL / model_timeout**, exact error
`provider request timed out`. No automatic retry occurred.

Request 1 returned HTTP 200 after 55.450 s; its open-gripper move produced
31 completed waypoints, settle and fresh observation (generation 1 -> 2).
Request 2 used the established connection, finished sending its request body
at 0.042869 s, then failed while receiving response headers at 60.103233 s;
the recorded exception class is `ReadTimeout`. This localizes the failure to
waiting for HTTP response headers, not connection setup, TLS negotiation or
request upload. It does not isolate relay versus upstream service latency,
nor prove that a longer deadline would succeed. No timeout settings changed.

There were two transmitted requests and one policy decision. No closure, lift,
transport or release sequence was reached, so autonomous validation of the new
closure rule remains incomplete. Known usage: 3,163 input + 125 output = 3,288
tokens; timed-out request usage is unknown. Episode duration was 126.411 s
excluding Kit startup/preflight; model transport total 115.554 s. Render,
validation, execution and settle were 1.089 / 0.018 / 3.198 / 1.283 s.
Evaluator success=false, grasp_seen=false, closed_command_seen=false.

Primary result, metadata-only phase timings, HDF5, actions, observations and
failure images remain in `.agents/runs/xtrainer-astra/mission-6-separate-close/`.
76 focused tests PASS; credential-content audit and diff review PASS.
No physical hardware access or push; no further episode was launched.

## Historical live checkpoint

**Offline integration and live connectivity PASS; first autonomous mission FAIL.**

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
established fix. These were reported velocity fields, originally interpreted as
persistent motion. The later native-substep investigation above supersedes that
interpretation: reported velocity alone does not establish transform motion
under contact. This historical episode remains failed under its original gate.

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
  uses upstream PickCube's local wrist +Z160 mm reference convention, not a
  measured contact-pressure center or real TCP. The reference base is root-aligned
  at (1.06,0,0); the USD shoulder-joint anchor is another 223.4 mm above it.
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

## Comparison with the successful MuJoCo block-to-box episode

This is a retrospective analysis of the frozen MuJoCo evidence
`nova_mujoco/evidence/sim_s5b/20261011T040204Z-postfix-live/` and Isaac
`mission-1/`, not a controlled engine comparison. No new model request or
simulation episode was used for this analysis.

- **Different starting difficulty:** MuJoCo's initial grasp-center XY was
  (-0.30,-0.15) m, exactly the blue block's XY. Isaac's initial horizontal
  grasp-center-to-cube offset was 239.1 mm, calculated in world coordinates from
  recorded EEF, the right-base translation and the recorded robot root pose.
  The Isaac policy spent five decisions approaching before its first close.
- **Different object/contact models:** the frozen MuJoCo scene used a 60 mm,
  100 g primitive cube and simplified box-pad gripper (197 mm nominal grasp
  offset, 95 mm maximum opening). Read-only USD inspection gives the Isaac cube
  45 mm visual bounds and 10 g authored mass; its gripper is the upstream USD
  assembly, with the upstream 160 mm grasp convention. MuJoCo finger position
  drives have kp=2500 and an 8 N per-finger force range; Isaac's upstream
  implicit gripper configuration has stiffness=400, damping=40 and effort
  limit=50. These are different contact/control configurations, not evidence
  that one engine is inherently better. Contact force traces for the Isaac
  failure were not recorded, so actual applied force cannot be inferred from
  the configured limit.
- **Grasp-height hypothesis:** at the last pre-close observation, the Isaac
  nominal grasp center was world (0.510010,-0.000014,0.175007) m; the cube center
  was approximately (0.505529,-0.002233,0.120115) m. Horizontal difference was
  about 5.0 mm but vertical difference 54.9 mm. The last completed close waypoint
  tracked its commanded center within 0.444 mm (all completed waypoints under
  3 mm). Thus Cartesian tracking passed while the intended contact pose may
  have been inappropriate. The 160 mm reference point is not proof that the
  fingertips' usable contact area coincides with it. Height/contact geometry
  needs an isolated offline check; this evidence alone does not prove a frame
  bug or justify substituting a different TCP value.
- **Different continuation rule:** MuJoCo held each chunk for a bounded
  gripper-travel-derived duration plus 0.25 s and returned fresh observations;
  object `settled=false` did not automatically terminate intermediate decisions.
  That episode briefly lifted/dropped the block, reopened/reapproached and
  regrasped, then succeeded after 10 decisions/188 waypoints, with maximum lift
  143.8 mm and `dropped_observed=true`. Isaac requires ten consecutive stable
  robot/object ticks within 120 control steps after every chunk. Persistent
  motion on the first close triggered an abort before a fresh failure-state
  observation/replan. This does not prove that allowing continuation would
  have succeeded, and is not permission to weaken safety or final evaluation.
- **Different success definitions:** MuJoCo tests geometric inclusion in the box,
  release and settling (including distractor checks). Upstream Isaac tests a
  plate-relative region and return to rest; the adapter additionally requires
  release/settling. This difference did not cause the observed early failure:
  Isaac never reached lift or placement. Upstream `object_grasped` is a
  center-distance/joint-position heuristic, not a contact-force detector.
- **Visual and statistical limits:** both expose three RGBs and EEF, but camera
  poses, lighting, backgrounds, object scale and gripper appearance differ.
  Wrist images show successive visual corrections, not a total failure to
  locate the cube. One successful MuJoCo episode and one failed Isaac episode
  cannot estimate relative success rates or establish identical upstream model
  weights through a relay.

The bounded zero-provider investigation below now executes that proposal.
Historical evidence, physics parameters and evaluator rules remain unchanged.
A stable scripted grasp/lift remains a prerequisite to a new autonomous trial.


## Offline grasp diagnosis — stable closure BLOCKED

Analysis baseline: `42c12d7990deb3494f5fe0c2a65d67b6b441bb58`.
Local evidence is in `.agents/runs/xtrainer-astra/grasp-diagnosis/`, with one
`comparison.json`, per-case collision geometry, compact physics/action traces,
terminal diagnostic images, and upstream HDF5 recordings. The original
`mission-1` input is read-only; its recording SHA256 remains
`5e9da2dc4b00464358256c006402ff46dff6314af8212c8fd1e4fbc06c23b365`.

**What was established:**

- The upstream HDF5 reader and a single `reset_to(initial_state)` reproduce all
  636 recorded 17-dimensional controls. There are no per-frame state writes.
  Separately, the six recorded Cartesian destinations through the official
  `move_to` interpolator and existing adapter reproduce the same failure.
  Controls, robot joint positions/velocities, and cube poses/velocities are
  element-for-element equal to `mission-1` in both replays (maximum difference 0).
  The failure is not a replay-interface discrepancy.
- An independent world-flange/quaternion/+160 mm reconstruction agrees with
  the adapter's measured reference position to within 9.13e-8 m over the traces.
  No coordinate-conversion defect was established. The old comment incorrectly
  called (1.06,0,0) the shoulder anchor; it is the root-aligned pedestal reference.
  This comment is corrected without changing the public frame or any target.
- At original closure, the reference is about 54.9 mm above the cube center.
  The distal collision tips start around world Z=138.0 mm, overlapping only
  about 4.7 mm of the cube's upper edge. Collision vertices identify the narrow
  distal finger region as wrist Z=168.292..197.000 mm; its geometric midpoint is
  182.646 mm. This is a diagnostic target construction, **not a TCP replacement**
  or a measured center of contact pressure. The public +160 mm offset stays fixed.
- Aligning that region to the settled cube gives reference base Z=42.761 mm,
  with candidates at 37.761/47.761 mm. Original approach XY, orientation, closure
  endpoint and all physical settings remain fixed. Collision-mesh table
  clearance predictions are positive (10.761/15.761/20.761 mm). These are mesh
  surface clearances, not calibrated PhysX contact margins: the settled cube
  bottom is about 7.5 mm above the table mesh, so actual contact/cooking behavior
  remains a limitation of this measurement.
- Original closure advances five interpolated commands in five control ticks
  (about 0.167 s). At command zero, center-candidate measured aperture is still
  0.5585. This command/readback lag justified exactly one comparison with two
  ticks per closing waypoint (about 0.333 s), the same endpoint and unchanged
  settle check. It also fails; slower closure is not adopted as a live fix.

All statistics below cover the final 120 settle control steps. Existing limits
remain joint speed <0.02 rad/s, cube linear speed <0.01 m/s and angular speed
<0.1 rad/s for ten consecutive steps.

| Case | Reference base Z (mm) | Cube speed (m/s), min–max | Angular speed (rad/s), min–max | Stable ticks |
| --- | ---: | ---: | ---: | ---: |
| Recorded controls / Cartesian replay | 75.000 | 0.0511–0.1598 | 2.480–10.049 | 0/120 |
| Collision-derived center | 42.761 | 0.2233–0.2327 | 6.137–7.593 | 0/120 |
| Center −5 mm | 37.761 | 0.1152–0.1319 | 11.857–12.889 | 0/120 |
| Center +5 mm | 47.761 | 0.1456–0.3435 | 3.847–14.399 | 0/120 |
| Center, closure playback ×2 | 42.761 | 0.3264–0.3341 | 11.411–11.793 | 0/120 |

Every comparison stops at the first `settle_timeout` during closure. The
upstream distance/joint heuristic becomes `grasped_now=true` in the height
candidates, but **that does not demonstrate a physical grasp/lift**. Maximum
incidental cube rise is only 0.214 mm. No lift, hold, transport, release/rest
acceptance episode is run, because no candidate passes its unchanged prerequisite.
The unexecuted full-task prototype is not shipped as a validated runner mode.

**Unresolved cause:** persistent contact-state motion is measured, and changing
height or closure ramp alone is insufficient. The exact source among cooked
colliders, contact response and drive/solver dynamics is not isolated. PhysX
reports that `J2_8` is authored as both a merged and standalone collision; that
is a relevant warning, not proof of causation. Contact-pair identities and forces
are explicitly `unavailable` in the unchanged asset reporting configuration.
No collider, mass, friction, gain, solver, stability threshold or success rule
was changed. Investigation stops at the agreed boundary, rather than guessing a
physics fix or bypassing settle. Full scripted pick/place remains **NOT_RUN**.

There were eight bounded launches, including two harness failures before any
approach (missing optional HDF5 seed, and an Isaac 2.1 math API name mismatch).
Both harness defects were fixed. Six comparative episodes completed: two
reproductions, three heights and one slower closure. The raw replay then hit a
NumPy-integer JSON serialization error after recording; its complete HDF5,
physics trace and terminal images survive and were compared directly, without
re-running that episode. Serialization now has a focused regression test.
Each launch used an external 180 s timeout; completed Cartesian/candidate runner
times were 31.99–32.79 s excluding Kit startup. No ninth launch was made.

### Diagnostic entry point

The same Astra runner has a credential-free `diagnose` branch. Its container
uses `--network none`, no credential mount, and GPU devices only. It constructs
no provider client. `DIAGNOSE_COMPLETE` means the experiment was recorded, **not**
that grasping succeeded: inspect `diagnosis.outcome` and `diagnosis.failure`.
The Cartesian/height cases specifically reconstruct the frozen `mission-1`
six-destination approach; they are not a generic planner. Paths must be unique.

```bash
# Run from this component; these are future offline invocations, not permission
# to exceed the completed investigation's eight-launch limit.
timeout -k 10 180 scripts/astra/container.sh diagnose \
  --diagnostic-source .agents/runs/xtrainer-astra/mission-1 \
  --diagnostic-case raw \
  --output .agents/runs/xtrainer-astra/new-diagnosis/raw
# Other cases: --diagnostic-case cartesian
# or --diagnostic-case height --height-shift {-0.005,0,0.005}
# The single timing comparison additionally uses --slow-close.
```

Two production evidence defects are fixed: failed settle time is accumulated
in `finally`, and failure images are rendered from the stopped terminal state.
Diagnostic captures record simulation time, control ticks, sensor counters and
last policy generation; they neither step physics nor create a policy observation.
Privileged object/geometry data remains solely in diagnostic traces.

Validation: **46 focused adapter tests PASS** (32 existing plus 14 new, with
existing observation-isolation coverage extended), scoped Ruff/format, mypy,
compilation and shell syntax PASS. The original live/provider behavior is
unchanged. No MuJoCo or upstream-wide regression suite was rerun; no credentials
were loaded, no provider calls/hardware connections were made, and nothing was
pushed. A fresh paid Astra trial is not authorized by these offline results.

### Historical contact-force follow-up (then blocked)

The operator subsequently requested fixing the grasp before a single further
Astra trial. That trial remains conditional on two complete scripted pick/place
acceptances; none has passed, and no new provider request has been made.

Four additional bounded offline episodes used the same initial recording and
approach, with contact reporting enabled only for diagnosis. The reporting-only
control exactly reproduces all 651 earlier center-candidate controls, joint
positions/velocities and cube states (maximum absolute difference zero).
Original USD files and all physical parameters remain unchanged.

PhysX contact/rest offsets are 5.45/0 mm. The cube and fingers contact with
millimetres of reported penetration under the fully closed position target.
Reducing commanded closure reduces measured contact load, but does not meet
the existing stability criterion. The last comparison also centers XY on the
measured cube, removing the original approximately 5 mm horizontal offset.

Statistics below cover the last 120 **physics substeps**, not control steps.
Forces are the mean sum of absolute contact-normal forces per finger; they are
not net force, calibrated force sensing, or a new commanded-force interface.

| Aperture command | XY target | Finger forces (N) | Cube linear speed (m/s) | Angular speed (rad/s) | Stable substeps |
| ---: | --- | --- | --- | --- | --- |
| 0 | Original | 7.439 / 7.456 | 0.23003–0.23016 | 6.231–6.246 | 0/120 |
| 0.5 | Original | 1.604 / 1.581 | 0.04997–0.05006 | 5.224–5.255 | 0/120 |
| 0.625 | Original | 0.176 / 0.195 | 0.01165–0.01167 | 0.3765–0.3767 | 0/120 |
| 0.625 | Cube-centered | 0.173 / 0.199 | 0.00756–0.00758 | 0.3211–0.3219 | 0/120 |

The first case reports approximately 3.78/3.15 mm maximum finger penetration.
At its last 120 substeps, position-derived speed is only 1.05e-6–2.32e-5 m/s,
while PhysX reports approximately 0.230 m/s. Thus the discrepancy persists at
the native physics sampling interval; merely sampling twice per control step
does not resolve it. Neither pose differences nor the upstream grasp heuristic
are substituted for the unchanged velocity checks.

The USD inspection confirms that the `J2_8/collisions` merged collider and its
child mesh both declare CollisionAPI; native contact traces contain repeated
contact locations for this finger. **Causation is not yet established.** An
isolated control disabling the redundant standalone collision declaration needs
explicit authorization because it changes the protected upstream collision
configuration. No collider change or live adapter workaround has been adopted.
All four episodes stop at `settle_timeout`; no complete lift/hold/transport/
release/rest acceptance or Astra trial follows.

Evidence remains in the local task area:
`.agents/runs/xtrainer-astra/grasp-diagnosis/contact-followup/`, including one
compact `comparison.json`, contact properties, bounded contact/substep samples,
normal physics recordings and unchanged-state terminal images. The original
`mission-1` recording hash is unchanged. Raw traces are not published.

The diagnostic runner additionally supports `--diagnostic-case contact`,
`--diagnostic-aperture` (0..1), and `--center-cube`. Contact reporting is enabled
only in that offline case; it cannot enter the model observation. It retains at
most 256 closing substeps and restores the normal scene update function on exit.
These flags are investigative controls, not a validated grasp controller.
The four measurements used Isaac's contact-activation helper. Source review
then found it also authors a sleep-threshold attribute on the passed mount prim
(a non-rigid parent here). The shipped diagnostic spawner instead adds only
ContactReportAPI and its reporting threshold; it does not write sleep settings.
The recorded trajectory-equivalence check applies to the measured helper path.

Follow-up validation: **52 focused tests PASS**, scoped Ruff/format, mypy
(five own source modules), compilation and shell syntax PASS. A native USD
spawn comparison covers 29 rigid bodies with zero physics steps: only
ContactReportAPI/reporting-threshold additions differ; no sleep or collision
properties change. This validates instrumentation, not a successful grasp.
