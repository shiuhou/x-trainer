#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${1:-smoke}
shift || true
IMAGE=xtrainer-astra:isaac-2.1.0
# Frozen sources are prepared from Git archive by setup; never mount dirty references.
RUNTIME_DIR="$ROOT_DIR/.agents/runs/xtrainer-astra/runtime"
ARGS=(--rm --gpus all --shm-size=8g --cap-drop=ALL --cap-add=DAC_OVERRIDE --security-opt=no-new-privileges
  --ulimit core=0 --env PYTHONUNBUFFERED=1
  --volume xtrainer-astra-kit-cache:/isaac-sim/kit/cache
  --volume xtrainer-astra-nvidia-cache:/root/.cache/nvidia
  --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y
  --mount "type=bind,source=$ROOT_DIR,target=/workspace/x-trainer"
  --mount "type=bind,source=$RUNTIME_DIR/inspect-robots,target=/opt/inspect-robots,readonly"
  --mount "type=bind,source=$RUNTIME_DIR/s5,target=/opt/shared-s5,readonly"
  --workdir /workspace/x-trainer)
LIVE_ARGS=()
if [[ "$MODE" == connect || "$MODE" == mission ]]; then
  : "${ASTRA_CREDENTIAL_FILE:?Set ASTRA_CREDENTIAL_FILE to the operator credential file}"
  ARGS+=(--network host --mount "type=bind,source=$ASTRA_CREDENTIAL_FILE,target=/run/astra-relay.env,readonly")
  LIVE_ARGS=(--provider-runtime /opt/shared-s5 --credential-file /run/astra-relay.env)
fi
if [[ "$MODE" == keyboard ]]; then
  : "${DISPLAY:?A local graphical display is required}"
  ARGS+=(--env DISPLAY --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix,readonly)
  if [[ -n "${XAUTHORITY:-}" ]]; then
    ARGS+=(--env XAUTHORITY=/run/xauthority --mount "type=bind,source=$XAUTHORITY,target=/run/xauthority,readonly")
  fi
  exec docker run "${ARGS[@]}" --entrypoint /isaac-sim/python.sh "$IMAGE" \
    scripts/environments/teleoperation/teleop_se3_agent.py --task LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device bi_keyboard --num_envs 1 --device cuda --enable_cameras --multi_view "$@"
fi
docker run "${ARGS[@]}" --entrypoint /isaac-sim/python.sh "$IMAGE" \
  scripts/astra/run.py --mode "$MODE" --headless --enable_cameras "${LIVE_ARGS[@]}" "$@"
# Kit fast shutdown can return zero after a Python failure. Require the actual
# experiment result, which must exist and pass; never mistake process exit for PASS.
python3 - "$ROOT_DIR" "$MODE" "$@" <<'PY'
import argparse, json, pathlib, sys
root, mode, *args = sys.argv[1:]
parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parsed, _ = parser.parse_known_args(args)
result = json.loads((pathlib.Path(root) / parsed.output / 'result.json').read_text())
expected = {'original': 'ORIGINAL_BOOT_PASS', 'smoke': 'SMOKE_PASS', 'connect': 'CONNECT_PASS', 'mission': 'MISSION_PASS'}
print(result['status'])
raise SystemExit(0 if result['status'] == expected[mode] else 1)
PY
