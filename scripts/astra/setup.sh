#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE_DIR=$(cd "$ROOT_DIR/../.." && pwd)
RUNTIME_DIR="$ROOT_DIR/.agents/runs/xtrainer-astra/runtime"
mkdir -p "$RUNTIME_DIR/inspect-robots" "$RUNTIME_DIR/s5"
# Existing reviewed local sources, exported without dirty files or credentials.
git -C "$WORKSPACE_DIR/robocurve/references/inspect-robots" archive 7e506e3b6f39530f18e75cf97b23c4287aa90f12 | tar -x -C "$RUNTIME_DIR/inspect-robots"
git -C "$WORKSPACE_DIR/nova_mujoco" archive 32efb430d17c23e28e068b7751afda48509ab885 s5_agent.py s5_common.py configs/s5_official_stack.json | tar -x -C "$RUNTIME_DIR/s5"
python "$ROOT_DIR/scripts/astra/prepare_assets.py"
IMAGE=$(cat "$ROOT_DIR/docker/isaac-image.lock")
docker pull "$IMAGE"
docker build -f "$ROOT_DIR/docker/Dockerfile.astra" --build-arg "ISAAC_IMAGE=$IMAGE" -t xtrainer-astra:isaac-2.1.0 "$RUNTIME_DIR"
