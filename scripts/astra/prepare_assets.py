"""Fetch only the original X-Trainer robot and PickCube scene, pinned by revision."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REPOSITORY = "dstx123/xtrainer-leisaac"
REVISION = "fb140bf9d4bf7de8b71818df59b93a1cb4c0564a"
ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    url = f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    entries = []
    # Verified complete robot layers from the official Dobot dataset. The scene
    # repository robot snapshot has unresolved collider references; do not mix it.
    robot_revision = "02f008252001f654e28dee40d615c713ff79935a"
    robot_source = ROOT.parents[1] / "x-trainer/assets/robots"
    if not robot_source.is_dir():
        raise RuntimeError(
            "Local official robot assets unavailable; supply the pinned Dobot dataset"
        )
    for source in sorted(robot_source.rglob("*.usd")):
        relative = source.relative_to(robot_source)
        if any(part.startswith(".") for part in relative.parts):
            continue
        dest = ROOT / "assets/robots" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        entries.append(
            {
                "path": dest.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "repository": "Dobot-robot/X-trainer",
                "revision": robot_revision,
            }
        )
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if not name.startswith("assets/scenes/table_with_cube/"):
            continue
        if any(part.startswith(".") for part in Path(name).parts):
            continue
        dest = ROOT / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            with (
                urllib.request.urlopen(
                    f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}",
                    timeout=120,
                ) as source,
                dest.with_suffix(dest.suffix + ".part").open("wb") as target,
            ):
                shutil.copyfileobj(source, target)
            dest.with_suffix(dest.suffix + ".part").rename(dest)
        entries.append(
            {"path": name, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest()}
        )
        print(name, flush=True)
    if not (ROOT / "assets/scenes/table_with_cube/scene.usd").is_file():
        raise RuntimeError("Pinned upstream PickCube scene is missing")
    (ROOT / "assets/scenes/astra_asset_manifest.json").write_text(
        json.dumps(
            {"repository": REPOSITORY, "revision": REVISION, "files": entries}, indent=2
        )
    )


if __name__ == "__main__":
    main()
